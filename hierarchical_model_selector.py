import json
import torch
from tqdm import tqdm
import os
import numpy as np

def load_configs_from_json(filepath):
    """
    Loads model configurations from the specified JSON file.
    """
    print(f"Loading configurations from {filepath}...")
    try:
        with open(filepath, 'r') as f:
            configs = json.load(f)
        print(f"Loaded {len(configs)} configurations.")
        return configs
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return []

def get_stage_from_exit_location(exit_loc):
    """
    Determines the stage index based on the early exit location for a ResNet-110 like architecture.
    Stage boundaries are after 18, 36, 54 blocks.
    """
    # For resnet110, layers are [18, 18, 18]. Total 54 blocks.
    # Stage 0: blocks 0-17
    # Stage 1: blocks 18-35
    # Stage 2: blocks 36-53
    if exit_loc <= 18:
        return 0
    elif exit_loc <= 36:
        return 1
    else:
        return 2

def is_sub_model(config_sub, config_super):
    """
    Checks if config_sub represents a sub-model of config_super according to the strict
    hierarchical constraints. The logic is corrected to include the exit stage in the identity check.
    """
    # A sub-model cannot exit later than its super-model.
    if config_sub['early_exit_location'] > config_super['early_exit_location']:
        return False

    sub_exit_stage = get_stage_from_exit_location(config_sub['early_exit_location'])

    # For all stages up to and including the sub-model's exit stage, widths must be identical.
    # This was the location of the off-by-one bug.
    for i in range(sub_exit_stage + 1):
        if config_sub['width_multipliers'][i] != config_super['width_multipliers'][i]:
            return False

    # For all stages AFTER the sub-model's exit stage, the sub-model's width
    # must be less than or equal to the super-model's width.
    for i in range(sub_exit_stage + 1, len(config_sub['width_multipliers'])):
        if config_sub['width_multipliers'][i] > config_super['width_multipliers'][i]:
            return False

    return True

def find_best_config_for_distribution(all_configs, flops_constraints, participating_levels, beam_width=100):
    """
    Finds the best hierarchical configuration for a given distribution of client levels.

    Args:
        all_configs (list): A list of all possible model configurations from the JSON file.
        flops_constraints (dict): A dictionary mapping each level (int) to its FLOPs constraint (float).
        participating_levels (list): A sorted list of unique level indices that are active in the current round.
        beam_width (int): The width of the beam for the search.

    Returns:
        dict: A dictionary mapping each participating level to its chosen best model configuration, 
              or None if no valid hierarchy is found.
    """
    if not all_configs:
        print("Configuration list is empty. Cannot find a hierarchy.")
        return None
    if not participating_levels:
        print("Participating levels list is empty. No clients to configure.")
        return None
    
    # 对参与等级进行排序，确保我们从最低等级开始构建层级
    active_levels = sorted(list(set(participating_levels)))
    print(f"Searching for hierarchy for active levels: {active_levels}")

    print("Filtering and sorting candidates for each active level...")
    candidates_by_level = {}
    for level in active_levels:
        if level not in flops_constraints:
            print(f"Error: FLOPs constraint for Level {level} is not defined.")
            return None
        
        level_candidates = [c for c in all_configs if c['flops_m'] <= flops_constraints[level]]
        if not level_candidates:
            print(f"Error: No valid candidates found for Level {level} with FLOPs constraint {flops_constraints[level]} M.")
            return None
        
        # 核心优化：按核范数降序排序，这样我们总是优先尝试“最好”的候选模型
        level_candidates.sort(key=lambda x: x['total_conv_nuclear_norm'], reverse=True)
        candidates_by_level[level] = level_candidates
        print(f"Level {level} has {len(level_candidates)} valid candidates.")

    # --- Beam Search Start ---
    # 初始化束：从最低参与等级的 top `beam_width` 个候选模型开始
    start_level = active_levels[0]
    # 束中的每个元素是一个部分层级，现在用字典表示，更清晰: {level: config}
    beam = [{start_level: cand} for cand in candidates_by_level[start_level][:beam_width]]

    # 动态地、逐个参与等级地构建层级
    for i in range(1, len(active_levels)):
        current_level = active_levels[i]
        prev_level = active_levels[i-1]
        new_beam = []

        for partial_hierarchy in tqdm(beam, desc=f"Extending from L{prev_level} to L{current_level}"):
            # 父模型是当前部分层级中，等级最高的那个模型
            parent_model = partial_hierarchy[prev_level]
            
            found_count = 0
            # 在当前等级的候选者中寻找兼容的“超模型”
            for candidate in candidates_by_level[current_level]:
                # is_sub_model(parent, child) -> parent 是 child 的子模型
                if is_sub_model(parent_model, candidate):
                    # 复制并扩展当前的部分层级
                    new_partial_hierarchy = partial_hierarchy.copy()
                    new_partial_hierarchy[current_level] = candidate
                    new_beam.append(new_partial_hierarchy)
                    
                    found_count += 1
                    if found_count >= beam_width:  # 剪枝：为每个父节点只找 beam_width 个最佳子节点
                        break
        
        if not new_beam:
            print(f"\nCould not extend ANY hierarchy to Level {current_level}. Search failed.")
            return None

        # 根据整个部分层级的总核范数来对新束进行排序
        new_beam.sort(key=lambda h: sum(m['total_conv_nuclear_norm'] for m in h.values()), reverse=True)
        
        # 修剪束，只保留最优的 `beam_width` 个
        beam = new_beam[:beam_width]

    if not beam:
        print("\nNo valid hierarchical configuration could be found.")
        return None

    # 最终，最优的层级就是束中的第一个元素
    best_hierarchy_dict = beam[0]
    print("\nFound best hierarchical configuration!")
    
    # 返回的字典只包含参与等级的配置
    return best_hierarchy_dict
