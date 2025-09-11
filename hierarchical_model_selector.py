import json
import torch
from tqdm import tqdm
import os
import numpy as np
import hashlib
import pickle

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

def get_actual_channels(width_multipliers):
    """
    计算ResNet各stage的实际channel数量
    基础channels: [16, 32, 64]
    """
    base_channels = [16, 32, 64]
    return [int(base_channels[i] * width_multipliers[i]) for i in range(len(width_multipliers))]

def is_sub_model(config_sub, config_super):
    """
    检查config_sub是否是config_super的子模型
    使用实际channel数量比较，而不是width比例的精确相等
    """
    # 子模型不能比父模型退出得更晚
    if config_sub['early_exit_location'] > config_super['early_exit_location']:
        return False

    sub_exit_stage = get_stage_from_exit_location(config_sub['early_exit_location'])
    
    # 获取实际的channel数量
    sub_channels = get_actual_channels(config_sub['width_multipliers'])
    super_channels = get_actual_channels(config_super['width_multipliers'])

    # 对于子模型退出stage及之前的所有stage，实际channel数量必须相同
    for i in range(sub_exit_stage + 1):
        if sub_channels[i] != super_channels[i]:
            return False

    # 退出stage之后的stage不需要任何约束，因为子模型不会执行到那里

    return True


def generate_cache_key(participating_levels, beam_width, config_hash, flops_constraints=None, params_constraints=None):
    """Generate unique cache key based on search parameters"""
    # Create a deterministic string representation
    levels_str = "_".join(map(str, sorted(participating_levels)))
    
    # Add constraints if provided
    constraint_parts = []
    if flops_constraints:
        flops_str = "_".join([f"{k}:{v}f" for k, v in sorted(flops_constraints.items())])
        constraint_parts.append(flops_str)
    if params_constraints:
        params_str = "_".join([f"{k}:{v}p" for k, v in sorted(params_constraints.items())])
        constraint_parts.append(params_str)
    
    cache_str = f"{levels_str}_{'_'.join(constraint_parts)}_{beam_width}_{config_hash}"
    
    # Create hash for filename
    return hashlib.md5(cache_str.encode()).hexdigest()


def get_config_hash(all_configs):
    """Generate hash of configuration library for cache validation"""
    config_str = json.dumps(all_configs, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def load_cached_result(cache_file):
    """Load cached hierarchical selection result"""
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            print(f"✓ Loaded cached hierarchical result from: {cache_file}")
            return cached_data['result']
        return None
    except Exception as e:
        print(f"⚠ Failed to load cache {cache_file}: {e}")
        return None


def save_cached_result(cache_file, result, flops_constraints, participating_levels, params_constraints=None):
    """Save hierarchical selection result to cache"""
    try:
        os.makedirs('cache/hierarchical', exist_ok=True)
        cache_data = {
            'result': result,
            'flops_constraints': flops_constraints,
            'participating_levels': participating_levels,
            'params_constraints': params_constraints,
            'timestamp': os.path.getmtime(__file__)
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"✓ Saved hierarchical result to cache: {cache_file}")
    except Exception as e:
        print(f"⚠ Failed to save cache {cache_file}: {e}")

def find_best_config_for_distribution(all_configs, participating_levels, beam_width=500, flops_constraints=None, params_constraints=None):
    """
    Finds the best hierarchical configuration for a given distribution of client levels.
    Results are cached to avoid repeated computation.

    Args:
        all_configs (list): A list of all possible model configurations from the JSON file.
        participating_levels (list): A sorted list of unique level indices that are active in the current round.
        beam_width (int): The width of the beam for the search.
        flops_constraints (dict): Optional dictionary mapping each level (int) to its FLOPs constraint (float).
        params_constraints (dict): Optional dictionary mapping each level to its parameter constraint (float, in M).
        
    Note:
        At least one of flops_constraints or params_constraints must be provided.

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
    
    # Validate that at least one constraint is provided
    if not flops_constraints and not params_constraints:
        print("Error: At least one of flops_constraints or params_constraints must be provided.")
        return None
    
    # Generate cache key based on search parameters
    config_hash = get_config_hash(all_configs)
    cache_key = generate_cache_key(participating_levels, beam_width, config_hash, flops_constraints, params_constraints)
    cache_file = f"cache/hierarchical/hierarchy_{cache_key}.pkl"
    
    # Try to load cached result
    cached_result = load_cached_result(cache_file)
    if cached_result is not None:
        return cached_result
    
    print(f"Computing hierarchical configuration (will be cached as {cache_key[:8]}...)")
    
    # 对参与等级进行排序，确保我们从最低等级开始构建层级
    active_levels = sorted(list(set(participating_levels)))
    print(f"Searching for hierarchy for active levels: {active_levels}")

    print("Filtering and sorting candidates for each active level...")
    candidates_by_level = {}
    for level in active_levels:
        # Start with all configurations
        level_candidates = all_configs.copy()
        constraint_parts = []
        
        # Apply FLOPs constraint if provided
        if flops_constraints and level in flops_constraints:
            flops_limit = flops_constraints[level]
            level_candidates = [c for c in level_candidates if c['flops_m'] <= flops_limit]
            constraint_parts.append(f"FLOPs <= {flops_limit}M")
        
        # Apply params constraint if provided
        if params_constraints and level in params_constraints:
            params_limit = params_constraints[level]
            level_candidates = [c for c in level_candidates if c['num_params']/1e6 <= params_limit]
            constraint_parts.append(f"Params <= {params_limit}M")
        
        # Check if we have valid candidates
        if not level_candidates:
            constraint_msg = ", ".join(constraint_parts) if constraint_parts else "No constraints defined"
            print(f"Error: No valid candidates found for Level {level} with constraints: {constraint_msg}")
            return None
        
        # Validate that level has at least one constraint
        has_flops_constraint = flops_constraints and level in flops_constraints
        has_params_constraint = params_constraints and level in params_constraints
        if not has_flops_constraint and not has_params_constraint:
            print(f"Error: Level {level} has no constraints defined.")
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
    
    # 保存结果到缓存
    save_cached_result(cache_file, best_hierarchy_dict, flops_constraints, participating_levels, params_constraints)
    
    # 返回的字典只包含参与等级的配置
    return best_hierarchy_dict
