import json
import torch
from tqdm import tqdm
import os
import numpy as np
import hashlib
import pickle

def load_configs_from_json(filepath, model_type=None, dataset=None):
    """
    Loads model configurations from the specified JSON file with validation.

    Args:
        filepath: Path to the configuration JSON file
        model_type: Expected model type (e.g., 'resnet', 'vgg'). If provided, validates compatibility.
        dataset: Expected dataset. If provided, validates compatibility.
    """
    print(f"Loading configurations from {filepath}...")
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Handle both old format (direct list) and new format (with metadata)
        if isinstance(data, list):
            # Old format: direct list of configurations
            print(f"⚠ Warning: Using legacy config format without metadata")
            configs = data
        elif isinstance(data, dict) and "configurations" in data:
            # New format: with metadata
            metadata = data.get("metadata", {})
            configs = data["configurations"]

            # Validate model type compatibility
            if model_type and metadata.get("model_type"):
                file_model_type = metadata["model_type"]
                if file_model_type != model_type:
                    print(f"Warning: Config file is for {file_model_type} but loading for {model_type}")
                    print(f"This may cause incompatibility issues!")

            # Validate dataset compatibility
            if dataset and metadata.get("dataset"):
                file_dataset = metadata["dataset"]
                if file_dataset != dataset:
                    print(f"Warning: Config file is for {file_dataset} but loading for {dataset}")
                    print(f"This may cause incompatibility issues!")

            # Print metadata info
        else:
            print(f"Error: Invalid configuration file format")
            return []

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

def get_vgg_stage_from_exit_location(exit_loc):
    """
    Determines the stage index based on early exit location for VGG-D architecture.
    VGG-D stages: [64,64], [128,128], [256,256,256], [512,512,512], [512,512,512]
    """
    if exit_loc <= 2:    # Stage 0: layers 0-1 (64,64)
        return 0
    elif exit_loc <= 4:  # Stage 1: layers 2-3 (128,128)
        return 1
    elif exit_loc <= 7:  # Stage 2: layers 4-6 (256,256,256)
        return 2
    elif exit_loc <= 10: # Stage 3: layers 7-9 (512,512,512)
        return 3
    else:                # Stage 4: layers 10-12 (512,512,512)
        return 4

def vgg_layers_to_stages(layer_multipliers):
    """
    Convert 15 layer multipliers to 6 stage multipliers for VGG comparison
    VGG-D structure: [64,64], [128,128], [256,256,256], [512,512,512], [512,512,512] + [4096,4096]
    """
    if len(layer_multipliers) != 15:
        return layer_multipliers  # Not VGG format

    return [
        layer_multipliers[0],  # Stage 0: 64,64 (use first layer)
        layer_multipliers[2],  # Stage 1: 128,128 (use first layer)
        layer_multipliers[4],  # Stage 2: 256,256,256 (use first layer)
        layer_multipliers[7],  # Stage 3: 512,512,512 (use first layer)
        layer_multipliers[10], # Stage 4: 512,512,512 (use first layer)
        layer_multipliers[13]  # Stage 5: FC,FC (use first FC layer)
    ]

def get_actual_channels(width_multipliers, model_type="resnet"):
    """
    计算模型各层/stage的实际channel数量
    """
    if model_type == "resnet":
        # ResNet基础channels: [16, 32, 64] for 3 stages
        base_channels = [16, 32, 64]
        return [int(base_channels[i] * width_multipliers[i]) for i in range(len(width_multipliers))]
    elif model_type == "vgg":
        if len(width_multipliers) == 6:
            # VGG 6个stage的基础channels: [64, 128, 256, 512, 512, 4096]
            # Stage 0: 64, Stage 1: 128, Stage 2: 256, Stage 3: 512, Stage 4: 512, Stage 5: 4096 (FC)
            base_channels = [64, 128, 256, 512, 512, 4096]
        else:
            # VGG-D基础channels: 13 conv + 2 fc layers
            # VGG-D: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
            base_channels = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 4096, 4096]
        return [int(base_channels[i] * width_multipliers[i]) for i in range(len(width_multipliers))]
    elif model_type == "mobilenet":
        # MobileNetV2 8 stages base channels: [32, 16, 24, 32, 64, 96, 160, 320]
        base_channels = [32, 16, 24, 32, 64, 96, 160, 320]
        return [int(base_channels[i] * width_multipliers[i]) for i in range(len(width_multipliers))]
    else:
        # Fallback to resnet for unknown model types
        base_channels = [16, 32, 64]
        return [int(base_channels[i] * width_multipliers[i]) for i in range(min(len(width_multipliers), len(base_channels)))]

def is_sub_model(config_sub, config_super):
    """
    检查config_sub是否是config_super的子模型
    使用实际channel数量比较，而不是width比例的精确相等
    """
    # 子模型不能比父模型退出得更晚
    if config_sub['early_exit_location'] > config_super['early_exit_location']:
        return False

    # 获取实际的channel数量 - 从配置中推断模型类型
    # 检查是否包含VGG特有的字段
    if 'exit_stage' in config_sub or len(config_sub['width_multipliers']) == 15:
        model_type = "vgg"
    elif len(config_sub['width_multipliers']) == 8:
        model_type = "mobilenet"
    elif len(config_sub['width_multipliers']) == 3:
        model_type = "resnet"
    else:
        # 默认根据长度推断
        model_type = "resnet"

    sub_channels = get_actual_channels(config_sub['width_multipliers'], model_type)
    super_channels = get_actual_channels(config_super['width_multipliers'], model_type)

    if model_type == "vgg":
        # 对于VGG: 使用early_exit_location直接比较，不需要stage概念
        sub_exit_location = config_sub['early_exit_location']
        super_exit_location = config_super['early_exit_location']

        # 子模型的退出位置不能比父模型晚
        if sub_exit_location > super_exit_location:
            return False

        # 退出位置及之前的所有stage，channels必须完全匹配
        # 确定需要比较的stage数量（基于较早的退出位置）
        if 'exit_stage' in config_sub:
            # 6个stage格式：比较到退出位置对应的stage
            sub_exit_stage = config_sub['exit_stage']
            compare_stages = sub_exit_stage + 1
        else:
            # 15个layer格式：转换为stage进行比较
            sub_exit_stage = get_vgg_stage_from_exit_location(sub_exit_location)
            compare_stages = sub_exit_stage + 1

        # 比较对应stage的channels
        for i in range(compare_stages):
            if sub_channels[i] != super_channels[i]:
                return False
    elif model_type == "mobilenet":
        # 对于MobileNet: 直接比较所有 stages 的 channels
        # MobileNet 的 early_exit_location 对应到具体的 block
        # 所有 8 个 stages 的 channels 都要匹配
        for i in range(len(sub_channels)):
            if sub_channels[i] != super_channels[i]:
                return False
    else:
        # 对于ResNet: 按stage比较
        sub_exit_stage = get_stage_from_exit_location(config_sub['early_exit_location'])
        for i in range(sub_exit_stage + 1):
            if sub_channels[i] != super_channels[i]:
                return False

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
            print(f"[CACHE] Loaded cached hierarchical result from: {cache_file}")
            return cached_data['result']
        return None
    except Exception as e:
        print(f"[WARNING] Failed to load cache {cache_file}: {e}")
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
        print(f"[OK] Saved hierarchical result to cache: {cache_file}")
    except Exception as e:
        print(f"[WARNING] Failed to save cache {cache_file}: {e}")

def preprocess_vgg_configs(configs):
    """
    将VGG配置从15个layer multipliers折叠为6个stage multipliers
    同时映射exit_location到对应的stage
    """
    processed_configs = []
    for config in configs:
        if len(config['width_multipliers']) == 15:
            # 折叠为6个stage multipliers
            stage_multipliers = vgg_layers_to_stages(config['width_multipliers'])
            # 映射exit_location到stage
            exit_stage = get_vgg_stage_from_exit_location(config['early_exit_location'])

            processed_config = config.copy()
            processed_config['width_multipliers'] = stage_multipliers
            processed_config['exit_stage'] = exit_stage
            processed_config['original_early_exit_location'] = config['early_exit_location']
            processed_configs.append(processed_config)
        else:
            processed_configs.append(config)
    return processed_configs

def postprocess_vgg_results(result):
    """
    将VGG搜索结果从6个stage multipliers还原为15个layer multipliers
    """
    if result is None:
        return None

    restored_result = {}
    for level, config in result.items():
        if 'exit_stage' in config:
            # 展开6个stage multipliers为15个layer multipliers
            from ppo_architecture_generator import expand_vgg_stage_multipliers
            layer_multipliers = expand_vgg_stage_multipliers(config['width_multipliers'])

            restored_config = config.copy()
            restored_config['width_multipliers'] = layer_multipliers
            restored_config['early_exit_location'] = config['original_early_exit_location']
            # 清理临时字段
            del restored_config['exit_stage']
            del restored_config['original_early_exit_location']
            restored_result[level] = restored_config
        else:
            restored_result[level] = config

    return restored_result

def ensure_distinct_early_exits(hierarchy_dict, all_configs):
    """
    确保不同等级的early exit位置互不相同。
    从低等级开始，如果发现重复的eeloc，就将高等级的eeloc+1。
    注意：最后一个等级的eeloc不参与检查（因为模型生成时会去除）。
    重要：调整eeloc时，保持width_multipliers与前一等级一致。

    Args:
        hierarchy_dict: 层次化配置字典 {level: config}
        all_configs: 原始配置库，用于重新计算FLOPS和params

    Returns:
        调整后的层次化配置字典
    """
    if len(hierarchy_dict) <= 1:
        return hierarchy_dict

    # 获取排序后的等级列表，排除最后一个等级
    sorted_levels = sorted(hierarchy_dict.keys())
    levels_to_check = sorted_levels[:-1]  # 排除最后一个等级

    print(f"Checking early exit distinctness for levels: {levels_to_check}")

    # 从第二个等级开始检查
    for i in range(1, len(levels_to_check)):
        current_level = levels_to_check[i]
        prev_level = levels_to_check[i-1]

        current_eeloc = hierarchy_dict[current_level]['early_exit_location']
        prev_eeloc = hierarchy_dict[prev_level]['early_exit_location']

        # 如果当前等级的eeloc <= 前一个等级的eeloc，需要调整
        if current_eeloc <= prev_eeloc:
            new_eeloc = prev_eeloc + 1
            print(f"Level {current_level} early exit conflict: {current_eeloc} -> {new_eeloc}")

            # 创建新配置：使用前一等级的width_multipliers，但更新eeloc
            prev_config = hierarchy_dict[prev_level]
            current_config = hierarchy_dict[current_level].copy()

            # 关键：保持width_multipliers与前一等级一致
            current_config['early_exit_location'] = new_eeloc
            current_config['width_multipliers'] = prev_config['width_multipliers'].copy()

            print(f"  Using width_multipliers from Level {prev_level}: {prev_config['width_multipliers'][:5]}...")

            # 重新计算FLOPS和params
            updated_config = recalculate_config_metrics(current_config)
            hierarchy_dict[current_level] = updated_config

            print(f"  Updated config: eeloc={new_eeloc}, flops={updated_config['flops_m']:.2f}M, params={updated_config['num_params']:.0f}")

    # 验证最终结果
    final_eelocs = [hierarchy_dict[level]['early_exit_location'] for level in levels_to_check]
    print(f"Final early exit positions: {dict(zip(levels_to_check, final_eelocs))}")

    return hierarchy_dict


def recalculate_config_metrics(config):
    """
    重新计算给定配置的FLOPS和参数数量

    Args:
        config: 配置字典，包含width_multipliers和early_exit_location等

    Returns:
        更新后的配置字典，包含重新计算的flops_m和num_params
    """
    try:
        from utils.op_counter import measure_model
        import models
        import torch
        from args import arg_parser, modify_args

        # 创建临时args对象用于模型实例化
        # TODO: 应该从调用者传递正确的dataset，现在先硬编码cifar100
        temp_args = arg_parser.parse_args(['--data', 'cifar100', '--model', 'vgg', '--arch', 'vgg16_4'])
        temp_args = modify_args(temp_args)

        # 创建模型参数字典
        model_params = {
            'scale': 1.0,
            'ee_layer_locations': [config['early_exit_location']] if config['early_exit_location'] > 0 else []
        }

        # 根据配置创建模型
        from models.vgg import vgg_16_bn_eeloc

        # 创建临时模型进行计算
        test_model = vgg_16_bn_eeloc([0, 1], temp_args, model_params)

        # 手动设置width_multipliers（如果需要的话）
        if 'width_multipliers' in config:
            # 这里可能需要根据具体的模型实现来设置width multipliers
            pass

        # 计算FLOPS和参数
        H, W = temp_args.image_size[0], temp_args.image_size[1]
        flops, params = measure_model(test_model, H, W, exit_idx=1)  # exit_idx=1 对应early exit

        # 更新配置
        updated_config = config.copy()
        updated_config['flops_m'] = flops / 1e6  # 转换为百万
        updated_config['num_params'] = params

        # 清理内存
        del test_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return updated_config

    except Exception as e:
        print(f"Warning: Failed to recalculate metrics for config: {e}")
        print("Using original config with updated early_exit_location only")
        return config


def find_growth_config_for_level(normal_config, all_configs, model_type="resnet", available_exits=None):
    """
    为给定的 normal 配置找到最优的生长配置。

    生长配置满足：
    1. exit 比 normal 更深
    2. exit 必须在 available_exits 中（模型已有的 exit 位置）
    3. width 前缀匹配（锁定部分相同）
    4. 资源约束：冻结成本 + 训练成本 ≤ normal 的训练成本
    5. 核范数最大

    资源模型：
    - 冻结成本 ≈ 1 × params（只需存参数）
    - 训练成本 ≈ 4 × params（参数 + 梯度 + 优化器）
    - normal 预算 ≈ 4 × normal_params

    Args:
        normal_config: 正常模式的配置
        all_configs: 模型库中的所有配置
        model_type: 模型类型 (resnet/vgg/mobilenet)
        available_exits: 可用的 exit 位置列表（模型已有的 exit classifiers）

    Returns:
        dict: 最优的生长配置，包含额外字段：
            - 'frozen_range': (0, normal_exit)
            - 'active_range': (normal_exit, growth_exit)
            - 'growth_params': 生长部分的参数量
        或 None 如果没有可行的生长配置
    """
    normal_exit = normal_config['early_exit_location']
    normal_width = normal_config['width_multipliers']
    normal_params = normal_config['num_params']

    # 计算 normal 模式的预算（3 × params，对应 SGD+Momentum: 权重+梯度+动量）
    memory_budget = 3 * normal_params

    # 确定需要锁定的 stage 数量
    if model_type == "resnet":
        locked_stages = get_stage_from_exit_location(normal_exit) + 1
    elif model_type == "vgg":
        locked_stages = get_vgg_stage_from_exit_location(normal_exit) + 1
    else:
        locked_stages = len(normal_width)  # 锁定全部

    # 筛选候选配置
    candidates = []
    for config in all_configs:
        config_exit = config['early_exit_location']

        # 1. exit 必须更深
        if config_exit <= normal_exit:
            continue

        # 2. exit 必须在 available_exits 中（如果指定了）
        if available_exits is not None and config_exit not in available_exits:
            continue

        # 3. 前缀 width 必须匹配
        config_width = config['width_multipliers']
        prefix_match = True
        for i in range(min(locked_stages, len(normal_width), len(config_width))):
            if normal_width[i] != config_width[i]:
                prefix_match = False
                break
        if not prefix_match:
            continue

        # 4. 资源约束 (SGD+Momentum: K=3)
        # 冻结成本 = 1 × normal_params（只存权重）
        # 训练成本 = 3 × active_params（权重+梯度+动量）
        # 总成本 = normal_params + 3 × (config_params - normal_params)
        #        = 3 × config_params - 2 × normal_params
        config_params = config['num_params']
        growth_cost = 3 * config_params - 2 * normal_params

        if growth_cost > memory_budget:
            continue

        # 计算生长部分的参数量
        growth_params = config_params - normal_params

        candidates.append({
            **config,
            'frozen_range': (0, normal_exit),
            'active_range': (normal_exit, config_exit),
            'growth_params': growth_params,
            'growth_cost': growth_cost
        })

    if not candidates:
        return None

    # 选择核范数最大的
    best = max(candidates, key=lambda x: x['total_conv_nuclear_norm'])
    return best


def find_all_growth_configs(normal_configs, all_configs, model_type="resnet"):
    """
    为所有 level 的 normal 配置找到对应的生长配置。

    Growth 配置可以选择任意可行的 exit 位置（满足资源约束和宽度前缀匹配）。
    模型初始化时需要把这些 exit 位置也加入 ee_layer_locations。

    Args:
        normal_configs: dict, {level: normal_config}
        all_configs: 模型库中的所有配置
        model_type: 模型类型

    Returns:
        dict: {level: growth_config} 或 {level: None}
    """
    growth_configs = {}

    print(f"\n{'='*60}")
    print("TDD GROWTH CONFIG SELECTION")
    print(f"{'='*60}")

    for level in sorted(normal_configs.keys()):
        normal_config = normal_configs[level]
        # 不限制 available_exits，让算法自由选择最优的 exit 位置
        growth_config = find_growth_config_for_level(
            normal_config, all_configs, model_type, available_exits=None
        )
        growth_configs[level] = growth_config

        if growth_config:
            frozen_range = growth_config['frozen_range']
            active_range = growth_config['active_range']
            print(f"Level {level}: Normal exit={normal_config['early_exit_location']}, "
                  f"Growth exit={growth_config['early_exit_location']}, "
                  f"Frozen=[{frozen_range[0]},{frozen_range[1]}), "
                  f"Active=[{active_range[0]},{active_range[1]}), "
                  f"Growth params={growth_config['growth_params']:.0f}, "
                  f"Nuclear norm={growth_config['total_conv_nuclear_norm']:.2f}")
        else:
            print(f"Level {level}: No valid growth config found (exit={normal_config['early_exit_location']})")

    print(f"{'='*60}\n")

    return growth_configs


def find_best_config_independent(all_configs, participating_levels, flops_constraints=None, params_constraints=None):
    """
    Independent model selection: each level independently selects its best configuration
    (including width_multipliers and early_exit_location) without hierarchical constraints.

    Note: This is the "ideal" independent selection. Due to architectural constraints,
    the actual model will use only one width_multipliers (from highest level), but this
    selection process shows what each level would choose independently, demonstrating
    the mismatch between independent optimization and shared backbone requirements.

    Args:
        all_configs (list): All possible model configurations from the JSON file.
        participating_levels (list): Sorted list of unique level indices.
        flops_constraints (dict): Optional dictionary mapping each level to its FLOPs constraint.
        params_constraints (dict): Optional dictionary mapping each level to its parameter constraint.

    Returns:
        dict: A dictionary mapping each level to its independently-chosen best configuration.
              (Note: actual model will use highest level's width_multipliers for all levels)
    """
    if not all_configs:
        print("Configuration list is empty.")
        return None
    if not participating_levels:
        print("Participating levels list is empty.")
        return None

    # Validate that at least one constraint is provided
    if not flops_constraints and not params_constraints:
        print("Error: At least one of flops_constraints or params_constraints must be provided.")
        return None

    # Generate cache key based on search parameters (independent mode uses beam_width=-1 as marker)
    config_hash = get_config_hash(all_configs)
    cache_key = generate_cache_key(participating_levels, -1, config_hash, flops_constraints, params_constraints)
    cache_file = f"cache/hierarchical/independent_{cache_key}.pkl"

    # Try to load cached result
    cached_result = load_cached_result(cache_file)
    if cached_result is not None:
        return cached_result

    print(f"\n{'='*60}")
    print("INDEPENDENT MODEL SELECTION (No Hierarchical Constraints)")
    print(f"{'='*60}")
    print(f"Computing independent selection (will be cached as {cache_key[:8]}...)")
    print("Note: Each level independently selects its best config (including width).")
    print("      Actual model will use highest level's width for all levels.")

    result = {}
    active_levels = sorted(list(set(participating_levels)))

    for level in active_levels:
        # Filter candidates for this level
        level_candidates = all_configs.copy()
        constraint_parts = []

        # Apply FLOPs constraint
        if flops_constraints and level in flops_constraints:
            flops_limit = flops_constraints[level]
            level_candidates = [c for c in level_candidates if c['flops_m'] <= flops_limit]
            constraint_parts.append(f"FLOPs <= {flops_limit}M")

        # Apply params constraint
        if params_constraints and level in params_constraints:
            params_limit = params_constraints[level]
            level_candidates = [c for c in level_candidates if c['num_params']/1e6 <= params_limit]
            constraint_parts.append(f"Params <= {params_limit}M")

        # Check if we have valid candidates
        if not level_candidates:
            constraint_msg = ", ".join(constraint_parts) if constraint_parts else "No constraints"
            print(f"Error: No valid candidates for Level {level} with constraints: {constraint_msg}")
            return None

        # Select the one with maximum nuclear norm (independent of other levels)
        best_config = max(level_candidates, key=lambda x: x['total_conv_nuclear_norm'])
        result[level] = best_config

        width_preview = best_config['width_multipliers'][:3] if len(best_config['width_multipliers']) > 3 else best_config['width_multipliers']
        print(f"Level {level}: {len(level_candidates)} candidates, selected nuclear_norm={best_config['total_conv_nuclear_norm']:.2f}, "
              f"exit={best_config['early_exit_location']}, width={width_preview}...")

    print(f"\n[NOTE] Each level selected different width_multipliers.")
    print(f"[NOTE] Actual model will use Level {active_levels[-1]}'s width: {result[active_levels[-1]]['width_multipliers'][:3]}...")
    print(f"{'='*60}\n")

    # Save result to cache
    save_cached_result(cache_file, result, flops_constraints, participating_levels, params_constraints)

    return result


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

    # 检测模型类型并预处理VGG配置
    width_len = len(all_configs[0]['width_multipliers'])
    if width_len == 15:
        model_type = "vgg"
    elif width_len == 8:
        model_type = "mobilenet"
    elif width_len == 3:
        model_type = "resnet"
    else:
        model_type = "resnet"  # 默认

    if model_type == "vgg":
        print("Preprocessing VGG configs: folding 15 layers to 6 stages...")
        processed_configs = preprocess_vgg_configs(all_configs)
    else:
        processed_configs = all_configs

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
        # Start with processed configurations (folded for VGG)
        level_candidates = processed_configs.copy()
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
    
    # 对VGG结果进行后处理，还原为15个layer multipliers
    if model_type == "vgg":
        print("Postprocessing VGG results: unfolding 6 stages back to 15 layers...")
        best_hierarchy_dict = postprocess_vgg_results(best_hierarchy_dict)

    # Early exit位置去重处理（确保不同等级有不同的early exit位置）
    best_hierarchy_dict = ensure_distinct_early_exits(best_hierarchy_dict, all_configs)

    # 保存结果到缓存
    save_cached_result(cache_file, best_hierarchy_dict, flops_constraints, participating_levels, params_constraints)

    # 返回的字典只包含参与等级的配置
    return best_hierarchy_dict
