#!/usr/bin/env python3
"""
Test the new parameter constraint functionality
"""

from hierarchical_model_selector import find_best_config_for_distribution, load_configs_from_json

def test_params_constraint():
    """Test parameter constraint functionality"""
    
    # Load existing config library
    configs = load_configs_from_json("ppo_architecture_library.json")
    if not configs:
        print("No configurations found. Please generate PPO library first.")
        return
    
    print(f"Loaded {len(configs)} configurations")
    
    # Test cases
    test_cases = [
        {
            'name': 'Only FLOPs constraint (原始功能)',
            'flops_constraints': {0: 100, 1: 120, 2: 150},
            'params_constraints': None,
            'participating_levels': [0, 1, 2]
        },
        {
            'name': 'Only Params constraint (新功能)',
            'flops_constraints': None,
            'params_constraints': {0: 0.21, 1: 0.46, 2: 0.86},  # 参数约束(M)
            'participating_levels': [0, 1, 2]
        },
        {
            'name': 'Both FLOPs + Params (双重约束)',
            'flops_constraints': {0: 100, 1: 120, 2: 150},
            'params_constraints': {0: 0.5, 1: 0.8, 2: 1.2},
            'participating_levels': [0, 1, 2]
        },
        {
            'name': '严格参数约束 + 宽松FLOPs',
            'flops_constraints': {0: 200, 1: 220, 2: 250},  # 宽松FLOPs
            'params_constraints': {0: 0.3, 1: 0.4, 2: 0.5}, # 严格参数
            'participating_levels': [0, 1, 2]
        },
        {
            'name': '错误测试：无任何约束',
            'flops_constraints': None,
            'params_constraints': None,
            'participating_levels': [0, 1, 2]
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"测试案例 {i+1}: {test_case['name']}")
        print(f"{'='*60}")
        
        result = find_best_config_for_distribution(
            all_configs=configs,
            participating_levels=test_case['participating_levels'],
            flops_constraints=test_case['flops_constraints'],
            params_constraints=test_case['params_constraints']
        )
        
        if result:
            print("找到层次化配置:")
            for level, config in sorted(result.items()):
                params_m = config['num_params'] / 1e6
                print(f"  Level {level}:")
                print(f"    FLOPs: {config['flops_m']:.1f}M")
                print(f"    Params: {params_m:.2f}M")
                print(f"    Width: {config['width_multipliers']}")
                print(f"    Exit: {config['early_exit_location']}")
                
                # 检查约束是否满足
                flops_ok = True
                if test_case['flops_constraints'] and level in test_case['flops_constraints']:
                    flops_ok = config['flops_m'] <= test_case['flops_constraints'][level]
                
                params_ok = True
                if test_case['params_constraints'] and level in test_case['params_constraints']:
                    params_ok = params_m <= test_case['params_constraints'][level]
                
                status = "✓" if (flops_ok and params_ok) else "✗"
                constraint_details = []
                if test_case['flops_constraints'] and level in test_case['flops_constraints']:
                    constraint_details.append(f"FLOPs: {config['flops_m']:.1f} <= {test_case['flops_constraints'][level]} {'✓' if flops_ok else '✗'}")
                if test_case['params_constraints'] and level in test_case['params_constraints']:
                    constraint_details.append(f"Params: {params_m:.2f} <= {test_case['params_constraints'][level]} {'✓' if params_ok else '✗'}")
                
                print(f"    约束检查: {status}")
                if constraint_details:
                    for detail in constraint_details:
                        print(f"      {detail}")
        else:
            print("未找到满足约束的配置")

if __name__ == "__main__":
    test_params_constraint()