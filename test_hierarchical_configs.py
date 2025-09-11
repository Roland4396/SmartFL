#!/usr/bin/env python3
"""
Test FLOPs and parameters for hierarchical model configurations
"""

import torch
from models.searchable_resnet import SearchableResNet
from utils.metrics import calculate_flops, calculate_model_size, calculate_total_conv_nuclear_norm

def test_hierarchical_models():
    """Test 4-level hierarchical model configurations"""
    
    print("=== 层次化模型配置测试 ===")
    print("基于ResNet-110，测试不同等级的FLOPs和参数量")
    print()
    
    # 4个等级的配置
    configs = [
        {
            'level': 1,
            'width_multipliers': [0.7, 0.7, 0.7],
            'early_exit_location': 35,  # 66%深度
            'depth_ratio': 0.66,
            'description': '低端设备 - 窄模型+浅层'
        },
        {
            'level': 2, 
            'width_multipliers': [0.7, 0.7, 0.7],
            'early_exit_location': 41,  # 77%深度
            'depth_ratio': 0.77,
            'description': '中端设备 - 窄模型+中层'
        },
        {
            'level': 3,
            'width_multipliers': [0.75, 0.75, 0.75], 
            'early_exit_location': 47,  # 88%深度
            'depth_ratio': 0.88,
            'description': '高端设备 - 较宽模型+深层'
        },
        {
            'level': 4,
            'width_multipliers': [1.0, 1.0, 1.0],
            'early_exit_location': 53,  # 100%深度
            'depth_ratio': 1.0,
            'description': '顶端设备 - 全模型'
        }
    ]
    
    results = []
    
    print("配置详情:")
    print("-" * 80)
    
    for config in configs:
        print(f"等级 {config['level']}: {config['description']}")
        print(f"  宽度倍数: {config['width_multipliers']}")
        print(f"  早退位置: {config['early_exit_location']} (深度比例: {config['depth_ratio']:.0%})")
        
        # 创建模型
        model = SearchableResNet(
            num_blocks=[18, 18, 18],
            num_classes=100,
            width_multipliers=config['width_multipliers'],
            early_exit_location=config['early_exit_location']
        )
        
        # 计算指标
        dummy_input_size = (1, 3, 32, 32)
        
        # FLOPs (使用早退出)
        flops_m = calculate_flops(model, dummy_input_size, exit_idx=0)
        
        # 参数量 (新版本，支持早退出)
        params_early_exit = calculate_model_size(model, early_exit_location=config['early_exit_location'])
        
        # 参数量 (全模型，用于对比)
        params_full = calculate_model_size(model, early_exit_location=None)
        
        # 核范数
        nuclear_norm = calculate_total_conv_nuclear_norm(model, early_exit_location=config['early_exit_location'])
        
        result = {
            'level': config['level'],
            'width_multipliers': config['width_multipliers'],
            'early_exit_location': config['early_exit_location'],
            'depth_ratio': config['depth_ratio'],
            'flops_m': flops_m,
            'params_early_exit_m': params_early_exit / 1e6,
            'params_full_m': params_full / 1e6,
            'nuclear_norm': nuclear_norm,
            'description': config['description']
        }
        
        results.append(result)
        
        print(f"  FLOPs: {flops_m:.1f}M")
        print(f"  参数量(早退): {params_early_exit/1e6:.2f}M")
        print(f"  参数量(全模型): {params_full/1e6:.2f}M") 
        print(f"  参数比例: {params_early_exit/params_full:.1%}")
        print(f"  核范数: {nuclear_norm:.1f}")
        print()
    
    # 汇总对比表
    print("=" * 80)
    print("汇总对比表:")
    print("=" * 80)
    print(f"{'等级':<4} {'宽度倍数':<15} {'早退位置':<8} {'深度%':<6} {'FLOPs(M)':<10} {'参数(M)':<10} {'核范数':<10}")
    print("-" * 80)
    
    for r in results:
        width_str = f"{r['width_multipliers'][0]:.1f}×3"
        print(f"{r['level']:<4} {width_str:<15} {r['early_exit_location']:<8} {r['depth_ratio']:.0%}{'':^2} "
              f"{r['flops_m']:<10.1f} {r['params_early_exit_m']:<10.2f} {r['nuclear_norm']:<10.1f}")
    
    # 分析趋势
    print("\n" + "=" * 80)
    print("性能趋势分析:")
    print("=" * 80)
    
    base_flops = results[0]['flops_m']
    base_params = results[0]['params_early_exit_m'] 
    base_nuclear = results[0]['nuclear_norm']
    
    for i, r in enumerate(results):
        flops_ratio = r['flops_m'] / base_flops
        params_ratio = r['params_early_exit_m'] / base_params
        nuclear_ratio = r['nuclear_norm'] / base_nuclear
        
        print(f"等级{r['level']} 相对等级1的提升:")
        print(f"  FLOPs: {flops_ratio:.2f}x ({(flops_ratio-1)*100:+.0f}%)")
        print(f"  参数量: {params_ratio:.2f}x ({(params_ratio-1)*100:+.0f}%)")
        print(f"  核范数: {nuclear_ratio:.2f}x ({(nuclear_ratio-1)*100:+.0f}%)")
        print()
    
    return results

def analyze_width_vs_depth_impact():
    """分析宽度和深度对性能的影响"""
    
    print("=" * 80)
    print("宽度 vs 深度影响分析:")
    print("=" * 80)
    
    # 对比等级2和等级3：相同深度，不同宽度
    print("对比: 等级2(0.7宽度) vs 等级3(0.75宽度) - 相似深度的宽度影响")
    
    configs = [
        ([0.7, 0.7, 0.7], 41, "等级2"),
        ([0.75, 0.75, 0.75], 41, "等级3-修正")  # 用相同深度对比
    ]
    
    for width_mult, eeloc, name in configs:
        model = SearchableResNet(
            num_blocks=[18, 18, 18],
            num_classes=100, 
            width_multipliers=width_mult,
            early_exit_location=eeloc
        )
        
        flops_m = calculate_flops(model, (1, 3, 32, 32), exit_idx=0)
        params_m = calculate_model_size(model, early_exit_location=eeloc) / 1e6
        nuclear_norm = calculate_total_conv_nuclear_norm(model, early_exit_location=eeloc)
        
        print(f"{name}: FLOPs={flops_m:.1f}M, 参数={params_m:.2f}M, 核范数={nuclear_norm:.1f}")
    
    print("\n对比: 等级1 vs 等级2 - 相同宽度，不同深度的深度影响")
    
    configs = [
        ([0.7, 0.7, 0.7], 35, "等级1"),
        ([0.7, 0.7, 0.7], 41, "等级2")
    ]
    
    for width_mult, eeloc, name in configs:
        model = SearchableResNet(
            num_blocks=[18, 18, 18],
            num_classes=100,
            width_multipliers=width_mult, 
            early_exit_location=eeloc
        )
        
        flops_m = calculate_flops(model, (1, 3, 32, 32), exit_idx=0)
        params_m = calculate_model_size(model, early_exit_location=eeloc) / 1e6
        nuclear_norm = calculate_total_conv_nuclear_norm(model, early_exit_location=eeloc)
        
        print(f"{name}: FLOPs={flops_m:.1f}M, 参数={params_m:.2f}M, 核范数={nuclear_norm:.1f}")

if __name__ == "__main__":
    results = test_hierarchical_models()
    analyze_width_vs_depth_impact()