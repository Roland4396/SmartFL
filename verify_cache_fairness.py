#!/usr/bin/env python3
"""
验证缓存数据的公平性：对比预训练权重 vs 随机权重的FLOPs计算差异
"""

import pickle
import torch
import json
from models.searchable_resnet import SearchableResNet
from utils.metrics import calculate_flops, calculate_total_conv_nuclear_norm, calculate_model_size
from utils.op_counter import measure_model
from typing import Dict, Any

def load_cached_data(cache_path: str) -> Dict[str, Any]:
    """加载缓存的层次化配置数据"""
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ 成功加载缓存文件: {cache_path}")
        return data
    except Exception as e:
        print(f"✗ 加载缓存文件失败: {e}")
        return {}

def create_supernet_state_dict():
    """创建模拟的超网权重（用于对比）"""
    # 这里使用随机权重模拟预训练权重
    supernet = SearchableResNet(
        num_blocks=[18, 18, 18],
        num_classes=100,
        width_multipliers=[1.0, 1.0, 1.0],
        early_exit_location=None
    )
    return supernet.state_dict()

def get_sub_network_state_dict(subnet, supernet_state_dict):
    """获取子网络对应的权重（正确的权重切片逻辑）"""
    subnet_state_dict = {}
    
    for name, param in subnet.named_parameters():
        if name in supernet_state_dict:
            supernet_param = supernet_state_dict[name]
            
            if param.shape == supernet_param.shape:
                # 形状匹配，直接复制
                subnet_state_dict[name] = supernet_param.clone()
            else:
                # 需要权重切片
                if len(param.shape) == 4:  # Conv2d权重 [out_ch, in_ch, h, w]
                    out_ch, in_ch, h, w = param.shape
                    super_out_ch, super_in_ch, super_h, super_w = supernet_param.shape
                    
                    # 切片权重到目标大小
                    sliced_weight = supernet_param[:out_ch, :in_ch, :h, :w].clone()
                    subnet_state_dict[name] = sliced_weight
                    
                elif len(param.shape) == 2:  # Linear权重 [out_features, in_features]
                    out_feat, in_feat = param.shape
                    super_out_feat, super_in_feat = supernet_param.shape
                    
                    # 切片线性层权重
                    sliced_weight = supernet_param[:out_feat, :in_feat].clone()
                    subnet_state_dict[name] = sliced_weight
                    
                elif len(param.shape) == 1:  # BatchNorm或bias [channels]
                    ch = param.shape[0]
                    super_ch = supernet_param.shape[0]
                    
                    # 切片1D参数
                    sliced_param = supernet_param[:ch].clone()
                    subnet_state_dict[name] = sliced_param
                    
                else:
                    # 其他情况，尝试按第一维切片
                    target_size = param.shape[0] if len(param.shape) > 0 else param.numel()
                    super_size = supernet_param.shape[0] if len(supernet_param.shape) > 0 else supernet_param.numel()
                    
                    if target_size <= super_size:
                        if len(param.shape) == 0:  # 标量
                            subnet_state_dict[name] = supernet_param.clone()
                        else:
                            subnet_state_dict[name] = supernet_param[:target_size].clone()
                    else:
                        # 目标大小超过超网大小，填充零
                        new_param = torch.zeros_like(param)
                        if len(param.shape) == 0:
                            new_param = supernet_param.clone()
                        else:
                            new_param[:super_size] = supernet_param
                        subnet_state_dict[name] = new_param
        else:
            # 如果超网中没有这个参数，使用随机初始化
            subnet_state_dict[name] = param.clone()
    
    return subnet_state_dict

def verify_single_config(level: int, config: Dict[str, Any], supernet_state_dict: Dict) -> Dict[str, float]:
    """验证单个配置的FLOPs计算公平性"""
    
    width_multipliers = config['width_multipliers']
    early_exit_location = config['early_exit_location']
    cached_flops = config['flops_m']
    
    print(f"\n{'='*50}")
    print(f"验证 Level {level}:")
    print(f"  宽度: {width_multipliers}")
    print(f"  早退位置: {early_exit_location}")
    print(f"  缓存FLOPs: {cached_flops:.2f}M")
    
    # 方法1: 随机初始化权重 + op_counter
    print(f"\n方法1: 随机权重 + op_counter")
    random_subnet = SearchableResNet(
        num_blocks=[18, 18, 18],
        num_classes=100,
        width_multipliers=width_multipliers,
        early_exit_location=early_exit_location
    )
    random_subnet.eval()
    
    cls_ops, cls_params = measure_model(random_subnet, H=32, W=32, exit_idx=0)
    random_flops_opcounter = cls_ops[0] / 1e6 if cls_ops else 0.0
    print(f"  FLOPs: {random_flops_opcounter:.2f}M")
    
    # 方法2: 随机初始化权重 + calculate_flops
    print(f"\n方法2: 随机权重 + calculate_flops")
    random_flops_hook = calculate_flops(random_subnet, (1, 3, 32, 32), exit_idx=0)
    print(f"  FLOPs: {random_flops_hook:.2f}M")
    
    # 方法3: 预训练权重 + calculate_flops (模拟缓存中的计算方式)
    print(f"\n方法3: 预训练权重 + calculate_flops")
    pretrained_subnet = SearchableResNet(
        num_blocks=[18, 18, 18],
        num_classes=100,
        width_multipliers=width_multipliers,
        early_exit_location=early_exit_location
    )
    
    # 加载模拟的预训练权重
    try:
        sliced_state_dict = get_sub_network_state_dict(pretrained_subnet, supernet_state_dict)
        pretrained_subnet.load_state_dict(sliced_state_dict, strict=False)
    except Exception as e:
        print(f"  警告: 权重加载失败 {e}")
    
    pretrained_subnet.eval()
    pretrained_flops_hook = calculate_flops(pretrained_subnet, (1, 3, 32, 32), exit_idx=0)
    print(f"  FLOPs: {pretrained_flops_hook:.2f}M")
    
    # 方法4: 预训练权重 + op_counter
    print(f"\n方法4: 预训练权重 + op_counter")
    cls_ops_pretrained, _ = measure_model(pretrained_subnet, H=32, W=32, exit_idx=0)
    pretrained_flops_opcounter = cls_ops_pretrained[0] / 1e6 if cls_ops_pretrained else 0.0
    print(f"  FLOPs: {pretrained_flops_opcounter:.2f}M")
    
    # 计算差异
    print(f"\n差异分析:")
    diff1 = abs(cached_flops - random_flops_opcounter)
    diff2 = abs(cached_flops - random_flops_hook)
    diff3 = abs(cached_flops - pretrained_flops_hook)
    diff4 = abs(cached_flops - pretrained_flops_opcounter)
    
    print(f"  缓存 vs 随机+op_counter: {diff1:.2f}M ({diff1/cached_flops*100:.1f}%)")
    print(f"  缓存 vs 随机+hook: {diff2:.2f}M ({diff2/cached_flops*100:.1f}%)")
    print(f"  缓存 vs 预训练+hook: {diff3:.2f}M ({diff3/cached_flops*100:.1f}%)")
    print(f"  缓存 vs 预训练+op_counter: {diff4:.2f}M ({diff4/cached_flops*100:.1f}%)")
    
    # 权重影响分析
    weight_effect_hook = abs(pretrained_flops_hook - random_flops_hook)
    weight_effect_opcounter = abs(pretrained_flops_opcounter - random_flops_opcounter)
    
    print(f"\n权重影响分析:")
    print(f"  hook方法权重影响: {weight_effect_hook:.2f}M ({weight_effect_hook/random_flops_hook*100:.1f}%)")
    print(f"  op_counter权重影响: {weight_effect_opcounter:.2f}M ({weight_effect_opcounter/random_flops_opcounter*100:.1f}%)")
    
    return {
        'cached_flops': cached_flops,
        'random_opcounter': random_flops_opcounter,
        'random_hook': random_flops_hook,
        'pretrained_hook': pretrained_flops_hook,
        'pretrained_opcounter': pretrained_flops_opcounter,
        'weight_effect_hook': weight_effect_hook,
        'weight_effect_opcounter': weight_effect_opcounter
    }

def main():
    """主函数"""
    cache_path = "cache/hierarchical/hierarchy_92d23f50fcfe2c419cfdf77978788117.pkl"
    
    print("="*60)
    print("缓存数据公平性验证脚本")
    print("="*60)
    
    # 加载缓存数据
    cached_data = load_cached_data(cache_path)
    if not cached_data:
        return
    
    print(f"\n缓存数据概览:")
    result = cached_data.get('result', {})
    print(f"  包含层级: {list(result.keys())}")
    print(f"  FLOPs约束: {cached_data.get('flops_constraints', {})}")
    print(f"  参与层级: {cached_data.get('participating_levels', [])}")
    
    # 创建模拟的超网权重
    print(f"\n创建模拟超网权重...")
    supernet_state_dict = create_supernet_state_dict()
    print(f"超网权重创建完成")
    
    # 验证每个层级的配置
    all_results = {}
    for level, config in result.items():
        if isinstance(config, dict) and 'width_multipliers' in config:
            results = verify_single_config(level, config, supernet_state_dict)
            all_results[level] = results
    
    # 总结分析
    print(f"\n{'='*60}")
    print("总结分析")
    print(f"{'='*60}")
    
    print(f"\n各层级FLOPs对比:")
    print(f"{'Level':<6} {'缓存':<8} {'随机+op':<10} {'随机+hook':<12} {'预训练+hook':<14} {'权重影响%':<10}")
    print("-" * 70)
    
    for level, results in all_results.items():
        weight_effect_pct = results['weight_effect_hook'] / results['random_hook'] * 100
        print(f"{level:<6} {results['cached_flops']:<8.1f} {results['random_opcounter']:<10.1f} "
              f"{results['random_hook']:<12.1f} {results['pretrained_hook']:<14.1f} {weight_effect_pct:<10.1f}")
    
    # 判断哪种方法更接近缓存数据
    print(f"\n结论:")
    avg_diff_random_op = sum(abs(r['cached_flops'] - r['random_opcounter']) for r in all_results.values()) / len(all_results)
    avg_diff_pretrained_hook = sum(abs(r['cached_flops'] - r['pretrained_hook']) for r in all_results.values()) / len(all_results)
    
    print(f"  平均差异 - 缓存 vs 随机+op_counter: {avg_diff_random_op:.2f}M")
    print(f"  平均差异 - 缓存 vs 预训练+hook: {avg_diff_pretrained_hook:.2f}M")
    
    if avg_diff_random_op < avg_diff_pretrained_hook:
        print(f"  → 缓存数据更接近随机权重+op_counter的结果")
        print(f"  → 说明缓存可能已经使用了公平的计算方法")
    else:
        print(f"  → 缓存数据更接近预训练权重+hook的结果")
        print(f"  → 说明缓存可能使用了不公平的计算方法，建议修改")
    
    # 权重影响统计
    avg_weight_effect = sum(r['weight_effect_hook'] / r['random_hook'] for r in all_results.values()) / len(all_results) * 100
    print(f"  平均权重影响: {avg_weight_effect:.1f}%")
    
    if avg_weight_effect > 5:
        print(f"权重对FLOPs计算有显著影响，建议使用随机权重")
    else:
        print(f"权重对FLOPs计算影响较小")

if __name__ == "__main__":
    main()