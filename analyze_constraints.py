#!/usr/bin/env python3

import json
import numpy as np

def analyze_vgg_constraints():
    """分析VGG配置库中参数约束对应的FLOPs分布"""

    # 加载VGG配置库
    with open('vgg_cifar100_architecture_library.json', 'r') as f:
        data = json.load(f)

    configs = data['configurations']
    total_configs = len(configs)

    # 参数约束 (MB)
    params_constraints = [4.20625, 8.4125, 16.825, 33.65]

    print("VGG配置库参数与FLOPs分析")
    print("=" * 50)
    print(f"总配置数量: {total_configs}")
    print()

    # 分析每个参数约束level
    for level, params_limit in enumerate(params_constraints):
        print(f"Level {level}: 参数 <= {params_limit}M")

        # 筛选符合参数约束的配置
        valid_configs = [c for c in configs if c['num_params']/1e6 <= params_limit]
        valid_count = len(valid_configs)

        if valid_count == 0:
            print(f"  没有配置符合此参数约束")
            continue

        # 计算FLOPs统计信息
        flops_values = [c['flops_m'] for c in valid_configs]
        flops_min = min(flops_values)
        flops_max = max(flops_values)
        flops_mean = np.mean(flops_values)
        flops_median = np.median(flops_values)
        flops_25 = np.percentile(flops_values, 25)
        flops_75 = np.percentile(flops_values, 75)

        print(f"  符合约束的配置: {valid_count}/{total_configs} ({valid_count/total_configs*100:.1f}%)")
        print(f"  FLOPs分布 (M):")
        print(f"    最小值: {flops_min:.2f}")
        print(f"    25%分位: {flops_25:.2f}")
        print(f"    中位数: {flops_median:.2f}")
        print(f"    平均值: {flops_mean:.2f}")
        print(f"    75%分位: {flops_75:.2f}")
        print(f"    最大值: {flops_max:.2f}")

        # 分析early_exit_location分布
        exit_locations = {}
        for c in valid_configs:
            loc = c['early_exit_location']
            exit_locations[loc] = exit_locations.get(loc, 0) + 1

        print(f"  Early exit分布: {dict(sorted(exit_locations.items()))}")

        # 建议对应的FLOPs约束
        suggested_flops = flops_75  # 使用75%分位数作为建议约束
        print(f"  🎯 建议FLOPs约束: {suggested_flops:.2f}M (75%分位数)")
        print()

    # 对比当前FLOPs约束
    current_flops = [200.42, 288.01, 340.48, 511.42]
    print("当前FLOPs约束对比:")
    print("=" * 30)
    for level, (params_limit, flops_limit) in enumerate(zip(params_constraints, current_flops)):
        # 分别统计符合参数约束和FLOPs约束的配置数量
        params_valid = [c for c in configs if c['num_params']/1e6 <= params_limit]
        flops_valid = [c for c in configs if c['flops_m'] <= flops_limit]
        both_valid = [c for c in configs if c['num_params']/1e6 <= params_limit and c['flops_m'] <= flops_limit]

        print(f"Level {level}:")
        print(f"  参数约束 <= {params_limit}M: {len(params_valid)} 配置")
        print(f"  FLOPs约束 <= {flops_limit}M: {len(flops_valid)} 配置")
        print(f"  同时满足两个约束: {len(both_valid)} 配置")

        if len(params_valid) > 0:
            params_flops = [c['flops_m'] for c in params_valid]
            params_flops_max = max(params_flops)
            tight_ratio = flops_limit / params_flops_max
            print(f"  约束匹配度: FLOPs约束是参数约束下最大FLOPs的 {tight_ratio:.2f}x")
            if tight_ratio < 0.8:
                print(f"    ⚠️  FLOPs约束较紧")
            elif tight_ratio > 1.2:
                print(f"    ⚠️  FLOPs约束较松")
            else:
                print(f"    ✓  约束匹配较好")
        print()

if __name__ == "__main__":
    analyze_vgg_constraints()