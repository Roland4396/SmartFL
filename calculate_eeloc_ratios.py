#!/usr/bin/env python3
"""
Calculate early exit locations for ResNet-110 based on depth ratios
"""

def calculate_eeloc_for_ratios():
    """Calculate early exit locations for given depth ratios"""
    
    # ResNet-110 configuration
    total_blocks = 18 + 18 + 18  # 54 blocks total
    block_range = (0, total_blocks - 1)  # 0-53
    
    # Target depth ratios
    target_ratios = [0.88, 0.77, 0.66]
    
    print("=== ResNet-110 Early Exit Location Calculator ===")
    print(f"Total blocks: {total_blocks} (indices 0-{total_blocks-1})")
    print()
    
    results = {}
    
    for ratio in target_ratios:
        # Calculate the block position for this ratio
        # ratio = (current_block + 1) / total_blocks
        # So: current_block = ratio * total_blocks - 1
        eeloc_float = ratio * total_blocks - 1
        eeloc_rounded = round(eeloc_float)
        
        # Ensure within valid range
        eeloc_clipped = max(0, min(total_blocks - 1, eeloc_rounded))
        
        # Calculate actual ratio achieved
        actual_ratio = (eeloc_clipped + 1) / total_blocks
        
        results[ratio] = {
            'target_ratio': ratio,
            'calculated_eeloc': eeloc_float,
            'rounded_eeloc': eeloc_rounded,
            'final_eeloc': eeloc_clipped,
            'actual_ratio': actual_ratio,
            'ratio_error': abs(actual_ratio - ratio)
        }
        
        print(f"目标深度比例: {ratio:.2f}")
        print(f"  计算的eeloc: {eeloc_float:.2f}")
        print(f"  取整后eeloc: {eeloc_rounded}")
        print(f"  最终eeloc: {eeloc_clipped}")
        print(f"  实际深度比例: {actual_ratio:.3f}")
        print(f"  误差: {abs(actual_ratio - ratio):.3f}")
        print()
    
    return results

def verify_eeloc_mapping():
    """Verify the mapping between block indices and actual network depth"""
    
    print("=== Block位置到网络深度的映射验证 ===")
    
    # ResNet-110 structure: [18, 18, 18] blocks in 3 stages
    stage_blocks = [18, 18, 18]
    
    test_eelocs = [35, 41, 47]  # 基于上面计算的大概值
    
    for eeloc in test_eelocs:
        # Determine which stage this block belongs to
        cumulative = 0
        stage = 0
        for i, stage_size in enumerate(stage_blocks):
            if eeloc < cumulative + stage_size:
                stage = i
                block_in_stage = eeloc - cumulative
                break
            cumulative += stage_size
        else:
            stage = len(stage_blocks) - 1
            block_in_stage = eeloc - sum(stage_blocks[:-1])
        
        # Calculate depth ratio
        depth_ratio = (eeloc + 1) / sum(stage_blocks)
        
        print(f"Early exit location {eeloc}:")
        print(f"  在Stage {stage}中的第{block_in_stage}个block")
        print(f"  深度比例: {depth_ratio:.3f}")
        print(f"  累计block数: {eeloc + 1}/{sum(stage_blocks)}")
        print()

def calculate_precise_eelocs():
    """Calculate precise early exit locations for target ratios"""
    
    total_blocks = 54
    target_ratios = [0.88, 0.77, 0.66]
    
    print("=== 精确计算目标深度比例对应的eeloc ===")
    
    for ratio in target_ratios:
        # For ratio = (eeloc + 1) / total_blocks
        # So: eeloc = ratio * total_blocks - 1
        exact_eeloc = ratio * total_blocks - 1
        
        # Try both floor and ceil to see which is closer
        floor_eeloc = int(exact_eeloc)
        ceil_eeloc = floor_eeloc + 1
        
        floor_ratio = (floor_eeloc + 1) / total_blocks
        ceil_ratio = (ceil_eeloc + 1) / total_blocks
        
        floor_error = abs(floor_ratio - ratio)
        ceil_error = abs(ceil_ratio - ratio)
        
        if floor_error <= ceil_error:
            best_eeloc = floor_eeloc
            best_ratio = floor_ratio
            best_error = floor_error
        else:
            best_eeloc = ceil_eeloc
            best_ratio = ceil_ratio  
            best_error = ceil_error
        
        print(f"目标比例 {ratio:.2f}:")
        print(f"  精确eeloc: {exact_eeloc:.2f}")
        print(f"  Floor {floor_eeloc}: 比例={floor_ratio:.3f}, 误差={floor_error:.3f}")
        print(f"  Ceil  {ceil_eeloc}: 比例={ceil_ratio:.3f}, 误差={ceil_error:.3f}")
        print(f"  ✓ 最佳选择: eeloc={best_eeloc}, 比例={best_ratio:.3f}, 误差={best_error:.3f}")
        print()

if __name__ == "__main__":
    calculate_eeloc_for_ratios()
    print("-" * 60)
    verify_eeloc_mapping()
    print("-" * 60)
    calculate_precise_eelocs()