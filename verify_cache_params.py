#!/usr/bin/env python3
"""
Script to verify and recalculate parameters in cached hierarchical configurations
"""

import pickle
import torch
from models.searchable_resnet import SearchableResNet
from utils.metrics import calculate_model_size

def recalculate_cache_params(cache_path, supernet_path=None):
    """Recalculate parameters for cached configurations"""
    
    # Load cache
    try:
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Cache file not found: {cache_path}")
        return
    
    print("=== Cache Parameter Verification ===")
    print(f"Cache file: {cache_path}")
    print()
    
    if 'result' not in cached_data:
        print("[ERROR] No 'result' key in cache")
        return
    
    result = cached_data['result']
    
    # Load supernet if available (optional)
    supernet_state_dict = None
    if supernet_path and torch.cuda.is_available():
        try:
            supernet_state_dict = torch.load(supernet_path, map_location='cpu')
            print("✓ Supernet loaded for weight slicing")
        except:
            print("⚠ Could not load supernet, using random weights")
    
    print("Recalculating parameters for each level:")
    print("-" * 60)
    
    for level, config in sorted(result.items()):
        width_multipliers = config['width_multipliers']
        exit_location = config['early_exit_location']
        cached_params = config['num_params']
        
        # Create model with same configuration
        model = SearchableResNet(
            num_blocks=[18, 18, 18],
            num_classes=100,
            width_multipliers=width_multipliers,
            early_exit_location=exit_location
        )
        
        # Load weights if available
        if supernet_state_dict:
            try:
                # Simple weight slicing (simplified version)
                model_state_dict = model.state_dict()
                for key, model_param in model_state_dict.items():
                    if key in supernet_state_dict:
                        supernet_param = supernet_state_dict[key]
                        if supernet_param.shape == model_param.shape:
                            model_param.data.copy_(supernet_param.data)
                        else:
                            # Handle shape mismatch (width scaling)
                            if supernet_param.dim() > 1:
                                min_dims = [min(s, m) for s, m in zip(supernet_param.shape, model_param.shape)]
                                sliced = supernet_param
                                for i, dim_size in enumerate(min_dims):
                                    sliced = sliced.narrow(i, 0, dim_size)
                                model_param.data.copy_(sliced)
            except Exception as e:
                print(f"  ⚠ Weight loading failed for Level {level}: {e}")
        
        # Calculate parameters using old method (all parameters)
        old_params = sum(p.numel() for p in model.parameters())
        
        # Calculate parameters using new method (up to early exit)
        new_params = calculate_model_size(model, early_exit_location=exit_location)
        
        print(f"Level {level} (exit={exit_location}, widths={width_multipliers}):")
        print(f"  Cached:    {cached_params/1e6:.2f}M")
        print(f"  Old calc:  {old_params/1e6:.2f}M (全模型)")
        print(f"  New calc:  {new_params/1e6:.2f}M (到早退位置)")
        print(f"  Difference: {(new_params - cached_params)/1e6:.2f}M")
        
        # Verify if old calculation matches cached value
        if abs(old_params - cached_params) < 1000:
            print("  ✓ 缓存值匹配旧算法 (bug确认)")
        else:
            print("  ⚠ 缓存值与预期不匹配")
        print()

def analyze_layer_count_by_exit():
    """分析不同早退位置的层数"""
    print("\n=== 早退位置对应的层数分析 ===")
    
    for exit_loc in [44, 47, 53]:
        # 计算应该包含的卷积层数
        target_conv_layers = 1 + ((exit_loc + 1) * 2)
        print(f"Early exit {exit_loc}: 应包含 {target_conv_layers} 个卷积层")
        
        # 创建模型验证
        model = SearchableResNet(
            num_blocks=[18, 18, 18],
            num_classes=100, 
            width_multipliers=[0.5, 0.5, 1.0],
            early_exit_location=exit_loc
        )
        
        # 统计实际的卷积层数
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append(name)
        
        print(f"  模型中总卷积层数: {len(conv_layers)}")
        print(f"  前{target_conv_layers}层: {conv_layers[:target_conv_layers] if target_conv_layers <= len(conv_layers) else 'ERROR: 超出范围'}")
        
        # 计算参数
        all_params = sum(p.numel() for p in model.parameters())
        exit_params = calculate_model_size(model, early_exit_location=exit_loc)
        
        print(f"  全模型参数: {all_params/1e6:.2f}M")
        print(f"  早退参数: {exit_params/1e6:.2f}M")
        print(f"  参数比例: {exit_params/all_params:.1%}")
        print()

if __name__ == "__main__":
    cache_file = r"D:\compile\SmartFL\cache\hierarchical\hierarchy_74d5334c45f170a772ee05fe1a8fdac4.pkl"
    supernet_file = r"d:\compile\SmartFL\supernet.pth"  # 如果存在的话
    
    recalculate_cache_params(cache_file, supernet_file)
    analyze_layer_count_by_exit()