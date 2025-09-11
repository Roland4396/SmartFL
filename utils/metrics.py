import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


def calculate_flops(model: nn.Module, input_size: Tuple[int, ...], exit_idx: int = 0) -> float:
    """
    Calculate FLOPs for a given model with specific input size and exit index.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch_size, channels, height, width)
        exit_idx: Index of early exit (0 means final exit)
    
    Returns:
        FLOPs in millions
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_size)
    
    # Hook to count FLOPs
    flop_count = 0
    
    def conv_flop_count(module, input, output):
        nonlocal flop_count
        if isinstance(module, nn.Conv2d):
            batch_size = input[0].shape[0]
            output_dims = output.shape[2:]
            kernel_dims = module.kernel_size
            in_channels = module.in_channels
            out_channels = module.out_channels
            groups = module.groups
            
            filters_per_channel = out_channels // groups
            conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels // groups
            
            active_elements_count = batch_size * int(np.prod(output_dims))
            overall_conv_flops = conv_per_position_flops * active_elements_count * filters_per_channel
            
            bias_flops = 0
            if module.bias is not None:
                bias_flops = out_channels * active_elements_count
            
            flop_count += overall_conv_flops + bias_flops
    
    def linear_flop_count(module, input, output):
        nonlocal flop_count
        if isinstance(module, nn.Linear):
            batch_size = input[0].shape[0]
            flop_count += batch_size * module.in_features * module.out_features
    
    # Register hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_flop_count))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_flop_count))
    
    # Forward pass
    with torch.no_grad():
        if hasattr(model, 'forward') and 'manual_early_exit_index' in model.forward.__code__.co_varnames:
            _ = model(dummy_input, manual_early_exit_index=exit_idx)
        else:
            _ = model(dummy_input)
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    return flop_count / 1e6  # Return in millions


def calculate_total_conv_nuclear_norm(model: nn.Module, early_exit_location: Optional[int] = None) -> float:
    """
    Calculate the total nuclear norm of all convolutional layers up to a given early exit location.
    
    Args:
        model: PyTorch model
        early_exit_location: Block index where early exit occurs (None means use all layers)
                           For ResNet: block_index corresponds to BasicBlock position (0-53 for ResNet110)
    
    Returns:
        Total nuclear norm of all convolutional layers
    """
    model.eval()
    total_nuclear_norm = 0.0
    
    # Calculate target conv layer count based on early_exit_location
    target_conv_layers = None
    if early_exit_location is not None:
        # For SearchableResNet: each BasicBlock has 2 conv layers + 1 initial conv1
        # early_exit_location is the block index where we exit (0-based)
        # So we need to include conv layers up to and INCLUDING that block:
        # - conv1 (initial): 1 layer  
        # - Blocks 0 to early_exit_location (inclusive): each block has 2 conv layers
        target_conv_layers = 1 + ((early_exit_location + 1) * 2)  # +1 because we include the exit block
    
    conv_layer_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Check if we should stop processing layers
            if target_conv_layers is not None and conv_layer_count >= target_conv_layers:
                break
                
            # Get the weight tensor
            weight = module.weight.data
            
            # Reshape to 2D matrix: (out_channels, in_channels * kernel_h * kernel_w)
            weight_2d = weight.view(weight.size(0), -1)
            
            # Calculate nuclear norm (sum of singular values)
            try:
                # Use the new SVD function (torch.svd is deprecated)
                U, S, Vh = torch.linalg.svd(weight_2d, full_matrices=False)
                nuclear_norm = torch.sum(S).item()
                total_nuclear_norm += nuclear_norm
                conv_layer_count += 1
                        
            except Exception as e:
                print(f"Warning: SVD failed for layer {name}: {e}")
                # Fallback: use Frobenius norm as approximation
                try:
                    frobenius_norm = torch.norm(weight_2d, p='fro').item()
                    total_nuclear_norm += frobenius_norm
                    conv_layer_count += 1
                    print(f"  Using Frobenius norm as fallback: {frobenius_norm:.2f}")
                except Exception as e2:
                    print(f"  Both SVD and Frobenius norm failed for layer {name}: {e2}")
                    continue
    
    return total_nuclear_norm


def calculate_model_size(model: nn.Module, early_exit_location: Optional[int] = None) -> int:
    """
    Calculate the total number of parameters in the model up to a given early exit location.
    
    Args:
        model: PyTorch model
        early_exit_location: Block index where early exit occurs (None means use all layers)
                           For ResNet: block_index corresponds to BasicBlock position (0-53 for ResNet110)
    
    Returns:
        Total number of parameters up to the early exit location
    """
    if early_exit_location is None:
        # Calculate all parameters
        return sum(p.numel() for p in model.parameters())
    
    model.eval()
    total_params = 0
    
    # Calculate target conv layer count based on early_exit_location
    # For SearchableResNet: each BasicBlock has 2 conv layers + 1 initial conv1
    target_conv_layers = 1 + ((early_exit_location + 1) * 2)  # +1 because we include the exit block
    
    conv_layer_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Check if we should stop processing layers
            if conv_layer_count >= target_conv_layers:
                break
            conv_layer_count += 1
            
        # Count parameters for layers that should be included
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
            # Only count parameters up to the early exit location
            if isinstance(module, nn.Conv2d):
                if conv_layer_count <= target_conv_layers:
                    total_params += sum(p.numel() for p in module.parameters())
            elif isinstance(module, nn.BatchNorm2d):
                # BatchNorm layers follow Conv layers, so use the same logic
                if conv_layer_count <= target_conv_layers:
                    total_params += sum(p.numel() for p in module.parameters())
            elif isinstance(module, nn.Linear):
                # For early exit classifiers, always include them
                # For final classifier, only include if we're at the end
                if 'early_exit_classifier' in name or (conv_layer_count <= target_conv_layers):
                    total_params += sum(p.numel() for p in module.parameters())
    
    return total_params


def calculate_memory_usage(model: nn.Module, input_size: Tuple[int, ...]) -> float:
    """
    Estimate memory usage of the model for a given input size.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
    
    Returns:
        Estimated memory usage in MB
    """
    model.eval()
    
    # Calculate model parameters size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    # Calculate buffer size
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    # Estimate activation size (simplified)
    dummy_input = torch.randn(input_size)
    activations_size = 0
    
    def activation_hook(module, input, output):
        nonlocal activations_size
        if isinstance(output, torch.Tensor):
            activations_size += output.nelement() * output.element_size()
        elif isinstance(output, (list, tuple)):
            for o in output:
                if isinstance(o, torch.Tensor):
                    activations_size += o.nelement() * o.element_size()
    
    hooks = []
    for module in model.modules():
        hooks.append(module.register_forward_hook(activation_hook))
    
    with torch.no_grad():
        _ = model(dummy_input)
    
    for hook in hooks:
        hook.remove()
    
    total_size = param_size + buffer_size + activations_size
    return total_size / (1024 ** 2)  # Convert to MB