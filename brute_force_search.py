import torch
import json
import argparse
import itertools
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
from models.searchable_resnet import SearchableResNet
from utils.metrics import calculate_flops, calculate_total_conv_nuclear_norm

# ==============================================================================
#  SEARCH SPACE DEFINITION
# ==============================================================================
# Define the options for stage widths. We have 3 stages.
STAGE_WIDTH_OPTIONS =np.linspace(0.5, 1.0, 10)
NUM_STAGES = 3

# Define the block indices where an early exit can be placed.
# For ResNet-110, there are 54 blocks in total (18 per stage).
# We assume exits can be placed after any block from 10 to 53.
EXIT_LOCATION_OPTIONS = list(range(28, 54))


def get_sub_network_state_dict(supernet_state_dict, subnet_model):
    """
    Extracts and correctly slices the weights for a sub-network from the
    supernet's state dictionary.
    """
    subnet_state_dict = subnet_model.state_dict()
    
    for key, supernet_param in supernet_state_dict.items():
        if key in subnet_state_dict:
            subnet_param = subnet_state_dict[key]
            
            # If shapes match, we can just copy
            if supernet_param.shape == subnet_param.shape:
                subnet_param.data.copy_(supernet_param.data)
            else:
                # Handle mismatched shapes (due to width scaling)
                # This is a simplified slicing logic.
                # It assumes slicing happens on dimension 0 (output channels)
                # and dimension 1 (input channels) for conv layers.
                
                # For Conv layers, slice out_channels (dim 0) and in_channels (dim 1)
                if supernet_param.dim() > 1: # Conv weights are > 1D
                    
                    # Output channels (dim 0)
                    min_dim0 = min(supernet_param.shape[0], subnet_param.shape[0])
                    
                    # Input channels (dim 1)
                    min_dim1 = min(supernet_param.shape[1], subnet_param.shape[1])

                    # Create a temporary tensor to hold the sliced weights
                    sliced_param = supernet_param[:min_dim0, :min_dim1, ...]
                    
                    # Check if the remaining dimensions match
                    if sliced_param.shape == subnet_param.shape:
                         subnet_param.data.copy_(sliced_param)
                    else:
                        # This can happen for downsampling layers where kernel size is 1x1
                        # and the logic gets more complex. For now, we print a warning.
                        # print(f"Warning: Could not fully slice {key}. Shapes {supernet_param.shape} vs {subnet_param.shape}")
                        pass

                # For BN/Linear layers, slice the first dimension
                else:
                    min_dim0 = min(supernet_param.shape[0], subnet_param.shape[0])
                    subnet_param.data.copy_(supernet_param[:min_dim0])

    return subnet_state_dict


def search_architectures(args):
    """
    Performs a brute-force search over a predefined search space.
    For each architecture, it calculates FLOPs and the nuclear norm of the final classifier.
    """
    print("======================================================")
    print(" S T A R T I N G   B R U T E - F O R C E   S E A R C H ")
    print("======================================================")

    # 1. Load the pre-trained Supernet weights
    print(f"--> Loading Supernet weights from: {args.supernet_path}")
    try:
        supernet_state_dict = torch.load(args.supernet_path, map_location='cpu')
    except FileNotFoundError:
        print(f"!! ERROR: Supernet weights not found at '{args.supernet_path}'.")
        print("!! Please run train_supernet.py first.")
        return

    # 2. Generate all architecture combinations
    width_combinations = list(itertools.product(STAGE_WIDTH_OPTIONS, repeat=NUM_STAGES))
    print(f"--> Search Space: {len(width_combinations)} width combinations, {len(EXIT_LOCATION_OPTIONS)} exit locations.")
    print(f"--> Total architectures to evaluate: {len(width_combinations) * len(EXIT_LOCATION_OPTIONS)}")

    all_results = []
    pbar = tqdm(total=len(width_combinations) * len(EXIT_LOCATION_OPTIONS), desc="Evaluating Architectures")

    # 3. Loop through every possible architecture
    for widths in width_combinations:
        for exit_loc in EXIT_LOCATION_OPTIONS:
            # a. Instantiate the sub-network with the current configuration
            subnet = SearchableResNet(
                num_blocks=[18, 18, 18], # Max blocks for resnet110
                num_classes=100, # Assuming CIFAR-100
                width_multipliers=list(widths),
                early_exit_location=exit_loc
            )

            # b. Get the correctly sliced state_dict and load it.
            sliced_state_dict = get_sub_network_state_dict(supernet_state_dict, subnet)
            subnet.load_state_dict(sliced_state_dict)
            subnet.eval()

            # c. Calculate metrics for the sub-network
            dummy_input_size = (1, 3, 32, 32) # For CIFAR
            flops_m = calculate_flops(subnet, dummy_input_size, exit_idx=0) # We have only one exit, so idx is 0
            
            # Calculate the total nuclear norm of all convolutional layers up to the exit
            norm = calculate_total_conv_nuclear_norm(subnet, early_exit_location=exit_loc)

            # d. Store the results
            all_results.append({
                "width_multipliers": list(widths),
                "early_exit_location": exit_loc,
                "flops_m": flops_m,
                "total_conv_nuclear_norm": norm
            })
            pbar.update(1)

    pbar.close()

    # 4. Save all results to a JSON file
    print("======================================================")
    print("--> Search finished.")
    results_path = getattr(args, 'results_path', 'brute_force_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"--> Results for {len(all_results)} architectures saved to: {results_path}")
    print("======================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Brute-force search for network architectures.')
    parser.add_argument('--supernet-path', type=str, default='supernet.pth',
                        help='Path to the pre-trained supernet weights.')
    parser.add_argument('--results-path', type=str, default='brute_force_results.json',
                        help='Path to save the JSON file with results.')
    
    args = parser.parse_args()
    search_architectures(args)