#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from hierarchical_model_selector import find_best_config_for_distribution, load_configs_from_json, is_sub_model

def debug_hierarchical_search():
    """Debug VGG hierarchical search step by step"""

    # Load configs
    config_library_path = "vgg_cifar100_architecture_library.json"
    all_model_configs = load_configs_from_json(config_library_path, model_type="vgg", dataset="cifar100")
    print(f"Loaded {len(all_model_configs)} VGG configurations")

    # Test parameters
    participating_levels = [0, 1, 2, 3]
    flops_constraints = {0: 250.42, 1: 288.01, 2: 340.48, 3: 511.42}

    # Step 1: Check preprocessing
    from hierarchical_model_selector import preprocess_vgg_configs
    processed_configs = preprocess_vgg_configs(all_model_configs)

    print(f"\nStep 1: Preprocessing")
    print(f"Original config sample:")
    print(f"  width_multipliers length: {len(all_model_configs[0]['width_multipliers'])}")
    print(f"  early_exit_location: {all_model_configs[0]['early_exit_location']}")

    print(f"Processed config sample:")
    print(f"  width_multipliers length: {len(processed_configs[0]['width_multipliers'])}")
    print(f"  exit_stage: {processed_configs[0]['exit_stage']}")
    print(f"  early_exit_location: {processed_configs[0]['early_exit_location']}")

    # Step 2: Check level 0 candidates
    level_0_candidates = [c for c in processed_configs if c['flops_m'] <= 250.42]
    level_0_candidates.sort(key=lambda x: x['total_conv_nuclear_norm'], reverse=True)

    print(f"\nStep 2: Level 0 candidates")
    print(f"Total candidates: {len(level_0_candidates)}")
    print("Top 5 candidates:")
    for i, c in enumerate(level_0_candidates[:5]):
        print(f"  {i+1}: exit_stage={c['exit_stage']}, early_exit_location={c['early_exit_location']}")
        print(f"      width_multipliers={c['width_multipliers']}")
        print(f"      nuclear_norm={c['total_conv_nuclear_norm']:.2f}")

    # Step 3: Check level 1 candidates and compatibility
    level_1_candidates = [c for c in processed_configs if c['flops_m'] <= 288.01]
    level_1_candidates.sort(key=lambda x: x['total_conv_nuclear_norm'], reverse=True)

    print(f"\nStep 3: Level 1 compatibility check")
    print(f"Level 1 total candidates: {len(level_1_candidates)}")

    # Take the best level 0 candidate and check compatibility with level 1
    best_level_0 = level_0_candidates[0]
    print(f"Best level 0 config:")
    print(f"  exit_stage={best_level_0['exit_stage']}, early_exit_location={best_level_0['early_exit_location']}")
    print(f"  width_multipliers={best_level_0['width_multipliers']}")

    compatible_count = 0
    compatible_configs = []

    for i, level_1_config in enumerate(level_1_candidates[:20]):  # Check top 20
        if is_sub_model(best_level_0, level_1_config):
            compatible_count += 1
            compatible_configs.append(level_1_config)
            if compatible_count <= 3:  # Show first 3 compatible configs
                print(f"  Compatible {compatible_count}: exit_stage={level_1_config['exit_stage']}, early_exit_location={level_1_config['early_exit_location']}")
                print(f"    width_multipliers={level_1_config['width_multipliers']}")

    print(f"Found {compatible_count} compatible level 1 configs out of top 20")

    if compatible_count == 0:
        print("\nNo compatible level 1 configs found! This explains why all levels have the same exit location.")

        # Let's check why is_sub_model is failing
        print("\nDetailed compatibility check for first few level 1 configs:")
        for i, level_1_config in enumerate(level_1_candidates[:3]):
            print(f"\nChecking level_1_config {i+1}:")
            print(f"  Level 0 exit_location: {best_level_0['early_exit_location']}")
            print(f"  Level 1 exit_location: {level_1_config['early_exit_location']}")
            print(f"  Exit location check: {best_level_0['early_exit_location']} <= {level_1_config['early_exit_location']} = {best_level_0['early_exit_location'] <= level_1_config['early_exit_location']}")

            from hierarchical_model_selector import get_actual_channels
            level_0_channels = get_actual_channels(best_level_0['width_multipliers'], "vgg")
            level_1_channels = get_actual_channels(level_1_config['width_multipliers'], "vgg")

            compare_stages = best_level_0['exit_stage'] + 1
            print(f"  Comparing {compare_stages} stages:")
            for j in range(compare_stages):
                print(f"    Stage {j}: {level_0_channels[j]} == {level_1_channels[j]} = {level_0_channels[j] == level_1_channels[j]}")

if __name__ == "__main__":
    debug_hierarchical_search()