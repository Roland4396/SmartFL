#!/usr/bin/env python3
"""
Analysis script to identify under-explored regions in the architecture search space.
Compares different search methods and identifies gaps in exploration.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import os
from typing import Dict, List, Tuple, Any

def load_configurations(filepath: str) -> List[Dict]:
    """Load configurations from JSON file"""
    try:
        with open(filepath, 'r') as f:
            configs = json.load(f)
        print(f"✓ Loaded {len(configs)} configurations from {filepath}")
        return configs
    except FileNotFoundError:
        print(f"✗ File not found: {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"✗ Invalid JSON in {filepath}")
        return []

def analyze_width_distribution(configs: List[Dict]) -> Dict[str, Any]:
    """Analyze the distribution of width multipliers"""
    width_combinations = []
    width_stats = {'stage0': [], 'stage1': [], 'stage2': []}
    
    for config in configs:
        widths = config['width_multipliers']
        width_combinations.append(tuple(widths))
        width_stats['stage0'].append(widths[0])
        width_stats['stage1'].append(widths[1]) 
        width_stats['stage2'].append(widths[2])
    
    # Count unique combinations
    unique_combinations = set(width_combinations)
    combination_counts = Counter(width_combinations)
    
    # Available width options (from PPO code)
    available_widths = np.linspace(0.5, 1.0, 10).tolist()
    total_possible_combinations = len(available_widths) ** 3
    
    # Find missing combinations
    all_possible = set()
    for w1 in available_widths:
        for w2 in available_widths:
            for w3 in available_widths:
                all_possible.add((round(w1, 3), round(w2, 3), round(w3, 3)))
    
    # Round existing combinations for comparison
    rounded_combinations = set()
    for combo in unique_combinations:
        rounded_combo = tuple(round(w, 3) for w in combo)
        rounded_combinations.add(rounded_combo)
    
    missing_combinations = all_possible - rounded_combinations
    
    return {
        'total_configs': len(configs),
        'unique_combinations': len(unique_combinations),
        'total_possible_combinations': total_possible_combinations,
        'exploration_rate': len(unique_combinations) / total_possible_combinations,
        'missing_combinations': missing_combinations,
        'most_common_combinations': combination_counts.most_common(10),
        'stage_stats': {stage: {
            'min': min(values), 'max': max(values), 
            'mean': np.mean(values), 'std': np.std(values),
            'unique_count': len(set(values))
        } for stage, values in width_stats.items()}
    }

def analyze_depth_distribution(configs: List[Dict]) -> Dict[str, Any]:
    """Analyze the distribution of early exit locations"""
    exit_locations = [config['early_exit_location'] for config in configs]
    exit_counts = Counter(exit_locations)
    
    # Expected range: 28-53
    expected_range = list(range(28, 54))
    actual_range = list(set(exit_locations))
    missing_exits = set(expected_range) - set(actual_range)
    
    return {
        'total_configs': len(configs),
        'unique_exits': len(set(exit_locations)),
        'expected_exits': len(expected_range),
        'exploration_rate': len(set(exit_locations)) / len(expected_range),
        'missing_exits': sorted(missing_exits),
        'exit_distribution': dict(exit_counts),
        'most_common_exits': exit_counts.most_common(10),
        'least_common_exits': exit_counts.most_common()[-10:] if len(exit_counts) >= 10 else [],
        'range_stats': {
            'min': min(exit_locations),
            'max': max(exit_locations), 
            'mean': np.mean(exit_locations),
            'std': np.std(exit_locations)
        }
    }

def analyze_region_exploration(configs: List[Dict]) -> Dict[str, Any]:
    """Analyze exploration by early exit regions"""
    regions = {
        'shallow': (28, 35),   # Region 0
        'medium': (36, 44),    # Region 1  
        'deep': (45, 53)       # Region 2
    }
    
    region_counts = {name: 0 for name in regions.keys()}
    region_configs = {name: [] for name in regions.keys()}
    
    for config in configs:
        exit_loc = config['early_exit_location']
        for name, (min_exit, max_exit) in regions.items():
            if min_exit <= exit_loc <= max_exit:
                region_counts[name] += 1
                region_configs[name].append(config)
                break
    
    region_stats = {}
    for name, configs_in_region in region_configs.items():
        if configs_in_region:
            nuclear_norms = [c['total_conv_nuclear_norm'] for c in configs_in_region]
            flops = [c['flops_m'] for c in configs_in_region]
            region_stats[name] = {
                'count': len(configs_in_region),
                'nuclear_norm_stats': {
                    'min': min(nuclear_norms), 'max': max(nuclear_norms),
                    'mean': np.mean(nuclear_norms), 'std': np.std(nuclear_norms)
                },
                'flops_stats': {
                    'min': min(flops), 'max': max(flops),
                    'mean': np.mean(flops), 'std': np.std(flops)
                }
            }
        else:
            region_stats[name] = {'count': 0}
    
    return {
        'region_boundaries': regions,
        'region_counts': region_counts,
        'region_stats': region_stats,
        'total_configs': len(configs)
    }

def find_underexplored_combinations(configs: List[Dict], threshold: int = 2) -> Dict[str, List]:
    """Find width-depth combinations that are under-explored"""
    combination_counts = Counter()
    
    for config in configs:
        widths = tuple(round(w, 3) for w in config['width_multipliers'])
        exit_loc = config['early_exit_location']
        combination = (widths, exit_loc)
        combination_counts[combination] += 1
    
    under_explored = []
    never_explored = []
    
    # Check all possible combinations
    available_widths = [round(w, 3) for w in np.linspace(0.5, 1.0, 10)]
    
    for w1 in available_widths:
        for w2 in available_widths:
            for w3 in available_widths:
                for exit_loc in range(28, 54):
                    combo = ((w1, w2, w3), exit_loc)
                    count = combination_counts.get(combo, 0)
                    
                    if count == 0:
                        never_explored.append(combo)
                    elif count <= threshold:
                        under_explored.append((combo, count))
    
    return {
        'under_explored': under_explored,
        'never_explored': never_explored[:100],  # Show first 100 
        'total_never_explored': len(never_explored),
        'exploration_statistics': {
            'total_possible': len(available_widths)**3 * 26,  # 26 exit locations
            'actually_explored': len(combination_counts),
            'overall_exploration_rate': len(combination_counts) / (len(available_widths)**3 * 26)
        }
    }

def generate_exploration_report(data_files: List[str]) -> None:
    """Generate comprehensive exploration analysis report"""
    print("="*60)
    print("ARCHITECTURE SEARCH SPACE EXPLORATION ANALYSIS")
    print("="*60)
    
    for filepath in data_files:
        if not os.path.exists(filepath):
            print(f"\n⚠ Skipping {filepath} - file not found")
            continue
            
        print(f"\n{'='*40}")
        print(f"ANALYZING: {os.path.basename(filepath)}")
        print(f"{'='*40}")
        
        configs = load_configurations(filepath)
        if not configs:
            continue
        
        # Width distribution analysis
        print(f"\n📊 WIDTH MULTIPLIER ANALYSIS:")
        width_analysis = analyze_width_distribution(configs)
        print(f"  • Total configurations: {width_analysis['total_configs']}")
        print(f"  • Unique width combinations: {width_analysis['unique_combinations']}")
        print(f"  • Total possible combinations: {width_analysis['total_possible_combinations']}")
        print(f"  • Width exploration rate: {width_analysis['exploration_rate']:.1%}")
        print(f"  • Missing combinations: {len(width_analysis['missing_combinations'])}")
        
        print(f"\n  Stage-wise statistics:")
        for stage, stats in width_analysis['stage_stats'].items():
            print(f"    {stage}: unique={stats['unique_count']}, range=[{stats['min']:.3f}, {stats['max']:.3f}], std={stats['std']:.3f}")
        
        print(f"\n  Most common width combinations:")
        for combo, count in width_analysis['most_common_combinations'][:5]:
            print(f"    {combo}: {count} times")
        
        # Depth distribution analysis  
        print(f"\n📏 EARLY EXIT LOCATION ANALYSIS:")
        depth_analysis = analyze_depth_distribution(configs)
        print(f"  • Unique exit locations: {depth_analysis['unique_exits']}/{depth_analysis['expected_exits']}")
        print(f"  • Depth exploration rate: {depth_analysis['exploration_rate']:.1%}")
        print(f"  • Missing exit locations: {depth_analysis['missing_exits']}")
        print(f"  • Exit range: [{depth_analysis['range_stats']['min']}, {depth_analysis['range_stats']['max']}]")
        print(f"  • Exit std: {depth_analysis['range_stats']['std']:.1f}")
        
        print(f"\n  Most/Least common exit locations:")
        for exit_loc, count in depth_analysis['most_common_exits'][:3]:
            print(f"    Most: exit={exit_loc}, count={count}")
        for exit_loc, count in depth_analysis['least_common_exits'][:3]:
            print(f"    Least: exit={exit_loc}, count={count}")
        
        # Region analysis
        print(f"\n🏢 REGION-BASED ANALYSIS:")
        region_analysis = analyze_region_exploration(configs)
        for name, count in region_analysis['region_counts'].items():
            bounds = region_analysis['region_boundaries'][name]
            percentage = count / len(configs) * 100
            print(f"  • {name.capitalize()} region [{bounds[0]}-{bounds[1]}]: {count} configs ({percentage:.1f}%)")
            
            if name in region_analysis['region_stats'] and region_analysis['region_stats'][name]['count'] > 0:
                stats = region_analysis['region_stats'][name]
                print(f"    Nuclear norm: {stats['nuclear_norm_stats']['mean']:.1f}±{stats['nuclear_norm_stats']['std']:.1f}")
                print(f"    FLOPs: {stats['flops_stats']['mean']:.1f}±{stats['flops_stats']['std']:.1f}M")
        
        # Under-explored combinations
        print(f"\n🔍 UNDER-EXPLORED COMBINATIONS:")
        combo_analysis = find_underexplored_combinations(configs, threshold=2)
        stats = combo_analysis['exploration_statistics']
        print(f"  • Total possible combinations: {stats['total_possible']:,}")
        print(f"  • Actually explored: {stats['actually_explored']:,}")
        print(f"  • Overall exploration rate: {stats['overall_exploration_rate']:.1%}")
        print(f"  • Never explored: {combo_analysis['total_never_explored']:,}")
        print(f"  • Under-explored (≤2 times): {len(combo_analysis['under_explored']):,}")
        
        print(f"\n  Examples of never explored combinations:")
        for i, combo in enumerate(combo_analysis['never_explored'][:5]):
            widths, exit_loc = combo
            print(f"    {widths} + exit={exit_loc}")
        
        print(f"\n" + "="*40)

if __name__ == "__main__":
    # List of data files to analyze
    data_files = [
        "brute_force_results.json",
        "ppo_architecture_library.json",
        # Add more files as needed
    ]
    
    # Generate comprehensive report
    generate_exploration_report(data_files)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")