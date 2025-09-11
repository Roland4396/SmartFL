#!/usr/bin/env python3
"""
Simple script to read hierarchical cache file
"""

import pickle
import json
from pprint import pprint

def read_cache_file(cache_path):
    """Read and display cache file contents"""
    try:
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        print("=== Cache File Contents ===")
        print(f"File: {cache_path}")
        print()
        
        # Display cache metadata
        print("Cache Metadata:")
        for key, value in cached_data.items():
            if key != 'result':
                print(f"  {key}: {value}")
        print()
        
        # Display hierarchical result
        if 'result' in cached_data:
            result = cached_data['result']
            print("Hierarchical Configuration Result:")
            
            if isinstance(result, dict):
                for level, config in sorted(result.items()):
                    print(f"  Level {level}:")
                    print(f"    Width multipliers: {config['width_multipliers']}")
                    print(f"    Early exit location: {config['early_exit_location']}")
                    print(f"    FLOPs: {config['flops_m']:.1f}M")
                    print(f"    Nuclear norm: {config['total_conv_nuclear_norm']:.1f}")
                    print(f"    Parameters: {config['num_params']/1e6:.2f}M")
                    print()
            else:
                print(f"  Result type: {type(result)}")
                pprint(result)
        else:
            print("  No 'result' key found in cache")
            
    except FileNotFoundError:
        print(f"[ERROR] File not found: {cache_path}")
    except Exception as e:
        print(f"[ERROR] Error reading cache: {e}")

if __name__ == "__main__":
    cache_file = r"d:\compile\SmartFL\cache\hierarchical\hierarchy_92d23f50fcfe2c419cfdf77978788117.pkl"
    read_cache_file(cache_file)