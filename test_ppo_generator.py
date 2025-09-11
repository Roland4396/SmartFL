"""
Test PPO Architecture Generator

Simple test to verify the PPO architecture generator works correctly.
"""

import torch
import numpy as np
import json
from ppo_architecture_generator import ArchitectureSearchEnv, PPOArchitectureAgent


def create_dummy_supernet():
    """Create a dummy supernet for testing"""
    from models.searchable_resnet import SearchableResNet
    
    # Create a SearchableResNet as supernet
    supernet = SearchableResNet(
        num_blocks=[18, 18, 18],
        num_classes=100,
        width_multipliers=[1.0, 1.0, 1.0],  # Full width
        early_exit_location=28  # Arbitrary exit location
    )
    
    return supernet.state_dict()


def test_environment():
    """Test the PPO environment"""
    print("=== Testing PPO Environment ===")
    
    # Create dummy supernet
    supernet_state_dict = create_dummy_supernet()
    
    # Create environment
    env = ArchitectureSearchEnv(supernet_state_dict)
    
    # Test reset (no target_flops parameter needed)
    obs, info = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  Observation: {obs}")
    
    # Test random action
    action = env.action_space.sample()
    print(f"✓ Random action: {action}")
    
    # Test step
    obs, reward, done, _, info = env.step(action)
    print(f"✓ Environment step successful")
    print(f"  Reward: {reward:.3f}")
    print(f"  Done: {done}")
    
    if 'config' in info and info['config']:
        config = info['config']
        print(f"  Generated config:")
        print(f"    Width: {[f'{w:.2f}' for w in config['width_multipliers']]}")
        print(f"    Exit: {config['early_exit_location']}")
        print(f"    FLOPs: {config['flops_m']:.2f}M")
        print(f"    Nuclear norm: {config['total_conv_nuclear_norm']:.2f}")
        print(f"    Parameters: {config['num_params']}")
        return True
    else:
        print(f"  Error: {info.get('error', 'Unknown error')}")
        return False


def test_ppo_agent():
    """Test PPO agent"""
    print("\n=== Testing PPO Agent ===")
    
    # Create environment and agent
    supernet_state_dict = create_dummy_supernet()
    env = ArchitectureSearchEnv(supernet_state_dict)
    agent = PPOArchitectureAgent(env)
    
    # Test architecture generation (no target FLOPs)
    print(f"Generating diverse architecture...")
    
    best_config = agent.generate_architecture(num_episodes=100)
    
    if best_config:
        print(f"✓ Generated configuration:")
        print(f"  Width: {[f'{w:.2f}' for w in best_config.width_multipliers]}")
        print(f"  Exit: {best_config.early_exit_location}")
        print(f"  FLOPs: {best_config.flops_m:.2f}M")
        print(f"  Nuclear norm: {best_config.total_conv_nuclear_norm:.2f}")
        print(f"  Parameters: {best_config.num_params}")
        return True
    else:
        print("✗ No valid configuration found")
        return False


def test_library_generation():
    """Test library generation with small sample"""
    print("\n=== Testing Library Generation ===")
    
    # Create dummy supernet
    supernet_state_dict = create_dummy_supernet()
    
    # Save dummy supernet to file
    torch.save(supernet_state_dict, 'dummy_supernet.pth')
    
    try:
        from ppo_architecture_generator import generate_architecture_library
        
        # Generate small library (no flops_samples needed)
        configs = generate_architecture_library(
            supernet_path='dummy_supernet.pth',
            output_path='test_library.json',
            num_architectures=100,  # More architectures
            episodes_per_batch=100  # More episodes for diversity
        )
        
        print(f"✓ Generated {len(configs)} configurations")
        
        # Verify output format
        if configs:
            example_config = configs[0]
            required_keys = ['width_multipliers', 'early_exit_location', 'flops_m', 
                           'total_conv_nuclear_norm', 'num_params']
            
            missing_keys = [key for key in required_keys if key not in example_config]
            if not missing_keys:
                print("✓ Output format matches expected schema")
                
                # Show sample config
                print(f"  Sample config:")
                for key, value in example_config.items():
                    if isinstance(value, list):
                        print(f"    {key}: {[f'{v:.2f}' for v in value]}")
                    elif isinstance(value, float):
                        print(f"    {key}: {value:.2f}")
                    else:
                        print(f"    {key}: {value}")
                
                return True
            else:
                print(f"✗ Missing keys in output: {missing_keys}")
                return False
        else:
            print("✗ No configurations generated")
            return False
            
    except Exception as e:
        print(f"✗ Library generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    


def main():
    """Run all tests"""
    print("PPO Architecture Generator Test Suite")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    tests = [
        ("Environment Test", test_environment),
        ("PPO Agent Test", test_ppo_agent),
        ("Library Generation Test", test_library_generation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results:")
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! PPO Architecture Generator is ready.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()