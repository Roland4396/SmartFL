"""
PPO-based Architecture Generator for SmartFL

Replaces brute_force_search.py with intelligent PPO search.
Generates architecture configurations with the same output format.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from tqdm import tqdm
from datetime import datetime

from models.searchable_resnet import SearchableResNet
from models.searchable_vgg import searchable_vgg16
from models.searchable_mobilenet import searchable_mobilenet_v2
from models.searchable_convnext import searchable_convnext
from models.searchable_vit import (
    VIT_DEPTH,
    VIT_NUM_STAGES,
    VIT_SEARCH_EXIT_LOCATIONS,
    VIT_WIDTH_OPTIONS,
    is_vit_qkv_bias_key,
    is_vit_qkv_weight_key,
    slice_prefix_tensor,
    slice_vit_qkv_tensor,
    searchable_vit_small,
)
from utils.metrics import calculate_total_conv_nuclear_norm, calculate_model_size
from utils.op_counter import measure_model


def expand_vgg_stage_multipliers(stage_multipliers: List[float]) -> List[float]:
    """
    Expand 6 VGG stage multipliers to 15 layer multipliers
    VGG-D structure: [64,64], [128,128], [256,256,256], [512,512,512], [512,512,512] + [4096,4096]
    """
    if len(stage_multipliers) != 6:
        raise ValueError(f"Expected 6 stage multipliers for VGG, got {len(stage_multipliers)}")

    # Expand to 13 conv layers + 2 fc layers
    layer_multipliers = []

    # Stage 0: [64, 64] - 2 layers
    layer_multipliers.extend([stage_multipliers[0]] * 2)

    # Stage 1: [128, 128] - 2 layers
    layer_multipliers.extend([stage_multipliers[1]] * 2)

    # Stage 2: [256, 256, 256] - 3 layers
    layer_multipliers.extend([stage_multipliers[2]] * 3)

    # Stage 3: [512, 512, 512] - 3 layers
    layer_multipliers.extend([stage_multipliers[3]] * 3)

    # Stage 4: [512, 512, 512] - 3 layers
    layer_multipliers.extend([stage_multipliers[4]] * 3)

    # FC stage: independent FC layers control
    layer_multipliers.extend([stage_multipliers[5]] * 2)

    return layer_multipliers

def create_searchable_model(
        model_type: str,
        num_classes: int,
        width_multipliers: List[float],
        early_exit_location: int = None,
        image_size: int = 224):
    """Create a searchable model based on model type"""
    model_type = model_type.lower()

    if model_type == 'resnet':
        return SearchableResNet(
            num_blocks=[18, 18, 18],
            num_classes=num_classes,
            width_multipliers=width_multipliers,
            early_exit_location=early_exit_location
        )
    elif model_type == 'vgg':
        # For VGG, expand 6 stage multipliers to 15 layer multipliers
        if len(width_multipliers) == 6:
            expanded_multipliers = expand_vgg_stage_multipliers(width_multipliers)
        else:
            expanded_multipliers = width_multipliers  # Already expanded

        return searchable_vgg16(
            num_classes=num_classes,
            width_multipliers=expanded_multipliers,
            early_exit_location=early_exit_location
        )
    elif model_type == 'mobilenet':
        # MobileNetV2 uses 8 width stages and 17 bottleneck-level exit locations.
        return searchable_mobilenet_v2(
            num_classes=num_classes,
            width_multipliers=width_multipliers,
            early_exit_location=early_exit_location
        )
    elif model_type == 'convnext':
        return searchable_convnext(
            num_classes=num_classes,
            width_multipliers=width_multipliers,
            early_exit_location=early_exit_location
        )
    elif model_type == 'vit':
        return searchable_vit_small(
            num_classes=num_classes,
            width_multipliers=width_multipliers,
            early_exit_location=early_exit_location,
            image_size=image_size
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported: 'resnet', 'vgg', 'mobilenet', 'convnext', 'vit'")


def calculate_vit_token_nuclear_norm(model: nn.Module, probe_input: torch.Tensor, early_exit_location: int) -> float:
    """
    ViT analogue of feature-map nuclear norm.
    The representation matrix is formed from all batch tokens at the target exit.
    """
    if not hasattr(model, "extract_tokens_to_exit"):
        raise ValueError("ViT model must expose extract_tokens_to_exit for token nuclear norm")

    model.eval()
    with torch.no_grad():
        tokens = model.extract_tokens_to_exit(probe_input, early_exit_location=early_exit_location)
        if tokens.dim() != 3:
            raise ValueError(f"Expected token tensor [B, N, D], got shape {tuple(tokens.shape)}")
        token_matrix = tokens.reshape(-1, tokens.shape[-1]).float()
        token_matrix = token_matrix - token_matrix.mean(dim=0, keepdim=True)
        _, singular_values, _ = torch.linalg.svd(token_matrix, full_matrices=False)
    return torch.sum(singular_values).item()

@dataclass
class ArchConfig:
    """Architecture configuration"""
    width_multipliers: List[float]
    early_exit_location: int
    flops_m: float
    total_conv_nuclear_norm: float
    num_params: int = 0

    def to_dict(self) -> Dict:
        return {
            'width_multipliers': self.width_multipliers,
            'early_exit_location': self.early_exit_location,
            'flops_m': self.flops_m,
            'total_conv_nuclear_norm': self.total_conv_nuclear_norm,
            'num_params': self.num_params
        }


class ArchitectureSearchEnv(gym.Env):
    """
    PPO Environment for Architecture Search
    
    Task: Generate diverse, high-quality architectures (maximize nuclear norm)
    No FLOPs constraints - let hierarchical_model_selector handle filtering
    """
    
    def __init__(self,
                 supernet_state_dict: Dict,
                 model_type: str = "resnet",
                 num_classes: int = 100,
                 width_options: List[float] = None,
                 exit_location_range: Tuple[int, int] = None,
                 num_stages: int = None,
                 image_size: int = 32):
        
        super().__init__()

        self.supernet_state_dict = supernet_state_dict
        self.model_type = model_type.lower()
        self.num_classes = num_classes
        self.image_size = image_size

        # Set model-specific defaults
        if self.model_type == "resnet":
            self.width_options = width_options or np.linspace(0.5, 1.0, 10).tolist()
            self.exit_location_range = exit_location_range or (28, 54)
            self.num_stages = num_stages or 3
        elif self.model_type == "vgg":
            self.width_options = width_options or np.linspace(0.5, 1.0, 10).tolist()
            self.exit_location_range = exit_location_range or (4, 13)  # Conv layers 4-12 (stages 2-4 complete coverage)
            self.num_stages = num_stages or 6  # 5 conv stages + 1 FC stage: [64,64], [128,128], [256,256,256], [512,512,512], [512,512,512], [FC,FC]
        elif self.model_type == "mobilenet":
            self.width_options = width_options or np.linspace(0.5, 1.0, 10).tolist()
            self.exit_location_range = exit_location_range or (3, 17)  # bottleneck exits 3-16
            self.num_stages = num_stages or 8  # 8 stages: [32, 16, 24, 32, 64, 96, 160, 320]
        elif self.model_type == "convnext":
            self.width_options = width_options or np.linspace(0.5, 1.0, 10).tolist()
            self.exit_location_range = exit_location_range or (0, 4)  # 4 stage exits, final classifier is separate
            self.num_stages = num_stages or 4
        elif self.model_type == "vit":
            self.width_options = width_options or list(VIT_WIDTH_OPTIONS)
            self.exit_location_range = exit_location_range or (VIT_SEARCH_EXIT_LOCATIONS[0], VIT_DEPTH + 1)
            self.num_stages = num_stages or VIT_NUM_STAGES
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.exit_location_range = self.exit_location_range
        self.num_stages = self.num_stages
        if self.model_type == "vit":
            generator = torch.Generator().manual_seed(0)
            self.probe_input = torch.randn(8, 3, self.image_size, self.image_size, generator=generator)
        else:
            self.probe_input = None
        
        # Define action and observation spaces
        self._define_spaces()
    
    def _define_spaces(self):
        """Define action and observation spaces"""
        # Action space: width multipliers for each stage + early exit location
        width_low = np.array([min(self.width_options)] * self.num_stages, dtype=np.float32)
        width_high = np.array([max(self.width_options)] * self.num_stages, dtype=np.float32)
        
        self.action_space = spaces.Dict({
            'width_multipliers': spaces.Box(low=width_low, high=width_high, dtype=np.float32),
            'exit_location': spaces.Discrete(self.exit_location_range[1] - self.exit_location_range[0])
        })
        
        # Observation space: simple exploration signal (random noise)
        # PPO doesn't need complex state since task is just "generate good architectures"
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32  # Minimal state
        )
    
    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        # Random exploration signal - encourages diversity
        obs = np.random.uniform(0, 1, size=3).astype(np.float32)
        
        info = {}
        return obs, info
    
    def step(self, action: Dict[str, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step with configuration caching"""
        # Parse action
        width_multipliers = action['width_multipliers'].tolist()
        exit_location = self.exit_location_range[0] + int(action['exit_location'])
        
        # Quantize width multipliers to valid options
        quantized_widths = []
        for w in width_multipliers:
            quantized_w = min(self.width_options, key=lambda x: abs(x - w))
            quantized_widths.append(quantized_w)
        
        # Create config ID for caching
        config_id = (tuple(quantized_widths), exit_location)
        
        # Check cache first
        if hasattr(self, 'config_cache') and config_id in self.config_cache:
            # Use cached configuration - no expensive computation!
            config = self.config_cache[config_id]
            reward = self._calculate_reward(config)
            
            info = {
                'config': config.to_dict(),
                'reward_components': {
                    'nuclear_norm': config.total_conv_nuclear_norm,
                    'flops_m': config.flops_m,
                    'num_params': config.num_params
                },
                'from_cache': True
            }
        else:
            # Expensive computation only for new configurations
            try:
                config = self._evaluate_architecture(quantized_widths, exit_location)
                reward = self._calculate_reward(config)
                
                # Cache the result
                if hasattr(self, 'config_cache'):
                    self.config_cache[config_id] = config
                
                info = {
                    'config': config.to_dict(),
                    'reward_components': {
                        'nuclear_norm': config.total_conv_nuclear_norm,
                        'flops_m': config.flops_m,
                        'num_params': config.num_params
                    },
                    'from_cache': False
                }
                
            except Exception as e:
                # Invalid architecture
                reward = -100.0
                info = {'error': str(e), 'config': None, 'from_cache': False}
        
        # Episode is always done after one step (single-step task)
        obs = np.random.uniform(0, 1, size=3).astype(np.float32)  # New random state
        
        return obs, reward, True, False, info
    
    def _evaluate_architecture(self, width_multipliers: List[float], exit_location: int) -> ArchConfig:
        """Evaluate a single architecture"""

        # Create subnet with random initialization (not pretrained weights)
        subnet = create_searchable_model(
            model_type=self.model_type,
            num_classes=self.num_classes,
            width_multipliers=width_multipliers,
            early_exit_location=exit_location,
            image_size=self.image_size
        )
        subnet.eval()

        # Calculate FLOPs using op_counter (fair comparison without pretrained weights)
        cls_ops, cls_params = measure_model(subnet, H=self.image_size, W=self.image_size, exit_idx=0)
        flops_m = cls_ops[0] / 1e6 if cls_ops else 0.0

        # For nuclear norm calculation, we still need pretrained weights
        # Create a separate subnet with pretrained weights for nuclear norm only
        pretrained_subnet = create_searchable_model(
            model_type=self.model_type,
            num_classes=self.num_classes,
            width_multipliers=width_multipliers,
            early_exit_location=exit_location,
            image_size=self.image_size
        )
        sliced_state_dict = self._get_sub_network_state_dict(pretrained_subnet)
        pretrained_subnet.load_state_dict(sliced_state_dict)

        if self.model_type == "vit":
            nuclear_norm = calculate_vit_token_nuclear_norm(
                pretrained_subnet,
                self.probe_input,
                early_exit_location=exit_location,
            )
        else:
            nuclear_norm = calculate_total_conv_nuclear_norm(pretrained_subnet, early_exit_location=exit_location)
        num_params = calculate_model_size(subnet, early_exit_location=exit_location)
        
        # For VGG, ensure we save the expanded 15-element multipliers for compatibility
        if self.model_type == "vgg" and len(width_multipliers) == 6:
            expanded_multipliers = expand_vgg_stage_multipliers(width_multipliers)
        else:
            expanded_multipliers = width_multipliers

        return ArchConfig(
            width_multipliers=expanded_multipliers,
            early_exit_location=exit_location,
            flops_m=flops_m,
            total_conv_nuclear_norm=nuclear_norm,
            num_params=num_params
        )
    
    def _get_sub_network_state_dict(self, subnet_model):
        """Extract subnet weights from supernet (simplified version)"""
        subnet_state_dict = subnet_model.state_dict()
        
        for key, supernet_param in self.supernet_state_dict.items():
            if key in subnet_state_dict:
                subnet_param = subnet_state_dict[key]
                
                if supernet_param.shape == subnet_param.shape:
                    subnet_param.data.copy_(supernet_param.data)
                else:
                    if self.model_type == "vit" and (
                        is_vit_qkv_weight_key(key) or is_vit_qkv_bias_key(key)
                    ):
                        sliced_param = slice_vit_qkv_tensor(supernet_param, subnet_param.shape)
                        if sliced_param is not None and sliced_param.shape == subnet_param.shape:
                            subnet_param.data.copy_(sliced_param)
                        continue

                    # Handle width scaling for Conv/Linear/Norm/positional tensors.
                    if supernet_param.dim() == subnet_param.dim():
                        sliced_param = slice_prefix_tensor(
                            supernet_param,
                            subnet_param.shape,
                        ) if self.model_type == "vit" else None
                        if sliced_param is None:
                            slices = tuple(
                                slice(0, min(src, dst))
                                for src, dst in zip(supernet_param.shape, subnet_param.shape)
                            )
                            sliced_param = supernet_param[slices]
                        if sliced_param.shape == subnet_param.shape:
                            subnet_param.data.copy_(sliced_param)
        
        return subnet_state_dict
    
    def _calculate_reward(self, config: ArchConfig) -> float:
        """
        计算架构奖励：最大化核范数 + 鼓励探索多样性
        不考虑FLOPs和参数量约束，但会记录这些值供后续使用
        """
        
        # 主要奖励：核范数最大化（归一化）
        nuclear_norm_reward = config.total_conv_nuclear_norm / 1000.0
        
        # 探索奖励：鼓励宽度配置的多样性
        width_variance = np.var(config.width_multipliers)
        diversity_bonus = width_variance * 0.3
        
        # 探索激励：鼓励探索不同的宽度范围
        min_width = min(config.width_multipliers)
        max_width = max(config.width_multipliers)
        exploration_bonus = (max_width - min_width) * 0.2
        
        # 基础有效性奖励
        validity_bonus = 1.0
        
        total_reward = nuclear_norm_reward + diversity_bonus + exploration_bonus + validity_bonus
        
        return total_reward


class PPOArchitectureNetwork(nn.Module):
    """PPO Network for Architecture Search"""
    
    def __init__(
            self,
            obs_dim: int,
            width_action_dim: int,
            exit_action_dim: int,
            exit_location_range: Tuple[int, int],
            width_min: float = 0.5,
            width_max: float = 1.0,
            hidden_dim: int = 128):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.width_action_dim = width_action_dim
        self.exit_action_dim = exit_action_dim
        self.exit_location_range = exit_location_range
        self.width_min = width_min
        self.width_max = width_max
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Width multipliers actor (continuous)
        self.width_actor_mean = nn.Linear(hidden_dim, width_action_dim)
        self.width_actor_logstd = nn.Parameter(torch.zeros(1, width_action_dim))
        
        # Exit location actor (discrete)
        self.exit_actor = nn.Linear(hidden_dim, exit_action_dim)
        
        # Critic
        self.critic = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, obs: torch.Tensor):
        """Forward pass"""
        features = self.feature_extractor(obs)
        
        # Width multipliers
        width_mean = torch.sigmoid(self.width_actor_mean(features))  # [0, 1]
        width_mean = width_mean * (self.width_max - self.width_min) + self.width_min
        width_logstd = self.width_actor_logstd.expand_as(width_mean)
        
        # Exit location
        exit_logits = self.exit_actor(features)
        
        # Value
        value = self.critic(features)
        
        return width_mean, width_logstd, exit_logits, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False, exploration_bonus: float = 0.0, eeloc_region: tuple = None):
        """Enhanced action sampling with exploration bonus and optional eeloc region constraint"""
        width_mean, width_logstd, exit_logits, _ = self.forward(obs)
        
        if deterministic:
            width_action = width_mean
            exit_action = torch.argmax(exit_logits, dim=-1)
        else:
            # Dynamic exploration adjustment
            base_std = torch.exp(width_logstd)
            exploration_std = base_std * (1.0 + exploration_bonus)
            
            width_dist = torch.distributions.Normal(width_mean, exploration_std)
            width_action = width_dist.sample()
            width_action = torch.clamp(width_action, self.width_min, self.width_max)
            
            # Temperature sampling for exit location with optional region constraint
            temperature = 1.0 + exploration_bonus
            
            if eeloc_region is not None:
                # Apply region constraint to exit location
                min_exit, max_exit = eeloc_region
                # Create mask for valid exit locations
                mask = torch.full_like(exit_logits, -float('inf'))
                # Calculate offset based on exit_location_range start
                offset = self.exit_location_range[0]
                mask[:, min_exit-offset:max_exit-offset+1] = 0  # Adjust for 0-based indexing
                constrained_logits = exit_logits + mask
                exit_probs = F.softmax(constrained_logits / temperature, dim=-1)
            else:
                exit_probs = F.softmax(exit_logits / temperature, dim=-1)
            
            exit_dist = torch.distributions.Categorical(exit_probs)
            exit_action = exit_dist.sample()
        
        # Calculate log probabilities using original parameters (for training stability)
        width_log_prob = -0.5 * (((width_action - width_mean) / torch.exp(width_logstd)) ** 2 + 
                                2 * width_logstd + np.log(2 * np.pi))
        width_log_prob = width_log_prob.sum(dim=-1)
        
        exit_log_prob = F.log_softmax(exit_logits, dim=-1).gather(1, exit_action.unsqueeze(-1)).squeeze(-1)
        
        total_log_prob = width_log_prob + exit_log_prob
        
        action = {
            'width_multipliers': width_action,
            'exit_location': exit_action
        }
        
        return action, total_log_prob
    
    def get_value(self, obs: torch.Tensor):
        """Get value estimate"""
        _, _, _, value = self.forward(obs)
        return value.squeeze(-1)


class PPOArchitectureAgent:
    """Enhanced PPO Agent for Architecture Search with improved exploration"""
    
    def __init__(self, env: ArchitectureSearchEnv,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5):
        
        self.env = env
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Network
        obs_dim = env.observation_space.shape[0]
        width_action_dim = env.num_stages
        exit_action_dim = env.exit_location_range[1] - env.exit_location_range[0]
        
        self.network = PPOArchitectureNetwork(
            obs_dim,
            width_action_dim,
            exit_action_dim,
            env.exit_location_range,
            width_min=min(env.width_options),
            width_max=max(env.width_options),
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Training data storage
        self.experiences = []
        
        # Enhanced exploration parameters
        self.reset_frequency = 50  # Reset every 50 batches
        self.reset_ratio = 0.3     # Reset 30% of parameters
        self.generation_history = []  # Track generated configurations for diversity calculation
        
        # Configuration cache to avoid recomputing metrics
        self.config_cache = {}  # {config_id: ArchConfig}
        
        # TCP-like exploration control (AIMD: Additive Increase, Multiplicative Decrease)
        self.exploration_strength = 0.8  # Start with high exploration
        self.min_exploration = 0.05
        self.max_exploration = 1.0
        self.decrease_rate = 0.02  # Slow additive decrease when successful
        self.increase_factor = 2.0  # Fast multiplicative increase when duplicates detected
        
        # Region-based search strategy (model-specific)
        self.region_stats = {
            0: {'generated': 0, 'unique': 0, 'success_rate': 0.0},
            1: {'generated': 0, 'unique': 0, 'success_rate': 0.0},
            2: {'generated': 0, 'unique': 0, 'success_rate': 0.0}
        }

        # Set region boundaries based on model type
        if env.model_type == "resnet":
            # ResNet regions: early, middle, late blocks
            self.region_boundaries = [(28, 35), (36, 44), (45, 53)]
        elif env.model_type == "vgg":
            # VGG regions: stage 2, stage 3, stage 4 (complete stage coverage)
            self.region_boundaries = [(4, 6), (7, 9), (10, 12)]
        elif env.model_type == "mobilenet":
            # MobileNetV2 regions over bottleneck exits.
            self.region_boundaries = [(3, 7), (8, 12), (13, 16)]
        elif env.model_type == "convnext":
            # ConvNeXt regions: early, middle, late stage exits
            self.region_boundaries = [(0, 0), (1, 1), (2, 3)]
        elif env.model_type == "vit":
            self.region_boundaries = [(3, 5), (6, 8), (9, 12)]
        else:
            # Default to resnet boundaries
            self.region_boundaries = [(28, 35), (36, 44), (45, 53)]

        self.exploration_phase_batches = 30  # First 30 batches use rotation
    
    def periodic_reset(self, batch_num):
        """Periodically reset part of network parameters to prevent over-convergence"""
        if batch_num % self.reset_frequency == 0 and batch_num > 0:
            print(f"  [RESET] Periodic reset at batch {batch_num}")
            
            with torch.no_grad():
                for name, param in self.network.named_parameters():
                    if 'actor' in name:  # Only reset actor parameters
                        # Randomly select parameters to reset
                        reset_mask = torch.rand_like(param) < self.reset_ratio
                        if reset_mask.any():
                            # Reinitialize selected parameters
                            init_values = torch.randn_like(param) * 0.1
                            param.data = torch.where(reset_mask, init_values, param.data)
    
    def update_exploration_strength(self, batch_unique_ratio: float):
        """TCP-like exploration control: slow decrease when good, fast increase when bad"""
        if batch_unique_ratio > 0.7:  # High success rate (like low packet loss)
            # Slow additive decrease (缓慢下降)
            self.exploration_strength = max(
                self.min_exploration, 
                self.exploration_strength - self.decrease_rate
            )
        elif batch_unique_ratio < 0.3:  # High duplicate rate (like packet loss detected)
            # Fast multiplicative increase (快速上升)
            self.exploration_strength = min(
                self.max_exploration, 
                self.exploration_strength * self.increase_factor
            )
        # If 0.3 <= ratio <= 0.7, keep current strength (stable state)
        
        return self.exploration_strength
    
    def _calculate_exploration_bonus(self, batch_num: int = 0) -> float:
        """Get current exploration strength (TCP-like controlled)"""
        return self.exploration_strength
    
    def choose_search_region(self, batch_num: int) -> int:
        """Choose which eeloc region to search based on hybrid strategy"""
        if batch_num <= self.exploration_phase_batches:
            # Phase 1: Rotation strategy - explore all regions equally
            region = (batch_num - 1) % 3
            return region
        else:
            # Phase 2: Adaptive strategy - focus on successful regions
            # Calculate success rates
            for region_id, stats in self.region_stats.items():
                if stats['generated'] > 0:
                    stats['success_rate'] = stats['unique'] / stats['generated']
                else:
                    stats['success_rate'] = 0.0
            
            # Choose region with weighted probability based on success rate
            success_rates = [self.region_stats[i]['success_rate'] for i in range(3)]
            
            # Add small base probability to avoid completely abandoning regions
            base_prob = 0.1
            adjusted_rates = [rate + base_prob for rate in success_rates]
            total = sum(adjusted_rates)
            probabilities = [rate / total for rate in adjusted_rates]
            
            # Sample region based on probabilities
            region = np.random.choice(3, p=probabilities)
            return region
    
    def update_region_stats(self, region: int, generated_count: int, unique_count: int):
        """Update statistics for a region"""
        self.region_stats[region]['generated'] += generated_count
        self.region_stats[region]['unique'] += unique_count
    
    def _calculate_diversity_from_history(self, config: ArchConfig) -> float:
        """Calculate diversity reward based on distance from historical configurations"""
        if len(self.generation_history) < 10:
            return 1.0  # High reward for early configurations
        
        min_distance = float('inf')
        recent_history = self.generation_history[-100:]  # Only consider recent 100 configs
        
        for hist_config in recent_history:
            # Calculate distance in configuration space
            width_dist = sum((a - b) ** 2 for a, b in 
                            zip(config.width_multipliers, hist_config['width_multipliers']))
            exit_dist = (config.early_exit_location - hist_config['early_exit_location']) ** 2
            
            distance = np.sqrt(width_dist + exit_dist * 0.01)  # Normalize exit distance
            min_distance = min(min_distance, distance)
        
        return min_distance  # Higher distance = higher diversity reward
    
    def _is_duplicate(self, config: ArchConfig, configs: list) -> bool:
        """Check if configuration is duplicate"""
        config_id = (tuple(config.width_multipliers), config.early_exit_location)
        existing_ids = {(tuple(c.width_multipliers), c.early_exit_location) for c in configs[-20:]}
        return config_id in existing_ids
    
    def generate_architecture(self, num_episodes: int = 100) -> ArchConfig:
        """Generate best architecture through exploration"""
        
        best_config = None
        best_reward = float('-inf')
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            # Get action
            action, log_prob = self.network.get_action(obs_tensor, deterministic=False)  # Always explore
            
            # Convert to env format
            action_env = {
                'width_multipliers': action['width_multipliers'].squeeze(0).detach().numpy(),
                'exit_location': action['exit_location'].squeeze(0).detach().numpy()
            }
            
            # Step environment
            _, reward, done, _, info = self.env.step(action_env)
            
            # Store experience for training
            if 'config' in info and info['config'] is not None:
                value = self.network.get_value(obs_tensor).item()
                self.experiences.append({
                    'obs': obs,
                    'action': action,
                    'reward': reward,
                    'log_prob': log_prob.item(),
                    'value': value
                })
                
                # Track best config
                if reward > best_reward:
                    best_reward = reward
                    config_data = info['config']
                    best_config = ArchConfig(
                        width_multipliers=config_data['width_multipliers'],
                        early_exit_location=config_data['early_exit_location'],
                        flops_m=config_data['flops_m'],
                        total_conv_nuclear_norm=config_data['total_conv_nuclear_norm'],
                        num_params=config_data['num_params']
                    )
        
        return best_config

    def generate_diverse_architectures(self, num_episodes: int = 100, batch_num: int = 0, search_region: int = None) -> List[ArchConfig]:
        """Generate diverse architectures with enhanced exploration and optional region constraint"""
        
        configs = []
        consecutive_duplicates = 0
        
        # Get region bounds if specified
        eeloc_region = None
        if search_region is not None:
            eeloc_region = self.region_boundaries[search_region]
        
        # Calculate exploration bonus based on batch progress
        exploration_bonus = self._calculate_exploration_bonus(batch_num)
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            # Increase exploration if too many consecutive duplicates
            current_exploration_bonus = exploration_bonus
            if consecutive_duplicates > 5:
                current_exploration_bonus += 0.2
                consecutive_duplicates = 0
            
            # Get action with exploration bonus and optional region constraint
            action, log_prob = self.network.get_action(obs_tensor, 
                                                     deterministic=False,
                                                     exploration_bonus=current_exploration_bonus,
                                                     eeloc_region=eeloc_region)
            
            # Convert to env format
            action_env = {
                'width_multipliers': action['width_multipliers'].squeeze(0).detach().numpy(),
                'exit_location': action['exit_location'].squeeze(0).detach().numpy()
            }
            
            # Step environment
            _, base_reward, done, _, info = self.env.step(action_env)
            
            # Store experience for training (ALWAYS store for PPO learning)
            if 'config' in info and info['config'] is not None:
                config_data = info['config']
                config = ArchConfig(
                    width_multipliers=config_data['width_multipliers'],
                    early_exit_location=config_data['early_exit_location'],
                    flops_m=config_data['flops_m'],
                    total_conv_nuclear_norm=config_data['total_conv_nuclear_norm'],
                    num_params=config_data['num_params']
                )
                
                # Enhanced reward with diversity bonus
                diversity_reward = self._calculate_diversity_from_history(config)
                enhanced_reward = base_reward + 0.3 * diversity_reward
                
                value = self.network.get_value(obs_tensor).item()
                self.experiences.append({
                    'obs': obs,
                    'action': action,
                    'reward': enhanced_reward,  # Use enhanced reward
                    'log_prob': log_prob.item(),
                    'value': value
                })
                
                configs.append(config)
                
                # Update generation history for future diversity calculations
                self.generation_history.append(config_data)
                if len(self.generation_history) > 200:  # Keep only recent history
                    self.generation_history.pop(0)
                
                # Check for duplicates
                if self._is_duplicate(config, configs):
                    consecutive_duplicates += 1
                else:
                    consecutive_duplicates = 0
        
        return configs
    
    def update_policy(self):
        """Update policy using collected experiences"""
        if len(self.experiences) < 32:  # Minimum batch size
            return
        
        # Prepare batch data
        batch_size = min(64, len(self.experiences))
        batch_indices = np.random.choice(len(self.experiences), batch_size, replace=False)
        
        obs_batch = torch.FloatTensor([self.experiences[i]['obs'] for i in batch_indices])
        rewards_batch = torch.FloatTensor([self.experiences[i]['reward'] for i in batch_indices])
        old_log_probs_batch = torch.FloatTensor([self.experiences[i]['log_prob'] for i in batch_indices])
        old_values_batch = torch.FloatTensor([self.experiences[i]['value'] for i in batch_indices])
        
        # Reconstruct actions
        width_actions = torch.FloatTensor([self.experiences[i]['action']['width_multipliers'].squeeze(0).detach().numpy() 
                                         for i in batch_indices])
        exit_actions = torch.LongTensor([self.experiences[i]['action']['exit_location'].squeeze(0).item() 
                                       for i in batch_indices])
        
        actions_batch = {
            'width_multipliers': width_actions,
            'exit_location': exit_actions
        }
        
        # PPO update
        for _ in range(5):  # PPO epochs
            # Forward pass
            width_mean, width_logstd, exit_logits, values = self.network.forward(obs_batch)
            
            # Calculate new log probabilities
            width_log_prob = -0.5 * (((width_actions - width_mean) / torch.exp(width_logstd)) ** 2 + 
                                    2 * width_logstd + np.log(2 * np.pi))
            width_log_prob = width_log_prob.sum(dim=-1)
            
            exit_log_prob = F.log_softmax(exit_logits, dim=-1).gather(1, exit_actions.unsqueeze(-1)).squeeze(-1)
            new_log_probs = width_log_prob + exit_log_prob
            
            # PPO loss
            ratio = torch.exp(new_log_probs - old_log_probs_batch)
            advantages = rewards_batch - old_values_batch
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(values.squeeze(-1), rewards_batch)
            
            # Entropy for exploration
            width_entropy = 0.5 * (1 + np.log(2 * np.pi)) + width_logstd.mean()
            exit_entropy = -(F.softmax(exit_logits, dim=-1) * F.log_softmax(exit_logits, dim=-1)).sum(dim=-1).mean()
            entropy = width_entropy + exit_entropy
            
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear experiences
        self.experiences = []


def _dataset_num_classes_and_image_size(dataset: str) -> Tuple[int, int]:
    if dataset == 'cifar10':
        return 10, 32
    if dataset == 'cifar100':
        return 100, 32
    if dataset == 'tiny_imagenet':
        return 200, 64
    if dataset == 'imagenet':
        return 1000, 224
    if dataset == 'sst2':
        return 2, 32
    if dataset == 'ag_news':
        return 4, 32
    print(f"[WARN] Unknown dataset '{dataset}', defaulting to 100 classes and 32x32 images")
    return 100, 32


def _load_matching_weights(subnet_model, supernet_state_dict: Dict) -> Dict:
    subnet_state_dict = subnet_model.state_dict()
    for key, supernet_param in supernet_state_dict.items():
        if key not in subnet_state_dict:
            continue
        subnet_param = subnet_state_dict[key]
        if supernet_param.shape == subnet_param.shape:
            subnet_param.data.copy_(supernet_param.data)
            continue
        if is_vit_qkv_weight_key(key) or is_vit_qkv_bias_key(key):
            sliced_param = slice_vit_qkv_tensor(supernet_param, subnet_param.shape)
            if sliced_param is not None and sliced_param.shape == subnet_param.shape:
                subnet_param.data.copy_(sliced_param)
            continue
        if supernet_param.dim() == subnet_param.dim():
            sliced_param = slice_prefix_tensor(supernet_param, subnet_param.shape)
            if sliced_param is None:
                slices = tuple(slice(0, min(src, dst)) for src, dst in zip(supernet_param.shape, subnet_param.shape))
                sliced_param = supernet_param[slices]
            if sliced_param.shape == subnet_param.shape:
                subnet_param.data.copy_(sliced_param)
    return subnet_state_dict


def generate_vit_architecture_library(
        supernet_path: str,
        output_path: str,
        dataset: str = "cifar10") -> List[Dict]:
    """
    Generate a deterministic ViT-Small architecture library over the same
    width and exit search space used by the generic Stage 2 search.
    """
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"=== ViT-Small Width/Depth Library Generation [{start_time}] ===")
    print("Backbone: timm vit_small_patch16_224")
    print(f"Search space: widths {list(VIT_WIDTH_OPTIONS)}, exits after blocks {list(VIT_SEARCH_EXIT_LOCATIONS)}")

    try:
        supernet_state_dict = torch.load(supernet_path, map_location='cpu')
        print("[OK] Supernet loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load supernet: {e}")
        return []

    num_classes, image_size = _dataset_num_classes_and_image_size(dataset)
    all_configs = []

    generator = torch.Generator().manual_seed(0)
    probe_input = torch.randn(8, 3, image_size, image_size, generator=generator)

    for width in VIT_WIDTH_OPTIONS:
        for exit_location in VIT_SEARCH_EXIT_LOCATIONS:
            stage_widths = [width] * VIT_NUM_STAGES
            subnet = create_searchable_model(
                model_type="vit",
                num_classes=num_classes,
                width_multipliers=stage_widths,
                early_exit_location=exit_location,
                image_size=image_size,
            )
            subnet.eval()

            cls_ops, _ = measure_model(subnet, H=image_size, W=image_size, exit_idx=0)
            flops_m = cls_ops[0] / 1e6 if cls_ops else 0.0

            pretrained_subnet = create_searchable_model(
                model_type="vit",
                num_classes=num_classes,
                width_multipliers=stage_widths,
                early_exit_location=exit_location,
                image_size=image_size,
            )
            pretrained_subnet.load_state_dict(_load_matching_weights(pretrained_subnet, supernet_state_dict))
            nuclear_norm = calculate_vit_token_nuclear_norm(
                pretrained_subnet,
                probe_input,
                early_exit_location=exit_location,
            )
            num_params = calculate_model_size(subnet, early_exit_location=exit_location)

            config = ArchConfig(
                width_multipliers=stage_widths,
                early_exit_location=exit_location,
                flops_m=flops_m,
                total_conv_nuclear_norm=nuclear_norm,
                num_params=num_params,
            )
            all_configs.append(config.to_dict())
            print(
                f"[OK] width={width:.4f}, exit={exit_location}, FLOPs={flops_m:.2f}M, "
                f"norm={nuclear_norm:.2f}, params={num_params}"
            )

    config_with_metadata = {
        "metadata": {
            "model_type": "vit",
            "dataset": dataset,
            "num_classes": num_classes,
            "search_method": "alignfl_width_depth_vit",
            "backbone": "vit_small_patch16_224",
            "architecture_space": "vit_stage_hidden_width",
            "exit_granularity": "transformer_block",
            "width_options": list(VIT_WIDTH_OPTIONS),
            "exit_locations": list(VIT_SEARCH_EXIT_LOCATIONS),
            "total_configs": len(all_configs),
        },
        "configurations": all_configs,
    }

    with open(output_path, 'w') as f:
        json.dump(config_with_metadata, f, indent=4)

    print(f"\n=== ViT Configuration Library Saved ===")
    print(f"Generated {len(all_configs)} width/depth configurations")
    print(f"Saved to: {output_path}")
    return all_configs


def generate_architecture_library(supernet_path: str,
                                output_path: str = 'ppo_architecture_library.json',
                                num_architectures: int = 500,
                                episodes_per_batch: int = 200,
                                model_type: str = "resnet",
                                dataset: str = "cifar10") -> List[Dict]:
    """
    Generate diverse architecture library using PPO search
    
    Args:
        supernet_path: Path to trained supernet
        output_path: Output JSON file path  
        num_architectures: Target number of architectures to generate
        episodes_per_batch: Episodes per generation batch
    
    Returns:
        List of architecture configurations
    """
    start_time_obj = datetime.now()
    start_time = start_time_obj.strftime("%Y-%m-%d %H:%M:%S")
    print(f"=== PPO Architecture Library Generation [{start_time}] ===")
    print("Goal: Generate diverse, high-quality architectures")
    print("No FLOPs constraints - hierarchical_model_selector will filter later")
    
    # Load supernet
    print(f"Loading supernet from: {supernet_path}")
    try:
        supernet_state_dict = torch.load(supernet_path, map_location='cpu')
        print("[OK] Supernet loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load supernet: {e}")
        return []
    
    print(f"Target architectures: {num_architectures}")
    print(f"Episodes per batch: {episodes_per_batch}")
    print(f"Model type: {model_type}")
    print(f"Dataset: {dataset}")

    # Determine number of classes based on dataset (matching args.py logic)
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'imagenet':
        num_classes = 1000
    elif dataset == 'tiny_imagenet':
        num_classes = 200
    elif dataset == 'sst2':
        num_classes = 2
    elif dataset == 'ag_news':
        num_classes = 4
    else:
        # Default fallback
        print(f"[WARN] Unknown dataset '{dataset}', defaulting to 100 classes")
        num_classes = 100
    image_size = 64 if dataset == 'tiny_imagenet' else 32

    # Create environment and agent
    env = ArchitectureSearchEnv(
        supernet_state_dict,
        model_type=model_type,
        num_classes=num_classes,
        image_size=image_size
    )
    agent = PPOArchitectureAgent(env)
    
    # Share cache between agent and environment
    env.config_cache = agent.config_cache
    
    all_configs = []
    seen_configs = set()  # Avoid duplicates

    # Generate architectures until the requested target is reached.
    batch = 0
    max_batches = 1000  # Safety cap for duplicate-heavy searches.

    while batch < max_batches and len(all_configs) < num_architectures:
        batch += 1
        
        # Choose search region using hybrid strategy
        search_region = agent.choose_search_region(batch)
        region_bounds = agent.region_boundaries[search_region]
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n--- Batch {batch}/{max_batches} (Total Configs: {len(all_configs)}) [{current_time}] ---")
        if batch <= agent.exploration_phase_batches:
            print(f"[ROTATION] Searching eeloc region {search_region} [{region_bounds[0]}-{region_bounds[1]}]")
        else:
            success_rates = [f"{agent.region_stats[i]['success_rate']:.1%}" for i in range(3)]
            print(f"[ADAPTIVE] Chosen region {search_region} [{region_bounds[0]}-{region_bounds[1]}] (Success rates: {success_rates})")
        
        # Apply periodic reset for enhanced exploration
        agent.periodic_reset(batch)
        
        # Generate diverse architectures with batch and region context
        batch_configs = agent.generate_diverse_architectures(episodes_per_batch, batch_num=batch, search_region=search_region)
        
        new_configs_count = 0
        for config in batch_configs:
            config_id = (tuple(config.width_multipliers), config.early_exit_location)
            
            if config_id not in seen_configs:
                seen_configs.add(config_id)
                config_dict = config.to_dict()
                if len(all_configs) < num_architectures:
                    all_configs.append(config_dict)
                else:
                    break
                new_configs_count += 1
                
                print(f"[OK] Generated: width={[f'{w:.2f}' for w in config.width_multipliers]}, "
                      f"exit={config.early_exit_location}, "
                      f"FLOPs={config.flops_m:.1f}M, "
                      f"norm={config.total_conv_nuclear_norm:.1f}, "
                      f"params={config.num_params}")
        
        # Calculate batch unique ratio for TCP-like control
        batch_unique_ratio = new_configs_count / len(batch_configs) if batch_configs else 0.0
        
        # Update exploration strength based on success rate
        new_exploration = agent.update_exploration_strength(batch_unique_ratio)
        
        # Update region statistics
        agent.update_region_stats(search_region, len(batch_configs), new_configs_count)
        
        if new_configs_count == 0:
            print(f"[WARN] No new unique configs in this batch (found {len(batch_configs)} total)")
            print(f"  Current unique count: {len(all_configs)}/{num_architectures}")
            print(f"  Unique ratio: {batch_unique_ratio:.1%} -> Exploration: {new_exploration:.3f} (increasing)")
            print(f"  Cache size: {len(agent.config_cache)} cached configurations")
        else:
            print(f"[OK] Added {new_configs_count} new configs from {len(batch_configs)} generated")
            print(f"  Unique ratio: {batch_unique_ratio:.1%} -> Exploration: {new_exploration:.3f}")
            print(f"  Cache size: {len(agent.config_cache)} cached configurations")
        
        # Update policy periodically
        if batch % 3 == 0:
            agent.update_policy()
            print("  [UPDATE] Policy updated")
    
    # Report final statistics
    end_time_obj = datetime.now()
    end_time = end_time_obj.strftime("%Y-%m-%d %H:%M:%S")
    total_duration = end_time_obj - start_time_obj
    total_seconds = total_duration.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    print(f"\n=== PPO Search Completed [{end_time}] ===")
    print(f"Completed {batch} batches")
    print(f"Generated {len(all_configs)} unique configurations")
    print(f"Total time: {hours}h {minutes}m {seconds}s ({total_seconds:.1f} seconds)")
    avg_per_batch = len(all_configs) / batch if batch > 0 else 0
    print(f"Average {avg_per_batch:.1f} unique configs per batch")
    configs_per_second = len(all_configs) / total_seconds if total_seconds > 0 else 0
    print(f"Speed: {configs_per_second:.2f} configs/second")
    
    # Add metadata to the configuration file
    config_with_metadata = {
        "metadata": {
            "model_type": model_type,
            "dataset": dataset,
            "num_classes": num_classes,
            "search_method": "ppo",
            "architecture_space": "vit_stage_hidden_width" if model_type.lower() == "vit" else "stage_width",
            "nuclear_norm_source": "token_representation" if model_type.lower() == "vit" else "conv_weight",
            "generation_timestamp": str(torch.cuda.current_device() if torch.cuda.is_available() else "cpu"),
            "exit_granularity": "bottleneck" if model_type.lower() == "mobilenet" else "default",
            "exit_location_range": list(env.exit_location_range),
            "total_configs": len(all_configs)
        },
        "configurations": all_configs
    }

    # Save results
    with open(output_path, 'w') as f:
        json.dump(config_with_metadata, f, indent=4)
    
    print(f"\n=== Configuration Library Saved ===")
    print(f"Generated {len(all_configs)} unique architecture configurations")
    print(f"Completed {batch} batches with {avg_per_batch:.1f} configs/batch")
    print(f"Saved to: {output_path}")
    if all_configs:
        print(f"Range of FLOPs: {min(c['flops_m'] for c in all_configs):.1f}M - {max(c['flops_m'] for c in all_configs):.1f}M")
        print(f"Range of nuclear norm: {min(c['total_conv_nuclear_norm'] for c in all_configs):.1f} - {max(c['total_conv_nuclear_norm'] for c in all_configs):.1f}")
    
    return all_configs


def generate_random_architecture_library(supernet_path: str,
                                        output_path: str = 'random_architecture_library.json',
                                        num_architectures: int = 50000,
                                        model_type: str = "resnet",
                                        dataset: str = "cifar10") -> List[Dict]:
    """
    Generate architecture library using PURE RANDOM SEARCH (no optimization)

    This is the true random baseline - completely uniform sampling without any
    gradient-based optimization, reward guidance, or diversity mechanisms.
    """
    print("=== PURE RANDOM SEARCH (Baseline) ===")
    print("No optimization, no reward, no PPO - just uniform random sampling")

    # Load supernet
    print(f"Loading supernet from: {supernet_path}")
    try:
        supernet_state_dict = torch.load(supernet_path, map_location='cpu')
        print("[OK] Supernet loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load supernet: {e}")
        return []

    print(f"Target unique architectures: {num_architectures}")
    print(f"Model type: {model_type}")
    print(f"Dataset: {dataset}")

    # Determine number of classes
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'imagenet':
        num_classes = 1000
    elif dataset == 'tiny_imagenet':
        num_classes = 200
    elif dataset == 'sst2':
        num_classes = 2
    elif dataset == 'ag_news':
        num_classes = 4
    else:
        print(f"[WARN] Unknown dataset '{dataset}', defaulting to 100 classes")
        num_classes = 100

    # Set model-specific parameters
    width_options = list(VIT_WIDTH_OPTIONS) if model_type.lower() == "vit" else np.linspace(0.5, 1.0, 10).tolist()

    if model_type.lower() == "resnet":
        exit_location_range = (28, 54)
        num_stages = 3
    elif model_type.lower() == "vgg":
        exit_location_range = (4, 13)
        num_stages = 6
    elif model_type.lower() == "mobilenet":
        exit_location_range = (3, 17)
        num_stages = 8
    elif model_type.lower() == "convnext":
        exit_location_range = (0, 4)
        num_stages = 4
    elif model_type.lower() == "vit":
        exit_location_range = (VIT_SEARCH_EXIT_LOCATIONS[0], VIT_DEPTH + 1)
        num_stages = VIT_NUM_STAGES
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Create environment (only for evaluation, not for policy)
    env = ArchitectureSearchEnv(
        supernet_state_dict,
        model_type=model_type,
        num_classes=num_classes,
        width_options=width_options,
        exit_location_range=exit_location_range,
        num_stages=num_stages,
        image_size=64 if dataset == 'tiny_imagenet' else 32,
    )

    all_configs = []
    seen_configs = set()
    config_cache = {}
    env.config_cache = config_cache

    # 计算理论配置空间大小（用于统计）
    theoretical_max = (len(width_options) ** num_stages) * (exit_location_range[1] - exit_location_range[0])
    print(f"Theoretical configuration space: {theoretical_max:,}")
    print(f"Target configurations: {num_architectures:,}")

    attempts = 0
    max_attempts = num_architectures

    print("\nGenerating random architectures...")
    pbar = tqdm(total=max_attempts, desc="Random Search")

    while len(all_configs) < num_architectures and attempts < max_attempts:
        attempts += 1

        # PURE RANDOM SAMPLING - no optimization, no guidance
        width_multipliers = [np.random.choice(width_options) for _ in range(num_stages)]
        exit_location = np.random.randint(exit_location_range[0], exit_location_range[1])

        config_id = (tuple(width_multipliers), exit_location)

        if config_id in seen_configs:
            continue

        seen_configs.add(config_id)

        # Evaluate architecture (compute FLOPs and nuclear norm)
        try:
            action_env = {
                'width_multipliers': np.array(width_multipliers, dtype=np.float32),
                'exit_location': np.array(exit_location - exit_location_range[0])
            }

            _, _, _, _, info = env.step(action_env)

            if 'config' in info and info['config'] is not None:
                config_dict = info['config']
                all_configs.append(config_dict)
                pbar.update(1)

                if len(all_configs) % 1000 == 0:
                    print(f"\n[{len(all_configs)}/{num_architectures}] "
                          f"Attempts: {attempts}, Cache: {len(config_cache)}")

        except Exception as e:
            continue

    pbar.close()

    print(f"\n=== Random Search Completed ===")
    print(f"Generated {len(all_configs)} unique configurations")
    print(f"Target was: {num_architectures}")
    print(f"Theoretical max: {theoretical_max}")
    print(f"Coverage: {len(all_configs)/theoretical_max*100:.1f}% of configuration space")
    print(f"Total attempts: {attempts}")
    print(f"Success rate: {len(all_configs)/attempts*100:.1f}%")

    # Add metadata
    config_with_metadata = {
        "metadata": {
            "model_type": model_type,
            "dataset": dataset,
            "num_classes": num_classes,
            "search_method": "pure_random_search",
            "architecture_space": "vit_stage_hidden_width" if model_type.lower() == "vit" else "stage_width",
            "nuclear_norm_source": "token_representation" if model_type.lower() == "vit" else "conv_weight",
            "exit_granularity": "bottleneck" if model_type.lower() == "mobilenet" else "default",
            "exit_location_range": list(exit_location_range),
            "total_configs": len(all_configs),
            "theoretical_max": theoretical_max,
            "coverage_percent": len(all_configs)/theoretical_max*100,
            "total_attempts": attempts
        },
        "configurations": all_configs
    }

    # Save results
    with open(output_path, 'w') as f:
        json.dump(config_with_metadata, f, indent=4)

    print(f"\n=== Configuration Library Saved ===")
    print(f"Saved to: {output_path}")
    if all_configs:
        print(f"FLOPs range: {min(c['flops_m'] for c in all_configs):.1f}M - {max(c['flops_m'] for c in all_configs):.1f}M")
        print(f"Nuclear norm range: {min(c['total_conv_nuclear_norm'] for c in all_configs):.1f} - {max(c['total_conv_nuclear_norm'] for c in all_configs):.1f}")

    return all_configs


if __name__ == "__main__":
    # Example usage
    generate_architecture_library(
        supernet_path='supernet.pth',
        output_path='ppo_architecture_library.json'
    )
