"""
Time-Domain Decomposition (TDD) for SmartFL

Core idea: Allow small devices (Level 0) to contribute to deep layer training
by trading time for space.

- Frozen Base: Freeze blocks up to own exit location
- Active Head: Train borrowed blocks from higher level
- Dynamic Rotation: Periodically switch between normal and borrow modes

Example (with actual beam search results):
  Level 0: exit=38, Level 1: exit=43, Level 2: exit=51, Level 3: exit=53

  Phase A (Round 0-9): Normal training
    Level 0: blocks 0-37 [train]
    Level 1: blocks 0-42 [train]
    Level 2: blocks 0-50 [train]
    Level 3: blocks 0-52 [train]

  Phase B (Round 10-19): Borrow mode
    Level 0: blocks 0-37 [frozen] + blocks 38-42 [borrow from Level 1]
    Level 1: blocks 0-42 [frozen] + blocks 43-50 [borrow from Level 2]
    Level 2: blocks 0-50 [frozen] + blocks 51-52 [borrow from Level 3]
    Level 3: blocks 0-52 [normal train]
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple


# ResNet-110 configuration: 3 stages, each with 18 blocks
RESNET110_BLOCKS_PER_STAGE = [18, 18, 18]  # Total 54 blocks
RESNET110_STAGE_BOUNDARIES = [18, 36, 54]  # Cumulative


def get_stage_and_block_within_stage(global_block_idx: int) -> Tuple[int, int]:
    """
    Convert global block index to (stage_idx, block_within_stage).

    Args:
        global_block_idx: Block index in range [0, 53] for ResNet-110

    Returns:
        (stage_idx, block_within_stage) tuple
    """
    if global_block_idx < 18:
        return 0, global_block_idx
    elif global_block_idx < 36:
        return 1, global_block_idx - 18
    else:
        return 2, global_block_idx - 36


def get_global_block_idx(stage_idx: int, block_within_stage: int) -> int:
    """
    Convert (stage_idx, block_within_stage) to global block index.
    """
    if stage_idx == 0:
        return block_within_stage
    elif stage_idx == 1:
        return 18 + block_within_stage
    else:
        return 36 + block_within_stage


class DynamicScheduler:
    """
    Periodic rotation scheduler for Time-Domain Decomposition.
    """

    def __init__(self, rotation_period: int = 10, enable_tdd: bool = True):
        self.rotation_period = rotation_period
        self.enable_tdd = enable_tdd

    def get_phase(self, round_idx: int) -> int:
        """
        Returns 0 for normal phase, 1 for borrow phase.
        """
        if not self.enable_tdd:
            return 0
        return (round_idx // self.rotation_period) % 2

    def get_phase_info(self, round_idx: int) -> Dict[str, Any]:
        phase = self.get_phase(round_idx)
        phase_start = (round_idx // self.rotation_period) * self.rotation_period
        phase_end = phase_start + self.rotation_period - 1
        rounds_in_phase = round_idx - phase_start + 1

        return {
            'phase': phase,
            'phase_name': 'normal' if phase == 0 else 'borrow',
            'phase_start': phase_start,
            'phase_end': phase_end,
            'rounds_completed_in_phase': rounds_in_phase,
            'rotation_period': self.rotation_period,
            'tdd_enabled': self.enable_tdd
        }


class TimeDomainDecomposer:
    """
    Time-Domain Decomposition Manager with block-level granularity.
    """

    def __init__(self, scheduler: DynamicScheduler, level_configs: Dict[int, Dict]):
        """
        Args:
            scheduler: DynamicScheduler instance
            level_configs: Dictionary mapping level -> configuration from beam search
                          Each config contains 'early_exit_location', 'width_multipliers', etc.
        """
        self.scheduler = scheduler
        self.level_configs = level_configs
        self.num_levels = len(level_configs)

        # Store exit locations for each level
        self._level_exit_blocks = {}
        for level, config in level_configs.items():
            # early_exit_location is 1-indexed, represents the block where exit happens
            # blocks 0 to (exit_loc - 1) are used for training
            self._level_exit_blocks[level] = config['early_exit_location']

    def get_training_config(self, round_idx: int, level: int) -> Dict[str, Any]:
        """
        Returns the training configuration for a given level at a given round.

        Returns dict with:
            - mode: 'normal' or 'borrow'
            - model_config: Configuration to use for model creation
            - frozen_blocks: Range of blocks to freeze (start, end) - exclusive end
            - active_blocks: Range of blocks to train (start, end) - exclusive end
            - source_level: Which level's config we're borrowing from
        """
        phase = self.scheduler.get_phase(round_idx)
        sorted_levels = sorted(self.level_configs.keys())
        max_level = max(sorted_levels)

        own_exit = self._level_exit_blocks[level]

        # Normal mode for all levels in phase 0, or highest level in borrow phase
        if phase == 0 or level == max_level:
            return {
                'mode': 'normal',
                'model_config': self.level_configs[level],
                'frozen_blocks': (0, 0),  # No freezing
                'active_blocks': (0, own_exit),  # Train blocks 0 to own_exit-1
                'source_level': level
            }

        # Borrow mode: find next higher level
        higher_level = None
        for l in sorted_levels:
            if l > level:
                higher_level = l
                break

        if higher_level is None:
            return {
                'mode': 'normal',
                'model_config': self.level_configs[level],
                'frozen_blocks': (0, 0),
                'active_blocks': (0, own_exit),
                'source_level': level
            }

        higher_exit = self._level_exit_blocks[higher_level]

        return {
            'mode': 'borrow',
            'model_config': self.level_configs[higher_level],
            'frozen_blocks': (0, own_exit),  # Freeze blocks 0 to own_exit-1
            'active_blocks': (own_exit, higher_exit),  # Train blocks own_exit to higher_exit-1
            'source_level': higher_level
        }

    def freeze_blocks(self, model: nn.Module, freeze_range: Tuple[int, int]) -> int:
        """
        Freeze specific blocks in the model.

        Args:
            model: The neural network model (ResNet)
            freeze_range: (start_block, end_block) - freeze blocks in [start, end)

        Returns:
            Number of parameters frozen
        """
        start_block, end_block = freeze_range
        if start_block >= end_block:
            return 0

        frozen_count = 0

        # Freeze initial conv if block 0 is in freeze range
        if start_block == 0:
            if hasattr(model, 'conv1'):
                for param in model.conv1.parameters():
                    param.requires_grad = False
                    frozen_count += param.numel()
            if hasattr(model, 'bn1'):
                for param in model.bn1.parameters():
                    param.requires_grad = False
                    frozen_count += param.numel()

        # Freeze blocks in each stage
        if hasattr(model, 'layers'):
            current_block = 0

            for stage_idx, stage in enumerate(model.layers):
                # stage is nn.ModuleList of nn.Sequential (sub-layers divided by early exits)
                for sublayer_idx, sublayer in enumerate(stage):
                    # sublayer is nn.Sequential containing BasicBlocks
                    for block in sublayer.modules():
                        if block.__class__.__name__ in ['BasicBlock', 'Bottleneck']:
                            if start_block <= current_block < end_block:
                                for param in block.parameters():
                                    param.requires_grad = False
                                    frozen_count += param.numel()
                            current_block += 1

        # Also freeze early exit classifiers for frozen blocks
        if hasattr(model, 'ee_classifiers'):
            # ee_classifiers are placed at early exit locations
            # We need to freeze classifiers that correspond to frozen exits
            current_block = 0
            for stage_idx, ee_stage in enumerate(model.ee_classifiers):
                for classifier in ee_stage:
                    # This classifier is at some exit point
                    # For simplicity, freeze if the exit point is within frozen range
                    # (This is approximate - exact mapping requires more info)
                    pass  # Skip for now - classifiers are small

        return frozen_count

    def freeze_blocks_v2(self, model: nn.Module, freeze_range: Tuple[int, int]) -> int:
        """
        Freeze specific blocks using parameter name matching.
        More robust approach that works with different model structures.

        Args:
            model: The neural network model (ResNet)
            freeze_range: (start_block, end_block) - freeze blocks in [start, end)

        Returns:
            Number of parameters frozen
        """
        start_block, end_block = freeze_range
        if start_block >= end_block:
            return 0

        frozen_count = 0

        # First, count total blocks to understand the model structure
        total_blocks = self._count_blocks(model)
        print(f"  [TDD] Model has {total_blocks} blocks, freezing range [{start_block}, {end_block})")

        # Freeze initial conv if starting from block 0
        if start_block == 0:
            for name, param in model.named_parameters():
                if name.startswith('conv1') or name.startswith('bn1') or name.startswith('scaler'):
                    param.requires_grad = False
                    frozen_count += param.numel()

        # Build block index mapping
        block_idx = 0
        if hasattr(model, 'layers'):
            for stage_idx, stage in enumerate(model.layers):
                for sublayer_idx, sublayer in enumerate(stage):
                    # Count blocks in this sublayer
                    blocks_in_sublayer = self._count_blocks_in_module(sublayer)

                    # Determine overlap with freeze range
                    sublayer_start = block_idx
                    sublayer_end = block_idx + blocks_in_sublayer

                    # Check if this sublayer overlaps with freeze range
                    if sublayer_end > start_block and sublayer_start < end_block:
                        # Need to freeze some or all blocks in this sublayer
                        for local_idx, block in enumerate(sublayer):
                            global_idx = block_idx + local_idx
                            if start_block <= global_idx < end_block:
                                for param in block.parameters():
                                    param.requires_grad = False
                                    frozen_count += param.numel()

                    block_idx += blocks_in_sublayer

        return frozen_count

    def _count_blocks(self, model: nn.Module) -> int:
        """Count total number of BasicBlock/Bottleneck in the model."""
        count = 0
        for module in model.modules():
            if module.__class__.__name__ in ['BasicBlock', 'Bottleneck']:
                count += 1
        return count

    def _count_blocks_in_module(self, module: nn.Module) -> int:
        """Count BasicBlock/Bottleneck in a specific module."""
        count = 0
        for m in module.modules():
            if m.__class__.__name__ in ['BasicBlock', 'Bottleneck']:
                count += 1
        return count

    def unfreeze_all(self, model: nn.Module) -> None:
        """Unfreeze all layers in the model."""
        for param in model.parameters():
            param.requires_grad = True

    def log_config(self, round_idx: int) -> str:
        """Generate a log string describing the current TDD configuration."""
        phase_info = self.scheduler.get_phase_info(round_idx)

        lines = [
            f"\n=== Time-Domain Decomposition (Round {round_idx}) ===",
            f"Phase: {phase_info['phase_name']} ({phase_info['rounds_completed_in_phase']}/{phase_info['rotation_period']})",
            f"TDD Enabled: {phase_info['tdd_enabled']}",
            ""
        ]

        sorted_levels = sorted(self.level_configs.keys())
        for level in sorted_levels:
            config = self.get_training_config(round_idx, level)
            own_exit = self._level_exit_blocks[level]

            if config['mode'] == 'normal':
                lines.append(f"Level {level}: Normal mode, training blocks 0-{own_exit-1}")
            else:
                frozen = config['frozen_blocks']
                active = config['active_blocks']
                lines.append(
                    f"Level {level}: Borrow from Level {config['source_level']}, "
                    f"frozen=[0-{frozen[1]-1}], active=[{active[0]}-{active[1]-1}]"
                )

        return '\n'.join(lines)


def create_tdd_components(args, level_configs: Dict[int, Dict]) -> Optional[TimeDomainDecomposer]:
    """Factory function to create TDD components from args."""
    enable_tdd = getattr(args, 'enable_tdd', False)
    rotation_period = getattr(args, 'rotation_period', 10)

    if not enable_tdd:
        return None

    scheduler = DynamicScheduler(
        rotation_period=rotation_period,
        enable_tdd=True
    )

    decomposer = TimeDomainDecomposer(scheduler, level_configs)

    return decomposer
