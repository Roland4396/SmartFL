# git-disl/scale-fl/scale-fl-c2084e461cef83751b99a958198aced66c1d7d7c/fed.py

import copy
import datetime as dt
import os
import pickle as pkl
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from data_tools.dataloader import get_client_dataloader
from predict import local_validate
from train import execute_epoch
from utils.grad_traceback import get_downscale_index
from utils.phase_timing import append_phase_timing_rows, phase_timing_enabled
from utils.utils import save_checkpoint
from time_domain_decomposition import create_tdd_components, DynamicScheduler, TimeDomainDecomposer
from hierarchical_model_selector import (
    find_best_config_for_distribution,
    find_best_config_independent,
    load_configs_from_json,
    find_all_growth_configs
)

mp.set_start_method('spawn', force=True)


class Federator:
    def __init__(self, global_model, args, client_groups=[]):
        self.global_model = global_model
        self.args = args  # Store args for TDD initialization

        self.vertical_scale_ratios = args.vertical_scale_ratios
        self.horizontal_scale_ratios = args.horizontal_scale_ratios
        self.client_split_ratios = args.client_split_ratios

        assert len(self.vertical_scale_ratios) == len(self.horizontal_scale_ratios) == len(self.client_split_ratios)

        self.num_rounds = args.num_rounds
        self.num_clients = args.num_clients
        self.sample_rate = args.sample_rate
        self.alpha = args.alpha
        self.num_levels = len(self.vertical_scale_ratios)
        self.idx_dicts = [get_downscale_index(self.global_model, args, s) for s in self.vertical_scale_ratios]
        self.client_groups = client_groups

        self.use_gpu = args.use_gpu

        # Time-Domain Decomposition (TDD) components
        self.tdd_enabled = getattr(args, 'enable_tdd', 0) == 1
        self.tdd_scheduler = None
        self.tdd_decomposer = None
        self.tdd_level_configs = None  # Will be initialized on first round with participating levels

        # New TDD: Device-level mixing (Normal + Growth)
        self.tdd_normal_configs = None  # Normal configs for each level
        self.tdd_growth_configs = None  # Growth configs for each level
        self.tdd_growth_ratio = getattr(args, 'tdd_growth_ratio', 0.5)  # 50% devices use Growth
        self.tdd_all_model_configs = None  # Cache the full model library

        if self.tdd_enabled:
            rotation_period = getattr(args, 'rotation_period', 10)
            self.tdd_scheduler = DynamicScheduler(rotation_period=rotation_period, enable_tdd=True)
            print(f"[TDD] Time-Domain Decomposition ENABLED")
            print(f"[TDD] Device-level mixing: {self.tdd_growth_ratio*100:.0f}% Growth, {(1-self.tdd_growth_ratio)*100:.0f}% Normal")

    def _empty_val_results(self):
        nan = float('nan')
        return nan, nan, nan, np.array([nan]), np.array([nan])

    def _empty_local_val_results(self):
        nan = float('nan')
        return [[nan, nan, nan] for _ in range(self.num_levels + 1)]

    def _should_validate_round(self, args, round_idx):
        validate_every = max(1, getattr(args, 'validate_every', 1))
        if validate_every == 1:
            return True
        if round_idx == args.start_round:
            return True
        if round_idx == self.num_rounds - 1:
            return True
        return (round_idx + 1) % validate_every == 0

    def fed_train(self, train_set, val_set, user_groups, criterion, args, batch_size, train_params):

        scores = ['epoch\ttrain_loss\tval_loss\tval_acc1\tval_acc5\tlocal_val_acc1\tlocal_val_acc5' +
                  '\tlocal_val_acc1' * self.num_levels]
        best_acc1, best_round = 0.0, 0
        last_val_results = None
        last_local_val_results = None
        last_validated_round = None

        # pre-assignment of levels to clients (needs to be saved for inference)
        if not self.client_groups:
            client_idxs = np.arange(self.num_clients)
            np.random.seed(args.seed)
            shuffled_client_idxs = np.random.permutation(client_idxs)
            client_groups = []
            s = 0
            for ratio in self.client_split_ratios:
                e = s + int(len(shuffled_client_idxs) * ratio)
                client_groups.append(shuffled_client_idxs[s: e])
                s = e
            self.client_groups = client_groups

            with open(os.path.join(args.save_path, 'client_groups.pkl'), 'wb') as f:
                pkl.dump(self.client_groups, f)

        for round_idx in range(args.start_round, self.num_rounds):

            print(f'\n | Global Training Round : {round_idx + 1} |\n')
            print(' | Regenerating masks for the current round... |\n')
            self.idx_dicts = [get_downscale_index(self.global_model, args, s) for s in self.vertical_scale_ratios]

            train_loss, val_results, local_val_results, did_validate = \
                self.execute_round(train_set, val_set, user_groups, criterion, args, batch_size,
                                   train_params, round_idx)
            if did_validate:
                last_val_results = val_results
                last_local_val_results = local_val_results
                last_validated_round = round_idx
            else:
                if last_val_results is None:
                    last_val_results = self._empty_val_results()
                if last_local_val_results is None:
                    last_local_val_results = self._empty_local_val_results()
                print(
                    f"[VALIDATION] Skipped round {round_idx}; "
                    f"reusing metrics from round {last_validated_round if last_validated_round is not None else 'N/A'}"
                )
                val_results = last_val_results
                local_val_results = last_local_val_results

            val_loss, val_acc1, val_acc5, _, _ = val_results

            scores.append(('{}' + '\t{:.4f}' * int(6 + self.num_levels))
                          .format(round_idx, train_loss, val_loss, val_acc1, val_acc5,
                                  local_val_results[-1][1], local_val_results[-1][2],
                                  *[l[1] for l in local_val_results[:-1]]))

            is_best = did_validate and val_acc1 > best_acc1
            if is_best:
                best_acc1 = val_acc1
                best_round = round_idx
                print('Best var_acc1 {}'.format(best_acc1))

            model_filename = 'checkpoint_%03d.pth.tar' % round_idx
            save_checkpoint({
                'round': round_idx,
                'arch': args.arch,
                'state_dict': self.global_model.state_dict(),
                'best_acc1': best_acc1,
            }, args, is_best, model_filename, scores)

        return best_acc1, best_round

    def get_level(self, client_idx):
        # Return the complexity level of given client, starts with 0
        try:
            level = np.where([client_idx in c for c in self.client_groups])[0][0]
        except:
            # client will be skipped
            level = -1

        return level

    def _freeze_blocks_by_range(self, model, freeze_range):
        """
        Freeze specific blocks in the model by setting requires_grad=False.

        Args:
            model: The neural network model (ResNet/VGG)
            freeze_range: (start_block, end_block) - freeze blocks in [start, end)

        Returns:
            Number of parameters frozen
        """
        start_block, end_block = freeze_range
        if start_block >= end_block:
            return 0

        frozen_count = 0
        total_blocks = 0

        # Freeze initial conv and bn if starting from block 0
        if start_block == 0:
            if hasattr(model, 'conv1'):
                for param in model.conv1.parameters():
                    param.requires_grad = False
                    frozen_count += param.numel()
            if hasattr(model, 'bn1'):
                for param in model.bn1.parameters():
                    param.requires_grad = False
                    frozen_count += param.numel()

        # Freeze blocks in model.layers (ResNet structure)
        if hasattr(model, 'layers'):
            current_block = 0
            print(f"    [DEBUG] Model has {len(model.layers)} stages")
            for stage_idx, stage in enumerate(model.layers):
                stage_blocks = 0
                # stage is nn.ModuleList of nn.Sequential
                for sublayer in stage:
                    # sublayer is nn.Sequential containing BasicBlocks
                    for block in sublayer:
                        if hasattr(block, 'conv1'):  # It's a BasicBlock or Bottleneck
                            if start_block <= current_block < end_block:
                                for param in block.parameters():
                                    param.requires_grad = False
                                    frozen_count += param.numel()
                            current_block += 1
                            stage_blocks += 1
                print(f"    [DEBUG] Stage {stage_idx}: {stage_blocks} blocks, total so far: {current_block}")
            total_blocks = current_block
        elif hasattr(model, 'features'):
            current_block = 0
            print(f"    [DEBUG] VGG has {len(model.features)} feature modules")
            for feature_idx, layer in enumerate(model.features):
                has_conv = False
                for module in layer.modules():
                    if module.__class__.__name__ == 'Conv2d':
                        has_conv = True
                        break
                if not has_conv:
                    continue

                if start_block <= current_block < end_block:
                    for param in layer.parameters():
                        param.requires_grad = False
                        frozen_count += param.numel()
                current_block += 1
                print(f"    [DEBUG] Feature block {feature_idx}: counted as logical block {current_block}")
            total_blocks = current_block
        elif hasattr(model, 'block'):
            if start_block == 0 and hasattr(model, 'pre'):
                for param in model.pre.parameters():
                    param.requires_grad = False
                    frozen_count += param.numel()

            total_blocks = len(model.block)
            print(f"    [DEBUG] MobileNet has {total_blocks} blocks")
            for block_idx, block in enumerate(model.block):
                if start_block <= block_idx < end_block:
                    for param in block.parameters():
                        param.requires_grad = False
                        frozen_count += param.numel()
                print(
                    f"    [DEBUG] MobileNet block {block_idx}: "
                    f"{'frozen' if start_block <= block_idx < end_block else 'active'}"
                )

        # NOTE: 不冻结 early exit classifiers！
        # Growth 模式需要用到最后一个 exit 的 classifier 来计算 loss
        # 只冻结 backbone blocks，保持所有 classifiers 可训练

        # Debug: count trainable params
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"    [DEBUG] freeze_range={freeze_range}, total_blocks={total_blocks}")
        print(f"    [DEBUG] frozen_params={frozen_count}, trainable_params={trainable_params}, total_params={total_params}")

        return frozen_count

    def _find_exit_index_for_location(self, exit_location):
        """
        找到 exit_location 对应的 exit index。

        Args:
            exit_location: early exit 的 block 位置

        Returns:
            exit index，如果找不到返回 None
        """
        # 构建完整的 exit 位置列表（normal + growth）
        all_exits = set()

        # 从 normal configs 获取
        if self.tdd_normal_configs:
            for cfg in self.tdd_normal_configs.values():
                all_exits.add(cfg['early_exit_location'])

        # 从 growth configs 获取
        if self.tdd_growth_configs:
            for cfg in self.tdd_growth_configs.values():
                if cfg is not None:
                    all_exits.add(cfg['early_exit_location'])

        # 排序并去掉最大的（最大的是完整模型输出，不在 ee_layer_locations 中）
        all_exits = sorted(all_exits)
        if all_exits:
            max_exit = all_exits[-1]
            ee_locs = [e for e in all_exits if e < max_exit]

            if exit_location in ee_locs:
                return ee_locs.index(exit_location)
            # 如果是最大 exit，返回最后一个 index
            if exit_location == max_exit:
                return len(ee_locs)

        return None

    def _get_model_for_exit(self, exit_idx, scale):
        """
        获取指定 exit index 的模型副本。

        Args:
            exit_idx: exit 的 index
            scale: 模型宽度缩放

        Returns:
            配置好的模型副本
        """
        import copy
        model = copy.deepcopy(self.global_model)

        # 如果需要缩放宽度
        if scale != 1 and hasattr(model, 'stored_inp_kwargs'):
            model_kwargs = copy.deepcopy(model.stored_inp_kwargs)
            if 'scale' in model_kwargs:
                model_kwargs['scale'] = scale
            elif 'params' in model_kwargs and 'scale' in model_kwargs['params']:
                model_kwargs['params']['scale'] = scale

            # 重新创建模型
            local_model = type(self.global_model)(**model_kwargs)

            # 复制权重
            local_state_dict = local_model.state_dict()
            for n, p in self.global_model.state_dict().items():
                if n in local_state_dict:
                    if local_state_dict[n].shape == p.shape:
                        local_state_dict[n] = p
                    elif 'num_batches_tracked' in n:
                        local_state_dict[n] = p
            local_model.load_state_dict(local_state_dict)
            model = local_model

        return model

    def _get_level_configs(self, participating_levels, args):
        """
        Get level configurations from beam search.
        This mirrors the logic in model __init__ methods (e.g., ResNet, VGG).

        Args:
            participating_levels: List of participating level indices
            args: Arguments namespace

        Returns:
            Dictionary mapping level -> configuration, or None if failed
        """
        try:
            # Determine model type and config library path
            if hasattr(args, 'arch') and args.arch:
                if 'mobilenet' in args.arch:
                    model_name = 'mobilenet'
                elif 'resnet' in args.arch:
                    model_name = 'resnet'
                elif 'vgg' in args.arch:
                    model_name = 'vgg'
                else:
                    model_name = getattr(args, 'model', 'resnet')
            else:
                model_name = getattr(args, 'model', 'resnet')

            dataset_name = getattr(args, 'data', 'cifar100')

            if args.config_library_path:
                config_library_path = args.config_library_path
            else:
                config_library_path = f"{model_name}_{dataset_name}_architecture_library.json"

            # Load configurations
            all_model_configs = load_configs_from_json(
                config_library_path,
                model_type=model_name,
                dataset=dataset_name
            )

            if not all_model_configs:
                print("[TDD] Warning: No configurations loaded from library")
                return None

            # Build constraints
            flops_constraints = None
            if args.flops_constraints:
                flops_constraints = {i: val for i, val in enumerate(args.flops_constraints)}

            params_constraints = None
            if args.params_constraints:
                params_constraints = {i: val for i, val in enumerate(args.params_constraints)}

            # Get best configs using beam search (or independent selection)
            if getattr(args, 'independent_selection', False):
                best_configs = find_best_config_independent(
                    all_model_configs,
                    participating_levels,
                    flops_constraints=flops_constraints,
                    params_constraints=params_constraints
                )
            else:
                best_configs = find_best_config_for_distribution(
                    all_model_configs,
                    participating_levels,
                    beam_width=500,
                    flops_constraints=flops_constraints,
                    params_constraints=params_constraints
                )

            return best_configs

        except Exception as e:
            print(f"[TDD] Warning: Failed to get level configs: {e}")
            return None

    def execute_round(self, train_set, val_set, user_groups, criterion, args, batch_size, train_params, round_idx):
        timing_enabled = phase_timing_enabled(args)
        round_start = time.perf_counter()

        self.global_model.train()
        m = max(int(self.sample_rate * self.num_clients), 1)
        client_idxs = np.random.choice(range(self.num_clients), m, replace=False)

        client_loader_start = time.perf_counter()
        client_train_loaders = [get_client_dataloader(train_set, user_groups[0][client_idx], args, batch_size, loader_role='train') for
                                client_idx in client_idxs]
        client_loader_time = time.perf_counter() - client_loader_start
        levels = [self.get_level(client_idx) for client_idx in client_idxs]
        scales = [self.vertical_scale_ratios[level] for level in levels]
        levels_in_round = [self.get_level(cid) for cid in client_idxs]
        participating_levels = sorted(list(set(l for l in levels_in_round if l != -1)))

        # === TDD: Initialize configs on first round ===
        if self.tdd_enabled and self.tdd_normal_configs is None:
            # Get normal configs from beam search
            self.tdd_normal_configs = self._get_level_configs(participating_levels, args)

            if self.tdd_normal_configs is not None:
                print(f"\n[TDD] Normal configs from beam search:")
                for level, config in sorted(self.tdd_normal_configs.items()):
                    print(f"  Level {level}: exit={config['early_exit_location']}, "
                          f"width={config['width_multipliers']}, params={config['num_params']:.0f}")

                # Load full model library and find growth configs
                model_name = 'resnet' if 'resnet' in args.arch else ('vgg' if 'vgg' in args.arch else 'mobilenet')
                dataset_name = getattr(args, 'data', 'cifar100')
                config_path = args.config_library_path or f"{model_name}_{dataset_name}_architecture_library.json"
                self.tdd_all_model_configs = load_configs_from_json(config_path, model_type=model_name)

                # Find growth configs for each level
                self.tdd_growth_configs = find_all_growth_configs(
                    self.tdd_normal_configs,
                    self.tdd_all_model_configs,
                    model_type=model_name
                )

                # === TDD: Update horizontal_scale_ratios based on actual exit structure ===
                # horizontal_scale_ratios 表示每个 level 使用多少个 exit
                # 需要根据 tdd_normal_configs 的 early_exit_location 来计算
                new_h_scale_ratios = []
                for level in range(self.num_levels):
                    if level in self.tdd_normal_configs:
                        exit_location = self.tdd_normal_configs[level]['early_exit_location']
                        # 找到这个 exit_location 对应的 exit index
                        exit_idx = self._find_exit_index_for_location(exit_location)
                        if exit_idx is not None:
                            # horizontal_scale_ratios 需要的是 exit 数量（1-based）
                            # exit_idx 是 0-based，所以 +1
                            new_h_scale_ratios.append(exit_idx + 1)
                        else:
                            # Fallback: 使用原始值
                            new_h_scale_ratios.append(self.horizontal_scale_ratios[level])
                    else:
                        new_h_scale_ratios.append(self.horizontal_scale_ratios[level])

                # 更新 horizontal_scale_ratios
                old_ratios = self.horizontal_scale_ratios.copy()
                self.horizontal_scale_ratios = new_h_scale_ratios
                print(f"\n[TDD] Updated horizontal_scale_ratios: {old_ratios} -> {new_h_scale_ratios}")

        # === TDD: Assign Normal/Growth mode to each client ===
        tdd_client_modes = {}  # {client_idx: 'normal' or 'growth'}
        if self.tdd_enabled and self.tdd_normal_configs is not None:
            print(f"\n[TDD] Round {round_idx} - Device-level mixing:")
            for i, level in enumerate(levels):
                if level == -1:
                    continue

                # Check if this level has a growth config
                growth_config = self.tdd_growth_configs.get(level) if self.tdd_growth_configs else None

                if growth_config is None:
                    # No growth config available, use normal
                    tdd_client_modes[i] = 'normal'
                else:
                    # Randomly assign Normal or Growth based on ratio
                    if np.random.random() < self.tdd_growth_ratio:
                        tdd_client_modes[i] = 'growth'
                    else:
                        tdd_client_modes[i] = 'normal'

            # Log assignment summary
            normal_count = sum(1 for m in tdd_client_modes.values() if m == 'normal')
            growth_count = sum(1 for m in tdd_client_modes.values() if m == 'growth')
            print(f"  Assigned: {normal_count} Normal, {growth_count} Growth")

        # === Get local models (with TDD support) ===
        local_models = []
        tdd_frozen_blocks = []  # Track frozen block ranges for each client
        tdd_exit_indices = []  # Track exit index for each client (for training)
        tdd_is_growth = []  # Track whether each client is in Growth mode
        local_model_build_start = time.perf_counter()
        for i in range(len(client_idxs)):
            level = levels[i]
            scale = scales[i]

            if self.tdd_enabled and i in tdd_client_modes:
                mode = tdd_client_modes[i]

                if mode == 'growth' and self.tdd_growth_configs.get(level) is not None:
                    # Growth mode: use same model as Normal, but train with growth_exit
                    growth_config = self.tdd_growth_configs[level]
                    frozen_range = growth_config['frozen_range']
                    active_range = growth_config['active_range']
                    growth_exit = growth_config['early_exit_location']

                    # 找到 growth_exit 对应的 exit index
                    growth_exit_idx = self._find_exit_index_for_location(growth_exit)

                    if growth_exit_idx is not None:
                        # Growth 模式使用和 Normal 模式相同的模型（正确应用 scale）
                        # 但训练时使用 growth_exit_idx
                        local_model = self.get_local_split(level, scale, participating_levels)

                        # Freeze the blocks in frozen_range
                        frozen_count = self._freeze_blocks_by_range(local_model, frozen_range)
                        tdd_frozen_blocks.append(frozen_range)
                        tdd_exit_indices.append(growth_exit_idx)  # 训练时使用 growth exit index
                        tdd_is_growth.append(True)  # 是 Growth 模式

                        print(f"  [TDD] Client {i} (Level {level}): GROWTH mode, "
                              f"exit={growth_exit} (idx={growth_exit_idx}), "
                              f"frozen=[0,{frozen_range[1]}), active=[{active_range[0]},{active_range[1]}), "
                              f"frozen {frozen_count} params")
                    else:
                        # Fallback: growth_exit not found, use normal mode
                        local_model = self.get_local_split(level, scale, participating_levels)
                        tdd_frozen_blocks.append((0, 0))
                        tdd_exit_indices.append(level)  # 使用原始 level
                        tdd_is_growth.append(False)
                        print(f"  [TDD] Client {i} (Level {level}): NORMAL mode (growth_exit {growth_exit} not in model)")
                else:
                    # Normal mode
                    local_model = self.get_local_split(level, scale, participating_levels)
                    tdd_frozen_blocks.append((0, 0))
                    tdd_exit_indices.append(level)  # 使用原始 level
                    tdd_is_growth.append(False)
                    if mode == 'growth':
                        print(f"  [TDD] Client {i} (Level {level}): NORMAL mode (no growth config)")
                    else:
                        print(f"  [TDD] Client {i} (Level {level}): NORMAL mode")
            else:
                # TDD disabled or not applicable
                local_model = self.get_local_split(level, scale, participating_levels)
                tdd_frozen_blocks.append((0, 0))
                tdd_exit_indices.append(level)  # 使用原始 level
                tdd_is_growth.append(False)

            local_models.append(local_model)
        local_model_build_time = time.perf_counter() - local_model_build_start

        h_scale_ratios = [self.horizontal_scale_ratios[level] for level in levels]

        pool_args = [train_set, user_groups, criterion, args, batch_size, train_params, round_idx]
        local_weights = []
        local_losses = []
        local_grad_flags = []
        client_train_total_time = 0.0
        client_weight_upload_time = 0.0
        client_timing_rows = []
        pool_args.append(None)

        for i, client_idx in enumerate(client_idxs):
            # 使用 tdd_exit_indices 作为 exit index（Growth 模式用 growth exit，Normal 模式用原始 level）
            exit_idx_for_training = tdd_exit_indices[i] if tdd_exit_indices else levels[i]
            is_growth_mode = tdd_is_growth[i] if tdd_is_growth else False

            # Growth 模式下，h_scale_ratio 需要与 exit_idx_for_training 匹配
            # h_scale_ratio 表示使用多少个 exit（传给 execute_epoch 作为 h_level）
            if is_growth_mode:
                # Growth 模式：exit_idx_for_training 是 0-based index，h_scale_ratio 需要 +1
                h_scale_ratio_for_client = exit_idx_for_training + 1
            else:
                h_scale_ratio_for_client = h_scale_ratios[i]

            client_args = pool_args + [local_models[i], client_train_loaders[i], exit_idx_for_training, scales[i], h_scale_ratio_for_client, client_idx, is_growth_mode]
            client_train_start = time.perf_counter()
            result = execute_client_round(client_args)
            client_train_duration = time.perf_counter() - client_train_start
            client_train_total_time += client_train_duration

            weight_upload_start = time.perf_counter()
            if args.use_gpu:
                for k, v in result[0].items():
                    result[0][k] = v.cuda(0)
            weight_upload_duration = time.perf_counter() - weight_upload_start
            client_weight_upload_time += weight_upload_duration

            local_weights.append(result[0])
            local_grad_flags.append(result[1])
            local_losses.append(result[2])
            if timing_enabled:
                client_timing_rows.append([
                    round_idx,
                    i,
                    client_idx,
                    levels[i],
                    int(is_growth_mode),
                    client_train_duration,
                    weight_upload_duration,
                ])
            print(f'Client {i+1}/{len(client_idxs)} completely finished')

        train_loss = sum(local_losses) / len(client_idxs)

        # Update the global model
        # Growth 模式现在使用 get_local_split（和 Normal 模式相同），所以聚合时使用原始 levels
        aggregate_start = time.perf_counter()
        global_weights = self.average_weights(local_weights, local_grad_flags, levels, self.global_model, args)
        self.global_model.load_state_dict(global_weights)
        aggregate_time = time.perf_counter() - aggregate_start

        did_validate = self._should_validate_round(args, round_idx)
        validation_timing = {}
        validate_time = 0.0
        val_results = None
        local_val_results = None
        if did_validate:
            # Validation for all clients
            if self.client_split_ratios[-1] == 0:
                level = np.where(self.client_split_ratios)[0].tolist()[-1]
                scale = self.vertical_scale_ratios[level]
                global_model = self.get_local_split(level, scale,participating_levels)
                if self.use_gpu:
                    global_model = global_model.cuda()
            else:
                global_model = copy.deepcopy(self.global_model)

            validate_start = time.perf_counter()
            validate_output = local_validate(
                self,
                participating_levels,
                val_set,
                user_groups[1],
                criterion,
                args,
                512,
                global_model,
                return_timing=timing_enabled,
            )
            validate_time = time.perf_counter() - validate_start
            if timing_enabled:
                val_results, local_val_results, validation_timing = validate_output
            else:
                val_results, local_val_results = validate_output
        else:
            print(
                f"[VALIDATION] Round {round_idx} skipped "
                f"(validate_every={max(1, getattr(args, 'validate_every', 1))})"
            )

        if timing_enabled:
            round_total_time = time.perf_counter() - round_start
            validation_timing = validation_timing if 'validation_timing' in locals() else {}
            validation_loader_time = validation_timing.get('loader_s', 0.0)
            validation_model_build_time = validation_timing.get('model_build_s', 0.0)
            validation_single_model_time = validation_timing.get('single_model_validate_s', 0.0)
            validation_dual_model_time = validation_timing.get('dual_model_validate_s', 0.0)
            accounted_time = (
                client_loader_time +
                local_model_build_time +
                client_train_total_time +
                client_weight_upload_time +
                aggregate_time +
                validate_time
            )
            round_other_time = max(0.0, round_total_time - accounted_time)
            append_phase_timing_rows(
                args.save_path,
                'phase_timing_rounds.tsv',
                [
                    'round_idx',
                    'sampled_clients',
                    'client_loader_s',
                    'local_model_build_s',
                    'client_train_total_s',
                    'client_train_avg_s',
                    'client_weight_upload_s',
                    'aggregate_s',
                    'validate_total_s',
                    'validate_loader_s',
                    'validate_model_build_s',
                    'validate_single_model_s',
                    'validate_dual_model_s',
                    'validate_clients',
                    'round_other_s',
                    'round_total_s',
                ],
                [[
                    round_idx,
                    len(client_idxs),
                    client_loader_time,
                    local_model_build_time,
                    client_train_total_time,
                    client_train_total_time / max(len(client_idxs), 1),
                    client_weight_upload_time,
                    aggregate_time,
                    validate_time,
                    validation_loader_time,
                    validation_model_build_time,
                    validation_single_model_time,
                    validation_dual_model_time,
                    validation_timing.get('client_count', 0),
                    round_other_time,
                    round_total_time,
                ]],
            )
            append_phase_timing_rows(
                args.save_path,
                'phase_timing_clients.tsv',
                [
                    'round_idx',
                    'client_order',
                    'client_idx',
                    'level',
                    'is_growth_mode',
                    'train_s',
                    'weight_upload_s',
                ],
                client_timing_rows,
            )
            print(
                f"[TIMING] round={round_idx} total={round_total_time:.2f}s "
                f"prep(loader={client_loader_time:.2f}s, model={local_model_build_time:.2f}s) "
                f"train={client_train_total_time:.2f}s agg={aggregate_time:.2f}s "
                f"val={validate_time:.2f}s other={round_other_time:.2f}s"
            )

        return train_loss, val_results, local_val_results, did_validate

    def average_weights(self, w, grad_flags, levels, model, args):
        w_avg = copy.deepcopy(model.state_dict())

        for key in w_avg.keys():

            if 'num_batches_tracked' in key:
                w_avg[key] = w[0][key]
                continue

            if 'running' in key:
                w_avg[key] = sum([w_[key] for w_ in w]) / len(w)
                continue

            tmp = torch.zeros_like(w_avg[key])
            count = torch.zeros_like(tmp, dtype=torch.int64)

            for i in range(len(w)):
                if key not in grad_flags[i]:
                    continue
                if grad_flags[i][key]:
                    idx = self.idx_dicts[levels[i]][key]
                    idx = self.fix_idx_array(idx, w[i][key].shape)
                    tmp[idx] += w[i][key].flatten()
                    count[idx] += 1

            w_avg[key][count != 0] = tmp[count != 0]
            count[count == 0] = 1
            w_avg[key] = w_avg[key] / count

        return w_avg

    def get_idx_shape(self, inp, local_shape):
        # Return the output shape for binary mask input
        # [[1, 1, 0], [1, 1, 0], [0, 0, 0,]] -> [2, 2]
        if any([s == 0 for s in inp.shape]):
            print('Indexing error')
            raise RuntimeError

        if len(local_shape) == 4:
            dim_1 = inp.shape[2] // 2
            dim_2 = inp.shape[3] // 2
            idx_shape = (inp[:, 0, dim_1, dim_2].sum().item(),
                         inp[0, :, dim_1, dim_2].sum().item(), *local_shape[2:])
        elif len(local_shape) == 2:
            idx_shape = (inp[:, 0].sum().item(),
                         inp[0, :].sum().item())
        else:
            idx_shape = (inp.sum(),)

        return idx_shape

    def fix_idx_array(self, idx_array, local_shape):
        idx_shape = self.get_idx_shape(idx_array, local_shape)
        if all([idx_shape[i] >= local_shape[i] for i in range(len(local_shape))]):
            pass
        else:
            idx_array = idx_array[idx_array.sum(dim=1).argmax()].repeat((idx_array.shape[0], 1))
            idx_shape = self.get_idx_shape(idx_array, local_shape)

        ind_list = [slice(None)] * len(idx_array.shape)
        for i in range(len(local_shape)):

            lim = idx_array.shape[i]
            while idx_shape[i] != local_shape[i]:
                lim -= 1
                ind_list[i] = slice(0, lim)
                idx_shape = self.get_idx_shape(idx_array[tuple(ind_list)], local_shape)

        tmp = torch.zeros_like(idx_array, dtype=bool)
        tmp[tuple(ind_list)] = idx_array[tuple(ind_list)]
        idx_array = tmp

        if len(idx_array.shape) == 4:
            dim_1 = idx_array.shape[2] // 2
            dim_2 = idx_array.shape[3] // 2
            if idx_array.sum(dim=0).sum(dim=0)[0, 0] != idx_array.sum(dim=0).sum(dim=0)[dim_1, dim_2]:
                idx_array = idx_array[:, :, dim_1, dim_2].repeat(idx_array.shape[2], idx_array.shape[3], 1, 1).permute(
                    2, 3, 0, 1)
        return idx_array

    def get_local_split(self, level, scale,participating_levels):
        if scale == 1:
            return copy.deepcopy(self.global_model)

        model_kwargs = copy.deepcopy(self.global_model.stored_inp_kwargs)
        if 'scale' in model_kwargs.keys():
            model_kwargs['scale'] = scale
        else:
            model_kwargs['params']['scale'] = scale
        local_model = type(self.global_model)(**model_kwargs,)
        if 'bert' in str(type(local_model)):
            local_model.add_exits(model_kwargs['ee_layer_locations'])

        local_state_dict = local_model.state_dict()
        global_state_dict = self.global_model.state_dict()
        level_idx_dict = self.idx_dicts[level]

        for n, p in global_state_dict.items():

            if 'num_batches_tracked' in n:
                local_state_dict[n] = p
                continue

            # Skip if key doesn't exist in local model (e.g., shortcut missing due to scale)
            if n not in local_state_dict.keys():
                continue

            global_shape = p.shape
            local_shape = local_state_dict[n].shape

            if len(global_shape) != len(local_shape):
                print('Models are not alignable!')
                raise RuntimeError

            idx_array = self.fix_idx_array(level_idx_dict[n], local_shape)
            local_state_dict[n] = p[idx_array].reshape(local_shape)

        local_model.load_state_dict(local_state_dict)

        return local_model


def execute_client_round(args):
    train_set, user_groups, criterion, args, batch_size, train_params, round_idx, global_model, \
    local_model, client_train_loader, level, scale, h_scale_ratio, client_idx, is_growth_mode = args

    if args.use_gpu:
        local_model = local_model.cuda()

    # Growth 模式下，只优化可训练的参数
    base_params = [v for k, v in local_model.named_parameters() if 'ee_' not in k and v.requires_grad]
    exit_params = [v for k, v in local_model.named_parameters() if 'ee_' in k and v.requires_grad]

    optimizer = torch.optim.SGD([{'params': base_params},
                                 {'params': exit_params}],
                                lr=train_params['lr'],
                                momentum=train_params['momentum'],
                                weight_decay=train_params['weight_decay'])

    loss = 0.0
    for epoch in range(train_params['num_epoch']):
        print(f'{client_idx}-{epoch}-{dt.datetime.now()}')
        iter_idx = round_idx
        loss = execute_epoch(local_model, client_train_loader, criterion, optimizer, iter_idx, epoch,
                             args, train_params, h_scale_ratio, level, global_model, is_growth_mode=is_growth_mode)

    print(f'Finished epochs for {client_idx}')
    state_dict = local_model.state_dict(keep_vars=True)
    local_weights = {k: v.detach().cpu() for k, v in state_dict.items()}
    local_grad_flags = {k: v.grad is not None for k, v in state_dict.items()}

    del state_dict
    del optimizer
    del local_model

    return local_weights, local_grad_flags, loss
