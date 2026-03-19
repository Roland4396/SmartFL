# git-disl/scale-fl/scale-fl-c2084e461cef83751b99a958198aced66c1d7d7c/fed.py

import copy
import datetime as dt
import os
import pickle as pkl

import numpy as np
import torch
import torch.multiprocessing as mp

from data_tools.dataloader import get_client_dataloader
from predict import local_validate
from train import execute_epoch
from utils.grad_traceback import get_downscale_index
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

    def fed_train(self, train_set, val_set, user_groups, criterion, args, batch_size, train_params):

        scores = ['epoch\ttrain_loss\tval_loss\tval_acc1\tval_acc5\tlocal_val_acc1\tlocal_val_acc5' +
                  '\tlocal_val_acc1' * self.num_levels]
        best_acc1, best_round = 0.0, 0

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

            train_loss, val_results, local_val_results = \
                self.execute_round(train_set, val_set, user_groups, criterion, args, batch_size,
                                   train_params, round_idx)

            val_loss, val_acc1, val_acc5, _, _ = val_results

            scores.append(('{}' + '\t{:.4f}' * int(6 + self.num_levels))
                          .format(round_idx, train_loss, val_loss, val_acc1, val_acc5,
                                  local_val_results[-1][1], local_val_results[-1][2],
                                  *[l[1] for l in local_val_results[:-1]]))

            is_best = val_acc1 > best_acc1
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
        # DEBUG: Log round start
        import os
        debug_log_path = os.path.join(args.save_path, 'debug_execute_round.log')
        with open(debug_log_path, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"[DEBUG execute_round] Round {round_idx} starting...\n")
            f.write(f"  Vertical scale ratios: {self.vertical_scale_ratios}\n")
            f.write(f"  global_model.stored_inp_kwargs id: {id(self.global_model.stored_inp_kwargs)}\n")
            if 'scale' in self.global_model.stored_inp_kwargs:
                f.write(f"  global_model.stored_inp_kwargs['scale']: {self.global_model.stored_inp_kwargs['scale']}\n")
            elif 'params' in self.global_model.stored_inp_kwargs and 'scale' in self.global_model.stored_inp_kwargs['params']:
                f.write(f"  global_model.stored_inp_kwargs['params']['scale']: {self.global_model.stored_inp_kwargs['params']['scale']}\n")

        self.global_model.train()
        m = max(int(self.sample_rate * self.num_clients), 1)
        client_idxs = np.random.choice(range(self.num_clients), m, replace=False)

        client_train_loaders = [get_client_dataloader(train_set, user_groups[0][client_idx], args, batch_size) for
                                client_idx in client_idxs]
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

        h_scale_ratios = [self.horizontal_scale_ratios[level] for level in levels]

        pool_args = [train_set, user_groups, criterion, args, batch_size, train_params, round_idx]
        local_weights = []
        local_losses = []
        local_grad_flags = []
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
            result = execute_client_round(client_args)

            if args.use_gpu:
                for k, v in result[0].items():
                    result[0][k] = v.cuda(0)

            local_weights.append(result[0])
            local_grad_flags.append(result[1])
            local_losses.append(result[2])
            print(f'Client {i+1}/{len(client_idxs)} completely finished')

        train_loss = sum(local_losses) / len(client_idxs)

        # Update the global model
        # Growth 模式现在使用 get_local_split（和 Normal 模式相同），所以聚合时使用原始 levels
        global_weights = self.average_weights(local_weights, local_grad_flags, levels, self.global_model, args)
        self.global_model.load_state_dict(global_weights)

        # Validation for all clients
        if self.client_split_ratios[-1] == 0:
            level = np.where(self.client_split_ratios)[0].tolist()[-1]
            scale = self.vertical_scale_ratios[level]
            global_model = self.get_local_split(level, scale,participating_levels)
            if self.use_gpu:
                global_model = global_model.cuda()
        else:
            global_model = copy.deepcopy(self.global_model)

        val_results, local_val_results = local_validate(self,participating_levels,val_set, user_groups[1], criterion, args, 512,
                                                        global_model)

        return train_loss, val_results, local_val_results

    def average_weights(self, w, grad_flags, levels, model, args):
        # DEBUG: Check for NaN in local weights before averaging
        import os
        debug_log_path = os.path.join(args.save_path, 'debug_average_weights.log')
        os.makedirs(os.path.dirname(debug_log_path), exist_ok=True)

        # Track first call to log dtype info early
        if not hasattr(self, '_dtype_logged'):
            self._dtype_logged = False

        nan_detected_in_local = False
        with open(debug_log_path, 'a') as f:
            f.write(f"\n[DEBUG average_weights] Checking {len(w)} local weights from levels {levels}\n")
            for i, local_w in enumerate(w):
                for key, val in local_w.items():
                    if 'num_batches_tracked' in key:
                        continue
                    if torch.isnan(val).any():
                        f.write(f"  [ERROR] NaN detected in LOCAL weight {i} (level {levels[i]}), key: {key}\n")
                        nan_detected_in_local = True
                    elif torch.isinf(val).any():
                        f.write(f"  [ERROR] Inf detected in LOCAL weight {i} (level {levels[i]}), key: {key}\n")
                        nan_detected_in_local = True
                    # Check for very large values
                    max_val = val.abs().max().item()
                    if max_val > 1e5:
                        f.write(f"  [WARNING] Very large value in LOCAL weight {i} (level {levels[i]}), key: {key}, max={max_val:.2e}\n")

            if not nan_detected_in_local:
                f.write(f"  All local weights are clean (no NaN/Inf)\n")

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

            # DEBUG: Check if any client has abnormal values for THIS key
            debug_this_key = False
            for i in range(len(w)):
                # Skip if key doesn't exist in this client's grad_flags
                if key not in grad_flags[i]:
                    continue
                if grad_flags[i][key]:
                    if w[i][key].abs().max().item() > 1e4:  # Any client with large value
                        debug_this_key = True
                        break

            # Force logging for first round on deep layers
            if not self._dtype_logged and 'features.1' in key and '.weight' in key:
                debug_this_key = True
                self._dtype_logged = True
                with open(debug_log_path, 'a') as f:
                    f.write(f"\n  [FIRST ROUND DTYPE CHECK] Logging details for {key}:\n")
                    f.write(f"    w_avg[key].shape: {w_avg[key].shape}\n")

            if debug_this_key and w[0][key].abs().max().item() > 1e4:
                with open(debug_log_path, 'a') as f:
                    f.write(f"\n  [ALERT] Found large values in {key}, logging details:\n")
                    f.write(f"    w_avg[key].shape: {w_avg[key].shape}\n")

            # Store client info for retrospective logging
            client_info = []
            idx_dtypes = []  # Track idx dtypes
            for i in range(len(w)):
                # Skip if key doesn't exist in this client's grad_flags
                if key not in grad_flags[i]:
                    continue
                if grad_flags[i][key]:
                    idx = self.idx_dicts[levels[i]][key]
                    idx = self.fix_idx_array(idx, w[i][key].shape)
                    idx_dtypes.append((i, levels[i], idx.dtype, idx.min().item(), idx.max().item()))
                    client_max = w[i][key].abs().max().item()
                    idx_sum = idx.sum().item()
                    idx_numel = idx.numel()
                    client_shape = w[i][key].shape
                    client_info.append((i, levels[i], client_max, idx_sum, idx_numel, client_shape, True))

                    # DEBUG: Log if this key needs debugging
                    if debug_this_key:
                        with open(debug_log_path, 'a') as f:
                            f.write(f"    Client {i} (level {levels[i]}): w[i][key] max={client_max:.2e}, ")
                            f.write(f"idx.sum()={idx_sum}/{idx_numel}, local shape={client_shape}\n")

                    tmp[idx] += w[i][key].flatten()
                    count[idx] += 1
                else:
                    client_info.append((i, levels[i], 0, 0, 0, None, False))
                    if debug_this_key:
                        with open(debug_log_path, 'a') as f:
                            f.write(f"    Client {i} (level {levels[i]}): SKIPPED (grad_flag=False)\n")

            # Store aggregation stats before assignment
            tmp_max = tmp.abs().max().item()
            count_max = count.max().item()
            count_min = count.min().item()
            count_unique = count.unique().tolist()
            count_zero_cnt = (count == 0).sum().item()
            w_avg_before = w_avg[key].abs().max().item()

            # DEBUG: Check result if this key was flagged
            if debug_this_key:
                with open(debug_log_path, 'a') as f:
                    f.write(f"    idx dtypes and ranges:\n")
                    for ci, cl, idtype, idmin, idmax in idx_dtypes:
                        f.write(f"      Client {ci} (level {cl}): dtype={idtype}, min={idmin}, max={idmax}\n")
                    f.write(f"    tmp.dtype: {tmp.dtype}, count.dtype: {count.dtype}\n")
                    f.write(f"    After aggregation: tmp max: {tmp_max:.2e}\n")
                    f.write(f"    count: max={count_max}, min={count_min}, unique={count_unique}\n")
                    f.write(f"    count>0 positions: {count.numel() - count_zero_cnt}/{count.numel()}\n")
                    f.write(f"    Before assignment: w_avg[key] max: {w_avg_before:.2e}\n")

            w_avg[key][count != 0] = tmp[count != 0]
            count[count == 0] = 1
            w_avg[key] = w_avg[key] / count

            # DEBUG: Check result if this key was flagged OR if result is large
            final_max = w_avg[key].abs().max().item()
            if debug_this_key or final_max > 1e4:
                with open(debug_log_path, 'a') as f:
                    f.write(f"    After division: w_avg[key] max: {final_max:.2e}\n")
                    if final_max > 1e4 and not debug_this_key:
                        # Retrospective detailed logging
                        f.write(f"  [ALERT] {key} became large AFTER aggregation (was normal before)!\n")
                        f.write(f"  [RETROSPECTIVE] Full aggregation details:\n")
                        f.write(f"    w_avg[key].shape: {w_avg[key].shape}\n")
                        f.write(f"    w_avg[key].dtype: {w_avg[key].dtype}\n")
                        f.write(f"    tmp.dtype: {tmp.dtype}, count.dtype: {count.dtype}\n")
                        for ci, cl, cmax, isum, inumel, cshape, participated in client_info:
                            if participated:
                                f.write(f"    Client {ci} (level {cl}): w[i][key] max={cmax:.2e}, ")
                                f.write(f"idx.sum()={isum}/{inumel}, shape={cshape}\n")
                            else:
                                f.write(f"    Client {ci} (level {cl}): SKIPPED (grad_flag=False)\n")
                        f.write(f"    idx dtypes and ranges:\n")
                        for ci, cl, idtype, idmin, idmax in idx_dtypes:
                            f.write(f"      Client {ci} (level {cl}): dtype={idtype}, min={idmin}, max={idmax}\n")
                        f.write(f"    After aggregation: tmp max: {tmp_max:.2e}\n")
                        f.write(f"    count: max={count_max}, min={count_min}, unique={count_unique}\n")
                        f.write(f"    count==0 positions: {count_zero_cnt}/{count.numel()}\n")
                        f.write(f"    Before assignment: w_avg[key] max: {w_avg_before:.2e}\n")

        # DEBUG: Check for NaN in averaged weights
        nan_detected_in_avg = False
        with open(debug_log_path, 'a') as f:
            f.write(f"\n  Checking averaged weights for NaN/Inf:\n")
            for key, val in w_avg.items():
                if 'num_batches_tracked' in key:
                    continue
                if torch.isnan(val).any():
                    f.write(f"  [ERROR] NaN detected in AVERAGED weight, key: {key}\n")
                    nan_detected_in_avg = True
                elif torch.isinf(val).any():
                    f.write(f"  [ERROR] Inf detected in AVERAGED weight, key: {key}\n")
                    nan_detected_in_avg = True
                # Check for very large values
                max_val = val.abs().max().item()
                if max_val > 1e5:
                    f.write(f"  [WARNING] Very large value in AVERAGED weight, key: {key}, max={max_val:.2e}\n")

            if not nan_detected_in_avg:
                f.write(f"  All averaged weights are clean (no NaN/Inf)\n")

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
        model = copy.deepcopy(self.global_model)

        # DEBUG: Track stored_inp_kwargs before modification
        import os
        debug_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'debug_get_local_split.log')
        os.makedirs(os.path.dirname(debug_log_path), exist_ok=True)
        with open(debug_log_path, 'a') as f:
            f.write(f"\n[DEBUG get_local_split] Level={level}, scale={scale}\n")
            f.write(f"  global_model.stored_inp_kwargs id: {id(self.global_model.stored_inp_kwargs)}\n")
            if 'scale' in self.global_model.stored_inp_kwargs:
                f.write(f"  BEFORE: global_model.stored_inp_kwargs['scale']: {self.global_model.stored_inp_kwargs['scale']}\n")
            elif 'params' in self.global_model.stored_inp_kwargs and 'scale' in self.global_model.stored_inp_kwargs['params']:
                f.write(f"  BEFORE: global_model.stored_inp_kwargs['params']['scale']: {self.global_model.stored_inp_kwargs['params']['scale']}\n")

        if scale == 1:
            return model

        model_kwargs = model.stored_inp_kwargs
        # DEBUG: Check if this is a reference or copy
        with open(debug_log_path, 'a') as f:
            f.write(f"  model_kwargs id: {id(model_kwargs)}\n")
            f.write(f"  Are they same object? {id(model_kwargs) == id(self.global_model.stored_inp_kwargs)}\n")

        if 'scale' in model_kwargs.keys():
            model_kwargs['scale'] = scale
        else:
            model_kwargs['params']['scale'] = scale
        local_model = type(self.global_model)(**model_kwargs,)
        if 'bert' in str(type(local_model)):
            local_model.add_exits(model_kwargs['ee_layer_locations'])

        local_state_dict = local_model.state_dict()

        for n, p in self.global_model.state_dict().items():

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

            idx_array = self.fix_idx_array(self.idx_dicts[level][n], local_shape)
            local_state_dict[n] = p[idx_array].reshape(local_shape)

        local_model.load_state_dict(local_state_dict)

        # DEBUG: Check if global model's stored_inp_kwargs was modified
        with open(debug_log_path, 'a') as f:
            if 'scale' in self.global_model.stored_inp_kwargs:
                f.write(f"  AFTER: global_model.stored_inp_kwargs['scale']: {self.global_model.stored_inp_kwargs['scale']}\n")
            elif 'params' in self.global_model.stored_inp_kwargs and 'scale' in self.global_model.stored_inp_kwargs['params']:
                f.write(f"  AFTER: global_model.stored_inp_kwargs['params']['scale']: {self.global_model.stored_inp_kwargs['params']['scale']}\n")

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
    local_weights = {k: v.cpu() for k, v in local_model.state_dict(keep_vars=True).items()}
    local_grad_flags = {k: v.grad is not None for k, v in local_model.state_dict(keep_vars=True).items()}

    del local_model
    torch.cuda.empty_cache()

    return local_weights, local_grad_flags, loss
