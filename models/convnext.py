import copy

import torch
import torch.nn as nn

from args import arg_parser, modify_args
from hierarchical_model_selector import (
    find_all_growth_configs,
    find_best_config_for_distribution,
    find_best_config_independent,
    load_configs_from_json,
)
from models.convnext_utils import DropPath, GRN, LayerNorm
from models.model_utils import Scaler


CONVNEXT_V2_PICO_DIMS = [64, 128, 256, 512]
CONVNEXT_V2_PICO_DEPTHS = [2, 2, 6, 2]
MAX_CONVNEXT_STAGE_IDX = 3
DEFAULT_CONVNEXT_EXIT_LOCATIONS = [0, 1, 2]


def _sorted_unique_exit_locations(exit_locations):
    return sorted(set(exit_locations))


def _normalize_exit_locations(exit_locations):
    if exit_locations is None:
        return []

    normalized = _sorted_unique_exit_locations(exit_locations)
    invalid = [loc for loc in normalized if loc < 0 or loc > MAX_CONVNEXT_STAGE_IDX]
    if invalid:
        raise ValueError(
            f"ConvNeXt V2 early-exit locations must be stage indices in [0, {MAX_CONVNEXT_STAGE_IDX}], got {invalid}"
        )
    return normalized


def _scaled_dim(base_dim, width_multiplier):
    scaled = int(base_dim * width_multiplier)
    return max(16, scaled)


class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim, drop_path=0.0, scale=1.0):
        super().__init__()
        self.scaler = Scaler(scale) if scale < 1 else nn.Identity()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1, bias=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(self.scaler(x))
        x = self.norm(x)
        x = self.pwconv1(self.scaler(x))
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(self.scaler(x))
        return residual + self.drop_path(x)


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale=1.0):
        super().__init__()
        self.scaler = Scaler(scale) if scale < 1 else nn.Identity()
        self.norm = LayerNorm(in_channels, eps=1e-6, data_format="channels_first")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(self.scaler(x))
        return x


class ExitHead(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.norm(x)
        x = x.mean(dim=(-2, -1))
        return self.fc(x)


class ConvNeXt(nn.Module):
    """
    ConvNeXt V2-Pico backbone adapted to the SmartFL multi-exit setting.
    Official macro design is preserved:
    - patch stem
    - ConvNeXt V2 blocks with GRN
    - depths = [2, 2, 6, 2]
    - dims = [64, 128, 256, 512]
    """

    def __init__(
        self,
        participating_levels,
        num_channels=3,
        num_classes=100,
        trs=True,
        scale=1.0,
        ee_layer_locations=None,
        args=None,
    ):
        super().__init__()
        self.stored_inp_kwargs = copy.deepcopy(locals())
        del self.stored_inp_kwargs['self']
        del self.stored_inp_kwargs['__class__']

        requested_ee_layer_locations = _normalize_exit_locations(ee_layer_locations)

        if args is None:
            args = arg_parser.parse_args()
            args = modify_args(args)

        if args.config_library_path is None:
            if hasattr(args, 'arch') and args.arch and 'convnext' in args.arch:
                model_name = 'convnext'
            else:
                model_name = getattr(args, 'model', 'convnext')
            dataset_name = getattr(args, 'data', 'cifar100')
            config_library_path = f"{model_name}_{dataset_name}_architecture_library.json"
        else:
            config_library_path = args.config_library_path

        all_model_configs = load_configs_from_json(
            config_library_path,
            model_type="convnext",
            dataset=getattr(args, 'data', None),
        )
        flops_constraints = None
        if args.flops_constraints:
            flops_constraints = {i: val for i, val in enumerate(args.flops_constraints)}
        params_constraints = None
        if args.params_constraints:
            params_constraints = {i: val for i, val in enumerate(args.params_constraints)}

        if args.independent_selection:
            best_configs_for_round = find_best_config_independent(
                all_model_configs,
                participating_levels,
                flops_constraints=flops_constraints,
                params_constraints=params_constraints,
            )
        else:
            best_configs_for_round = find_best_config_for_distribution(
                all_model_configs,
                participating_levels,
                beam_width=500,
                flops_constraints=flops_constraints,
                params_constraints=params_constraints,
            )

        if best_configs_for_round is None:
            print("Warning: No valid hierarchical configuration found. Using default ConvNeXt V2-Pico configuration.")
            if requested_ee_layer_locations:
                ee_loc_list = requested_ee_layer_locations
            else:
                ee_loc_list = DEFAULT_CONVNEXT_EXIT_LOCATIONS[:max(len(participating_levels) - 1, 0)]
            wide_scales = [1.0] * 4
        else:
            normal_exit_locations = []
            for level in sorted(best_configs_for_round.keys()):
                config = best_configs_for_round[level]
                normal_exit_locations.append(config['early_exit_location'])
            wide_scales = [config['width_multipliers'] for config in best_configs_for_round.values()][-1]
            ee_loc_list = normal_exit_locations[:-1]

            if getattr(args, 'enable_tdd', 0) == 1:
                growth_configs = find_all_growth_configs(
                    best_configs_for_round,
                    all_model_configs,
                    model_type="convnext",
                    growth_budget_scale=getattr(args, 'tdd_growth_budget_scale', 1.0),
                )
                growth_exit_locations = []
                for growth_config in growth_configs.values():
                    if growth_config is None:
                        continue
                    growth_exit_locations.append(growth_config['early_exit_location'])
                ee_loc_list.extend(_sorted_unique_exit_locations(growth_exit_locations))

        ee_loc_list = _sorted_unique_exit_locations(ee_loc_list)
        if getattr(args, 'enable_tdd', 0) == 1:
            print(f"[TDD] Added growth exits, all exits: {ee_loc_list}")

        self.scale = scale
        self.wide_scales = wide_scales
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.trs = trs
        self.ee_layer_locations = _normalize_exit_locations(ee_loc_list)
        self.num_blocks = len(self.ee_layer_locations) + 1

        self.base_dims = CONVNEXT_V2_PICO_DIMS
        self.depths = CONVNEXT_V2_PICO_DEPTHS
        self.stage_dims = [_scaled_dim(base, width) for base, width in zip(self.base_dims, self.wide_scales)]

        total_blocks = sum(self.depths)
        dp_rates = torch.linspace(0, 0.1, total_blocks).tolist()

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(num_channels, self.stage_dims[0], kernel_size=4, stride=4),
                LayerNorm(self.stage_dims[0], eps=1e-6, data_format="channels_first"),
            )
        )

        for stage_idx in range(1, len(self.stage_dims)):
            self.downsample_layers.append(
                DownsampleLayer(self.stage_dims[stage_idx - 1], self.stage_dims[stage_idx], scale=scale)
            )

        self.block = nn.ModuleList()
        dp_cursor = 0
        for stage_idx, (dim, depth) in enumerate(zip(self.stage_dims, self.depths)):
            blocks = []
            for _ in range(depth):
                blocks.append(ConvNeXtV2Block(dim, drop_path=dp_rates[dp_cursor], scale=scale))
                dp_cursor += 1
            self.block.append(nn.Sequential(*blocks))

        self.ee_classifiers = nn.ModuleList()
        self.exit_to_classifier_idx = {}
        for classifier_idx, exit_loc in enumerate(self.ee_layer_locations):
            self.ee_classifiers.append(ExitHead(self.stage_dims[exit_loc], num_classes))
            self.exit_to_classifier_idx[exit_loc] = classifier_idx

        self.classifier = ExitHead(self.stage_dims[-1], num_classes)
        self._initialize_weights()

    def get_conv_layer_cutoff(self, exit_location):
        cutoff = 1
        for stage_idx, depth in enumerate(self.depths):
            if stage_idx > 0:
                cutoff += 1
            cutoff += depth * 3
            if stage_idx == exit_location:
                return cutoff
        return cutoff

    def count_params_to_exit(self, exit_location):
        total = sum(p.numel() for p in self.downsample_layers[0].parameters())
        for stage_idx in range(exit_location + 1):
            if stage_idx > 0:
                total += sum(p.numel() for p in self.downsample_layers[stage_idx].parameters())
            total += sum(p.numel() for p in self.block[stage_idx].parameters())

        classifier_idx = self.exit_to_classifier_idx.get(exit_location)
        if classifier_idx is not None:
            total += sum(p.numel() for p in self.ee_classifiers[classifier_idx].parameters())
        return total

    def forward(self, x, manual_early_exit_index=0):
        preds = []

        for stage_idx in range(len(self.block)):
            x = self.downsample_layers[stage_idx](x)
            x = self.block[stage_idx](x)

            classifier_idx = self.exit_to_classifier_idx.get(stage_idx)
            if classifier_idx is not None:
                preds.append(self.ee_classifiers[classifier_idx](x))
                if manual_early_exit_index and len(preds) >= manual_early_exit_index:
                    return preds

        preds.append(self.classifier(x))
        if manual_early_exit_index:
            return preds[:manual_early_exit_index]
        return preds

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def convnext_1(args, params):
    num_classes = args.num_classes
    track_running_stats = args.track_running_stats if hasattr(args, 'track_running_stats') else False
    scale = params.get('scale', 1.0)
    return ConvNeXt(
        participating_levels=[],
        num_channels=3,
        num_classes=num_classes,
        trs=track_running_stats,
        scale=scale,
        ee_layer_locations=[],
    )


def convnext_4(participating_levels, args, params):
    num_classes = args.num_classes
    track_running_stats = args.track_running_stats if hasattr(args, 'track_running_stats') else False
    scale = params.get('scale', 1.0)
    ee_layer_locations = params.get('ee_layer_locations', DEFAULT_CONVNEXT_EXIT_LOCATIONS.copy())
    return ConvNeXt(
        participating_levels=participating_levels,
        num_channels=3,
        num_classes=num_classes,
        trs=track_running_stats,
        scale=scale,
        ee_layer_locations=ee_layer_locations,
        args=args,
    )
