import copy
import os
import torch
import torch.nn as nn
from args import arg_parser, modify_args
from hierarchical_model_selector import (
    find_best_config_for_distribution,
    find_best_config_independent,
    load_configs_from_json,
    find_all_growth_configs,
)


def _sorted_unique_exit_locations(exit_locations):
    return sorted(set(exit_locations))


class LinearBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t, trs):
        super(LinearBottleNeck, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t, track_running_stats=trs),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t, track_running_stats=trs),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels, track_running_stats=trs)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual


class MobileNetV2(nn.Module):
    """
    MobileNetV2 implementation with early exit support for federated learning
    """

    def __init__(self, participating_levels, num_channels=3, num_classes=100, trs=True, scale=1.0, ee_layer_locations=[], args=None):
        super(MobileNetV2, self).__init__()
        self.stored_inp_kwargs = copy.deepcopy(locals())
        del self.stored_inp_kwargs['self']
        del self.stored_inp_kwargs['__class__']

        # Use provided args or parse from command line
        if args is None:
            args = arg_parser.parse_args()
            args = modify_args(args)

        # Use auto-generated config library path if not specified
        if args.config_library_path is None:
            # Extract model name from arch (matching main.py logic)
            if hasattr(args, 'arch') and args.arch and 'mobilenet' in args.arch:
                model_name = 'mobilenetv2'
            else:
                model_name = getattr(args, 'model', 'mobilenet')
            dataset_name = getattr(args, 'data', 'cifar100')
            config_library_path = f"{model_name}_{dataset_name}_architecture_library.json"
        else:
            config_library_path = args.config_library_path

        all_model_configs = load_configs_from_json(config_library_path, model_type="mobilenet", dataset=getattr(args, 'data', None))
        flops_constraints = None
        if args.flops_constraints:
            flops_constraints = {i: val for i, val in enumerate(args.flops_constraints)}
        params_constraints = None
        if args.params_constraints:
            params_constraints = {i: val for i, val in enumerate(args.params_constraints)}

        # Choose selection method based on args
        if args.independent_selection:
            # Independent selection: each level maximizes nuclear norm independently
            best_configs_for_round = find_best_config_independent(
                all_model_configs,
                participating_levels,
                flops_constraints=flops_constraints,
                params_constraints=params_constraints
            )
        else:
            # Hierarchical selection: maintains sub-model relationships
            best_configs_for_round = find_best_config_for_distribution(
                all_model_configs,
                participating_levels,
                beam_width=500,
                flops_constraints=flops_constraints,
                params_constraints=params_constraints
            )

        if best_configs_for_round is None:
            print("Warning: No valid hierarchical configuration found. Using default MobileNetV2 configuration.")
            # Use default configuration
            ee_loc_list = [6, 8] if len(participating_levels) > 1 else []
            wide_scales = [1.0] * 8  # Default multipliers for MobileNetV2 8 stages
        else:
            normal_exit_locations = []
            for level in sorted(best_configs_for_round.keys()):
                config = best_configs_for_round[level]
                normal_exit_locations.append(config['early_exit_location'])
            # Use highest level's width_multipliers (contains all actually-used stages)
            # Lower levels' width values after their early_exit are meaningless
            wide_scales = [config['width_multipliers'] for config in best_configs_for_round.values()][-1]
            ee_loc_list = normal_exit_locations[:-1]

            if getattr(args, 'enable_tdd', 0) == 1:
                growth_configs = find_all_growth_configs(
                    best_configs_for_round, all_model_configs, model_type="mobilenet"
                )
                growth_exit_locations = []
                for growth_config in growth_configs.values():
                    if growth_config is None:
                        continue
                    growth_exit_locations.append(growth_config['early_exit_location'])
                growth_exit_locations = _sorted_unique_exit_locations(growth_exit_locations)
                ee_loc_list.extend(growth_exit_locations[:-1])
        ee_loc_list = _sorted_unique_exit_locations(ee_loc_list)
        if getattr(args, 'enable_tdd', 0) == 1:
            print(f"[TDD] Added growth exits, all exits: {ee_loc_list}")

        ee_layer_locations = sorted(ee_loc_list)

        self.scale = scale
        self.wide_scales = wide_scales
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.trs = trs
        self.ee_layer_locations = ee_layer_locations

        if not self.ee_layer_locations:
            self.ee_layer_locations = [6, 8]

        # Backward-compatible aliases for existing utilities that inspect the first exits.
        self.exit1 = self.ee_layer_locations[0] if len(self.ee_layer_locations) > 0 else None
        self.exit2 = self.ee_layer_locations[1] if len(self.ee_layer_locations) > 1 else None
        self.exit3 = self.ee_layer_locations[2] if len(self.ee_layer_locations) > 2 else None

        self.pre = nn.Sequential(
            nn.Conv2d(num_channels, int(32 * wide_scales[0]), 3, padding=1),
            nn.BatchNorm2d(int(32 * wide_scales[0]), track_running_stats=trs),
            nn.ReLU6(inplace=True)
        )

        # Modified to use 8 stages: merge three 160-channel blocks to share wide_scales[6]
        magic_list = [0, 16 * wide_scales[1], 24 * wide_scales[2], 32 * wide_scales[3],
                      64 * wide_scales[4], 96 * wide_scales[5], 160 * wide_scales[6],
                      160 * wide_scales[6], 160 * wide_scales[6], 320 * wide_scales[7]]

        self.block = nn.Sequential(
            LinearBottleNeck(int(32 * wide_scales[0]), int(16 * wide_scales[1]), 1, 1, trs),
            self._make_stage(2, int(16 * wide_scales[1]), int(24 * wide_scales[2]), 2, 6, trs),
            self._make_stage(3, int(24 * wide_scales[2]), int(32 * wide_scales[3]), 2, 6, trs),
            self._make_stage(4, int(32 * wide_scales[3]), int(64 * wide_scales[4]), 2, 6, trs),
            self._make_stage(3, int(64 * wide_scales[4]), int(96 * wide_scales[5]), 1, 6, trs),
            LinearBottleNeck(int(96 * wide_scales[5]), int(160 * wide_scales[6]), 2, 6, trs),
            LinearBottleNeck(int(160 * wide_scales[6]), int(160 * wide_scales[6]), 1, 6, trs),
            LinearBottleNeck(int(160 * wide_scales[6]), int(160 * wide_scales[6]), 1, 6, trs),
            LinearBottleNeck(int(160 * wide_scales[6]), int(320 * wide_scales[7]), 1, 6, trs)
        )

        # Create one classifier for each configured early exit, followed by the final head.
        self.classifier = nn.ModuleList()
        self.exit_to_classifier_idx = {}
        for classifier_idx, exit_loc in enumerate(self.ee_layer_locations):
            exit_channels = int(magic_list[exit_loc])
            self.classifier.append(nn.Sequential(
                nn.Conv2d(exit_channels, exit_channels * 4, 1),
                nn.BatchNorm2d(exit_channels * 4, track_running_stats=trs),
                nn.ReLU6(inplace=True),
                nn.AdaptiveMaxPool2d((1, 1)),
                nn.Conv2d(exit_channels * 4, num_classes, 1),
                nn.Flatten()
            ))
            self.exit_to_classifier_idx[exit_loc] = classifier_idx

        # Final classifier is always placed after the last block.
        self.classifier.append(nn.Sequential(
            nn.Conv2d(int(magic_list[9]), int(magic_list[9] * 4), 1),
            nn.BatchNorm2d(int(magic_list[9] * 4), track_running_stats=trs),
            nn.ReLU6(inplace=True),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(int(magic_list[9] * 4), num_classes, 1),
            nn.Flatten()
        ))

    def _make_stage(self, n, in_channels, out_channels, stride, t, trs):
        layers = [LinearBottleNeck(in_channels, out_channels, stride, t, trs)]

        while n - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t, trs))
            n -= 1

        return nn.Sequential(*layers)

    def forward(self, x, manual_early_exit_index=0):
        preds = []
        output = self.pre(x)

        for i, layer in enumerate(self.block):
            output = layer(output)
            exit_loc = i + 1

            if exit_loc in self.exit_to_classifier_idx:
                classifier_idx = self.exit_to_classifier_idx[exit_loc]
                ee_out = self.classifier[classifier_idx](output)
                preds.append(ee_out)

                if manual_early_exit_index and len(preds) >= manual_early_exit_index:
                    return preds

        final_output = self.classifier[len(self.ee_layer_locations)](output)
        preds.append(final_output)

        if manual_early_exit_index:
            # Return only the requested number of outputs
            return preds[:manual_early_exit_index]

        return preds


def mobilenet_v2_1(args, params):
    """
    MobileNetV2 model with single block for federated learning
    """
    num_classes = args.num_classes
    track_running_stats = args.track_running_stats if hasattr(args, 'track_running_stats') else False
    scale = params.get('scale', 1.0)

    return MobileNetV2(
        participating_levels=[],
        num_channels=3,
        num_classes=num_classes,
        trs=track_running_stats,
        scale=scale,
        ee_layer_locations=[]
    )


def mobilenet_v2_4(participating_levels, args, params):
    """
    MobileNetV2 model with early exit locations for federated learning
    Args format matches ResNet/VGG for compatibility with main.py
    """
    num_classes = args.num_classes
    track_running_stats = args.track_running_stats if hasattr(args, 'track_running_stats') else False
    scale = params.get('scale', 1.0)
    ee_layer_locations = params.get('ee_layer_locations', [6, 8])

    return MobileNetV2(
        participating_levels=participating_levels,
        num_channels=3,
        num_classes=num_classes,
        trs=track_running_stats,
        scale=scale,
        ee_layer_locations=ee_layer_locations,
        args=args
    )


if __name__ == '__main__':
    # Test the model
    model = MobileNetV2(participating_levels=[], num_channels=3, num_classes=200, trs=False, scale=1.0, ee_layer_locations=[6, 8])
    print(model)

    # Test forward pass
    data = torch.rand(1, 3, 64, 64)
    output = model(data)
    print(f"Output shapes: {[out.shape for out in output]}")
