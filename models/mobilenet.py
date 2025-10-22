import copy
import torch
import torch.nn as nn
from args import arg_parser, modify_args
from hierarchical_model_selector import find_best_config_for_distribution, find_best_config_independent, load_configs_from_json


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

    def __init__(self, participating_levels, num_channels=3, num_classes=100, trs=True, scale=1.0, ee_layer_locations=[]):
        super(MobileNetV2, self).__init__()
        self.stored_inp_kwargs = copy.deepcopy(locals())
        del self.stored_inp_kwargs['self']
        del self.stored_inp_kwargs['__class__']

        args = arg_parser.parse_args()
        args = modify_args(args)

        # Use auto-generated config library path if not specified
        if args.config_library_path is None:
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
            wide_scales = [1.0] * 10  # Default multipliers for MobileNetV2 stages
        else:
            ee_loc_list = []
            for level in sorted(best_configs_for_round.keys()):
                config = best_configs_for_round[level]
                ee_loc_list.append(config['early_exit_location'])
            # Use highest level's width_multipliers (contains all actually-used stages)
            # Lower levels' width values after their early_exit are meaningless
            wide_scales = [config['width_multipliers'] for config in best_configs_for_round.values()][-1]

        ee_loc_list = ee_loc_list[:-1]
        ee_layer_locations = ee_loc_list

        self.scale = scale
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.trs = trs
        self.ee_layer_locations = ee_layer_locations

        # Set default exit points if not specified
        if not ee_layer_locations:
            self.exit1 = 6
            self.exit2 = 8
        elif len(ee_layer_locations) == 1:
            self.exit1 = ee_layer_locations[0]
            self.exit2 = 8
        else:
            self.exit1 = ee_layer_locations[0]
            self.exit2 = ee_layer_locations[1] if len(ee_layer_locations) > 1 else 8

        self.pre = nn.Sequential(
            nn.Conv2d(num_channels, int(32 * scale), 3, padding=1),
            nn.BatchNorm2d(int(32 * scale), track_running_stats=trs),
            nn.ReLU6(inplace=True)
        )

        magic_list = [0, 16 * scale, 24 * scale, 32 * scale, 64 * scale, 96 * scale, 160 * scale, 160 * scale,
                      160 * scale, 320 * scale]

        self.block = nn.Sequential(
            LinearBottleNeck(int(32 * scale), int(16 * scale), 1, 1, trs),
            self._make_stage(2, int(16 * scale), int(24 * scale), 2, 6, trs),
            self._make_stage(3, int(24 * scale), int(32 * scale), 2, 6, trs),
            self._make_stage(4, int(32 * scale), int(64 * scale), 2, 6, trs),
            self._make_stage(3, int(64 * scale), int(96 * scale), 1, 6, trs),
            self._make_stage(3, int(96 * scale), int(160 * scale), 2, 6, trs)[0],
            self._make_stage(3, int(96 * scale), int(160 * scale), 2, 6, trs)[1],
            self._make_stage(3, int(96 * scale), int(160 * scale), 2, 6, trs)[2],
            LinearBottleNeck(int(160 * scale), int(320 * scale), 1, 6, trs)
        )

        # Early exit classifiers
        self.ee_classifiers = nn.ModuleList()

        # Add classifiers for early exits
        for i, exit_point in enumerate([self.exit1, self.exit2]):
            self.ee_classifiers.append(nn.Sequential(
                nn.Conv2d(int(magic_list[exit_point]), int(magic_list[exit_point] * 4), 1),
                nn.BatchNorm2d(int(magic_list[exit_point] * 4), track_running_stats=trs),
                nn.ReLU6(inplace=True),
                nn.AdaptiveMaxPool2d((1, 1)),
                nn.Conv2d(int(magic_list[exit_point] * 4), num_classes, 1),
                nn.Flatten()
            ))

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(int(magic_list[9]), int(magic_list[9] * 4), 1),
            nn.BatchNorm2d(int(magic_list[9] * 4), track_running_stats=trs),
            nn.ReLU6(inplace=True),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(int(magic_list[9] * 4), num_classes, 1),
            nn.Flatten(),
        )

    def _make_stage(self, n, in_channels, out_channels, stride, t, trs):
        layers = [LinearBottleNeck(in_channels, out_channels, stride, t, trs)]

        while n - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t, trs))
            n -= 1

        return nn.Sequential(*layers)

    def forward(self, x, manual_early_exit_index=0):
        ee_outs = []
        output = x

        output = self.pre(output)

        # Process blocks and check for early exits
        for i, layer in enumerate(self.block):
            output = layer(output)

            # Check for early exits
            if i == self.exit1 - 1 and len(self.ee_classifiers) > 0:
                ee_out = self.ee_classifiers[0](output)
                ee_outs.append(ee_out)

                if manual_early_exit_index and len(ee_outs) >= manual_early_exit_index:
                    return ee_outs

            if i == self.exit2 - 1 and len(self.ee_classifiers) > 1:
                ee_out = self.ee_classifiers[1](output)
                ee_outs.append(ee_out)

                if manual_early_exit_index and len(ee_outs) >= manual_early_exit_index:
                    return ee_outs

        # Final output
        preds = ee_outs
        final_output = self.classifier(output)
        preds.append(final_output)

        if manual_early_exit_index:
            assert len(preds) <= manual_early_exit_index

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
        ee_layer_locations=ee_layer_locations
    )


if __name__ == '__main__':
    # Test the model
    model = MobileNetV2(participating_levels=[], num_channels=3, num_classes=200, trs=False, scale=1.0, ee_layer_locations=[6, 8])
    print(model)

    # Test forward pass
    data = torch.rand(1, 3, 64, 64)
    output = model(data)
    print(f"Output shapes: {[out.shape for out in output]}")