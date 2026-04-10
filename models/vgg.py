"""vgg in pytorch
[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

'''VGG11/13/16/19 in Pytorch.'''

import torch.nn as nn
from args import arg_parser, modify_args
from models.model_utils import Scaler, conv3x3
from hierarchical_model_selector import (
    find_best_config_for_distribution,
    find_best_config_independent,
    load_configs_from_json,
    find_all_growth_configs,
)

def expand_vgg_stage_multipliers(stage_multipliers):
    """
    Expand 6 VGG stage multipliers to 15 layer multipliers
    VGG-D structure: [64,64], [128,128], [256,256,256], [512,512,512], [512,512,512] + [4096,4096]
    """
    if len(stage_multipliers) != 6:
        return stage_multipliers  # Already expanded or different format

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

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}


def _sorted_unique_exit_locations(exit_locations):
    return sorted(set(exit_locations))


class Classifier(nn.Module):
    def __init__(self, in_planes, num_classes, num_conv_layers=3, reduction=1, scale=1.):
        super(Classifier, self).__init__()

        self.in_planes = in_planes
        self.num_classes = num_classes
        self.num_conv_layers = num_conv_layers
        self.reduction = reduction
        self.scale = scale

        if scale < 1:
            scaler = Scaler(scale)
        else:
            scaler = nn.Identity()

        if reduction == 1:
            conv_list = [conv3x3(in_planes, in_planes) for _ in range(num_conv_layers)]
        else:
            conv_list = [conv3x3(in_planes, int(in_planes/reduction))]
            in_planes = int(in_planes/reduction)
            conv_list.extend([conv3x3(in_planes, in_planes) for _ in range(num_conv_layers-1)])

        bn_list = [nn.BatchNorm2d(in_planes, track_running_stats=False) for _ in range(num_conv_layers)]
        relu_list = [nn.ReLU() for _ in range(num_conv_layers)]
        avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        flatten = nn.Flatten()

        layers = []
        for i in range(num_conv_layers):
            layers.append(conv_list[i])
            layers.append(scaler)
            layers.append(bn_list[i])
            layers.append(relu_list[i])
        layers.append(avg_pool)
        layers.append(flatten)

        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(in_planes, num_classes)

    def forward(self, inp, pred=None):
        output = self.layers(inp)
        output = self.fc(output)
        return output


class VGG(nn.Module):
    def __init__(self, participating_levels, features, num_class=100, num_channels=3, ee_layer_locations=[], scale=1., trs=False, args=None):
        super().__init__()
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
            if hasattr(args, 'arch') and args.arch and 'vgg' in args.arch:
                model_name = 'vgg'
            else:
                model_name = getattr(args, 'model', 'vgg')
            dataset_name = getattr(args, 'data', 'cifar100')
            config_library_path = f"{model_name}_{dataset_name}_architecture_library.json"
        else:
            config_library_path = args.config_library_path
        all_model_configs = load_configs_from_json(config_library_path, model_type="vgg", dataset=getattr(args, 'data', None))
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
                beam_width=1000,
                flops_constraints=flops_constraints,
                params_constraints=params_constraints
            )
        if best_configs_for_round is None:
            print("Warning: No valid hierarchical configuration found. Using default VGG configuration.")
            # Use default configuration
            ee_loc_list = [12] * (len(participating_levels) - 1) if len(participating_levels) > 1 else []
            wide_scales = [1.0] * 15  # Default 15-element multipliers
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
                    best_configs_for_round, all_model_configs, model_type="vgg"
                )
                growth_exit_locations = []
                for growth_config in growth_configs.values():
                    if growth_config is None:
                        continue
                    growth_exit_locations.append(growth_config['early_exit_location'])
                growth_exit_locations = _sorted_unique_exit_locations(growth_exit_locations)
                # Unlike ResNet, VGG growth exits such as 11/12 are still valid
                # pre-classifier exit points rather than the implicit final output.
                # Keep all discovered growth exits so growth clients do not get
                # silently redirected to the final classifier path.
                ee_loc_list.extend(growth_exit_locations)
        ee_loc_list = _sorted_unique_exit_locations(ee_loc_list)
        if getattr(args, 'enable_tdd', 0) == 1:
            print(f"[TDD] Added growth exits, all exits: {ee_loc_list}")
        print(ee_loc_list)
        ee_layer_locations = ee_loc_list

        print(f"[DEBUG VGG __init__] ee_layer_locations: {ee_layer_locations}")
        print(f"[DEBUG VGG __init__] wide_scales: {wide_scales}")
        self.scale = scale
        self.wide_scales = wide_scales
        self.num_classes = num_class
        self.num_channels = num_channels
        self.trs = trs
        self.ee_layer_locations = ee_layer_locations

        # Update features with proper wide_scales
        if hasattr(features, '_modules'):
            # Recreate features with wide_scales (first 13 for conv layers)
            cfg_copy = cfg['D'].copy()
            self.features = make_layers(cfg_copy, batch_norm=True, track_running_stats=trs,
                                      num_channels=num_channels, wide_scales=wide_scales[:13])
        else:
            self.features = features
        self.ee_classifiers = nn.ModuleList()

        for i, logical_ee_layer in enumerate(ee_layer_locations):
            # Map logical layer index to actual stage end position
            actual_features_idx = self._map_logical_layer_to_features_idx(logical_ee_layer)
            if actual_features_idx < len(self.features):
                in_planes = self._get_layer_channels(logical_ee_layer, wide_scales)
                self.ee_classifiers.append(
                    Classifier(in_planes, num_classes=num_class, reduction=1, scale=wide_scales[logical_ee_layer] if logical_ee_layer < len(wide_scales) else 1.0)
                )
                # Update ee_layer_locations to use actual features indices
                self.ee_layer_locations[i] = actual_features_idx
        self.exit_to_classifier_idx = {
            exit_loc: classifier_idx for classifier_idx, exit_loc in enumerate(self.ee_layer_locations)
        }

        if num_channels == 3:
            dim = 4096
        else:
            dim = 256

        if num_class == 200:
            self.features[-1].append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features[-1].append(nn.Flatten(start_dim=1, end_dim=-1))
        self.features.append(nn.Sequential(
            nn.Linear(int(512 * wide_scales[12]), int(dim * wide_scales[13])),  # 13th scale for first FC
            nn.LayerNorm(int(dim * wide_scales[13])),
            nn.ReLU(inplace=True),
            nn.Dropout()))
        self.features.append(nn.Sequential(
            nn.Linear(int(dim * wide_scales[13]), int(dim * wide_scales[14])),  # 14th scale for second FC
            nn.LayerNorm(int(dim * wide_scales[14])),
            nn.ReLU(inplace=True),
            nn.Dropout()))
        self.classifier = nn.Linear(int(dim * wide_scales[14]), num_class)  # Final FC uses 15th scale

    def _map_logical_layer_to_features_idx(self, logical_layer_idx):
        """
        Map logical conv layer index (0-12) to actual self.features index
        VGG-D structure: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]

        Logical layers to features mapping:
        0-1: Stage 0 → features[0], features[1]
        2-3: Stage 1 → features[2], features[3]
        4-6: Stage 2 → features[4], features[5], features[6]
        7-9: Stage 3 → features[7], features[8], features[9]
        10-12: Stage 4 → features[10], features[11], features[12]
        """
        # Direct mapping since make_layers creates one Sequential per conv layer
        return min(logical_layer_idx, len(self.features) - 1)

    def _get_layer_channels(self, layer_idx, wide_scales):
        # VGG-16: 13 conv layers + 2 fc layers = 15 total
        layer_configs = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 4096, 4096]
        if layer_idx < len(layer_configs) and layer_idx < len(wide_scales):
            return int(layer_configs[layer_idx] * wide_scales[layer_idx])
        return 512

    def forward(self, x, manual_early_exit_index=0):
        ee_outs = []
        output = x

        for i, layer in enumerate(self.features):
            output = layer(output)

            ee_idx = self.exit_to_classifier_idx.get(i)
            if ee_idx is not None and ee_idx < len(self.ee_classifiers):
                ee_out = self.ee_classifiers[ee_idx](output)
                ee_outs.append(ee_out)

                # Fixed: Only stop if h_level is within valid range
                # If h_level > num_exit_points, this is the largest model, execute all layers
                if manual_early_exit_index and manual_early_exit_index <= len(self.ee_layer_locations):
                    if len(ee_outs) >= manual_early_exit_index:
                        return ee_outs

        # Execute all layers for largest model or when no early exit is triggered
        preds = ee_outs
        final_output = self.classifier(output)
        preds.append(final_output)

        if manual_early_exit_index:
            # Allow h_level to exceed available exit points (for largest model)
            assert len(preds) <= manual_early_exit_index or manual_early_exit_index > len(self.ee_layer_locations)

        return preds

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False, track_running_stats=True, num_channels=3, wide_scales=None):
    layers = []
    input_channel = num_channels
    maxpool = None
    if num_channels == 3:
        cfg.append("M")
    index = 0

    if wide_scales is None:
        wide_scales = [1.0] * len([x for x in cfg if x != 'M'])

    for l in cfg:
        if l == 'M':
            maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            continue
        if index == 0:
            conv2d = nn.Conv2d(input_channel, int(l * wide_scales[index]), kernel_size=3, padding=1)
        else:
            conv2d = nn.Conv2d(int(input_channel * wide_scales[index - 1]), int(l * wide_scales[index]), kernel_size=3, padding=1)
        if batch_norm:
            seq = nn.Sequential(
                conv2d,
                nn.BatchNorm2d(int(l * wide_scales[index]), track_running_stats=track_running_stats),
                nn.ReLU(inplace=True)
            )
        else:
            seq = nn.Sequential(
                conv2d,
                nn.ReLU(inplace=True)
            )

        if maxpool is not None:
            seq.add_module('MaxPool2d', maxpool)
            maxpool = None

        layers.append(seq)
        input_channel = l
        index += 1

    if maxpool is not None:
        layers[-1].append(maxpool)

    return nn.Sequential(*layers)


def vgg_16_bn(num_classes, track_running_stats=True, num_channels=3, wide_scales=None):
    if wide_scales is None:
        wide_scales = [1.0] * 15  # 13 conv layers + 2 fc layers
    return VGG(
        participating_levels=[],
        features=make_layers(cfg['D'], batch_norm=True, track_running_stats=track_running_stats, num_channels=num_channels,
                    wide_scales=wide_scales[:13]),  # Only use first 13 for conv layers
        num_class=num_classes,
        num_channels=num_channels,
        ee_layer_locations=[],
        scale=1.,
        trs=track_running_stats)


def vgg_16_bn_eeloc(participating_levels, args, params):
    """
    VGG model with early exit locations for federated learning
    Args format matches ResNet for compatibility with main.py
    """
    # Extract parameters from args and params
    num_classes = args.num_classes
    track_running_stats = args.track_running_stats if hasattr(args, 'track_running_stats') else False
    scale = params.get('scale', 1.0)
    ee_layer_locations = params.get('ee_layer_locations', [])

    return VGG(
        participating_levels=participating_levels,
        features=make_layers(cfg['D'], batch_norm=True, track_running_stats=track_running_stats, num_channels=3,
                    wide_scales=None),  # wide_scales will be determined by hierarchical model selector
        num_class=num_classes,
        num_channels=3,
        ee_layer_locations=ee_layer_locations,
        scale=scale,
        trs=track_running_stats,
        args=args)


if __name__ == '__main__':
    model_1 = vgg_16_bn(10, True, 3,
                        [1] * 15)  # VGG-16 has 13 conv + 2 fc = 15 layers total
    print(model_1)
    # Random summon data to test model_1(CIFAR10)
    data = torch.rand(10, *(3, 64, 64))
    totalParam = []

    model = vgg_16_bn(10, True, 3,
                      [0.71]*15)
    LayerParams = np.array([])
    for layer in model_1.features:
        params = sum(p.numel() for p in layer.parameters())
        totalParam = np.append(totalParam, params)
    for layer in model.features:
        params = sum(p.numel() for p in layer.parameters())
        LayerParams = np.append(LayerParams, params)
    totalParam = np.append(totalParam, sum(p.numel() for p in model_1.classifier.parameters()))
    LayerParams = np.append(LayerParams, sum(p.numel() for p in model.classifier.parameters()))

    print(list(LayerParams/totalParam))
