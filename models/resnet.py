import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from args import arg_parser, modify_args
from models.model_utils import Scaler, conv3x3
from hierarchical_model_selector import find_best_config_for_distribution, find_best_config_independent, load_configs_from_json, find_all_growth_configs

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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, trs=False, scale=1.):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=trs)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=trs)

        if scale < 1:
            self.scaler = Scaler(scale)
        else:
            self.scaler = nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, track_running_stats=trs)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.scaler(self.conv1(x))))
        out = self.bn2(self.scaler(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, trs=False, scale=1.):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=trs)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=trs)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes, track_running_stats=trs)

        if scale < 1:
            self.scaler = Scaler(scale)
        else:
            self.scaler = nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=trs)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.scaler(self.conv1(x))))
        out = F.relu(self.bn2(self.scaler(self.conv2(out))))
        out = self.bn3(self.scaler(self.conv3(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self,participating_levels, block, layers, num_classes, ee_layer_locations=[], scale=1., trs=False, args=None):
        super(ResNet, self).__init__()
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
            if hasattr(args, 'arch') and args.arch and 'resnet' in args.arch:
                model_name = 'resnet'
            else:
                model_name = getattr(args, 'model', 'resnet')
            dataset_name = getattr(args, 'data', 'cifar100')
            config_library_path = f"{model_name}_{dataset_name}_architecture_library.json"
        else:
            config_library_path = args.config_library_path
        all_model_configs = load_configs_from_json(config_library_path, model_type="resnet", dataset=getattr(args, 'data', None))
        flops_constraints = None
        if args.flops_constraints:
            flops_constraints = {i: val for i, val in enumerate(args.flops_constraints)}
        # Process params_constraints if provided
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
            print("Warning: No valid hierarchical configuration found. Using default ResNet configuration.")
            # Use default configuration
            ee_loc_list = [3] * (len(participating_levels) - 1) if len(participating_levels) > 1 else []
            wide_scales = [1.0] * 4  # Default 4-element multipliers for ResNet stages
        else:
            ee_loc_list = []
            for level in sorted(best_configs_for_round.keys()):
                config = best_configs_for_round[level]
                ee_loc_list.append(config['early_exit_location'])
            wide_scales = [config['width_multipliers'] for config in best_configs_for_round.values()][-1]

            # TDD: 如果启用了 TDD，计算 Growth configs 并把它们的 exit 位置也加入
            if getattr(args, 'enable_tdd', 0) == 1:
                growth_configs = find_all_growth_configs(
                    best_configs_for_round, all_model_configs, model_type="resnet"
                )
                # 把 growth exit 位置加入 ee_loc_list
                for level, growth_config in growth_configs.items():
                    if growth_config is not None:
                        growth_exit = growth_config['early_exit_location']
                        if growth_exit not in ee_loc_list:
                            ee_loc_list.append(growth_exit)
                # 重新排序
                ee_loc_list = sorted(ee_loc_list)
                print(f"[TDD] Added growth exits, all exits: {ee_loc_list}")

        ee_loc_list=ee_loc_list[:-1]
        ee_layer_locations=ee_loc_list
        if num_classes == 200:
            factor = 4
        else:
            factor = 1
        print(ee_layer_locations)
        self.scale = scale
        # 使用 wide_scales[0] 而不是 scale 来初始化 in_planes
        # 这样可以避免全局模型和本地模型因为 shortcut 结构不同而导致权重复制问题
        self.in_planes = int(16 * wide_scales[0] * factor)
        self.num_blocks = len(ee_layer_locations) + 1
        self.num_classes = num_classes
        self.trs = trs

        if scale < 1:
            self.scaler = nn.Identity()
        else:
            self.scaler = nn.Identity()

        ee_block_list = []
        ee_layer_list = []

        for ee_layer_idx in ee_layer_locations:
            b, l = self.find_ee_block_and_layer(layers, ee_layer_idx)
            ee_block_list.append(b)
            ee_layer_list.append(l)

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes, track_running_stats=self.trs)

        layer1, ee1 = self._make_layer(wide_scales[0],block, int(16 * wide_scales[0] * factor), layers[0], stride=1,
                                       ee_layer_locations=[l for i, l in enumerate(ee_layer_list) if ee_block_list[i] == 0])
        layer2, ee2 = self._make_layer(wide_scales[1],block, int(32 * wide_scales[1] * factor), layers[1], stride=2,
                                       ee_layer_locations=[l for i, l in enumerate(ee_layer_list) if ee_block_list[i] == 1])
        layer3, ee3 = self._make_layer(wide_scales[2],block, int(64 * wide_scales[2] * factor), layers[2], stride=2,
                                       ee_layer_locations=[l for i, l in enumerate(ee_layer_list) if ee_block_list[i] == 2])
        self.layers = nn.ModuleList([layer1, layer2, layer3])
        self.ee_classifiers = nn.ModuleList([ee1, ee2, ee3])

        num_planes = int(64 * wide_scales[2] * factor) * block.expansion
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_layer(self,wide_scale, block_type, planes, num_block, stride, ee_layer_locations):
        strides = [stride] + [1] * (num_block - 1)

        ee_layer_locations_ = ee_layer_locations + [num_block]
        layers = [[] for _ in range(len(ee_layer_locations_))]

        ee_classifiers = []

        if len(ee_layer_locations_) > 1:
            start_layer = 0
            counter = 0
            for i, ee_layer_idx in enumerate(ee_layer_locations_):

                for _ in range(start_layer, ee_layer_idx):
                    layers[i].append(block_type(self.in_planes, planes, strides[counter], trs=self.trs, scale=wide_scale))
                    self.in_planes = planes * block_type.expansion
                    counter += 1
                start_layer = ee_layer_idx

                if ee_layer_idx == 0:
                    num_planes = self.in_planes
                else:
                    num_planes = planes * block_type.expansion

                if i < len(ee_layer_locations_) - 1:
                    ee_classifiers.append(Classifier(num_planes, num_classes=self.num_classes,
                                                     reduction=block_type.expansion, scale=wide_scale,
                                                     ))

        else:
            for i in range(num_block):
                layers[0].append(block_type(self.in_planes, planes, strides[i], trs=self.trs, scale=wide_scale))
                self.in_planes = planes * block_type.expansion

        return nn.ModuleList([nn.Sequential(*l) for l in layers]), nn.ModuleList(ee_classifiers)

    @staticmethod
    def find_ee_block_and_layer(layers, layer_idx):
        temp_array = np.zeros((sum(layers)), dtype=int)
        cum_array = np.cumsum(layers)
        for i in range(1, len(cum_array)):
            temp_array[cum_array[i-1]:] += 1
        block = temp_array[layer_idx]
        if block == 0:
            layer = layer_idx
        else:
            layer = layer_idx - cum_array[block-1]
        return block, layer

    def forward(self, x, manual_early_exit_index=0):
        final_out = F.relu(self.bn1(self.scaler(self.conv1(x))))
        if self.num_classes > 100:
            final_out = F.max_pool2d(final_out, kernel_size=3, stride=2, padding=1)
        ee_outs = []
        counter = 0

        while counter < len(self.layers):
            if final_out is not None:
                if manual_early_exit_index > sum([len(ee) for ee in self.ee_classifiers[:counter+1]]):
                    manual_early_exit_index_ = 0
                elif manual_early_exit_index:
                    manual_early_exit_index_ = manual_early_exit_index - sum([len(ee) for ee in self.ee_classifiers[:counter]])
                else:
                    manual_early_exit_index_ = manual_early_exit_index

                final_out = self._block_forward(self.layers[counter], self.ee_classifiers[counter], final_out, ee_outs, manual_early_exit_index_)
            counter += 1

        preds = ee_outs

        if final_out is not None:
            out = F.adaptive_avg_pool2d(final_out, (1, 1))
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            preds.append(out)

        if manual_early_exit_index:
            assert len(preds) <= manual_early_exit_index

        return preds

    def _block_forward(self, layers, ee_classifiers, x, outs, early_exit=0):
        for i in range(len(layers)-1):
            x = layers[i](x)
            if outs:
                outs.append(ee_classifiers[i](x, outs[-1]))
            else:
                outs.append(ee_classifiers[i](x))
            if early_exit == i + 1:
                break
        if early_exit == 0:
            final_out = layers[-1](x)
        else:
            final_out = None
        return final_out


def resnet110_1(args, params):
    return ResNet(BasicBlock, [18, 18, 18], args.num_classes, scale=params.get('scale', 1),
                  trs=args.track_running_stats)


def resnet110_4(participating_levels,args, params):
    return ResNet(participating_levels,BasicBlock, [18, 18, 18], args.num_classes, ee_layer_locations=args.ee_locs,
                  scale=params.get('scale', 1), trs=args.track_running_stats, args=args)
