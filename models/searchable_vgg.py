import torch
import torch.nn as nn
import math

# VGG配置
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

def make_searchable_layers(cfg, batch_norm=False, track_running_stats=True, num_channels=3, width_multipliers=None):
    """Create VGG layers with searchable width multipliers"""
    layers = []
    input_channel = num_channels
    maxpool = None
    if num_channels == 3:
        cfg.append("M")
    index = 0

    if width_multipliers is None:
        width_multipliers = [1.0] * len([x for x in cfg if x != 'M'])

    for l in cfg:
        if l == 'M':
            maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            continue
        if index == 0:
            conv2d = nn.Conv2d(input_channel, int(l * width_multipliers[index]), kernel_size=3, padding=1)
        else:
            conv2d = nn.Conv2d(int(input_channel * width_multipliers[index - 1]), int(l * width_multipliers[index]), kernel_size=3, padding=1)

        if batch_norm:
            seq = nn.Sequential(
                conv2d,
                nn.BatchNorm2d(int(l * width_multipliers[index]), track_running_stats=track_running_stats),
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

class SearchableVGG(nn.Module):
    def __init__(self, num_classes, width_multipliers, early_exit_location=None, num_channels=3):
        super(SearchableVGG, self).__init__()

        # Expand 6 stage multipliers to 15 layer multipliers if needed
        if len(width_multipliers) == 6:
            self.width_multipliers = expand_vgg_stage_multipliers(width_multipliers)
        else:
            self.width_multipliers = width_multipliers
        self.early_exit_location = early_exit_location
        self.num_classes = num_classes
        self.num_channels = num_channels

        # Create conv features with specified width multipliers
        self.features = make_searchable_layers(cfg['D'].copy(), batch_norm=True,
                                             track_running_stats=True,
                                             num_channels=num_channels,
                                             width_multipliers=width_multipliers[:13])

        # Determine dimensions for FC layers
        if num_channels == 3:
            dim = 4096
        else:
            dim = 256

        # Add global average pooling and flatten
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # Add fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(int(512 * width_multipliers[12]), int(dim * width_multipliers[13])),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(int(dim * width_multipliers[13]), int(dim * width_multipliers[14])),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        # Final classifier
        self.classifier = nn.Linear(int(dim * width_multipliers[14]), num_classes)

        # Early exit classifier if needed
        self.early_exit_classifier = None
        if self.early_exit_location is not None:
            exit_channels = self._get_channels_at_location(early_exit_location)
            self.early_exit_classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(exit_channels, num_classes)
            )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _get_channels_at_location(self, location):
        """Get number of channels at specific layer location"""
        # VGG-16 conv layer channels: [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        layer_channels = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        if location < len(layer_channels):
            return int(layer_channels[location] * self.width_multipliers[location])
        return int(512 * self.width_multipliers[12])  # Default to last conv layer

    def forward(self, x, manual_early_exit_index=None):
        layer_count = 0

        # Process through conv layers
        for i, layer in enumerate(self.features):
            x = layer(x)

            # Check for early exit (only for conv layers)
            if (self.early_exit_location is not None and
                layer_count == self.early_exit_location):
                return [self.early_exit_classifier(x)]  # Return as list to match SearchableResNet

            layer_count += 1

        # Global average pooling
        x = self.avgpool(x)
        x = self.flatten(x)

        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)

        # Final classification
        final_output = self.classifier(x)
        return final_output

def searchable_vgg16(num_classes, width_multipliers, early_exit_location=None, num_channels=3):
    """Create SearchableVGG16 model for supernet training (Phase 1)"""
    return SearchableVGG(num_classes, width_multipliers, early_exit_location, num_channels)