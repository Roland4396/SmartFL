import math

import torch
import torch.nn as nn


class LinearBottleNeck(nn.Module):
    """Inverted residual block for MobileNetV2."""

    def __init__(self, in_channels, out_channels, stride, t):
        super(LinearBottleNeck, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = self.residual(x)
        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x
        return residual


MOBILENET_BOTTLENECK_SPECS = [
    (0, 1, 1, 1),
    (1, 2, 2, 6),
    (2, 2, 1, 6),
    (2, 3, 2, 6),
    (3, 3, 1, 6),
    (3, 3, 1, 6),
    (3, 4, 2, 6),
    (4, 4, 1, 6),
    (4, 4, 1, 6),
    (4, 4, 1, 6),
    (4, 5, 1, 6),
    (5, 5, 1, 6),
    (5, 5, 1, 6),
    (5, 6, 2, 6),
    (6, 6, 1, 6),
    (6, 6, 1, 6),
    (6, 7, 1, 6),
]
MOBILENET_EXIT_STAGE_IDS = [spec[1] for spec in MOBILENET_BOTTLENECK_SPECS]


class SearchableMobileNetV2(nn.Module):
    """
    Searchable MobileNetV2 for supernet training and architecture evaluation.

    Exits are indexed on the 17 MobileNetV2 bottleneck units.
    Width multipliers control the 8 channel stages: [32, 16, 24, 32, 64, 96, 160, 320].
    """

    def __init__(self, num_classes, width_multipliers, early_exit_location=None, num_channels=3):
        super(SearchableMobileNetV2, self).__init__()

        if len(width_multipliers) != 8:
            raise ValueError(f"Expected 8 width multipliers for MobileNetV2, got {len(width_multipliers)}")

        self.num_classes = num_classes
        self.width_multipliers = width_multipliers
        self.early_exit_location = early_exit_location
        self.num_channels = num_channels

        self.stage_channels = [
            int(32 * width_multipliers[0]),
            int(16 * width_multipliers[1]),
            int(24 * width_multipliers[2]),
            int(32 * width_multipliers[3]),
            int(64 * width_multipliers[4]),
            int(96 * width_multipliers[5]),
            int(160 * width_multipliers[6]),
            int(320 * width_multipliers[7]),
        ]
        self.block_output_stage_ids = MOBILENET_EXIT_STAGE_IDS

        self.pre = nn.Sequential(
            nn.Conv2d(num_channels, self.stage_channels[0], 3, padding=1),
            nn.BatchNorm2d(self.stage_channels[0]),
            nn.ReLU6(inplace=True),
        )

        self.blocks = nn.ModuleList(self._make_bottleneck_blocks())

        final_channels = self.stage_channels[-1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Conv2d(final_channels, final_channels * 4, 1),
            nn.BatchNorm2d(final_channels * 4),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(final_channels * 4, num_classes, 1),
            nn.Flatten(),
        )

        self.early_exit_classifier = None
        if early_exit_location is not None:
            exit_channels = self._get_channels_at_location(early_exit_location)
            self.early_exit_classifier = nn.Sequential(
                nn.Conv2d(exit_channels, exit_channels * 4, 1),
                nn.BatchNorm2d(exit_channels * 4),
                nn.ReLU6(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(exit_channels * 4, num_classes, 1),
                nn.Flatten(),
            )

        self._initialize_weights()

    def _make_bottleneck_blocks(self):
        return [
            LinearBottleNeck(
                self.stage_channels[in_stage],
                self.stage_channels[out_stage],
                stride,
                expansion,
            )
            for in_stage, out_stage, stride, expansion in MOBILENET_BOTTLENECK_SPECS
        ]

    def _get_channels_at_location(self, location):
        if location < 0 or location >= len(self.block_output_stage_ids):
            raise ValueError(f"Unsupported MobileNetV2 bottleneck exit: {location}")
        return self.stage_channels[self.block_output_stage_ids[location]]

    def get_conv_layer_cutoff(self, early_exit_location):
        if early_exit_location < 0 or early_exit_location >= len(self.blocks):
            raise ValueError(f"Unsupported MobileNetV2 bottleneck exit: {early_exit_location}")
        return 1 + 3 * (early_exit_location + 1)

    def count_params_to_exit(self, early_exit_location):
        if early_exit_location < 0 or early_exit_location >= len(self.blocks):
            raise ValueError(f"Unsupported MobileNetV2 bottleneck exit: {early_exit_location}")

        total_params = sum(p.numel() for p in self.pre.parameters())
        for block in self.blocks[:early_exit_location + 1]:
            total_params += sum(p.numel() for p in block.parameters())
        if self.early_exit_classifier is not None:
            total_params += sum(p.numel() for p in self.early_exit_classifier.parameters())
        return total_params

    def forward(self, x, manual_early_exit_index=None):
        x = self.pre(x)

        for block_idx, block in enumerate(self.blocks):
            x = block(x)
            if self.early_exit_location is not None and block_idx == self.early_exit_location:
                return [self.early_exit_classifier(x)]

        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def searchable_mobilenet_v2(num_classes, width_multipliers, early_exit_location=None, num_channels=3):
    return SearchableMobileNetV2(num_classes, width_multipliers, early_exit_location, num_channels)


if __name__ == '__main__':
    model = searchable_mobilenet_v2(
        num_classes=100,
        width_multipliers=[1.0] * 8,
        early_exit_location=None,
        num_channels=3,
    )
    print(model)

    data = torch.rand(2, 3, 32, 32)
    output = model(data)
    print(f"Output shape: {output.shape}")

    model_with_exit = searchable_mobilenet_v2(
        num_classes=100,
        width_multipliers=[1.0] * 8,
        early_exit_location=10,
        num_channels=3,
    )
    output_early = model_with_exit(data)
    print(f"Early exit output: {[o.shape for o in output_early]}")
