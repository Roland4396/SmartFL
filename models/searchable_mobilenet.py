import torch
import torch.nn as nn
import math


class LinearBottleNeck(nn.Module):
    """Inverted Residual Block for MobileNetV2"""
    def __init__(self, in_channels, out_channels, stride, t):
        super(LinearBottleNeck, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1, bias=False),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t, bias=False),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual


class SearchableMobileNetV2(nn.Module):
    """
    Searchable MobileNetV2 for supernet training (Phase 1)

    Architecture:
    - Stage 0: Initial conv (3 -> 32)
    - Stage 1: Inverted residual blocks (32 -> 16 -> 24 -> 32 -> 64 -> 96 -> 160 -> 320)
    - Each stage can be scaled by width_multipliers

    Args:
        num_classes: Number of output classes
        width_multipliers: List of width multipliers for each stage [stage0, ..., stage7]
                          8 multipliers for MobileNetV2 stages (based on output channels)
                          [32, 16, 24, 32, 64, 96, 160, 320]
        early_exit_location: Block index for early exit (0-8 for 9 blocks)
        num_channels: Input channels (default: 3 for RGB)
    """

    def __init__(self, num_classes, width_multipliers, early_exit_location=None, num_channels=3):
        super(SearchableMobileNetV2, self).__init__()

        if len(width_multipliers) != 10:
            raise ValueError(f"Expected 10 width multipliers for MobileNetV2, got {len(width_multipliers)}")

        self.num_classes = num_classes
        self.width_multipliers = width_multipliers
        self.early_exit_location = early_exit_location
        self.num_channels = num_channels

        # Stage channel definitions with width_multipliers (10 stages total)
        # Matching mobilenet.py magic_list: [0, 16, 24, 32, 64, 96, 160, 160, 160, 320]
        # Each block has independent width control
        self.stage_channels = [
            int(32 * width_multipliers[0]),   # Stage 0: pre layer (initial conv)
            int(16 * width_multipliers[1]),   # Stage 1: block[0] output
            int(24 * width_multipliers[2]),   # Stage 2: block[1] output
            int(32 * width_multipliers[3]),   # Stage 3: block[2] output
            int(64 * width_multipliers[4]),   # Stage 4: block[3] output
            int(96 * width_multipliers[5]),   # Stage 5: block[4] output
            int(160 * width_multipliers[6]),  # Stage 6: block[5] output (first 160)
            int(160 * width_multipliers[7]),  # Stage 7: block[6] output (second 160)
            int(160 * width_multipliers[8]),  # Stage 8: block[7] output (third 160)
            int(320 * width_multipliers[9])   # Stage 9: block[8] output (final)
        ]

        # Initial convolution
        self.pre = nn.Sequential(
            nn.Conv2d(num_channels, self.stage_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(self.stage_channels[0]),
            nn.ReLU6(inplace=True)
        )

        # Build inverted residual blocks (9 blocks total, matching mobilenet.py structure)
        # Output channels: [16, 24, 32, 64, 96, 160, 160, 160, 320]
        # Each block has independent parameters (even if channel dimensions repeat)

        self.blocks = nn.ModuleList([
            # Block 0: 32->16 (t=1, no expansion)
            LinearBottleNeck(self.stage_channels[0], self.stage_channels[1], 1, 1),

            # Block 1: 16->24 (Sequential包含2个bottleneck, 整体作为一个block)
            self._make_stage(2, self.stage_channels[1], self.stage_channels[2], 2, 6),

            # Block 2: 24->32 (Sequential包含3个bottleneck, 整体作为一个block)
            self._make_stage(3, self.stage_channels[2], self.stage_channels[3], 2, 6),

            # Block 3: 32->64 (Sequential包含4个bottleneck, 整体作为一个block)
            self._make_stage(4, self.stage_channels[3], self.stage_channels[4], 2, 6),

            # Block 4: 64->96 (Sequential包含3个bottleneck, 整体作为一个block)
            self._make_stage(3, self.stage_channels[4], self.stage_channels[5], 1, 6),

            # Blocks 5-7: Each with INDEPENDENT parameters (matching mobilenet.py's 3 separate calls)
            LinearBottleNeck(self.stage_channels[5], self.stage_channels[6], 2, 6),  # Block 5: 96->160 (stride=2)
            LinearBottleNeck(self.stage_channels[6], self.stage_channels[7], 1, 6),  # Block 6: 160->160 (stride=1)
            LinearBottleNeck(self.stage_channels[7], self.stage_channels[8], 1, 6),  # Block 7: 160->160 (stride=1)

            # Block 8: 160->320 (final)
            LinearBottleNeck(self.stage_channels[8], self.stage_channels[9], 1, 6)
        ])

        # Global pooling and final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.stage_channels[9], num_classes)  # 320 channels

        # Early exit classifier if needed
        self.early_exit_classifier = None
        if early_exit_location is not None:
            exit_channels = self._get_channels_at_location(early_exit_location)
            self.early_exit_classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(exit_channels, num_classes)
            )

        # Initialize weights
        self._initialize_weights()

    def _make_stage(self, n, in_channels, out_channels, stride, t):
        """Create a stage with n blocks"""
        layers = [LinearBottleNeck(in_channels, out_channels, stride, t)]

        for _ in range(1, n):
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))

        return nn.Sequential(*layers)

    def _get_channels_at_location(self, location):
        """Get output channels at a specific block location"""
        # Map block location to corresponding channel (matching mobilenet.py magic_list)
        channel_map = [
            self.stage_channels[1],  # Block 0 output: 16
            self.stage_channels[2],  # Block 1 output: 24
            self.stage_channels[3],  # Block 2 output: 32
            self.stage_channels[4],  # Block 3 output: 64
            self.stage_channels[5],  # Block 4 output: 96
            self.stage_channels[6],  # Block 5 output: 160 (with multiplier[6])
            self.stage_channels[7],  # Block 6 output: 160 (with multiplier[7])
            self.stage_channels[8],  # Block 7 output: 160 (with multiplier[8])
            self.stage_channels[9],  # Block 8 output: 320 (with multiplier[9])
        ]

        if location < len(channel_map):
            return channel_map[location]
        return self.stage_channels[9]  # Default to final channels (320)

    def forward(self, x, manual_early_exit_index=None):
        """Forward pass with optional early exit"""
        x = self.pre(x)

        # Process through blocks
        for block_idx, block in enumerate(self.blocks):
            x = block(x)

            # Check for early exit
            if (self.early_exit_location is not None and
                block_idx == self.early_exit_location):
                return [self.early_exit_classifier(x)]  # Return as list to match other models

        # Final classification
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        final_output = self.classifier(x)

        return final_output

    def _initialize_weights(self):
        """Initialize model weights"""
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
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def searchable_mobilenet_v2(num_classes, width_multipliers, early_exit_location=None, num_channels=3):
    """Create SearchableMobileNetV2 model for supernet training (Phase 1)"""
    return SearchableMobileNetV2(num_classes, width_multipliers, early_exit_location, num_channels)


if __name__ == '__main__':
    # Test the searchable model
    model = searchable_mobilenet_v2(
        num_classes=100,
        width_multipliers=[1.0] * 10,  # Max width for all 10 stages
        early_exit_location=None,
        num_channels=3
    )
    print(model)

    # Test forward pass
    data = torch.rand(2, 3, 32, 32)
    output = model(data)
    print(f"Output shape: {output.shape}")

    # Test with early exit
    model_with_exit = searchable_mobilenet_v2(
        num_classes=100,
        width_multipliers=[1.0] * 10,  # 10 stages: [32,16,24,32,64,96,160,160,160,320]
        early_exit_location=4,
        num_channels=3
    )
    output_early = model_with_exit(data)
    print(f"Early exit output: {[o.shape for o in output_early]}")
