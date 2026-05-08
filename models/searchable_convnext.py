import math

import torch
import torch.nn as nn

from models.convnext_utils import DropPath, GRN, LayerNorm


CONVNEXT_V2_PICO_DIMS = [64, 128, 256, 512]
CONVNEXT_V2_PICO_DEPTHS = [2, 2, 6, 2]


def _scaled_dim(base_dim, width_multiplier):
    scaled = int(base_dim * width_multiplier)
    return max(16, scaled)


class SearchableConvNeXtV2Block(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1, bias=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + self.drop_path(x)


class SearchableConvNeXtV2(nn.Module):
    """
    Faithful small ConvNeXt V2 backbone using the official Pico layout:
    - depths = [2, 2, 6, 2]
    - dims = [64, 128, 256, 512]
    - patch stem and GRN-enabled ConvNeXt V2 blocks
    """

    def __init__(self, num_classes, width_multipliers, early_exit_location=None, num_channels=3):
        super().__init__()

        if len(width_multipliers) != 4:
            raise ValueError(f"Expected 4 width multipliers for ConvNeXt V2, got {len(width_multipliers)}")

        self.num_classes = num_classes
        self.width_multipliers = width_multipliers
        self.early_exit_location = early_exit_location
        self.num_channels = num_channels
        self.base_dims = CONVNEXT_V2_PICO_DIMS
        self.depths = CONVNEXT_V2_PICO_DEPTHS
        self.stage_dims = [_scaled_dim(base, width) for base, width in zip(self.base_dims, self.width_multipliers)]

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
                nn.Sequential(
                    LayerNorm(self.stage_dims[stage_idx - 1], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(self.stage_dims[stage_idx - 1], self.stage_dims[stage_idx], kernel_size=2, stride=2),
                )
            )

        self.stages = nn.ModuleList()
        dp_cursor = 0
        for stage_idx, (dim, depth) in enumerate(zip(self.stage_dims, self.depths)):
            blocks = []
            for _ in range(depth):
                blocks.append(SearchableConvNeXtV2Block(dim, drop_path=dp_rates[dp_cursor]))
                dp_cursor += 1
            self.stages.append(nn.Sequential(*blocks))

        self.exit_norms = nn.ModuleList([LayerNorm(dim, eps=1e-6, data_format="channels_first") for dim in self.stage_dims])
        self.classifier = nn.Linear(self.stage_dims[-1], num_classes)
        self.early_exit_classifier = None
        if early_exit_location is not None:
            if early_exit_location < 0 or early_exit_location >= len(self.stage_dims):
                raise ValueError(
                    f"ConvNeXt V2 early_exit_location must be in [0, {len(self.stage_dims) - 1}], got {early_exit_location}"
                )
            self.early_exit_classifier = nn.Linear(self.stage_dims[early_exit_location], num_classes)

        self._initialize_weights()

    def _forward_stage_features(self, x, target_stage):
        for stage_idx in range(target_stage + 1):
            x = self.downsample_layers[stage_idx](x)
            x = self.stages[stage_idx](x)
        return x

    def _pool_and_classify(self, x, stage_idx, classifier):
        x = self.exit_norms[stage_idx](x)
        x = x.mean(dim=(-2, -1))
        return classifier(x)

    def get_conv_layer_cutoff(self, exit_location):
        cutoff = 1  # patch stem conv
        for stage_idx, depth in enumerate(self.depths):
            if stage_idx > 0:
                cutoff += 1  # stage downsample conv
            cutoff += depth * 3  # dwconv + pwconv1 + pwconv2
            if stage_idx == exit_location:
                return cutoff
        return cutoff

    def count_params_to_exit(self, exit_location):
        total = sum(p.numel() for p in self.downsample_layers[0].parameters())
        for stage_idx in range(exit_location + 1):
            if stage_idx > 0:
                total += sum(p.numel() for p in self.downsample_layers[stage_idx].parameters())
            total += sum(p.numel() for p in self.stages[stage_idx].parameters())
            total += sum(p.numel() for p in self.exit_norms[stage_idx].parameters())

        if self.early_exit_classifier is not None:
            total += sum(p.numel() for p in self.early_exit_classifier.parameters())
        return total

    def forward(self, x, manual_early_exit_index=None):
        if self.early_exit_location is not None:
            x = self._forward_stage_features(x, self.early_exit_location)
            return [self._pool_and_classify(x, self.early_exit_location, self.early_exit_classifier)]

        x = self._forward_stage_features(x, len(self.stage_dims) - 1)
        return self._pool_and_classify(x, len(self.stage_dims) - 1, self.classifier)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def searchable_convnext(num_classes, width_multipliers, early_exit_location=None, num_channels=3):
    return SearchableConvNeXtV2(
        num_classes=num_classes,
        width_multipliers=width_multipliers,
        early_exit_location=early_exit_location,
        num_channels=num_channels,
    )
