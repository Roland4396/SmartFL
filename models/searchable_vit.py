import torch
import torch.nn as nn

try:
    import timm
    from timm.models.vision_transformer import VisionTransformer
except ImportError as exc:
    raise ImportError("ViT support requires timm. Install it in the active environment.") from exc


TIMM_VIT_MODEL = "vit_small_patch16_224"
VIT_DEPTH = 12
VIT_EMBED_DIM = 384
VIT_HEAD_DIM = 64
VIT_NUM_STAGES = 4
VIT_STAGE_DEPTHS = (3, 3, 3, 3)
VIT_BASE_MLP_RATIO = 4.0
VIT_MLP_HIDDEN_DIM = int(VIT_EMBED_DIM * VIT_BASE_MLP_RATIO)
VIT_OFFICIAL_EXIT_LOCATIONS = (8, 9, 10, 11, 12)
VIT_SEARCH_EXIT_LOCATIONS = tuple(range(3, VIT_DEPTH + 1))
VIT_WIDTH_OPTIONS = (
    0.125,
    0.25,
    1.0 / 3.0,
    5.0 / 12.0,
    0.5,
    2.0 / 3.0,
    0.75,
    7.0 / 8.0,
    11.0 / 12.0,
    1.0,
)

# Backward-compatible names for older local scripts.
VIT_TINY_DEPTH = VIT_DEPTH
VIT_TINY_EMBED_DIM = VIT_EMBED_DIM


def _normalize_exit_location(early_exit_location):
    if early_exit_location is None:
        return None
    if early_exit_location < 1 or early_exit_location > VIT_DEPTH:
        raise ValueError(
            f"ViT early_exit_location must be a block count in [1, {VIT_DEPTH}], "
            f"got {early_exit_location}"
        )
    return early_exit_location


def _normalize_width_multipliers(width_multipliers):
    if width_multipliers is None:
        return [1.0] * VIT_NUM_STAGES
    if isinstance(width_multipliers, (int, float)):
        return [float(width_multipliers)] * VIT_NUM_STAGES
    if len(width_multipliers) == 0:
        return [1.0] * VIT_NUM_STAGES
    if len(width_multipliers) == 1:
        return [float(width_multipliers[0])] * VIT_NUM_STAGES
    if len(width_multipliers) == VIT_NUM_STAGES:
        return [float(width) for width in width_multipliers]
    if len(width_multipliers) == VIT_DEPTH:
        stage_widths = []
        offset = 0
        for depth in VIT_STAGE_DEPTHS:
            stage_widths.append(float(width_multipliers[offset]))
            offset += depth
        return stage_widths
    raise ValueError(
        f"ViT width_multipliers must have length 1, {VIT_NUM_STAGES}, or {VIT_DEPTH}; "
        f"got {len(width_multipliers)}"
    )


def _normalize_width_multiplier(width_multipliers):
    return _normalize_width_multipliers(width_multipliers)[0]


def _scaled_embed_dim(width_multiplier):
    # Kept for compatibility with older callers. The searchable ViT no longer
    # changes the residual stream width; pruning happens inside each MLP block.
    return VIT_EMBED_DIM


def _stage_index_for_block(block_number):
    return min((block_number - 1) // VIT_STAGE_DEPTHS[0], VIT_NUM_STAGES - 1)


def _scaled_mlp_hidden_dim(width_multiplier):
    width_multiplier = max(min(float(width_multiplier), 1.0), min(VIT_WIDTH_OPTIONS))
    hidden_dim = int(round(VIT_MLP_HIDDEN_DIM * width_multiplier / VIT_HEAD_DIM)) * VIT_HEAD_DIM
    return max(VIT_HEAD_DIM, min(VIT_MLP_HIDDEN_DIM, hidden_dim))


def _num_heads_for_embed_dim(embed_dim):
    return max(1, embed_dim // VIT_HEAD_DIM)


def _resize_block_mlp(block, hidden_dim):
    current_hidden_dim = block.mlp.fc1.out_features
    if current_hidden_dim == hidden_dim:
        return
    in_features = block.mlp.fc1.in_features
    out_features = block.mlp.fc2.out_features
    block.mlp.fc1 = nn.Linear(in_features, hidden_dim, bias=block.mlp.fc1.bias is not None)
    block.mlp.fc2 = nn.Linear(hidden_dim, out_features, bias=block.mlp.fc2.bias is not None)


def _create_vit_backbone(
        num_classes,
        image_size,
        embed_dim=VIT_EMBED_DIM,
        pretrained=False,
        width_multipliers=None):
    stage_widths = _normalize_width_multipliers(width_multipliers)
    use_pretrained = pretrained and embed_dim == VIT_EMBED_DIM and all(width == 1.0 for width in stage_widths)
    if use_pretrained:
        return timm.create_model(
            TIMM_VIT_MODEL,
            pretrained=True,
            num_classes=num_classes,
            img_size=image_size,
        )

    model = VisionTransformer(
        img_size=image_size,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=VIT_DEPTH,
        num_heads=_num_heads_for_embed_dim(embed_dim),
        mlp_ratio=VIT_BASE_MLP_RATIO,
        qkv_bias=True,
    )

    for block_idx, block in enumerate(model.blocks, start=1):
        stage_idx = _stage_index_for_block(block_idx)
        hidden_dim = _scaled_mlp_hidden_dim(stage_widths[stage_idx])
        _resize_block_mlp(block, hidden_dim)
    return model


class ViTExitHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(self.norm(x[:, 0]))


class SearchableViT(nn.Module):
    """
    Search wrapper around timm's ViT-Small implementation.

    The residual stream keeps the official ViT-Small embedding dimension.
    Width multipliers prune the MLP hidden dimension in four transformer-block
    stages, while early_exit_location controls depth.
    """

    def __init__(
        self,
        num_classes,
        width_multipliers=None,
        early_exit_location=None,
        num_channels=3,
        image_size=224,
        pretrained=False,
    ):
        super().__init__()
        if num_channels != 3:
            raise ValueError("timm ViT-Small wrapper currently expects RGB inputs.")

        self.num_classes = num_classes
        self.width_multipliers = _normalize_width_multipliers(width_multipliers)
        self.embed_dim = VIT_EMBED_DIM
        self.early_exit_location = _normalize_exit_location(early_exit_location)
        self.num_channels = num_channels
        self.image_size = image_size
        self.pretrained = pretrained

        base = _create_vit_backbone(
            num_classes=num_classes,
            image_size=image_size,
            embed_dim=self.embed_dim,
            pretrained=pretrained,
            width_multipliers=self.width_multipliers,
        )

        self.patch_embed = base.patch_embed
        self.cls_token = base.cls_token
        self.pos_embed = base.pos_embed
        self.pos_drop = base.pos_drop
        self.patch_drop = base.patch_drop
        self.norm_pre = base.norm_pre
        self.block = nn.ModuleList(list(base.blocks))
        self.norm = base.norm
        self.fc_norm = base.fc_norm
        self.head_drop = base.head_drop
        self.classifier = base.head

        self.early_exit_classifier = None
        if self.early_exit_location is not None and self.early_exit_location < VIT_DEPTH:
            self.early_exit_classifier = ViTExitHead(self.embed_dim, num_classes)

    def _pos_embed(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        return self.pos_drop(x)

    def _forward_tokens(self, x, block_count):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for block in self.block[:block_count]:
            x = block(x)
        return x

    def extract_tokens_to_exit(self, x, early_exit_location=None):
        block_count = VIT_DEPTH if early_exit_location is None else _normalize_exit_location(early_exit_location)
        return self._forward_tokens(x, block_count)

    def _forward_final(self, x):
        x = self.norm(x)
        x = self.fc_norm(x[:, 0])
        x = self.head_drop(x)
        return self.classifier(x)

    def get_weight_layer_cutoff(self, exit_location):
        exit_location = _normalize_exit_location(exit_location)
        # patch projection + qkv/proj/fc1/fc2 for each executed transformer block
        return 1 + 4 * exit_location

    def iter_nuclear_norm_modules(self, early_exit_location=None):
        block_count = VIT_DEPTH if early_exit_location is None else _normalize_exit_location(early_exit_location)
        yield self.patch_embed.proj
        for block in self.block[:block_count]:
            yield block.attn.qkv
            yield block.attn.proj
            yield block.mlp.fc1
            yield block.mlp.fc2

    def count_params_to_exit(self, early_exit_location):
        block_count = _normalize_exit_location(early_exit_location)
        total = 0
        total += sum(p.numel() for p in self.patch_embed.parameters())
        total += self.cls_token.numel()
        total += self.pos_embed.numel()
        total += sum(p.numel() for p in self.pos_drop.parameters())
        total += sum(p.numel() for p in self.patch_drop.parameters())
        total += sum(p.numel() for p in self.norm_pre.parameters())
        for block in self.block[:block_count]:
            total += sum(p.numel() for p in block.parameters())
        if block_count < VIT_DEPTH and self.early_exit_classifier is not None:
            total += sum(p.numel() for p in self.early_exit_classifier.parameters())
        else:
            total += sum(p.numel() for p in self.norm.parameters())
            total += sum(p.numel() for p in self.fc_norm.parameters())
            total += sum(p.numel() for p in self.head_drop.parameters())
            total += sum(p.numel() for p in self.classifier.parameters())
        return total

    def forward(self, x, manual_early_exit_index=None):
        if self.early_exit_location is not None and self.early_exit_location < VIT_DEPTH:
            x = self._forward_tokens(x, self.early_exit_location)
            return [self.early_exit_classifier(x)]

        x = self._forward_tokens(x, VIT_DEPTH)
        return self._forward_final(x)


def searchable_vit_small(
        num_classes,
        width_multipliers=None,
        early_exit_location=None,
        num_channels=3,
        image_size=224,
        pretrained=False):
    return SearchableViT(
        num_classes=num_classes,
        width_multipliers=width_multipliers,
        early_exit_location=early_exit_location,
        num_channels=num_channels,
        image_size=image_size,
        pretrained=pretrained,
    )


def searchable_vit_tiny(*args, **kwargs):
    return searchable_vit_small(*args, **kwargs)
