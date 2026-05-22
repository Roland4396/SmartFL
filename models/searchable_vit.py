import torch
import torch.nn as nn

try:
    from timm.models.vision_transformer import Block
except ImportError as exc:
    raise ImportError("ViT support requires timm. Install it in the active environment.") from exc


TIMM_VIT_MODEL = "vit_small_patch16_224"
VIT_INPUT_SIZE = 224
VIT_DEPTH = 12
VIT_EMBED_DIM = 384
VIT_HEAD_DIM = 64
VIT_NUM_STAGES = 4
VIT_STAGE_DEPTHS = (3, 3, 3, 3)
VIT_BASE_MLP_RATIO = 4.0
VIT_MLP_HIDDEN_DIM = int(VIT_EMBED_DIM * VIT_BASE_MLP_RATIO)
VIT_OFFICIAL_EXIT_LOCATIONS = (8, 9, 10, 11, 12)
VIT_SEARCH_EXIT_LOCATIONS = tuple(range(3, VIT_DEPTH + 1))
VIT_EMBED_DIM_OPTIONS = (
    56,
    72,
    112,
    136,
    152,
    160,
    176,
    192,
    216,
    224,
    232,
    256,
    272,
    296,
    312,
    368,
    384,
)
VIT_WIDTH_OPTIONS = tuple(dim / VIT_EMBED_DIM for dim in VIT_EMBED_DIM_OPTIONS)

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


def _nearest_supported_embed_dim(width_multiplier):
    raw_dim = int(round(VIT_EMBED_DIM * float(width_multiplier)))
    return min(VIT_EMBED_DIM_OPTIONS, key=lambda dim: abs(dim - raw_dim))


def _scaled_embed_dim(width_multiplier):
    width_multiplier = max(min(float(width_multiplier), 1.0), min(VIT_WIDTH_OPTIONS))
    return _nearest_supported_embed_dim(width_multiplier)


def _stage_index_for_block(block_number):
    return min((block_number - 1) // VIT_STAGE_DEPTHS[0], VIT_NUM_STAGES - 1)


def _block_dims_from_widths(width_multipliers):
    stage_widths = _normalize_width_multipliers(width_multipliers)
    stage_dims = [_scaled_embed_dim(width) for width in stage_widths]
    block_dims = []
    for stage_dim, depth in zip(stage_dims, VIT_STAGE_DEPTHS):
        block_dims.extend([stage_dim] * depth)
    return stage_dims, block_dims


def _num_heads_for_embed_dim(embed_dim):
    for num_heads in (6, 4, 3, 2, 1):
        if embed_dim % num_heads == 0:
            return num_heads
    return 1


def _is_vit_qkv_key(key):
    return ".attn.qkv." in key


def is_vit_qkv_weight_key(key):
    return key.endswith(".attn.qkv.weight")


def is_vit_qkv_bias_key(key):
    return key.endswith(".attn.qkv.bias")


def slice_vit_qkv_tensor(full_tensor, target_shape):
    """Slice packed timm qkv tensors without mixing Q/K/V segments."""
    if full_tensor.dim() == 2:
        target_rows, target_cols = target_shape
        if target_rows % 3 != 0 or full_tensor.shape[0] % 3 != 0:
            return None
        full_dim = full_tensor.shape[0] // 3
        target_dim = target_rows // 3
        if target_cols != target_dim or target_dim > full_dim or target_cols > full_tensor.shape[1]:
            return None
        q = full_tensor[0:target_dim, 0:target_dim]
        k = full_tensor[full_dim:full_dim + target_dim, 0:target_dim]
        v = full_tensor[2 * full_dim:2 * full_dim + target_dim, 0:target_dim]
        return torch.cat([q, k, v], dim=0)

    if full_tensor.dim() == 1:
        target_rows = target_shape[0]
        if target_rows % 3 != 0 or full_tensor.shape[0] % 3 != 0:
            return None
        full_dim = full_tensor.shape[0] // 3
        target_dim = target_rows // 3
        if target_dim > full_dim:
            return None
        q = full_tensor[0:target_dim]
        k = full_tensor[full_dim:full_dim + target_dim]
        v = full_tensor[2 * full_dim:2 * full_dim + target_dim]
        return torch.cat([q, k, v], dim=0)

    return None


def make_vit_qkv_mask(full_shape, target_shape, device=None):
    """Boolean mask mapping a packed full qkv tensor to a packed subnet qkv tensor."""
    if len(full_shape) == 2:
        full_rows, full_cols = full_shape
        target_rows, target_cols = target_shape
        if full_rows % 3 != 0 or target_rows % 3 != 0:
            return None
        full_dim = full_rows // 3
        target_dim = target_rows // 3
        if target_cols != target_dim or target_dim > full_dim or target_cols > full_cols:
            return None
        mask = torch.zeros(full_shape, dtype=torch.bool, device=device)
        mask[0:target_dim, 0:target_dim] = True
        mask[full_dim:full_dim + target_dim, 0:target_dim] = True
        mask[2 * full_dim:2 * full_dim + target_dim, 0:target_dim] = True
        return mask

    if len(full_shape) == 1:
        full_rows = full_shape[0]
        target_rows = target_shape[0]
        if full_rows % 3 != 0 or target_rows % 3 != 0:
            return None
        full_dim = full_rows // 3
        target_dim = target_rows // 3
        if target_dim > full_dim:
            return None
        mask = torch.zeros(full_shape, dtype=torch.bool, device=device)
        mask[0:target_dim] = True
        mask[full_dim:full_dim + target_dim] = True
        mask[2 * full_dim:2 * full_dim + target_dim] = True
        return mask

    return None


def slice_prefix_tensor(full_tensor, target_shape):
    """Prefix-slice normal ViT tensors after hidden-width scaling."""
    if full_tensor.dim() != len(target_shape):
        return None
    if any(target > full for target, full in zip(target_shape, full_tensor.shape)):
        return None
    slices = tuple(slice(0, target) for target in target_shape)
    return full_tensor[slices]


def make_prefix_mask(full_shape, target_shape, device=None):
    """Boolean mask for normal prefix-sliced ViT tensors."""
    if len(full_shape) != len(target_shape):
        return None
    if any(target > full for target, full in zip(target_shape, full_shape)):
        return None
    mask = torch.zeros(full_shape, dtype=torch.bool, device=device)
    slices = tuple(slice(0, target) for target in target_shape)
    mask[slices] = True
    return mask


class PatchEmbed(nn.Module):
    def __init__(self, image_size, patch_size, num_channels, embed_dim):
        super().__init__()
        self.img_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (self.img_size[0] // patch_size, self.img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class HiddenDimAdapter(nn.Module):
    """Parameter-free hidden-dimension adapter for variable-width ViT stages."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        if self.out_dim == self.in_dim:
            return x
        if self.out_dim < self.in_dim:
            return x[..., :self.out_dim]
        pad_shape = list(x.shape)
        pad_shape[-1] = self.out_dim - self.in_dim
        return torch.cat([x, x.new_zeros(pad_shape)], dim=-1)


class ViTExitHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(self.norm(x[:, 0]))


class SearchableViT(nn.Module):
    """
    ViT-Small search wrapper using full hidden-width scaling.

    Width multipliers control the residual hidden dimension of each 3-block
    stage. The same dimension change applies to patch embedding, cls/pos
    tokens, qkv, attention projection, MLP, LayerNorm, and classifier heads.
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
            raise ValueError("ViT-Small wrapper currently expects RGB inputs.")
        if pretrained:
            raise ValueError(
                "Pretrained timm ViT weights cannot be loaded directly into variable-width ViT. "
                "Use the trained full-width supernet and QKV-aware slicing."
            )

        self.num_classes = num_classes
        self.width_multipliers = _normalize_width_multipliers(width_multipliers)
        self.stage_dims, self.block_dims = _block_dims_from_widths(self.width_multipliers)
        self.embed_dim = self.stage_dims[0]
        self.final_embed_dim = self.block_dims[-1]
        self.early_exit_location = _normalize_exit_location(early_exit_location)
        self.num_channels = num_channels
        self.image_size = image_size
        self.pretrained = pretrained

        self.patch_embed = PatchEmbed(image_size, patch_size=16, num_channels=num_channels, embed_dim=self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()

        self.block = nn.ModuleList()
        self.transitions = nn.ModuleDict()
        prev_dim = self.embed_dim
        for block_idx, dim in enumerate(self.block_dims, start=1):
            if dim != prev_dim:
                self.transitions[str(block_idx - 1)] = HiddenDimAdapter(prev_dim, dim)
            self.block.append(
                Block(
                    dim=dim,
                    num_heads=_num_heads_for_embed_dim(dim),
                    mlp_ratio=VIT_BASE_MLP_RATIO,
                    qkv_bias=True,
                )
            )
            prev_dim = dim

        self.norm = nn.LayerNorm(self.final_embed_dim)
        self.fc_norm = nn.Identity()
        self.head_drop = nn.Dropout(p=0.0)
        self.classifier = nn.Linear(self.final_embed_dim, num_classes)

        self.early_exit_classifier = None
        if self.early_exit_location is not None and self.early_exit_location < VIT_DEPTH:
            exit_dim = self.block_dims[self.early_exit_location - 1]
            self.early_exit_classifier = ViTExitHead(exit_dim, num_classes)

        self._init_weights()

    @property
    def blocks(self):
        return self.block

    @property
    def head(self):
        return self.classifier

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

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
        for block_idx, block in enumerate(self.block[:block_count], start=1):
            transition_key = str(block_idx - 1)
            if transition_key in self.transitions:
                x = self.transitions[transition_key](x)
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
        return 1 + 4 * exit_location

    def iter_nuclear_norm_modules(self, early_exit_location=None):
        block_count = VIT_DEPTH if early_exit_location is None else _normalize_exit_location(early_exit_location)
        yield self.patch_embed.proj
        for block_idx, block in enumerate(self.block[:block_count], start=1):
            transition_key = str(block_idx - 1)
            if transition_key in self.transitions:
                yield self.transitions[transition_key]
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
        for block_idx, block in enumerate(self.block[:block_count], start=1):
            transition_key = str(block_idx - 1)
            if transition_key in self.transitions:
                total += sum(p.numel() for p in self.transitions[transition_key].parameters())
            total += sum(p.numel() for p in block.parameters())
        if block_count < VIT_DEPTH and self.early_exit_classifier is not None:
            total += sum(p.numel() for p in self.early_exit_classifier.parameters())
        else:
            total += sum(p.numel() for p in self.norm.parameters())
            total += sum(p.numel() for p in self.classifier.parameters())
        return total

    def forward(self, x, manual_early_exit_index=None):
        if self.early_exit_location is not None and self.early_exit_location < VIT_DEPTH:
            x = self._forward_tokens(x, self.early_exit_location)
            return [self.early_exit_classifier(x)]

        x = self._forward_tokens(x, VIT_DEPTH)
        return self._forward_final(x)


def _create_vit_backbone(
        num_classes,
        image_size,
        embed_dim=VIT_EMBED_DIM,
        pretrained=False,
        width_multipliers=None):
    if width_multipliers is None:
        width_multipliers = [embed_dim / VIT_EMBED_DIM] * VIT_NUM_STAGES
    return SearchableViT(
        num_classes=num_classes,
        width_multipliers=width_multipliers,
        early_exit_location=None,
        num_channels=3,
        image_size=image_size,
        pretrained=pretrained,
    )


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
