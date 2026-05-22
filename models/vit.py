import copy

import torch
import torch.nn as nn

from args import arg_parser, modify_args
from hierarchical_model_selector import (
    find_all_growth_configs,
    find_best_config_for_distribution,
    find_best_config_independent,
    load_configs_from_json,
)
from models.searchable_vit import (
    TIMM_VIT_MODEL,
    VIT_DEPTH,
    VIT_EMBED_DIM,
    VIT_OFFICIAL_EXIT_LOCATIONS,
    ViTExitHead,
    _create_vit_backbone,
    _normalize_width_multipliers,
)

try:
    import timm
except ImportError as exc:
    raise ImportError("ViT support requires timm. Install it in the active environment.") from exc


DEFAULT_VIT_EXIT_LOCATIONS = [8, 10, 11]


def _sorted_unique_exit_locations(exit_locations):
    return sorted(set(exit_locations))


def _normalize_exit_locations(exit_locations):
    if exit_locations is None:
        return []
    normalized = _sorted_unique_exit_locations(exit_locations)
    invalid = [loc for loc in normalized if loc < 1 or loc >= VIT_DEPTH]
    if invalid:
        raise ValueError(
            f"ViT early-exit locations must be block counts in [1, {VIT_DEPTH - 1}], got {invalid}"
        )
    return normalized


class ViTSmall(nn.Module):
    """
    Multi-exit wrapper around a ViT-Small supernet.

    Stage 3 instantiates the highest selected width as the global model and
    attaches the exits selected from the Stage 2 architecture library.
    """

    def __init__(
        self,
        participating_levels,
        num_channels=3,
        num_classes=100,
        trs=True,
        scale=1.0,
        ee_layer_locations=None,
        args=None,
        image_size=224,
        pretrained=False,
        width_multipliers_override=None,
    ):
        super().__init__()
        self.stored_inp_kwargs = copy.deepcopy(locals())
        del self.stored_inp_kwargs["self"]
        del self.stored_inp_kwargs["__class__"]

        if num_channels != 3:
            raise ValueError("timm ViT-Small wrapper currently expects RGB inputs.")

        requested_ee_layer_locations = _normalize_exit_locations(ee_layer_locations)

        if args is None:
            args = arg_parser.parse_args()
            args = modify_args(args)

        if hasattr(args, "image_size"):
            image_size = args.image_size[0]
        self.stored_inp_kwargs["image_size"] = image_size
        self.stored_inp_kwargs["pretrained"] = False
        pretrained = False

        if args.config_library_path is None:
            if hasattr(args, "arch") and args.arch and "vit" in args.arch:
                model_name = "vit_small"
            else:
                model_name = getattr(args, "model", "vit")
            dataset_name = getattr(args, "data", "cifar100")
            config_library_path = f"{model_name}_{dataset_name}_architecture_library.json"
        else:
            config_library_path = args.config_library_path

        all_model_configs = load_configs_from_json(
            config_library_path,
            model_type="vit",
            dataset=getattr(args, "data", None),
        )
        flops_constraints = None
        if args.flops_constraints:
            flops_constraints = {i: val for i, val in enumerate(args.flops_constraints)}
        params_constraints = None
        if args.params_constraints:
            params_constraints = {i: val for i, val in enumerate(args.params_constraints)}

        if args.independent_selection:
            best_configs_for_round = find_best_config_independent(
                all_model_configs,
                participating_levels,
                flops_constraints=flops_constraints,
                params_constraints=params_constraints,
            )
        else:
            best_configs_for_round = find_best_config_for_distribution(
                all_model_configs,
                participating_levels,
                beam_width=500,
                flops_constraints=flops_constraints,
                params_constraints=params_constraints,
            )

        if best_configs_for_round is None:
            print("Warning: No valid hierarchical configuration found. Using default ViT-Small exits.")
            if requested_ee_layer_locations:
                ee_loc_list = requested_ee_layer_locations
            else:
                ee_loc_list = DEFAULT_VIT_EXIT_LOCATIONS[:max(len(participating_levels) - 1, 0)]
            wide_scales = [1.0] * 4
        else:
            normal_exit_locations = []
            sorted_configs = [
                best_configs_for_round[level]
                for level in sorted(best_configs_for_round.keys())
            ]
            for config in sorted_configs:
                normal_exit_locations.append(config["early_exit_location"])
            normalized_widths = [
                _normalize_width_multipliers(config["width_multipliers"])
                for config in sorted_configs
            ]
            wide_scales = [
                max(widths[stage_idx] for widths in normalized_widths)
                for stage_idx in range(4)
            ]
            ee_loc_list = [loc for loc in normal_exit_locations[:-1] if loc < VIT_DEPTH]

            if getattr(args, "enable_tdd", 0) == 1:
                growth_configs = find_all_growth_configs(
                    best_configs_for_round,
                    all_model_configs,
                    model_type="vit",
                    growth_budget_scale=getattr(args, "tdd_growth_budget_scale", 1.0),
                )
                growth_exit_locations = []
                for growth_config in growth_configs.values():
                    if growth_config is None:
                        continue
                    if growth_config["early_exit_location"] < VIT_DEPTH:
                        growth_exit_locations.append(growth_config["early_exit_location"])
                ee_loc_list.extend(_sorted_unique_exit_locations(growth_exit_locations))

        ee_loc_list = _normalize_exit_locations(ee_loc_list)
        if getattr(args, "enable_tdd", 0) == 1:
            print(f"[TDD] Added growth exits, all exits: {ee_loc_list}")

        self.scale = scale
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.trs = trs
        self.image_size = image_size
        self.pretrained = pretrained
        self.ee_layer_locations = ee_loc_list
        self.num_blocks = len(self.ee_layer_locations) + 1
        if width_multipliers_override is not None:
            wide_scales = width_multipliers_override
            scale = 1.0
        elif scale < 1.0:
            wide_scales = [scale] * 4
        self.width_multipliers = _normalize_width_multipliers(wide_scales)

        base = _create_vit_backbone(
            num_classes=num_classes,
            image_size=image_size,
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
        self.transitions = base.transitions
        self.norm = base.norm
        self.fc_norm = base.fc_norm
        self.head_drop = base.head_drop
        self.classifier = base.head
        self.stage_dims = base.stage_dims
        self.block_dims = base.block_dims
        self.embed_dim = base.embed_dim
        self.final_embed_dim = base.final_embed_dim

        self.ee_classifiers = nn.ModuleList()
        self.exit_to_classifier_idx = {}
        for classifier_idx, exit_loc in enumerate(self.ee_layer_locations):
            self.ee_classifiers.append(ViTExitHead(self.block_dims[exit_loc - 1], num_classes))
            self.exit_to_classifier_idx[exit_loc] = classifier_idx

    def _pos_embed(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        return self.pos_drop(x)

    def _forward_final(self, x):
        x = self.norm(x)
        x = self.fc_norm(x[:, 0])
        x = self.head_drop(x)
        return self.classifier(x)

    def extract_tokens_to_exit(self, x, early_exit_location=None):
        block_count = VIT_DEPTH if early_exit_location is None else early_exit_location
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

    def iter_nuclear_norm_modules(self, early_exit_location=None):
        block_count = VIT_DEPTH if early_exit_location is None else early_exit_location
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
        total = 0
        total += sum(p.numel() for p in self.patch_embed.parameters())
        total += self.cls_token.numel()
        total += self.pos_embed.numel()
        total += sum(p.numel() for p in self.pos_drop.parameters())
        total += sum(p.numel() for p in self.patch_drop.parameters())
        total += sum(p.numel() for p in self.norm_pre.parameters())
        for block_idx, block in enumerate(self.block[:early_exit_location], start=1):
            transition_key = str(block_idx - 1)
            if transition_key in self.transitions:
                total += sum(p.numel() for p in self.transitions[transition_key].parameters())
            total += sum(p.numel() for p in block.parameters())

        classifier_idx = self.exit_to_classifier_idx.get(early_exit_location)
        if classifier_idx is not None:
            total += sum(p.numel() for p in self.ee_classifiers[classifier_idx].parameters())
        else:
            total += sum(p.numel() for p in self.norm.parameters())
            total += sum(p.numel() for p in self.fc_norm.parameters())
            total += sum(p.numel() for p in self.head_drop.parameters())
            total += sum(p.numel() for p in self.classifier.parameters())
        return total

    def forward(self, x, manual_early_exit_index=0):
        preds = []
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for block_idx, block in enumerate(self.block, start=1):
            transition_key = str(block_idx - 1)
            if transition_key in self.transitions:
                x = self.transitions[transition_key](x)
            x = block(x)
            classifier_idx = self.exit_to_classifier_idx.get(block_idx)
            if classifier_idx is not None:
                preds.append(self.ee_classifiers[classifier_idx](x))
                if manual_early_exit_index and len(preds) >= manual_early_exit_index:
                    return preds

        preds.append(self._forward_final(x))
        if manual_early_exit_index:
            return preds[:manual_early_exit_index]
        return preds


def vit_small_1(args, params):
    num_classes = args.num_classes
    track_running_stats = args.track_running_stats if hasattr(args, "track_running_stats") else False
    image_size = args.image_size[0] if hasattr(args, "image_size") else 224
    return ViTSmall(
        participating_levels=[],
        num_channels=3,
        num_classes=num_classes,
        trs=track_running_stats,
        scale=params.get("scale", 1.0),
        ee_layer_locations=[],
        args=args,
        image_size=image_size,
    )


def vit_small_4(participating_levels, args, params):
    num_classes = args.num_classes
    track_running_stats = args.track_running_stats if hasattr(args, "track_running_stats") else False
    image_size = args.image_size[0] if hasattr(args, "image_size") else 224
    return ViTSmall(
        participating_levels=participating_levels,
        num_channels=3,
        num_classes=num_classes,
        trs=track_running_stats,
        scale=params.get("scale", 1.0),
        ee_layer_locations=params.get("ee_layer_locations", DEFAULT_VIT_EXIT_LOCATIONS.copy()),
        args=args,
        image_size=image_size,
    )


def vit_tiny_1(args, params):
    return vit_small_1(args, params)


def vit_tiny_4(participating_levels, args, params):
    return vit_small_4(participating_levels, args, params)
