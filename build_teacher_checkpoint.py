"""Utility to wrap published backbone weights into a full teacher checkpoint.

The official DINOv3 backbone checkpoints from Meta only provide a backbone
state_dict, while our fine-tuning code expects a consolidated "teacher"
checkpoint containing a ModuleDict with backbone + DINO/iBOT heads. This
script loads the backbone weights (ViT-H/16+ or ViT-7B/16), instantiates fresh
heads with architecture-appropriate defaults, and saves the combined state
under a top-level "teacher" key so SSLMetaArch can resume without missing-key
errors.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

# Local repo path so torch.hub can load from source
REPO_DIR = "/home/paul/dinov3"


@dataclass(frozen=True)
class HeadDefaults:
    dino_prototypes: int
    dino_bottleneck_dim: int
    dino_hidden_dim: int
    dino_nlayers: int
    ibot_prototypes: int
    ibot_bottleneck_dim: int
    ibot_hidden_dim: int
    ibot_nlayers: int


@dataclass(frozen=True)
class ModelSpec:
    name: str
    loader_name: str
    default_weights: str
    default_out: str
    head_defaults: HeadDefaults
    weight_hints: tuple[str, ...]


def _model_specs() -> dict[str, ModelSpec]:
    """Return supported model specs keyed by arch name."""
    return {
        "vith16plus": ModelSpec(
            name="vith16plus",
            loader_name="dinov3_vith16plus",
            default_weights="checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
            default_out="checkpoints/dinov3_vith16plus_saved_teacher.pth",
            head_defaults=HeadDefaults(
                dino_prototypes=262144,
                dino_bottleneck_dim=512,
                dino_hidden_dim=8192,
                dino_nlayers=3,
                ibot_prototypes=98304,
                ibot_bottleneck_dim=384,
                ibot_hidden_dim=4096,
                ibot_nlayers=3,
            ),
            weight_hints=("vith16plus", "vith", "h16plus", "7c1da9a5"),
        ),
        "vit7b16": ModelSpec(
            name="vit7b16",
            loader_name="dinov3_vit7b16",
            default_weights="checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
            default_out="checkpoints/dinov3_vit7b16_saved_teacher.pth",
            head_defaults=HeadDefaults(
                dino_prototypes=262144,
                dino_bottleneck_dim=1024,
                dino_hidden_dim=16384,
                dino_nlayers=3,
                ibot_prototypes=98304,
                ibot_bottleneck_dim=768,
                ibot_hidden_dim=8192,
                ibot_nlayers=3,
            ),
            weight_hints=("vit7b16", "vit7b", "a955f4ea", "a6675841"),
        ),
    }


def _infer_arch_from_weights(weights_path: str, specs: dict[str, ModelSpec]) -> Optional[str]:
    """Infer which backbone to use based on filename hints."""
    name = weights_path.lower()
    for arch, spec in specs.items():
        if any(hint in name for hint in spec.weight_hints):
            return arch
    return None


def _load_backbone(loader: Callable, weights: str):
    """Load a backbone with the provided weights, avoiding torch.hub extra deps."""
    backbone = loader(pretrained=True, weights=weights)
    backbone.eval()
    return backbone


def main():
    specs = _model_specs()
    parser = argparse.ArgumentParser(
        description="Load DINOv3 backbone (ViT-H/16+ or ViT-7B/16) and save a pretraining-style teacher checkpoint."
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="auto",
        choices=["auto", *specs.keys()],
        help="Backbone architecture to wrap; defaults to auto-detect from the weights path.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path for the saved teacher checkpoint (.pth). Defaults per-architecture.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=specs["vith16plus"].default_weights,
        help="Path or URL to pretrained backbone weights.",
    )
    # Head sizing defaults/overrides (avoids relying on any external YAML); None means use arch defaults.
    parser.add_argument("--dino-prototypes", type=int, default=None)
    parser.add_argument("--dino-bottleneck-dim", type=int, default=None)
    parser.add_argument("--dino-hidden-dim", type=int, default=None)
    parser.add_argument("--dino-nlayers", type=int, default=None)
    parser.add_argument("--ibot-prototypes", type=int, default=None)
    parser.add_argument("--ibot-bottleneck-dim", type=int, default=None)
    parser.add_argument("--ibot-hidden-dim", type=int, default=None)
    parser.add_argument("--ibot-nlayers", type=int, default=None)
    args = parser.parse_args()

    # Resolve architecture
    arch = args.arch
    if arch == "auto":
        arch = _infer_arch_from_weights(args.weights, specs)
        if arch is None:
            raise ValueError(
                f"Could not infer architecture from weights path '{args.weights}'. "
                f"Pass --arch explicitly (choices: {', '.join(specs.keys())})."
            )
    if arch not in specs:
        raise ValueError(f"Unsupported architecture '{arch}'. Supported: {', '.join(specs.keys())}.")
    spec = specs[arch]

    # Backfill defaults that depend on the selected architecture
    weights = args.weights or spec.default_weights
    out = args.out or spec.default_out
    hd = spec.head_defaults
    dino_prototypes = args.dino_prototypes or hd.dino_prototypes
    dino_bottleneck_dim = args.dino_bottleneck_dim or hd.dino_bottleneck_dim
    dino_hidden_dim = args.dino_hidden_dim or hd.dino_hidden_dim
    dino_nlayers = args.dino_nlayers or hd.dino_nlayers
    ibot_prototypes = args.ibot_prototypes or hd.ibot_prototypes
    ibot_bottleneck_dim = args.ibot_bottleneck_dim or hd.ibot_bottleneck_dim
    ibot_hidden_dim = args.ibot_hidden_dim or hd.ibot_hidden_dim
    ibot_nlayers = args.ibot_nlayers or hd.ibot_nlayers

    # Load backbone from local repo with the provided weights without torch.hub to avoid extra deps
    sys.path.insert(0, str(Path(REPO_DIR)))
    from dinov3.hub import backbones as hub_backbones  # noqa: E402

    loader = getattr(hub_backbones, spec.loader_name)
    backbone = _load_backbone(loader, weights)

    embed_dim = getattr(backbone, "embed_dim", None)
    if embed_dim is None:
        raise RuntimeError("Loaded backbone has no embed_dim attribute; cannot size heads.")

    from dinov3.layers.dino_head import DINOHead  # noqa: E402

    dino_head = DINOHead(
        in_dim=embed_dim,
        out_dim=dino_prototypes,
        hidden_dim=dino_hidden_dim,
        bottleneck_dim=dino_bottleneck_dim,
        nlayers=dino_nlayers,
    )
    ibot_head = DINOHead(
        in_dim=embed_dim,
        out_dim=ibot_prototypes,
        hidden_dim=ibot_hidden_dim,
        bottleneck_dim=ibot_bottleneck_dim,
        nlayers=ibot_nlayers,
    )
    dino_head.init_weights()
    ibot_head.init_weights()

    moduledict = torch.nn.ModuleDict(
        {
            "backbone": backbone,
            "dino_head": dino_head,
            "ibot_head": ibot_head,
        }
    )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    teacher_state = {k: v.cpu() for k, v in moduledict.state_dict().items()}
    torch.save({"teacher": teacher_state}, out_path)
    print(f"Saved teacher checkpoint to: {out_path.resolve()}")


if __name__ == "__main__":
    main()


# python /home/paul/dinov3/build_teacher_checkpoint.py \
#   --arch vit7b16 \
#   --weights /home/paul/dinov3/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth \
#   --out /home/paul/dinov3/checkpoints/dinov3_vit7b16_saved_teacher_smallheads.pth \
#   --dino-prototypes 131072 --dino-bottleneck-dim 384 --dino-hidden-dim 2048 --dino-nlayers 3 \
#   --ibot-prototypes 131072 --ibot-bottleneck-dim 256 --ibot-hidden-dim 2048 --ibot-nlayers 3
