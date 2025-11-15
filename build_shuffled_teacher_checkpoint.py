"""Build a randomly shuffled teacher checkpoint for sanity-check experiments."""

import argparse
import sys
from pathlib import Path

import torch

REPO_DIR = "/home/paul/dinov3"


def _should_shuffle(tensor: torch.Tensor) -> bool:
    return tensor is not None and tensor.is_floating_point() and tensor.numel() > 1


def shuffle_tensor_(tensor: torch.Tensor) -> None:
    if not _should_shuffle(tensor):
        return
    flat = tensor.view(-1)
    perm = torch.randperm(flat.numel(), device=flat.device)
    flat.copy_(flat[perm])


def shuffle_module_(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        shuffle_tensor_(parameter.data)
    for buffer in module.buffers():
        shuffle_tensor_(buffer)


def main():
    parser = argparse.ArgumentParser(description="Shuffle a pretrained DINOv3 ViT-H/16 teacher checkpoint.")
    parser.add_argument(
        "--out",
        type=str,
        default="checkpoints/dinov3_vith16plus_random_teacher.pth",
        help="Output path for the shuffled teacher checkpoint.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
        help="Path or URL to pretrained backbone weights.",
    )
    parser.add_argument("--dino-prototypes", type=int, default=262144)
    parser.add_argument("--dino-bottleneck-dim", type=int, default=512)
    parser.add_argument("--dino-hidden-dim", type=int, default=8192)
    parser.add_argument("--dino-nlayers", type=int, default=3)
    parser.add_argument("--ibot-prototypes", type=int, default=98304)
    parser.add_argument("--ibot-bottleneck-dim", type=int, default=384)
    parser.add_argument("--ibot-hidden-dim", type=int, default=4096)
    parser.add_argument("--ibot-nlayers", type=int, default=3)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(REPO_DIR)))
    from dinov3.hub.backbones import dinov3_vith16plus  # noqa: E402
    from dinov3.layers.dino_head import DINOHead  # noqa: E402

    backbone = dinov3_vith16plus(pretrained=True, weights=args.weights)
    embed_dim = getattr(backbone, "embed_dim", None)
    if embed_dim is None:
        raise RuntimeError("Loaded backbone has no embed_dim attribute; cannot size heads.")

    dino_head = DINOHead(
        in_dim=embed_dim,
        out_dim=args.dino_prototypes,
        hidden_dim=args.dino_hidden_dim,
        bottleneck_dim=args.dino_bottleneck_dim,
        nlayers=args.dino_nlayers,
    )
    ibot_head = DINOHead(
        in_dim=embed_dim,
        out_dim=args.ibot_prototypes,
        hidden_dim=args.ibot_hidden_dim,
        bottleneck_dim=args.ibot_bottleneck_dim,
        nlayers=args.ibot_nlayers,
    )
    moduledict = torch.nn.ModuleDict({"backbone": backbone, "dino_head": dino_head, "ibot_head": ibot_head})
    shuffle_module_(moduledict)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    teacher_state = {k: v.cpu() for k, v in moduledict.state_dict().items()}
    torch.save({"teacher": teacher_state}, out_path)
    print(f"Saved shuffled teacher checkpoint to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
