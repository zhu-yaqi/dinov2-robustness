from __future__ import annotations
import argparse
from pathlib import Path
import torch

from src.checkpoints import save_checkpoint
from src.data import build_loaders
from src.engine import evaluate, train_consistency_epoch
from src.models import FrozenFeatureClassifier
from src.utils import (
    format_count, get_device, set_seed,
    total_parameter_count, trainable_parameter_count,
)

def parse_args():
    p = argparse.ArgumentParser(description="Train DINOv2 with an occlusion-consistency adapter.")
    p.add_argument("--dino-source", choices=["hub","timm","auto"], default="auto")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--adapter-dim", type=int, default=256)
    p.add_argument("--lambda-consistency", type=float, default=0.5)
    p.add_argument("--occlusion-min", type=float, default=0.10)
    p.add_argument("--occlusion-max", type=float, default=0.40)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--train-subset", type=int, default=None)
    p.add_argument("--val-subset", type=int, default=None)
    p.add_argument("--test-subset", type=int, default=None)
    p.add_argument("--output", default="results/checkpoints/dinov2_consistency_l05.pt")
    p.add_argument("--label", default=None)
    return p.parse_args()

def main():
    args = parse_args()
    if args.lambda_consistency < 0:
        raise ValueError("--lambda-consistency must be >= 0")

    set_seed(args.seed)
    device = get_device(args.device)
    label = args.label or f"DINOv2+Adapter lambda={args.lambda_consistency:g}"

    loaders = build_loaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        test_subset=args.test_subset,
    )

    model = FrozenFeatureClassifier(
        backbone_name="dinov2_vits14",
        num_classes=10,
        adapter_dim=args.adapter_dim,
        dino_source=args.dino_source,
    ).to(device)

    print("Device:", device)
    print("Source:", model.actual_source)
    print("Total parameters:", format_count(total_parameter_count(model)))
    print("Trainable parameters:", format_count(trainable_parameter_count(model)))
    print("lambda_consistency:", args.lambda_consistency)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = -1.0
    for epoch in range(1, args.epochs+1):
        train_m = train_consistency_epoch(
            model, loaders.train, optimizer, device,
            lambda_consistency=args.lambda_consistency,
            occlusion_min=args.occlusion_min,
            occlusion_max=args.occlusion_max,
            amp=not args.no_amp,
        )
        val_m = evaluate(model, loaders.val, device, amp=not args.no_amp)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"loss={train_m.loss:.4f}, cons={train_m.consistency_loss:.4f}, "
            f"train acc={100*train_m.accuracy:.2f}% | "
            f"val acc={100*val_m.accuracy:.2f}%"
        )

        if val_m.accuracy > best_val:
            best_val = val_m.accuracy
            save_checkpoint(
                args.output, model, epoch, best_val, label, vars(args).copy()
            )
            print("Saved best:", args.output)

    print("Best val accuracy:", f"{100*best_val:.2f}%")
    print("Checkpoint:", Path(args.output).resolve())

if __name__ == "__main__":
    main()
