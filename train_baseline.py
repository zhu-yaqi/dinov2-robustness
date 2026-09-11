from __future__ import annotations
import argparse
from pathlib import Path
import torch

from src.checkpoints import save_checkpoint
from src.data import build_loaders
from src.engine import evaluate, train_baseline_epoch
from src.models import FrozenFeatureClassifier
from src.utils import (
    format_count, get_device, set_seed,
    total_parameter_count, trainable_parameter_count,
)

def parse_args():
    p = argparse.ArgumentParser(description="Train a frozen-backbone linear baseline.")
    p.add_argument("--model", choices=["resnet50","convnextv2_tiny","dinov2_vits14"], required=True)
    p.add_argument("--dino-source", choices=["hub","timm","auto"], default="auto")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--train-subset", type=int, default=None)
    p.add_argument("--val-subset", type=int, default=None)
    p.add_argument("--test-subset", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--label", default=None)
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    output = args.output or f"results/checkpoints/{args.model}.pt"
    label = args.label or args.model

    print("Device:", device)
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

    print("Loading pretrained model...")
    model = FrozenFeatureClassifier(
        backbone_name=args.model,
        num_classes=10,
        adapter_dim=0,
        dino_source=args.dino_source,
    ).to(device)
    print("Source:", model.actual_source)
    print("Total parameters:", format_count(total_parameter_count(model)))
    print("Trainable parameters:", format_count(trainable_parameter_count(model)))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = -1.0
    for epoch in range(1, args.epochs+1):
        train_m = train_baseline_epoch(
            model, loaders.train, optimizer, device, amp=not args.no_amp
        )
        val_m = evaluate(model, loaders.val, device, amp=not args.no_amp)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={train_m.loss:.4f}, train acc={100*train_m.accuracy:.2f}% | "
            f"val loss={val_m.loss:.4f}, val acc={100*val_m.accuracy:.2f}%"
        )

        if val_m.accuracy > best_val:
            best_val = val_m.accuracy
            save_checkpoint(
                output, model, epoch, best_val, label, vars(args).copy()
            )
            print("Saved best:", output)

    print("Best val accuracy:", f"{100*best_val:.2f}%")
    print("Checkpoint:", Path(output).resolve())

if __name__ == "__main__":
    main()
