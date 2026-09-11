from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import torch

from src.checkpoints import load_checkpoint
from src.constants import CORRUPTION_SPECS
from src.data import build_loaders
from src.engine import evaluate
from src.utils import ensure_parent, get_device, set_seed

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate clean and corrupted CIFAR-10.")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--corruption-seed", type=int, default=1234)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--test-subset", type=int, default=None)
    p.add_argument("--dino-source-override", choices=["hub","timm","auto"], default=None)
    p.add_argument("--output", default="results/tables/robustness_main.csv")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    test_loader = build_loaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_subset=1,
        val_subset=1,
        test_subset=args.test_subset,
    ).test

    rows = []
    for ckpt_path in args.checkpoints:
        print("\n" + "="*72)
        model, payload = load_checkpoint(
            ckpt_path, device, args.dino_source_override
        )
        label = payload.get("label", Path(ckpt_path).stem)
        print("Model:", label)
        print("Source:", model.actual_source)

        m = evaluate(model, test_loader, device, amp=not args.no_amp)
        rows.append({
            "model": label, "checkpoint": ckpt_path, "corruption": "clean",
            "level": 0, "parameter": 0.0, "accuracy": m.accuracy,
            "accuracy_percent": 100*m.accuracy, "loss": m.loss,
        })
        print(f"clean: {100*m.accuracy:.2f}%")

        for corruption, specs in CORRUPTION_SPECS.items():
            for level, parameter in specs.items():
                m = evaluate(
                    model, test_loader, device, amp=not args.no_amp,
                    corruption=corruption, level=level,
                    corruption_seed=args.corruption_seed,
                )
                rows.append({
                    "model": label, "checkpoint": ckpt_path,
                    "corruption": corruption, "level": int(level),
                    "parameter": float(parameter), "accuracy": m.accuracy,
                    "accuracy_percent": 100*m.accuracy, "loss": m.loss,
                })
                print(
                    f"{corruption:16s} level={level} "
                    f"param={parameter} acc={100*m.accuracy:.2f}%"
                )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    ensure_parent(args.output)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print("\nSaved:", Path(args.output).resolve())
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
