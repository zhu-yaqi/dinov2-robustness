from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Run the coursework experiment pipeline.")
    p.add_argument("--mode", choices=["quick","main","full"], default="main")
    p.add_argument("--dino-source", choices=["hub","timm","auto"], default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--force", action="store_true")
    return p.parse_args()

def run(cmd):
    print("\n" + "="*80)
    print("RUN:", " ".join(cmd))
    print("="*80)
    subprocess.run(cmd, check=True)

def train_if_missing(path, cmd, force):
    if Path(path).exists() and not force:
        print("SKIP existing checkpoint:", path)
        return
    run(cmd)

def main():
    args = parse_args()
    py = sys.executable
    for d in ["results/checkpoints","results/tables","results/figures"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    if args.mode == "quick":
        batch = str(min(args.batch_size, 32))
        dino = "results/checkpoints/quick_dinov2.pt"
        ours = "results/checkpoints/quick_ours.pt"

        train_if_missing(dino, [
            py, "train_baseline.py",
            "--model", "dinov2_vits14",
            "--dino-source", args.dino_source,
            "--epochs", "1",
            "--batch-size", batch,
            "--num-workers", str(args.num_workers),
            "--train-subset", "1200",
            "--val-subset", "400",
            "--output", dino,
            "--label", "DINOv2 quick",
        ], args.force)

        train_if_missing(ours, [
            py, "train_consistency.py",
            "--dino-source", args.dino_source,
            "--epochs", "1",
            "--batch-size", batch,
            "--num-workers", str(args.num_workers),
            "--train-subset", "1200",
            "--val-subset", "400",
            "--lambda-consistency", "0.5",
            "--output", ours,
            "--label", "Ours quick",
        ], args.force)

        csv = "results/tables/robustness_quick.csv"
        run([
            py, "evaluate_robustness.py",
            "--checkpoints", dino, ours,
            "--batch-size", batch,
            "--num-workers", str(args.num_workers),
            "--test-subset", "500",
            "--dino-source-override", args.dino_source,
            "--output", csv,
        ])
        run([
            py, "plot_results.py",
            "--input", csv,
            "--output-dir", "results/figures/quick",
        ])
        return

    common = [
        "--epochs", "20",
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
    ]

    jobs = [
        (
            "results/checkpoints/resnet50.pt",
            [py,"train_baseline.py","--model","resnet50",*common,
             "--output","results/checkpoints/resnet50.pt","--label","ResNet50"]
        ),
        (
            "results/checkpoints/convnextv2_tiny.pt",
            [py,"train_baseline.py","--model","convnextv2_tiny",*common,
             "--output","results/checkpoints/convnextv2_tiny.pt","--label","ConvNeXt V2 Tiny"]
        ),
        (
            "results/checkpoints/dinov2.pt",
            [py,"train_baseline.py","--model","dinov2_vits14",
             "--dino-source",args.dino_source,*common,
             "--output","results/checkpoints/dinov2.pt","--label","DINOv2 ViT-S/14"]
        ),
        (
            "results/checkpoints/dinov2_consistency_l05.pt",
            [py,"train_consistency.py","--dino-source",args.dino_source,*common,
             "--lambda-consistency","0.5",
             "--output","results/checkpoints/dinov2_consistency_l05.pt",
             "--label","DINOv2 + Ours"]
        ),
    ]

    for path, cmd in jobs:
        train_if_missing(path, cmd, args.force)

    ckpts = [p for p,_ in jobs]

    if args.mode == "full":
        ablations = [
            ("results/checkpoints/dinov2_adapter_l00.pt","0.0","DINOv2 + Adapter lambda=0"),
            ("results/checkpoints/dinov2_consistency_l01.pt","0.1","DINOv2 + Adapter lambda=0.1"),
            ("results/checkpoints/dinov2_consistency_l10.pt","1.0","DINOv2 + Adapter lambda=1.0"),
        ]
        for path, lam, label in ablations:
            train_if_missing(path, [
                py,"train_consistency.py","--dino-source",args.dino_source,*common,
                "--lambda-consistency",lam,"--output",path,"--label",label
            ], args.force)
            ckpts.append(path)
        csv = "results/tables/robustness_full.csv"
    else:
        csv = "results/tables/robustness_main.csv"

    run([
        py,"evaluate_robustness.py",
        "--checkpoints",*ckpts,
        "--batch-size",str(args.batch_size),
        "--num-workers",str(args.num_workers),
        "--dino-source-override",args.dino_source,
        "--output",csv,
    ])
    run([
        py,"plot_results.py",
        "--input",csv,
        "--output-dir","results/figures",
    ])

if __name__ == "__main__":
    main()
