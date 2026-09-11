from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/tables/robustness_main.csv")
    p.add_argument("--output-dir", default="results/figures")
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)

    clean = (
        df[df["corruption"]=="clean"][["model","accuracy_percent"]]
        .drop_duplicates("model")
        .set_index("model")["accuracy_percent"].to_dict()
    )

    corruptions = [c for c in df["corruption"].drop_duplicates() if c != "clean"]
    for corruption in corruptions:
        fig, ax = plt.subplots(figsize=(7.2,4.8))
        sub = df[df["corruption"]==corruption]
        for model_name, group in sub.groupby("model", sort=False):
            group = group.sort_values("level")
            xs = [0] + group["level"].astype(int).tolist()
            ys = [clean[model_name]] + group["accuracy_percent"].tolist()
            ax.plot(xs, ys, marker="o", label=model_name)
        ax.set_title(f"Robustness under {corruption.replace('_',' ').title()}")
        ax.set_xlabel("Severity Level")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks([0,1,2,3,4,5])
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = out_dir / f"accuracy_{corruption}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print("Saved:", path)

    corrupted = df[df["corruption"]!="clean"]
    mean_corr = (
        corrupted.groupby("model", as_index=False)["accuracy_percent"].mean()
        .rename(columns={"accuracy_percent":"mean_corruption_accuracy_percent"})
    )
    clean_df = (
        df[df["corruption"]=="clean"][["model","accuracy_percent"]]
        .drop_duplicates("model")
        .rename(columns={"accuracy_percent":"clean_accuracy_percent"})
    )
    summary = clean_df.merge(mean_corr, on="model", how="left")
    summary["robustness_drop_points"] = (
        summary["clean_accuracy_percent"] -
        summary["mean_corruption_accuracy_percent"]
    )
    summary_path = out_dir / "summary_by_model.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("Saved:", summary_path)

    fig, ax = plt.subplots(figsize=(8,4.8))
    ax.bar(summary["model"], summary["mean_corruption_accuracy_percent"])
    ax.set_title("Mean Accuracy across Corruptions")
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Accuracy (%)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = out_dir / "mean_corruption_accuracy.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("Saved:", path)

if __name__ == "__main__":
    main()
