from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TABLE_DIR = Path("results/tables")
FIGURE_DIR = Path("results/figures")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# Figure 1: Clean vs Corrupted Accuracy
# ==========================================

df = pd.read_csv(TABLE_DIR / "main_results.csv")

models = df["model"]
clean = df["clean"]
corrupted = df["corruption_mean"]

x = range(len(models))
width = 0.36

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    clean,
    width=width,
    label="Clean"
)

plt.bar(
    [i + width / 2 for i in x],
    corrupted,
    width=width,
    label="Mean Corrupted"
)

plt.xticks(list(x), models, rotation=10)
plt.ylabel("Accuracy (%)")
plt.title("Clean vs. Corrupted Accuracy")
plt.ylim(60, 100)
plt.legend()
plt.grid(axis="y", alpha=0.25)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "clean_vs_corrupted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# Figure 2: Lambda Ablation
# ==========================================

ablation = pd.read_csv(TABLE_DIR / "ablation_lambda.csv")

configs = ablation["configuration"]

plt.figure(figsize=(10, 6))

plt.plot(
    configs,
    ablation["clean"],
    marker="o",
    linewidth=2,
    label="Clean"
)

plt.plot(
    configs,
    ablation["occlusion_mean"],
    marker="o",
    linewidth=2,
    label="Occlusion Mean"
)

plt.plot(
    configs,
    ablation["corruption_mean"],
    marker="o",
    linewidth=2,
    label="Corruption Mean"
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Configuration")
plt.title("Consistency Weight Ablation")
plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "lambda_ablation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Figures generated successfully!")
print("Saved to:", FIGURE_DIR)
