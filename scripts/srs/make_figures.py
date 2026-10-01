"""Chart figures of the SRS manuscript from the CSVs in outputs/srs/.

  Fig_benchmark.png      macro IoU per model, binary and multi-class change
  Fig_confusion_9x9.png  row-normalised change-class confusion matrix, Clay v1.0

Needs no raw data. Usage: python scripts/srs/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUTS

FIG_DIR = OUTPUTS / "figures"
ORDER = ["Clay v1.0", "DOFA", "Clay v1.5", "TerraMind", "Prithvi", "U-Net"]
SHORT_LABELS = {
    "No change": "No change",
    "Mature Tree Density Loss": "Mature Density Loss",
    "Old Growth Density Loss": "Old-growth Loss",
    "Forest Setback YoungLoss": "Setback/Young Loss",
    "Forest Stage Progression": "Stage Progression",
    "Forest Density Gain": "Density Gain",
    "Early Forest Establishment": "Early Establishment",
    "Clearcut Loss": "Clearcut Loss",
    "Other Transition": "Other",
}

mpl.rcParams.update({"font.size": 9, "axes.grid": False})


def benchmark() -> None:
    ci = pd.read_csv(OUTPUTS / "metrics_ci.csv").set_index(["model", "task"])
    colors = ["#d6604d" if m == "U-Net" else "#2166ac" for m in ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, task, title in [(axes[0], "change-binary", "Binary change"),
                            (axes[1], "change-multiclass", "Multi-class change")]:
        x = np.arange(len(ORDER))
        ax.bar(x, [ci.loc[(m, task), "macro_IoU"] for m in ORDER], color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(ORDER, rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 0.5)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Macro IoU")
    fig.suptitle("Cross-temporal change detection (2013→2020), RGB+NIR+nDSM", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig_benchmark.png", dpi=600, bbox_inches="tight")
    print("wrote Fig_benchmark.png")


def confusion() -> None:
    # The 9x9 matrix incl. "No change" keeps the diagonal equal to per-class recall.
    cm = pd.read_csv(OUTPUTS / "confusion_matrix_change_9x9_rownorm.csv", index_col=0)
    labels = [SHORT_LABELS[c] for c in cm.index]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(cm.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference")
    for i in range(n):
        for j in range(n):
            v = cm.values[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "#333333", fontsize=6.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized (recall)")
    ax.set_title("Clay v1.0 change-class confusion (2013→2020)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig_confusion_9x9.png", dpi=600, bbox_inches="tight")
    print("wrote Fig_confusion_9x9.png")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    benchmark()
    confusion()
