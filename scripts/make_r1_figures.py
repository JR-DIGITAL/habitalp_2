"""Result figures for the MDPI revision (round 1) into paper/srs-mdpi/.

Replaces, with new file names so the submitted figures stay for latexdiff:
  Fig_benchmark.png      -> Fig_3_r1_model_comparison.png
      Masked split, best checkpoint, 3 seeds: seed mean, seed SD and the individual
      seeds for 2020 segmentation, multi-class CD and binary CD macro IoU (top row);
      paired hierarchical bootstrap (tiles and seeds) difference to MiT-B2 U-Net with 95% CI; filled = CI excludes 0
      (bottom row).
  Fig_confusion_9x9.png  -> Fig_4_r1_confusion_9x9.png
      Row-normalised 9x9 change confusion matrix for Clay v1.0, revision runs (masked
      split, best checkpoint), pooled over the 3 seeds (equal pixel counts per seed,
      so this equals the mean of the per-seed row-normalised matrices).
  Fig_4_r1_confusion_9x9_phaseE.png
      Same layout for the submitted Clay v1.0 Phase E run after the B0 dwarf-forest
      fix (outputs/r1/confusion_matrix_change_9x9.csv), in case the per-class table
      stays on that run.

Inputs: outputs/r1/b9_summary.csv, b9_per_run.csv, b1_paired.csv, b9_tile_cms.npz,
confusion_matrix_change_9x9.csv. Run: .venv314/bin/python scripts/make_r1_figures.py
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
R1 = ROOT / "outputs" / "r1"
OUT = ROOT / "paper" / "srs-mdpi"
DPI = 300
TEXTWIDTH_IN = 13.86 / 2.54  # MDPI \textwidth (21 cm - 5.87 cm - 1.27 cm)
TEST_YEAR = "2019/20"  # test-epoch label (year naming decided 2026-10-27)

INK, INK2, GRID = "#1a1a1a", "#52514e", "#e4e3df"
# Two categorical slots of the dataviz reference palette (validated, light mode,
# all pairs): GFMs blue, supervised baselines orange.
C_GFM, C_SUP = "#2a78d6", "#eb6834"

mpl.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0, "font.family": "DejaVu Sans",
})

# Fixed model order (top to bottom): reference baseline, second baseline, GFMs.
MODELS = ["MiT-B2 U-Net", "ResNet-50 U-Net", "Clay v1.0", "Clay v1.5", "DOFA",
          "Prithvi", "TerraMind"]
SUPERVISED = {"MiT-B2 U-Net", "ResNet-50 U-Net"}
B1_NAME = {"Prithvi": "Prithvi-EO-2.0"}
METRICS = [("seg2020_mIoU", f"Segmentation\n{TEST_YEAR}"),
           ("cd_mc_IoU", "Multi-class change\n2013 to 2019/20"),
           ("cd_bin_IoU", "Binary change\n2013 to 2019/20")]


def color(m):
    return C_SUP if m in SUPERVISED else C_GFM


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", pad_inches=0.03,
                facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


def style_axis(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=4)


# ------------------------------------------------------------------ Fig 3
def fig3():
    summ = pd.read_csv(R1 / "b9_summary.csv")
    summ = summ[(summ.split == "masked") & (summ.ckpt == "best")].set_index(["model", "metric"])
    runs = pd.read_csv(R1 / "b9_per_run.csv")
    runs = runs[(runs.split == "masked") & (runs.ckpt == "best")]
    b1 = pd.read_csv(R1 / "b1_paired.csv")
    b1 = b1[(b1.family == "main") & (b1.model_b == "MiT-B2 U-Net")].set_index(["model_a", "metric"])

    y_top = np.arange(len(MODELS))[::-1]
    diff_models = MODELS[1:]
    y_bot = np.arange(len(diff_models))[::-1]

    fig, axes = plt.subplots(
        2, 3, figsize=(TEXTWIDTH_IN, 4.55), sharey="row",
        gridspec_kw={"height_ratios": [len(MODELS), len(diff_models)], "hspace": 0.5,
                     "wspace": 0.16})

    for col, (metric, title) in enumerate(METRICS):
        ax = axes[0, col]
        style_axis(ax)
        lo_all, hi_all = [], []
        for m, y in zip(MODELS, y_top):
            s = summ.loc[(m, metric)]
            seeds = runs.loc[runs.model == m, metric].to_numpy()
            assert len(seeds) == 3 and abs(seeds.mean() - s["mean"]) < 1e-9, (m, metric)
            c = color(m)
            ax.plot([s["mean"] - s["sd"], s["mean"] + s["sd"]], [y, y], color=c, lw=1.6,
                    solid_capstyle="round", zorder=2)
            ax.scatter(seeds, [y] * 3, marker="|", s=55, color=INK2, lw=1.0, zorder=3)
            ax.scatter([s["mean"]], [y], s=34, color=c, edgecolor="white", lw=0.8, zorder=4)
            lo_all += [seeds.min(), s["mean"] - s["sd"]]
            hi_all += [seeds.max(), s["mean"] + s["sd"]]
        lo, hi = min(lo_all), max(hi_all)
        pad = 0.08 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3, steps=[1, 2, 4, 5]))
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
        ax.set_title(title, fontweight="bold", pad=5)
        if col == 1:
            ax.set_xlabel("Macro IoU (mean ± seed SD, single seeds as ticks)", color=INK2)
        ax.set_ylim(-0.6, len(MODELS) - 0.4)

        ax = axes[1, col]
        style_axis(ax)
        ax.axvline(0, color=INK2, lw=0.8, zorder=1)
        for m, y in zip(diff_models, y_bot):
            r = b1.loc[(B1_NAME.get(m, m), metric)]
            sig = bool(r["excludes_zero"])
            c = color(m)
            ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=c, lw=1.6,
                    solid_capstyle="round", zorder=2)
            ax.scatter([r["diff"]], [y], s=34, zorder=3, lw=1.1,
                       color=c if sig else "white", edgecolor=c)
        ax.set_xlim(-0.100, 0.068)
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([-0.05, 0.0, 0.05]))
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
            lambda v, _: "0" if abs(v) < 1e-9 else f"{v:+.2f}"))
        if col == 1:
            ax.set_xlabel("Difference in macro IoU to MiT-B2 U-Net (paired 95% CI over tiles and seeds)", color=INK2)
        ax.set_ylim(-0.6, len(diff_models) - 0.4)

    axes[0, 0].set_yticks(y_top, MODELS)
    axes[1, 0].set_yticks(y_bot, diff_models)
    for ax in axes[:, 0]:
        for t in ax.get_yticklabels():
            t.set_color(INK)
    fig.text(0.0, 0.985, "a", fontweight="bold", fontsize=10, va="top")
    fig.text(0.0, 0.43, "b", fontweight="bold", fontsize=10, va="top")

    handles = [
        Line2D([], [], marker="o", ls="-", color=C_GFM, mec="white", ms=6, lw=1.6,
               label="GFM: mean ± SD"),
        Line2D([], [], marker="o", ls="-", color=C_SUP, mec="white", ms=6, lw=1.6,
               label="Supervised: mean ± SD"),
        Line2D([], [], marker="|", ls="", color=INK2, ms=7, mew=1.0, label="Single seed"),
        Line2D([], [], marker="o", ls="", color=INK2, ms=6, label="95% CI excludes 0"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=6, mew=1.1,
               label="95% CI includes 0"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.55, -0.11), handletextpad=0.4, columnspacing=1.0,
               handlelength=1.6)
    save(fig, "Fig_3_r1_model_comparison.png")


# ------------------------------------------------------------------ Fig 4
CD_LABELS = ["No change", "Mature tree density loss", "Old-growth density loss",
             "Forest setback / young loss", "Forest stage progression",
             "Forest density gain", "Early forest establishment", "New clearcut",
             "Other transition"]
CSV_ORDER = ["No change", "Mature Tree Density Loss", "Old Growth Density Loss",
             "Forest Setback YoungLoss", "Forest Stage Progression", "Forest Density Gain",
             "Early Forest Establishment", "Clearcut Loss", "Other Transition"]


def plot_cm(cm, name):
    rn = cm / cm.sum(1, keepdims=True)
    n = len(CD_LABELS)
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 4.6))
    im = ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), CD_LABELS, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n), CD_LABELS)
    ax.tick_params(axis="both", length=0, labelsize=8)
    ax.tick_params(axis="y", pad=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Reference class")
    for i in range(n):
        for j in range(n):
            v = rn[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.5 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Share of reference pixels (recall on diagonal)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5, width=0.6)
    cb.outline.set_visible(False)
    save(fig, name)


def fig4():
    d = np.load(R1 / "b9_tile_cms.npz")
    cms = [d[f"r1-clay-v1-masked-s{s}__cd"].sum(0).astype(float) for s in range(3)]
    assert len({c.sum() for c in cms}) == 1  # equal pixel counts per seed
    cm = sum(cms)
    # sanity: macro IoU of the pooled matrix is close to the seed mean in b9_summary
    iou = np.diag(cm) / (cm.sum(0) + cm.sum(1) - np.diag(cm))
    ref = pd.read_csv(R1 / "b9_summary.csv").query(
        "model == 'Clay v1.0' and split == 'masked' and ckpt == 'best' and metric == 'cd_mc_IoU'")
    print(f"Clay v1.0 pooled macro IoU {iou.mean():.4f}, seed mean {ref['mean'].item():.4f}")
    plot_cm(cm, "Fig_4_r1_confusion_9x9.png")

    pe = pd.read_csv(R1 / "confusion_matrix_change_9x9.csv", index_col=0)
    assert list(pe.index) == CSV_ORDER and list(pe.columns) == CSV_ORDER
    plot_cm(pe.to_numpy(float), "Fig_4_r1_confusion_9x9_phaseE.png")


if __name__ == "__main__":
    fig3()
    fig4()
