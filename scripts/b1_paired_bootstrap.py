#!/usr/bin/env python
"""B1 (R1-M1): paired spatial block bootstrap of macro IoU differences.

Input: outputs/r1/b9_tile_cms.npz (written by b9_seed_aggregate.py), per-tile confusion
matrices of every revision run on the 88 non-empty 500 m tiles:
  <dir>__seg2020 (88, 24, 24)  2020 segmentation, macro over classes 1..23
  <dir>__cd      (88, 9, 9)    raw change detection; binary = collapse_binary (no change
                               vs any change), as in B0/B2/B9
Only best-checkpoint rasters are used (<dir> without "-last").

Method (decided 2026-10-27):
  * 10,000 resamples of the 88 tiles with replacement (seed 0). One tile index set per
    resample is applied to every raster and every metric, so all models are scored on
    identical tiles (paired).
  * Fixed class set with redraw, as block_bootstrap_ci_fixed (B2): K = classes with
    reference pixels in the full test set; a resample in which a class of K has no
    reference pixel (in any raster, for any metric) is redrawn. The redraw rate is reported.
  * Seeds are not matched across models: per resample, macro IoU per seed raster from the
    summed resampled CMs, mean over the model's seeds, difference of the seed means (a - b).
  * Hierarchical (revised 2026-10-27 after review): in each resample also the
    seeds of every model are resampled with replacement (independently per model, the same
    draw for all metrics), so the CI covers test-tile sampling and training randomness, as
    GEO-Bench resamples seeds. With 3 seeds this is approximate.
  * Observed diff = difference of seed means on the full test set; 95% percentile CI,
    unadjusted. No p-values and no multiplicity correction (field practice: GEO-Bench and
    PANGAEA report intervals only). The tiles-only CI (seeds fixed) is kept in the csv as
    ci_low_tiles/ci_high_tiles for a supplementary table.

Outputs: outputs/r1/b1_paired.csv (format of make_r1_tables.py b1), outputs/r1/b1_note.txt.

Run:
  OMP_NUM_THREADS=4 .venv314/bin/python scripts/b1_paired_bootstrap.py [--n-boot 10000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from src.trainers.bootstrap import collapse_binary  # noqa: E402

OUT = BASE / "outputs" / "r1"
METRICS = ["seg2020_mIoU", "cd_mc_IoU", "cd_bin_IoU"]
MET_LABEL = {"seg2020_mIoU": "2020 segmentation", "cd_mc_IoU": "CD multi-class",
             "cd_bin_IoU": "CD binary"}

# (name in csv, model_key in b9_summary, split, seeds)
MODELS = {
    "Clay v1.0": ("clay-v1", "masked", [0, 1, 2]),
    "Clay v1.5": ("clay-v1.5", "masked", [0, 1, 2]),
    "Prithvi-EO-2.0": ("prithvi-v2-300", "masked", [0, 1, 2]),
    "DOFA": ("dofa-base", "masked", [0, 1, 2]),
    "TerraMind": ("terramind", "masked", [0, 1, 2]),
    "ResNet-50 U-Net": ("unet-resnet50", "masked", [0, 1, 2]),
    "MiT-B2 U-Net": ("unet-mit-b2", "masked", [0, 1, 2]),
    "Clay v1.0, U-Net decoder": ("clay-v1-unetdec", "masked", [0]),
    "Prithvi-EO-2.0, U-Net decoder": ("prithvi-v2-300-unetdec", "masked", [0]),
    "MiT-B2 U-Net, original split": ("unet-mit-b2", "orig", [0, 1, 2]),
    "MiT-B2 U-Net, block-drop split": ("unet-mit-b2", "blockdrop", [0, 1, 2]),
}
BASELINE = "MiT-B2 U-Net"
COMPARISONS = [(m, BASELINE, "main") for m in
               ["Clay v1.0", "Clay v1.5", "Prithvi-EO-2.0", "DOFA", "TerraMind", "ResNet-50 U-Net"]]
COMPARISONS += [("Clay v1.0", "Prithvi-EO-2.0", "post hoc")]
# post hoc (2026-10-28, R2-M3): every GFM against the convolutional baseline; MiT-B2 vs ResNet-50
# is already a main row (sign flipped). Adding rows does not change the draws of the other rows.
COMPARISONS += [(m, "ResNet-50 U-Net", "cnn") for m in
                ["Clay v1.0", "Clay v1.5", "Prithvi-EO-2.0", "DOFA", "TerraMind"]]
COMPARISONS += [("Clay v1.0, U-Net decoder", "Prithvi-EO-2.0, U-Net decoder", "supporting"),
                ("MiT-B2 U-Net", "MiT-B2 U-Net, original split", "supporting"),
                ("MiT-B2 U-Net", "MiT-B2 U-Net, block-drop split", "supporting"),
                ("MiT-B2 U-Net, original split", "MiT-B2 U-Net, block-drop split", "supporting")]


def run_dir(key: str, split: str, seed: int) -> str:
    return f"r1-{key}-{split}-s{seed}"


def macro_iou_batch(cms: np.ndarray, k_mask: np.ndarray) -> np.ndarray:
    """Macro IoU over the fixed class set for a stack of CMs (..., C, C)."""
    tp = np.diagonal(cms, axis1=-2, axis2=-1).astype(np.float64)
    fp = cms.sum(axis=-2) - tp
    fn = cms.sum(axis=-1) - tp
    iou = tp / (tp + fp + fn)
    return iou[..., k_mask].mean(axis=-1)


def collapse_binary_batch(cms: np.ndarray) -> np.ndarray:
    out = np.empty(cms.shape[:-2] + (2, 2), dtype=cms.dtype)
    out[..., 0, 0] = cms[..., 0, 0]
    out[..., 0, 1] = cms[..., 0, 1:].sum(-1)
    out[..., 1, 0] = cms[..., 1:, 0].sum(-1)
    out[..., 1, 1] = cms[..., 1:, 1:].sum((-1, -2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    B = args.n_boot

    z = np.load(OUT / "b9_tile_cms.npz")
    rasters = sorted({run_dir(k, s, sd) for k, s, sds in MODELS.values() for sd in sds})
    # tile CMs per metric; seg: drop row/col 0 (nodata, always 0) -> classes 1..23
    cms = {"seg2020_mIoU": {r: z[f"{r}__seg2020"][:, 1:, 1:] for r in rasters},
           "cd_mc_IoU": {r: z[f"{r}__cd"] for r in rasters}}
    cms["cd_bin_IoU"] = {r: collapse_binary_batch(c) for r, c in cms["cd_mc_IoU"].items()}
    assert np.array_equal(cms["cd_bin_IoU"][rasters[0]][5], collapse_binary(cms["cd_mc_IoU"][rasters[0]][5]))
    n = z["tile_rc"].shape[0]

    # fixed class sets and per-tile support (minimum over rasters)
    k_mask, support = {}, []
    for m in METRICS:
        ref = np.stack([cms[m][r].sum(axis=2) for r in rasters])  # (rasters, tiles, classes)
        ks = ref.sum(axis=1) > 0
        assert (ks == ks[0]).all(), f"class set differs across rasters for {m}"
        k_mask[m] = ks[0]
        support.append((ref.min(axis=0) > 0)[:, k_mask[m]])
    support = np.concatenate(support, axis=1)  # (tiles, all classes of all metrics)
    support_same = all(np.array_equal(cms[m][r].sum(axis=2) > 0, cms[m][rasters[0]].sum(axis=2) > 0)
                       for m in METRICS for r in rasters)

    # resamples with redraw (same generator and draw as fixed_class_resamples)
    rng = np.random.default_rng(args.seed)
    keep, draws = [], 0
    while len(keep) < B:
        idx = rng.integers(0, n, size=n)
        draws += 1
        if support[idx].any(axis=0).all():
            keep.append(idx)
    idx_sets = np.stack(keep)
    redraw = 1.0 - B / draws
    W = np.zeros((B, n), dtype=np.float64)  # tile multiplicities
    np.add.at(W, (np.repeat(np.arange(B), n), idx_sets.ravel()), 1.0)

    # per-raster observed and bootstrap macro IoU
    obs, boot = {m: {} for m in METRICS}, {m: {} for m in METRICS}
    for m in METRICS:
        for r in rasters:
            t = cms[m][r].astype(np.float64)
            c = t.shape[-1]
            obs[m][r] = float(macro_iou_batch(t.sum(axis=0), k_mask[m]))
            boot[m][r] = macro_iou_batch((W @ t.reshape(n, -1)).reshape(B, c, c), k_mask[m])

    # seed resampling per model (hierarchical bootstrap), same draw for every metric
    rng_s = np.random.default_rng(args.seed + 1)
    seed_idx = {name: rng_s.integers(0, len(sds), size=(B, len(sds))) for name, (_, _, sds) in MODELS.items()}

    def seed_mean(name: str, m: str, which: str):
        key, split, sds = MODELS[name]
        src = obs if which == "obs" else boot
        vals = [src[m][run_dir(key, split, s)] for s in sds]
        if which == "hier":
            v = np.stack([boot[m][run_dir(key, split, s)] for s in sds])  # (seeds, B)
            return v[seed_idx[name].T, np.arange(B)].mean(axis=0), vals
        return np.mean(vals, axis=0), vals

    # sanity vs b9_summary.csv
    summ = pd.read_csv(OUT / "b9_summary.csv")
    summ = summ[summ.ckpt == "best"]
    max_dev = 0.0
    for name, (key, split, sds) in MODELS.items():
        for m in METRICS:
            row = summ[(summ.model_key == key) & (summ.split == split) & (summ.metric == m)]
            assert len(row) == 1, (key, split, m)
            dev = abs(seed_mean(name, m, "obs")[0] - row["mean"].iloc[0])
            max_dev = max(max_dev, dev)
    assert max_dev < 1e-9, f"observed seed means differ from b9_summary.csv: {max_dev}"

    rows = []
    for a, b, fam in COMPARISONS:
        for m in METRICS:
            d_obs = seed_mean(a, m, "obs")[0] - seed_mean(b, m, "obs")[0]
            d_t = seed_mean(a, m, "boot")[0] - seed_mean(b, m, "boot")[0]
            d = seed_mean(a, m, "hier")[0] - seed_mean(b, m, "hier")[0]
            lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
            rows.append(dict(model_a=a, model_b=b, metric=m, diff=d_obs, ci_low=lo, ci_high=hi,
                             excludes_zero=bool(lo > 0 or hi < 0), family=fam,
                             n_seeds_a=len(MODELS[a][2]), n_seeds_b=len(MODELS[b][2]),
                             ci_low_tiles=np.percentile(d_t, 2.5), ci_high_tiles=np.percentile(d_t, 97.5),
                             boot_sd=d.std(ddof=1), boot_sd_tiles=d_t.std(ddof=1)))
    df = pd.DataFrame(rows)
    cols = ["model_a", "model_b", "metric", "diff", "ci_low", "ci_high", "excludes_zero", "family",
            "n_seeds_a", "n_seeds_b", "ci_low_tiles", "ci_high_tiles", "boot_sd", "boot_sd_tiles"]
    df[cols].to_csv(OUT / "b1_paired.csv", index=False, float_format="%.6g")

    # ---------------------------------------------------------------- note
    L = ["B1: paired spatial block bootstrap of macro IoU differences (R1-M1)",
         "Script: scripts/b1_paired_bootstrap.py; input outputs/r1/b9_tile_cms.npz (best checkpoints); "
         "output b1_paired.csv",
         f"Resamples: {B} of the {n} tiles (500 m blocks), seed {args.seed}; one tile index set per resample "
         "for all rasters and metrics.",
         f"Fixed class set with redraw (block_bootstrap_ci_fixed rule): classes seg {k_mask['seg2020_mIoU'].sum()}, "
         f"CD multi {k_mask['cd_mc_IoU'].sum()}, CD binary {k_mask['cd_bin_IoU'].sum()}; "
         f"{draws} draws for {B} accepted, redraw rate {redraw:.4f}. "
         f"Per-tile reference support identical across rasters: {support_same} (minimum used).",
         "Per resample: macro IoU per seed raster, mean over the model's seeds, difference of the means "
         "(a - b); seeds resampled with replacement per model (hierarchical). CI: 95% percentile, "
         "unadjusted; no p-values, no multiplicity correction.",
         f"Sanity: observed seed means equal b9_summary.csv (max |diff| {max_dev:.1e}).", ""]
    L.append("Per-seed macro IoU (seeds 0;1;2), best checkpoint:")
    for name in MODELS:
        parts = []
        for m in METRICS:
            mean, vals = seed_mean(name, m, "obs")
            parts.append(f"{MET_LABEL[m]} {mean:.4f} ({';'.join(f'{v:.4f}' for v in vals)})")
        L.append(f"  {name}: " + "; ".join(parts))
    L.append("")
    for fam_title, fams in [("Planned comparisons", ["main", "post hoc"]),
                            ("Against the ResNet-50 U-Net (post hoc)", ["cnn"]),
                            ("Supporting", ["supporting"])]:
        L.append(fam_title + ":")
        L.append("  comparison | metric | diff | 95% CI tiles+seeds | excl. 0 | 95% CI tiles only")
        for _, r in df[df.family.isin(fams)].iterrows():
            tag = " (post hoc)" if r.family in ("post hoc", "cnn") else ""
            L.append(f"  {r.model_a} - {r.model_b}{tag} | {MET_LABEL[r.metric]} | {r['diff']:+.4f} | "
                     f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}] | {'yes' if r.excludes_zero else 'no'} | "
                     f"[{r.ci_low_tiles:+.4f}, {r.ci_high_tiles:+.4f}]")
        L.append("")
    L += ["Note: one-seed comparisons (U-Net decoder pair) have no seed resampling; their CI covers",
          "tiles only. With three seeds the training-randomness part of the CI is approximate",
          "(GEO-Bench recommends at least 10 seeds)."]
    (OUT / "b1_note.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
