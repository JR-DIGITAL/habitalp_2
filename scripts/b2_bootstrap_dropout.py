#!/usr/bin/env python
"""B2 (R2-M7): class drop-out in the block bootstrap, and a fixed method.

The block bootstrap of scripts/mdpi/bootstrap_ci.py / B0 resamples the 88 spatial tiles (2500 px
blocks at 0.2 m) with replacement and computes macro IoU with macro_iou_f1, which takes a
nanmean over classes. A class that is absent from the reference of a resample then gets:

  - IoU = 0 if the model predicts it somewhere in the resample (tp = 0, union = fp > 0);
  - IoU = NaN, i.e. it is silently dropped from the macro mean, if it is not predicted.

So the class set behind each bootstrap value changes from resample to resample and from
model to model. This script measures how often that happens per class (on the B0 maps in
<model_output>/<experiment>/r1_dwarffix/) and compares the B0 CIs with a fixed method:

  fixed class set K = classes present in the reference of the full test set (all 9);
  a resample in which any class of K has no reference pixel is redrawn (the bootstrap is
  conditional on every class of K being supported), and the redraw rate is reported
  next to the CI. Point estimates are unchanged; the CIs stay comparable to B0 (same
  tiles, same seed, same 1000 accepted resamples).

Also reported, for the note: the CI with an absent class scored as IoU 0 (the most
pessimistic reading). Tile confusion matrices are cached in outputs/r1/b2_tile_cms.npz
(also usable by B1).

Run: nice -n 19 python scripts/b2_bootstrap_dropout.py [--workers 3]
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

BASE_FOLDER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_FOLDER))
from scripts.b0_dwarffix_cd import (
    BLOCK, EXPERIMENTS, MC_NAMES, MO, MODELS, NEW_DIR, OUT, REF_CT, metrics_from_csv,
)
from src.trainers.bootstrap import (
    collapse_binary, fixed_class_resamples, load_aligned_2d, macro_iou_f1,
    tile_confusion_matrices,
)

N_BOOT = 1000
CACHE = OUT / "b2_tile_cms.npz"


def job_key(exp: str, postproc: str) -> str:
    return f"{exp}__{postproc}"


def map_path(exp: str, postproc: str) -> Path:
    sfx = "_after_postprocessing" if postproc == "after" else ""
    return MO / exp / NEW_DIR / f"change_map_2013-2020{sfx}.tif"


def tile_job(job: tuple[str, str]) -> tuple[str, np.ndarray]:
    exp, postproc = job
    ref2d, prd2d = load_aligned_2d(REF_CT, map_path(exp, postproc))
    cms = tile_confusion_matrices(ref2d, prd2d, 9, block=BLOCK, ignore_index=255)
    print(f"tiles {exp} {postproc}: {cms.shape[0]}", flush=True)
    return job_key(exp, postproc), cms


def per_class_iou(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    union = cm.sum(axis=0) + cm.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        return tp / union


def resample_indices(n: int, seed: int = 0) -> np.ndarray:
    """The B0 draws: block_bootstrap_ci uses default_rng(seed).integers(0, n, n) per draw."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.integers(0, n, size=n) for _ in range(N_BOOT)])


def ci(vals: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    jobs = [(e, p) for e in EXPERIMENTS for p in ["before", "after"]]
    if CACHE.exists():
        cache = dict(np.load(CACHE))
    else:
        with Pool(args.workers) as pool:
            cache = dict(pool.imap(tile_job, jobs))
        np.savez_compressed(CACHE, **cache)
    n_tiles = {k: v.shape[0] for k, v in cache.items()}
    assert len(set(n_tiles.values())) == 1, n_tiles
    n = next(iter(n_tiles.values()))

    # reference support per tile, for the per-class drop-out rate
    ref_rows = {k: v.sum(axis=2) for k, v in cache.items()}  # (tiles, classes) ref pixels
    ref0 = next(iter(ref_rows.values()))
    same_ref = all(np.array_equal(r > 0, ref0 > 0) for r in ref_rows.values())
    # a class counts for a tile only if it has reference pixels there in every map (the valid
    # area differs by a few edge pixels where a prediction has nodata)
    support = np.logical_and.reduce([r > 0 for r in ref_rows.values()])
    draws = resample_indices(n)
    absent = ~np.stack([support[idx].any(axis=0) for idx in draws])  # (N_BOOT, classes)
    px_ha = 0.199897019943870 ** 2 / 1e4

    cls_rows = []
    for c, name in enumerate(MC_NAMES):
        cls_rows.append({
            "class": c, "name": name,
            "ref_area_ha": float(ref0[:, c].sum() * px_ha),
            "tiles_with_class": int(support[:, c].sum()), "n_tiles": n,
            "dropout_rate_b0_draws": float(absent[:, c].mean()),
            "dropout_rate_expected": float((1 - support[:, c].mean()) ** n),
        })
    cls_df = pd.DataFrame(cls_rows)
    any_absent = float(absent.any(axis=1).mean())

    # the method itself lives in src/trainers/bootstrap.py (block_bootstrap_ci_fixed)
    fixed_idx, _, redraw_rate = fixed_class_resamples(next(iter(cache.values())), N_BOOT)
    n_draws = round(N_BOOT / (1 - redraw_rate))

    rows, treat_rows = [], []
    for exp, postproc in jobs:
        cms = cache[job_key(exp, postproc)]
        sfx = "_after_postprocessing" if postproc == "after" else ""
        for task in ["binary", "multiclass"]:
            is_bin = task == "binary"
            k = 2 if is_bin else 9
            csv = MO / exp / NEW_DIR / f"change_metrics_{task}_2013-2020{sfx}.csv"
            point = metrics_from_csv(csv)["IoU"]

            def boot(index_sets, absent_as_zero=False):
                ious, n_zero, n_nan, any_nan = [], np.zeros(k), np.zeros(k), 0
                for idx in index_sets:
                    cm = cms[idx].sum(axis=0)
                    cm = collapse_binary(cm) if is_bin else cm
                    iou = per_class_iou(cm)
                    no_ref = cm.sum(axis=1) == 0
                    n_zero += no_ref & ~np.isnan(iou)
                    n_nan += np.isnan(iou)
                    any_nan += bool(np.isnan(iou).any())
                    if absent_as_zero:
                        iou = np.where(np.isnan(iou), 0.0, iou)
                    ious.append(np.nanmean(iou))
                m = len(index_sets)
                return np.array(ious), n_zero / m, n_nan / m, any_nan / m

            b0, zero_rate, nan_rate, b0_any_nan = boot(draws)
            b0z = boot(draws, absent_as_zero=True)[0]
            fx = boot(fixed_idx)[0]
            full = cms.sum(axis=0)
            iou_cm = macro_iou_f1(collapse_binary(full) if is_bin else full)[0]
            assert abs(iou_cm - point) <= 0.01, (exp, postproc, task, iou_cm, point)
            row = {"model": MODELS[exp.split("-rgb")[0]], "experiment": exp, "phase": exp[-1],
                   "postproc": postproc, "task": f"change-{task}", "IoU": point,
                   "IoU_cm": iou_cm}
            row["IoU_ci_low_b0"], row["IoU_ci_high_b0"] = ci(b0)
            row["IoU_ci_low_fixed"], row["IoU_ci_high_fixed"] = ci(fx)
            row["IoU_ci_low_absent0"], row["IoU_ci_high_absent0"] = ci(b0z)
            row["resamples_with_class_dropped_b0"] = b0_any_nan
            row["redraw_rate_fixed"] = redraw_rate if not is_bin else 0.0
            rows.append(row)
            if not is_bin:
                for c in range(k):
                    treat_rows.append({"experiment": exp, "postproc": postproc, "class": c,
                                       "absent_scored_0": zero_rate[c],
                                       "absent_dropped_nan": nan_rate[c]})

    df = pd.DataFrame(rows)
    df["ci_width_b0"] = df.IoU_ci_high_b0 - df.IoU_ci_low_b0
    df["ci_width_fixed"] = df.IoU_ci_high_fixed - df.IoU_ci_low_fixed
    df.to_csv(OUT / "b2_bootstrap_ci_methods.csv", index=False)

    tr = pd.DataFrame(treat_rows)
    agg = tr.groupby("class")[["absent_scored_0", "absent_dropped_nan"]].agg(["min", "max"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    cls_df = cls_df.merge(agg, left_on="class", right_index=True)
    cls_df.to_csv(OUT / "b2_class_dropout.csv", index=False)
    tr.to_csv(OUT / "b2_class_dropout_per_experiment.csv", index=False)

    with open(OUT / "b2_summary.txt", "w") as f:
        f.write(f"tiles per map: {n} (all 24 maps); per-tile class support identical across maps: {same_ref} (edge tiles only; the minimum is used)\n")
        f.write(f"B0 draws (seed 0, {N_BOOT}): share with >= 1 class absent from the reference: {any_absent:.3f}\n")
        f.write(f"fixed class set: {n_draws} draws for {N_BOOT} accepted, redraw rate {redraw_rate:.3f}\n")
        mc = df[df.task == "change-multiclass"]
        d_lo = (mc.IoU_ci_low_fixed - mc.IoU_ci_low_b0).abs().max()
        d_hi = (mc.IoU_ci_high_fixed - mc.IoU_ci_high_b0).abs().max()
        f.write(f"multi-class CI bounds, fixed vs B0: max |diff| low {d_lo:.4f}, high {d_hi:.4f}\n")
        f.write(cls_df.round(4).to_string(index=False) + "\n")
    print(open(OUT / "b2_summary.txt").read())


if __name__ == "__main__":
    main()
