#!/usr/bin/env python
"""B4 (R1-M2): LiDAR (nDSM) ablation for single-date segmentation, 2020 and 2013.

Phase E (RGB+NIR+nDSM) against Phase F (RGB+NIR) of the six paper models, scored on the
existing 2013 and 2020 prediction rasters (no training, no inference). The nDSM gain
(E minus F) is split by class group, because the CC<80 / CC>=80 split of pole, mature and
old forest comes from the LiDAR canopy-cover product that the nDSM is derived from too:

  cc_split      classes 8, 9, 11-18 (the ten classes that carry a canopy-cover threshold)
  other         all other classes 1-7, 10, 19-23
  other_forest  part of "other": 6, 7 (young growth), 10 (broadleaf pole, no CC split),
                19 (clearcut), 20 (mountain dwarf forest)
  non_forest    part of "other": 1-5, 21-23
  cc_merged     the cc_split classes with each CC pair merged (8+9, 11+12, 13+14, 15+16,
                17+18; 5 stage classes), as in B3: what is left of the gain once the
                LiDAR threshold no longer has to be predicted
  all           all 23 classes (the paper's mIoU and balanced accuracy)

Evaluation as scripts/mdpi/evaluate.py (compute_final_metrics): reference
2013 = classes_v4_2013_test_cells.tif, 2020 = classes_v4_2020.tif (0.2 m), the 0.5 m
prediction is warped onto the reference intersection window with nearest resampling,
ignore_index 0 drops reference and prediction nodata, macro over classes 1-23. To fit the
2013 test area (7.7 G reference pixels) in memory, the same window is processed in strips
of BLOCK rows, and per-tile confusion matrices are built with tile_confusion_matrices
(BLOCK = 2500 px = 500 m tiles, as in scripts/mdpi/bootstrap_ci.py / B0-B3). The macro and
per-class values from the summed matrices are checked against the stored
classification_metrics_<year>.csv of each experiment.

Uncertainty: paired spatial block bootstrap of the E-F difference (same tile resamples
for E and F, fixed class set with redraw, fixed_class_resamples / macro_iou_f1_fixed of
src/trainers/bootstrap.py, 1000 resamples). It covers test-set sampling only, not training
randomness; the seed SD of the revision runs (B9) is the yardstick for single-run deltas.

Outputs (never overwriting): tile matrices in <model_output>/<experiment>/r1_b4/,
outputs/r1/b4_ndsm_ablation.csv (model, year, group), outputs/r1/b4_per_class_iou.csv,
outputs/r1/b4_check_stored_csv.csv.

Run: nice -n 19 python scripts/b4_ndsm_ablation_2020.py [--workers 4]
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.coords import BoundingBox
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window

BASE_FOLDER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_FOLDER))
from scripts.b0_dwarffix_cd import BLOCK, DATA_ROOT, EXPERIMENTS, MO, MODELS, OUT
from src.trainers.bootstrap import fixed_class_resamples, tile_confusion_matrices

NEW_DIR = "r1_b4"
REFS = {
    2013: DATA_ROOT / "processed/mask/classes_v4_2013_test_cells.tif",
    2020: DATA_ROOT / "processed/mask/classes_v4_2020.tif",
}
NC = 24  # 0 = nodata (ignored), classes 1-23
CLASS_NAMES = [
    "nodata", "Waterbody", "Gravel bank, shoal, fluviatile", "Erosion area, gully",
    "Debris-covered areas", "Rock", "Young conif. (growth, thicket)",
    "Young broadl. (growth, thicket)", "Conif. pole timber CC<80", "Conif. pole timber CC>=80",
    "Broadl. pole timber", "Conif. mature forest CC<80", "Conif. mature forest CC>=80",
    "Broadl. mature forest CC<80", "Broadl. mature forest CC>=80",
    "Old conif. forest, multilayered CC<80", "Old conif. forest, multilayered CC>=80",
    "Old broadl. forest, multilayered CC<80", "Old broadl. forest, multilayered CC>=80",
    "Clearcut areas", "Mountain dwarf forest", "Grassland, buffer strip",
    "Alpine grassland, heath", "Low importance or small extent",
]
CC_PAIRS = {9: 8, 12: 11, 14: 13, 16: 15, 18: 17}  # CC>=80 -> CC<80 code, as in B3
CC_SPLIT = [8, 9, 11, 12, 13, 14, 15, 16, 17, 18]
GROUPS = {
    "all": list(range(1, NC)),
    "cc_split": CC_SPLIT,
    "other": [c for c in range(1, NC) if c not in CC_SPLIT],
    "other_forest": [6, 7, 10, 19, 20],
    "non_forest": [1, 2, 3, 4, 5, 21, 22, 23],
}
MERGED = [8, 11, 13, 15, 17]  # the cc_merged group, on the merged matrix
N_BOOT = 1000

MERGE = np.eye(NC, dtype=np.int64)  # cm_merged = MERGE.T @ cm @ MERGE
for hi, lo in CC_PAIRS.items():
    MERGE[hi, hi] = 0
    MERGE[hi, lo] = 1


def strip_tile_cms(ref_path: Path, pred_path: Path) -> np.ndarray:
    """Per-tile 24x24 confusion matrices on the compute_final_metrics grid, built strip by
    strip: reference read over the (fractional) intersection window, prediction warped onto
    its window transform with nearest resampling, 0 ignored in both."""
    with rasterio.open(ref_path) as m, rasterio.open(pred_path) as p:
        bm, bp = m.bounds, p.bounds
        inter = BoundingBox(left=max(bm.left, bp.left), bottom=max(bm.bottom, bp.bottom),
                            right=min(bm.right, bp.right), top=min(bm.top, bp.top))
        win = m.window(*inter)
        full = win.crop(m.height, m.width).round_lengths()
        height, width = int(full.height), int(full.width)
        transform = m.window_transform(win)
        assert m.res != p.res  # the warp path of compute_final_metrics
        cms = []
        for r0 in range(0, height, BLOCK):
            h = min(BLOCK, height - r0)
            ref = m.read(1, window=Window(win.col_off, win.row_off + r0, width, h))
            assert ref.shape == (h, width), (ref.shape, h, width)
            pred = np.empty((h, width), dtype=p.dtypes[0])
            reproject(source=rasterio.band(p, 1), destination=pred, src_transform=p.transform,
                      src_crs=p.crs, dst_transform=transform * transform.translation(0, r0),
                      dst_crs=m.crs, resampling=Resampling.nearest)
            if ((ref != 0) & (pred != 0)).any():
                cms.append(tile_confusion_matrices(ref, pred, NC, block=BLOCK, ignore_index=0))
            del ref, pred
    return np.concatenate(cms)


def tile_job(job: tuple[str, int]) -> tuple[str, int, np.ndarray]:
    exp, year = job
    folder = MO / exp
    out = folder / NEW_DIR / f"tile_cms_seg_{year}.npy"
    if out.exists():
        return exp, year, np.load(out)
    out.parent.mkdir(exist_ok=True)
    pred = next(folder.glob(f"{year}_*.tif"))
    cms = strip_tile_cms(REFS[year], pred)
    np.save(out, cms)
    print(f"done {exp} {year}: {cms.shape[0]} tiles", flush=True)
    return exp, year, cms


def per_class(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (cm.sum(0) + cm.sum(1) - tp)
        rec = tp / cm.sum(1)
    return iou, rec


def group_scores(cm: np.ndarray, k: np.ndarray) -> dict[str, tuple[float, float]]:
    """(mean IoU, mean recall = balanced accuracy) per group over the classes of K."""
    iou, rec = per_class(cm)
    res = {g: (float(iou[[c for c in cls if k[c]]].mean()), float(rec[[c for c in cls if k[c]]].mean()))
           for g, cls in GROUPS.items()}
    km = (MERGE.T @ k.astype(np.int64)) > 0
    iou_m, rec_m = per_class(MERGE.T @ cm @ MERGE)
    sel = [c for c in MERGED if km[c]]
    res["cc_merged"] = (float(iou_m[sel].mean()), float(rec_m[sel].mean()))
    return res


def check_stored(exp: str, year: int, cm: np.ndarray) -> dict:
    """Max abs difference to the stored classification_metrics_<year>.csv."""
    df = pd.read_csv(MO / exp / f"classification_metrics_{year}.csv", index_col=0)
    df.index = df.index.astype(str)
    iou, rec = per_class(cm)
    cls = [str(c) for c in range(1, NC)]
    d_cls = max(np.nanmax(np.abs(df.loc[cls, "JaccardIndex"].to_numpy() - np.nan_to_num(iou[1:]))),
                np.nanmax(np.abs(df.loc[cls, "Recall"].to_numpy() - np.nan_to_num(rec[1:]))))
    return {"experiment": exp, "year": year,
            "mIoU_stored": df.loc["macro", "JaccardIndex"], "mIoU_recomputed": float(np.nanmean(iou[1:])),
            "BA_stored": df.loc["macro", "Recall"], "BA_recomputed": float(np.nanmean(rec[1:])),
            "max_abs_diff_per_class_iou_recall": float(d_cls), "valid_px": int(cm.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    jobs = [(e, y) for y in [2020, 2013] for e in EXPERIMENTS]
    with Pool(args.workers) as pool:
        tiles = {(e, y): c for e, y, c in pool.imap_unordered(tile_job, jobs)}

    checks, pcs, rows = [], [], []
    boot = {}  # (year, group) -> {model: diffs}, for the mean over models
    for year in [2020, 2013]:
        for m_key, model in MODELS.items():
            e, f_ = f"{m_key}-rgb-nir-ndsm-phaseE", f"{m_key}-rgb-nir-phaseF"
            ce, cf = tiles[(e, year)], tiles[(f_, year)]
            assert ce.shape == cf.shape, (e, f_, year, ce.shape, cf.shape)
            # the 2020 E rasters sit 0.125 m east of the F rasters (same shape), so each is
            # scored on its own intersection window, as in the stored csvs; the tile grids
            # then differ by a fraction of a reference pixel and the reference pixels per
            # tile by < 0.1%, close enough to pair tiles by index
            ref_diff = int(np.abs(ce.sum(2) - cf.sum(2)).sum())
            assert ref_diff < 1e-3 * ce.sum(), (e, f_, year, ref_diff)
            for exp, c in [(e, ce), (f_, cf)]:
                checks.append(check_stored(exp, year, c.sum(0)))
                iou, rec = per_class(c.sum(0))
                for cl in range(1, NC):
                    pcs.append({"model": model, "experiment": exp, "phase": exp[-1], "year": year,
                                "class": cl, "name": CLASS_NAMES[cl],
                                "group": "cc_split" if cl in CC_SPLIT else "other",
                                "IoU": iou[cl], "recall": rec[cl], "ref_px": int(c.sum(0)[cl].sum())})
            idx, k, redraw = fixed_class_resamples(ce, n_boot=N_BOOT)
            se, sf = group_scores(ce.sum(0), k), group_scores(cf.sum(0), k)
            bs = [(group_scores(ce[i].sum(0), k), group_scores(cf[i].sum(0), k)) for i in idx]
            for g in se:
                diffs = np.array([b[0][g][0] - b[1][g][0] for b in bs])
                boot.setdefault((year, g), {})[model] = diffs
                n_cls = len(MERGED) if g == "cc_merged" else int(k[GROUPS[g]].sum())
                rows.append({
                    "model": model, "year": year, "group": g, "n_classes": n_cls,
                    "IoU_E": se[g][0], "IoU_F": sf[g][0], "dIoU": se[g][0] - sf[g][0],
                    "dIoU_ci_low": float(np.percentile(diffs, 2.5)),
                    "dIoU_ci_high": float(np.percentile(diffs, 97.5)),
                    "BA_E": se[g][1], "BA_F": sf[g][1], "dBA": se[g][1] - sf[g][1],
                    "n_tiles": ce.shape[0], "redraw_rate": redraw,
                    "ref_px_diff_E_vs_F": ref_diff / ce.sum(),
                })
    res = pd.DataFrame(rows)

    # mean over the six models; a joint CI (mean of the per-model differences on the same
    # resamples) only where all models share the tile grid, else left empty
    mean_rows = []
    for (year, g), per_model in boot.items():
        sub = res[(res.year == year) & (res.group == g)]
        row = {"model": "mean (6 models)", "year": year, "group": g,
               "n_classes": int(sub.n_classes.iloc[0])}
        for col in ["IoU_E", "IoU_F", "dIoU", "BA_E", "BA_F", "dBA"]:
            row[col] = sub[col].mean()
        # same E reference tiles for all models -> same seed-0 resamples for all models
        refs = [tiles[(f"{m}-rgb-nir-ndsm-phaseE", year)].sum(2) for m in MODELS]
        if all(r.shape == refs[0].shape and np.array_equal(r, refs[0]) for r in refs):
            d = np.mean(np.stack(list(per_model.values())), axis=0)
            row["dIoU_ci_low"], row["dIoU_ci_high"] = np.percentile(d, [2.5, 97.5])
        row["n_tiles"] = int(sub.n_tiles.iloc[0])
        mean_rows.append(row)
    res = pd.concat([res, pd.DataFrame(mean_rows)], ignore_index=True)
    order = {g: i for i, g in enumerate(["all", "cc_split", "other", "other_forest", "non_forest", "cc_merged"])}
    res = res.sort_values(["year", "model", "group"], key=lambda s: s.map(order) if s.name == "group" else s,
                          ascending=[False, True, True])
    res.to_csv(OUT / "b4_ndsm_ablation.csv", index=False)
    pd.DataFrame(pcs).to_csv(OUT / "b4_per_class_iou.csv", index=False)
    chk = pd.DataFrame(checks)
    chk.to_csv(OUT / "b4_check_stored_csv.csv", index=False)

    pd.set_option("display.width", 200)
    print(chk.round(5).to_string(index=False))
    cols = ["model", "year", "group", "IoU_E", "IoU_F", "dIoU", "dIoU_ci_low", "dIoU_ci_high", "dBA"]
    print(res[res.group.isin(["all", "cc_split", "other", "cc_merged"])][cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
