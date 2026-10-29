#!/usr/bin/env python
"""B3 (R2-M1, R1-M2): change detection with the canopy-cover split collapsed.

The CC>=80 / CC<80 split of pole, mature and old forest comes from the LiDAR canopy-cover
product, and the models of Phase E see the nDSM from the same acquisitions. Here the
split is collapsed in the reference and in the predictions before the change rule is
applied (8+9, 11+12, 13+14, 15+16, 17+18, each pair mapped to its CC<80 code), then:

  - mature tree density loss (1), old-growth density loss (2) and forest density gain (5)
    cannot occur any more (same class before and after = no change);
  - forest setback (3) keeps only the true stage regressions (about 0.7 ha in the
    reference) and is merged into other transition (8), else it is absent from most
    bootstrap tiles.

The collapsed scheme has 5 classes (no change, stage progression, early forest
establishment, new clearcut, other transition) against 9 in the original scheme, so only
deltas (Phase E RGB+NIR+nDSM minus Phase F RGB+NIR, per model) are compared, not levels.
The change rule is the B0 one (dwarf-forest fix), get_change_array in src/trainers/utils.py.

Outputs (never overwriting): collapsed reference change map in
<data root>/habitalp_change/r1_cccollapse/, collapsed predicted change maps and metric
CSVs in <model_output>/<experiment>/r1_cccollapse/, tables in outputs/r1/b3_*.csv. The
framing numbers (area per reference change class, how much of it is LiDAR-threshold) come
from the cross-tabulation of the original against the collapsed reference change map.

Run: nice -n 19 python scripts/b3_cc_collapsed_cd.py [--workers 3]
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

BASE_FOLDER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_FOLDER))
from scripts.b0_dwarffix_cd import (
    BLOCK, DATA_ROOT, EXPERIMENTS, MC_NAMES, MO, MODELS, NEW_DIR as B0_DIR, OUT, PP_DIR,
    REF_2013, REF_2020, REF_CT,
)
from src.trainers.bootstrap import (
    block_bootstrap_ci_fixed, collapse_binary, fixed_class_resamples, load_aligned_2d,
    macro_iou_f1_fixed, tile_confusion_matrices,
)
from src.trainers.utils import (
    compute_final_metrics, get_change_array, read_intersecting_area_of_two_rasters,
)

NEW_DIR = "r1_cccollapse"
REF_CC = DATA_ROOT / "habitalp_change" / NEW_DIR / "habitalp_change_v4_2013-2020_cccollapsed.tif"
CC_PAIRS = {9: 8, 12: 11, 14: 13, 16: 15, 18: 17}  # CC>=80 -> CC<80 code of the same stage
LUT = np.arange(256, dtype=np.uint8)
for hi, lo in CC_PAIRS.items():
    LUT[hi] = lo
LIDAR_CLASSES = [1, 2, 3, 5]
KEPT = [0, 4, 6, 7, 8]
PX_HA = 0.199897019943870 ** 2 / 1e4


def collapsed_change(a13: np.ndarray, a20: np.ndarray, in_nodata: int) -> np.ndarray:
    ch = get_change_array(LUT[a13], LUT[a20], in_nodata=in_nodata, out_nodata=255)
    return np.where(ch == 3, 8, ch).astype(np.uint8)


def generate_collapsed_change_map(mask_path: Path, prediction_path: Path, output_path: Path,
                                  n_batches: int = 20, check_against: Path | None = None) -> dict:
    """As generate_change_map (src/inference/eval.py), with the CC split collapsed. With
    check_against, also recomputes the uncollapsed map and counts pixels that differ."""
    mask, prediction, meta = read_intersecting_area_of_two_rasters(mask_path, prediction_path)
    in_nodata = meta.get("nodata", 0)
    meta.update({"compress": "lzw", "nodata": 255, "tiled": True, "predictor": 2, "bigtiff": True})
    height, width = mask.shape
    chunk = height // n_batches
    n_diff = 0
    ref = rasterio.open(check_against) if check_against else None
    with rasterio.open(output_path, "w", **meta) as dst:
        for i in range(n_batches):
            r0, r1 = i * chunk, (i + 1) * chunk if i < n_batches - 1 else height
            m, p = mask[r0:r1], prediction[r0:r1]
            win = rasterio.windows.Window(0, r0, width, r1 - r0)
            dst.write(collapsed_change(m, p, in_nodata), 1, window=win)
            if ref is not None:
                assert ref.shape == (height, width)
                orig = get_change_array(m, p, in_nodata=in_nodata, out_nodata=255)
                n_diff += int((orig != ref.read(1, window=win)).sum())
    if ref is not None:
        ref.close()
    return {"uncollapsed_px_differing_from_reference": n_diff}


def write_metrics(reference: Path, change_map: Path, out_csv: Path, binary: bool) -> pd.DataFrame:
    """compute_final_metrics as in B0; with 9 class slots, torchmetrics leaves out the
    classes that are absent from reference and prediction, so the macro is over 5."""
    if not out_csv.exists():
        mc, each, _ = compute_final_metrics(
            reference_file_path=reference, prediction_file_path=change_map,
            num_classes=2 if binary else 9,
            class_names=["No change", "Change"] if binary else MC_NAMES, ignore_index=255)
        each.pop("MulticlassConfusionMatrix")
        pd.concat([pd.DataFrame([mc], index=["macro"]).astype(float),
                   pd.DataFrame(each).astype(float)]).to_csv(out_csv)
    df = pd.read_csv(out_csv, index_col=0)
    df.index = df.index.astype(str)
    return df


def reference_job() -> dict:
    REF_CC.parent.mkdir(exist_ok=True)
    res = {}
    if not REF_CC.exists():
        res = generate_collapsed_change_map(REF_2013, REF_2020, REF_CC, check_against=REF_CT)
    orig, coll = load_aligned_2d(REF_CT, REF_CC)
    valid = orig != 255
    assert np.array_equal(valid, coll != 255)
    xt = np.bincount(orig[valid].astype(np.int64) * 9 + coll[valid], minlength=81).reshape(9, 9)
    pd.DataFrame(xt * PX_HA, index=MC_NAMES, columns=MC_NAMES).to_csv(OUT / "b3_reference_crosstab_ha.csv")
    res["crosstab"] = xt
    return res


def run_job(job: tuple[str, str]) -> dict:
    import torch
    torch.set_num_threads(3)
    exp, postproc = job
    folder = MO / exp
    new_dir = folder / NEW_DIR
    new_dir.mkdir(exist_ok=True)
    sfx = "_after_postprocessing" if postproc == "after" else ""
    pred = folder / PP_DIR / "2020_re-rasterized.tif" if postproc == "after" else next(folder.glob("2020_*.tif"))
    cmap = new_dir / f"change_map_2013-2020{sfx}.tif"
    cms_file = new_dir / f"tile_cms{sfx}.npy"
    if not cmap.exists():
        generate_collapsed_change_map(REF_2013, pred, cmap)
    out = {"experiment": exp, "postproc": postproc}
    for t in ["binary", "multiclass"]:
        out[t] = write_metrics(REF_CC, cmap, new_dir / f"change_metrics_{t}_2013-2020{sfx}.csv", t == "binary")
    if cms_file.exists():
        out["cms_cc"] = np.load(cms_file)
    else:
        ref2d, prd2d = load_aligned_2d(REF_CC, cmap)
        out["cms_cc"] = tile_confusion_matrices(ref2d, prd2d, 9, block=BLOCK, ignore_index=255)
        np.save(cms_file, out["cms_cc"])
    ref2d, prd2d = load_aligned_2d(REF_CT, folder / B0_DIR / f"change_map_2013-2020{sfx}.tif")
    out["cms_9"] = tile_confusion_matrices(ref2d, prd2d, 9, block=BLOCK, ignore_index=255)
    print(f"done {exp} {postproc}", flush=True)
    return out


def macro(cm: np.ndarray, k: np.ndarray) -> dict:
    tp = np.diag(cm).astype(np.float64)
    rec = tp / cm.sum(axis=1)
    iou, f1 = macro_iou_f1_fixed(cm, k)
    return {"IoU": iou, "F1": f1, "BA": float(rec[k].mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ref = reference_job()
    xt = ref["crosstab"]
    area = xt.sum(axis=1) * PX_HA
    change_ha = area[1:].sum()
    lidar_ha = area[LIDAR_CLASSES].sum()
    cls = pd.DataFrame({
        "class": range(9), "name": MC_NAMES, "area_ha": area,
        "to_no_change_ha": xt[:, 0] * PX_HA, "to_other_transition_ha": xt[:, 8] * PX_HA,
        "kept_same_ha": np.diag(xt) * PX_HA,
    })
    cls.to_csv(OUT / "b3_reference_class_areas.csv", index=False)
    with open(OUT / "b3_framing.txt", "w") as f:
        for k, v in ref.items():
            if k != "crosstab":
                f.write(f"{k}: {v}\n")
        f.write(f"reference change area (classes 1-8): {change_ha:.1f} ha\n")
        f.write(f"LiDAR-threshold classes 1, 2, 3, 5: {lidar_ha:.1f} ha ({lidar_ha / change_ha:.1%} of change)\n")
        f.write(f"setback (3): {area[3]:.2f} ha, of which {xt[3, 0] * PX_HA:.2f} ha CC-only (to no change), "
                f"{xt[3, 8] * PX_HA:.2f} ha true stage regression (to other transition)\n")
        f.write(f"collapsed reference change area: {(xt.sum(axis=0) * PX_HA)[1:].sum():.1f} ha\n")
        f.write(cls.round(2).to_string(index=False) + "\n")

    jobs = [(e, p) for e in EXPERIMENTS for p in ["before", "after"]]
    with Pool(args.workers) as pool:
        res = list(pool.imap(run_job, jobs))
    res = {(r["experiment"], r["postproc"]): r for r in res}

    rows = []
    for (exp, pp), r in res.items():
        for t in ["binary", "multiclass"]:
            is_bin = t == "binary"
            df = r[t]
            row = {"model": MODELS[exp.split("-rgb")[0]], "experiment": exp, "phase": exp[-1],
                   "postproc": pp, "task": f"change-{t}", "IoU_cc": df.loc["macro", "JaccardIndex"],
                   "F1_cc": df.loc["macro", "F1Score"], "BA_cc": df.loc["macro", "Recall"]}
            ci = block_bootstrap_ci_fixed(r["cms_cc"], binary=is_bin)
            row.update({"IoU_cc_ci_low": ci["IoU_ci_low"], "IoU_cc_ci_high": ci["IoU_ci_high"],
                        "n_classes_cc": ci["n_classes"], "redraw_rate_cc": ci["redraw_rate"]})
            cm = r["cms_cc"].sum(axis=0)
            cm = collapse_binary(cm) if is_bin else cm
            m = macro(cm, cm.sum(axis=1) > 0)
            assert abs(m["IoU"] - row["IoU_cc"]) < 1e-3, (exp, pp, t, m, row["IoU_cc"])
            cm9 = r["cms_9"].sum(axis=0)
            cm9 = collapse_binary(cm9) if is_bin else cm9
            row.update({f"{k}_9": v for k, v in macro(cm9, cm9.sum(axis=1) > 0).items()})
            if not is_bin:
                for c in KEPT:
                    row[f"c{c}_IoU_cc"] = df.loc[str(c), "JaccardIndex"]
            rows.append(row)
    met = pd.DataFrame(rows)
    met.to_csv(OUT / "b3_cc_collapsed_metrics.csv", index=False)

    # nDSM gain, E minus F per model, both schemes, with a paired block bootstrap
    drows = []
    for m_key, model in MODELS.items():
        e, f_ = f"{m_key}-rgb-nir-ndsm-phaseE", f"{m_key}-rgb-nir-phaseF"
        for pp in ["before", "after"]:
            for t in ["binary", "multiclass"]:
                is_bin = t == "binary"
                row = {"model": model, "postproc": pp, "task": f"change-{t}"}
                for scheme, key in [("9", "cms_9"), ("cc", "cms_cc")]:
                    ce, cf = res[(e, pp)][key], res[(f_, pp)][key]
                    if is_bin:
                        ce = np.stack([collapse_binary(c) for c in ce])
                        cf = np.stack([collapse_binary(c) for c in cf])
                    # common resamples: redraw if a class is missing from the reference
                    # (the reference, so the class support, is the same for E and F)
                    idx, k, _ = fixed_class_resamples(ce)
                    diffs = np.array([macro_iou_f1_fixed(ce[i].sum(0), k)[0] - macro_iou_f1_fixed(cf[i].sum(0), k)[0]
                                      for i in idx])
                    me, mf = macro(ce.sum(0), k), macro(cf.sum(0), k)
                    for metric in ["IoU", "F1", "BA"]:
                        row[f"{metric}_E_{scheme}"] = me[metric]
                        row[f"{metric}_F_{scheme}"] = mf[metric]
                        row[f"d{metric}_{scheme}"] = me[metric] - mf[metric]
                    row[f"dIoU_{scheme}_ci_low"] = float(np.percentile(diffs, 2.5))
                    row[f"dIoU_{scheme}_ci_high"] = float(np.percentile(diffs, 97.5))
                drows.append(row)
    d = pd.DataFrame(drows)
    d.to_csv(OUT / "b3_ndsm_gain_9_vs_collapsed.csv", index=False)
    cols = ["model", "postproc", "task", "dIoU_9", "dIoU_9_ci_low", "dIoU_9_ci_high",
            "dIoU_cc", "dIoU_cc_ci_low", "dIoU_cc_ci_high", "dBA_9", "dBA_cc"]
    print(open(OUT / "b3_framing.txt").read())
    print(d[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
