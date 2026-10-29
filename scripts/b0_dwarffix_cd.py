#!/usr/bin/env python
"""B0 (R2-S5): recompute change maps and CD metrics with the dwarf-forest fix.

Reuses the evaluate-06 path (generate_change_map: reference 2013 -> predicted 2020,
compute_final_metrics for multi-class and binary) and the block bootstrap of
scripts/mdpi/bootstrap_ci.py. Never overwrites existing files: new outputs go to
<model_output>/<experiment>/r1_dwarffix/, the summary to outputs/r1/.

Run: python scripts/b0_dwarffix_cd.py [--workers 3] [--skip-reference-check]
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, from_bounds

BASE_FOLDER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_FOLDER))
from src.trainers.bootstrap import (
    block_bootstrap_ci, collapse_binary, load_aligned_2d, macro_iou_f1,
    tile_confusion_matrices,
)
from src.trainers.utils import compute_final_metrics, get_change_array
from src.inference.eval import generate_change_map

DATA_ROOT = Path(os.environ.get(
    "HABITALP_DATA_ROOT", BASE_FOLDER / "data/Originaldaten"))
MO = DATA_ROOT / "model_output"
REF_2013 = DATA_ROOT / "processed/mask/classes_v4_2013.tif"
REF_2020 = DATA_ROOT / "processed/mask/classes_v4_2020.tif"
REF_CT = DATA_ROOT / "habitalp_change/habitalp_change_v4_2013-2020.tif"
OUT = BASE_FOLDER / "outputs/r1"
NEW_DIR = "r1_dwarffix"
PP_DIR = "constraint_resolution_+_polygonfix_cc_updated"
BLOCK = 2500  # px @ 0.2 m, same blocks as scripts/mdpi/bootstrap_ci.py

MODELS = {
    "clay-v1": "Clay v1.0", "clay-v1.5": "Clay v1.5", "prithvi-v2-300": "Prithvi",
    "dofa-base": "DOFA", "terramind": "TerraMind", "unet-mit-b2": "U-Net",
}
EXPERIMENTS = [f"{m}-rgb-nir-ndsm-phaseE" for m in MODELS] + [f"{m}-rgb-nir-phaseF" for m in MODELS]
MC_NAMES = [
    "No change", "Mature Tree Density Loss", "Old Growth Density Loss",
    "Forest Setback YoungLoss", "Forest Stage Progression", "Forest Density Gain",
    "Early Forest Establishment", "Clearcut Loss", "Other Transition",
]
FOREST = np.arange(6, 19)
DWARF = 20


def reference_check(rows_per_tile: int = 2048) -> dict:
    """Recompute the reference change map tile-wise from the 2013/2020 class rasters with
    the fixed rule and count pixels that differ from the stored reference."""
    res = {"pixels": 0, "diff_fixed_rule": 0, "diff_old_rule": 0, "dwarf_to_forest": 0,
           "forest_to_dwarf": 0}
    with (rasterio.open(REF_2013) as s13, rasterio.open(REF_2020) as s20,
          rasterio.open(REF_CT) as sct):
        assert s20.transform == sct.transform and s20.shape == sct.shape
        w13 = from_bounds(*s20.bounds, transform=s13.transform).round_offsets().round_lengths()
        assert (w13.height, w13.width) == s20.shape, (w13, s20.shape)
        in_nodata = s13.nodata
        for r0 in range(0, s20.height, rows_per_tile):
            h = min(rows_per_tile, s20.height - r0)
            a13 = s13.read(1, window=Window(w13.col_off, w13.row_off + r0, s20.width, h))
            a20 = s20.read(1, window=Window(0, r0, s20.width, h))
            ct = sct.read(1, window=Window(0, r0, s20.width, h))
            new = get_change_array(a13, a20, in_nodata=in_nodata, out_nodata=255)
            d2f = (a13 == DWARF) & np.isin(a20, FOREST)
            old = np.where(d2f, 6, new)  # old rule: dwarf forest -> forest was 6
            res["pixels"] += ct.size
            res["diff_fixed_rule"] += int((new != ct).sum())
            res["diff_old_rule"] += int((old != ct).sum())
            res["dwarf_to_forest"] += int(d2f.sum())
            res["forest_to_dwarf"] += int((np.isin(a13, FOREST) & (a20 == DWARF)).sum())
    return res


def metrics_from_csv(path: Path) -> dict:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    d = {"IoU": df.loc["macro", "JaccardIndex"], "F1": df.loc["macro", "F1Score"],
         "BA": df.loc["macro", "Recall"]}
    if len(df) > 3:  # multi-class: per-class values for 6 and 8
        for c in ["6", "8"]:
            d[f"c{c}_IoU"] = df.loc[c, "JaccardIndex"]
            d[f"c{c}_precision"] = df.loc[c, "Precision"]
            d[f"c{c}_recall"] = df.loc[c, "Recall"]
    return d


def write_metrics_csv(change_map: Path, out_csv: Path, binary: bool) -> None:
    if out_csv.exists():
        return
    multiclass_metrics, metrics_each_label, _ = compute_final_metrics(
        reference_file_path=REF_CT,
        prediction_file_path=change_map,
        num_classes=2 if binary else 9,
        class_names=["No change", "Change"] if binary else MC_NAMES,
        ignore_index=255,
    )
    metrics_each_label.pop("MulticlassConfusionMatrix")
    metrics_df = pd.concat([
        pd.DataFrame([multiclass_metrics], index=["macro"]).astype(float),
        pd.DataFrame(metrics_each_label).astype(float),
    ])
    metrics_df.to_csv(out_csv)


def tile_cms(change_map: Path) -> np.ndarray:
    ref2d, prd2d = load_aligned_2d(REF_CT, change_map)
    return tile_confusion_matrices(ref2d, prd2d, 9, block=BLOCK, ignore_index=255)


def run_job(job: tuple[str, str]) -> list[dict]:
    import torch
    torch.set_num_threads(4)
    exp, postproc = job
    folder = MO / exp
    new_dir = folder / NEW_DIR
    new_dir.mkdir(exist_ok=True)
    sfx = "_after_postprocessing" if postproc == "after" else ""
    if postproc == "after":
        pred_2020 = folder / PP_DIR / "2020_re-rasterized.tif"
        old_map = folder / PP_DIR / "change_map_2013-2020.tif"
        old_csv = {t: folder / PP_DIR / f"change_metrics_{t}_2013-2020_after_postprocessing.csv"
                   for t in ["binary", "multiclass"]}
    else:
        pred_2020 = next(folder.glob("2020_*.tif"))
        old_map = folder / "change_map_2013-2020.tif"
        old_csv = {t: folder / f"change_metrics_{t}_2013-2020.csv" for t in ["binary", "multiclass"]}
    new_map = new_dir / f"change_map_2013-2020{sfx}.tif"
    new_csv = {t: new_dir / f"change_metrics_{t}_2013-2020{sfx}.csv" for t in ["binary", "multiclass"]}
    rows_csv = new_dir / f"b0_summary_rows{sfx}.csv"
    if rows_csv.exists():  # job finished in an earlier run
        return pd.read_csv(rows_csv).to_dict("records")

    if not new_map.exists():
        generate_change_map(mask_path=REF_2013, prediction_path=pred_2020, output_path=new_map)
    for t in ["binary", "multiclass"]:
        write_metrics_csv(new_map, new_csv[t], binary=(t == "binary"))

    # pixels whose change class moved between the old and the new map
    old2d, new2d = load_aligned_2d(old_map, new_map)
    changed = old2d != new2d
    n_changed = int(changed.sum())
    n_6_to_8 = int((changed & (old2d == 6) & (new2d == 8)).sum())
    del old2d, new2d, changed

    # block bootstrap CIs (as scripts/mdpi/bootstrap_ci.py), old and new maps
    cms = {"old": tile_cms(old_map), "new": tile_cms(new_map)}
    cm9_new = cms["new"].sum(axis=0)
    # same, inside the evaluated area (reference change map valid)
    n_6_to_8_eval = int(cms["old"].sum(axis=0)[:, 6].sum() - cm9_new[:, 6].sum())
    pd.DataFrame(cm9_new, index=MC_NAMES, columns=MC_NAMES).to_csv(
        new_dir / f"confusion_matrix_change_9x9{sfx}.csv")

    rows = []
    model = MODELS[exp.split("-rgb")[0]]
    phase = exp[-1]
    for t in ["binary", "multiclass"]:
        is_bin = t == "binary"
        row = {"model": model, "experiment": exp, "phase": phase, "postproc": postproc,
               "task": f"change-{t}", "pixels_changed_old_to_new": n_changed,
               "pixels_6_to_8": n_6_to_8, "pixels_6_to_8_eval_area": n_6_to_8_eval}
        for ver in ["old", "new"]:
            m = metrics_from_csv(old_csv[t] if ver == "old" else new_csv[t])
            cm = cms[ver].sum(axis=0)
            cm = collapse_binary(cm) if is_bin else cm
            iou_cm, _ = macro_iou_f1(cm)
            ci = block_bootstrap_ci(cms[ver], n_boot=1000, binary=is_bin)
            assert abs(iou_cm - m["IoU"]) <= 0.01, (exp, postproc, t, ver, iou_cm, m["IoU"])
            row.update({f"{k}_{ver}": v for k, v in m.items()})
            row.update({f"{k}_{ver}": ci[k] for k in
                        ["IoU_ci_low", "IoU_ci_high", "F1_ci_low", "F1_ci_high"]})
        row["n_tiles"] = ci["n_tiles"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(rows_csv, index=False)
    print(f"done {exp} {postproc}: {n_changed} px changed ({n_6_to_8} 6->8)", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--skip-reference-check", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if not args.skip_reference_check:
        ref = reference_check()
        print("reference check:", ref, flush=True)
        pd.DataFrame([ref]).to_csv(OUT / "b0_reference_check.csv", index=False)

    jobs = [(e, p) for e in EXPERIMENTS for p in ["before", "after"]]
    with Pool(args.workers) as pool:
        rows = [r for rs in pool.imap(run_job, jobs) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "b0_dwarffix_cd_metrics.csv", index=False)
    print(f"wrote {OUT / 'b0_dwarffix_cd_metrics.csv'}")

    # 9x9 and 8x8 confusion matrices for Clay v1.0 phase E, as scripts/mdpi/bootstrap_ci.py
    cm = pd.read_csv(MO / "clay-v1-rgb-nir-ndsm-phaseE" / NEW_DIR / "confusion_matrix_change_9x9.csv",
                     index_col=0).to_numpy()
    pd.DataFrame(cm, index=MC_NAMES, columns=MC_NAMES).to_csv(OUT / "confusion_matrix_change_9x9.csv")
    pd.DataFrame(cm / cm.sum(axis=1, keepdims=True), index=MC_NAMES, columns=MC_NAMES).to_csv(
        OUT / "confusion_matrix_change_9x9_rownorm.csv")
    cm8 = cm[1:, 1:]
    pd.DataFrame(cm8, index=MC_NAMES[1:], columns=MC_NAMES[1:]).to_csv(
        OUT / "confusion_matrix_change_8x8_transitions.csv")
    pd.DataFrame(cm8 / cm8.sum(axis=1, keepdims=True), index=MC_NAMES[1:], columns=MC_NAMES[1:]).to_csv(
        OUT / "confusion_matrix_change_8x8_transitions_rownorm.csv")


if __name__ == "__main__":
    main()
