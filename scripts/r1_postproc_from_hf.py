#!/usr/bin/env python
"""Change-detection metrics before and after post-processing from the released dataset.

Recomputes the post-processing table of the revised manuscript (columns "_new" of
outputs/r1/b0_dwarffix_cd_metrics.csv, written by b0_dwarffix_cd.py) from the files of the
HabitAlp 2.0 dataset on Hugging Face (v4.1), without the internal folder layout:

  labels/classes_2013.tif, labels/habitalp_change_2013_2020.tif
  model_outputs/<experiment>/prediction_2020.tif               (raw model output, "before")
  model_outputs/<experiment>/prediction_2020_postprocessed.tif (post-processed, "after")

Same path as b0: change map from the 2013 reference and the 2020 prediction
(generate_change_map, change rule of src/trainers/utils.py::get_change_array), multi-class and
binary metrics (compute_final_metrics), 95 % CIs from the spatial block bootstrap over 500 m tiles.
Writes outputs/r1/hf_postproc_metrics.csv and, if b0_dwarffix_cd_metrics.csv is present, compares
with it and exits with an error if any value differs by more than TOL.

Run: python scripts/r1_postproc_from_hf.py --hf-root <dataset folder> [--workers 3]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

BASE_FOLDER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_FOLDER))
from scripts.b0_dwarffix_cd import BLOCK, EXPERIMENTS, MC_NAMES, MODELS, metrics_from_csv
from src.inference.eval import generate_change_map
from src.trainers.bootstrap import (
    block_bootstrap_ci, collapse_binary, load_aligned_2d, macro_iou_f1, tile_confusion_matrices,
)
from src.trainers.utils import compute_final_metrics

OUT = BASE_FOLDER / "outputs/r1"
TOL = 1e-6  # metrics and CIs are deterministic (bootstrap seed 0)
PRED = {"before": "prediction_2020.tif", "after": "prediction_2020_postprocessed.tif"}


def write_metrics_csv(ref_ct: Path, change_map: Path, out_csv: Path, binary: bool) -> None:
    multiclass_metrics, metrics_each_label, _ = compute_final_metrics(
        reference_file_path=ref_ct,
        prediction_file_path=change_map,
        num_classes=2 if binary else 9,
        class_names=["No change", "Change"] if binary else MC_NAMES,
        ignore_index=255,
    )
    metrics_each_label.pop("MulticlassConfusionMatrix")
    pd.concat([
        pd.DataFrame([multiclass_metrics], index=["macro"]).astype(float),
        pd.DataFrame(metrics_each_label).astype(float),
    ]).to_csv(out_csv)


def run_job(job: tuple[Path, Path, str, str]) -> list[dict]:
    import torch
    torch.set_num_threads(4)
    hf_root, work, exp, postproc = job
    ref_2013 = hf_root / "labels/classes_2013.tif"
    ref_ct = hf_root / "labels/habitalp_change_2013_2020.tif"
    change_map = work / f"{exp}_{postproc}_change_map_2013-2020.tif"
    generate_change_map(mask_path=ref_2013, prediction_path=hf_root / "model_outputs" / exp / PRED[postproc],
                        output_path=change_map)
    ref2d, prd2d = load_aligned_2d(ref_ct, change_map)
    cms = tile_confusion_matrices(ref2d, prd2d, 9, block=BLOCK, ignore_index=255)
    del ref2d, prd2d

    rows = []
    for t in ["binary", "multiclass"]:
        is_bin = t == "binary"
        csv = work / f"{exp}_{postproc}_change_metrics_{t}.csv"
        write_metrics_csv(ref_ct, change_map, csv, binary=is_bin)
        m = metrics_from_csv(csv)
        cm = cms.sum(axis=0)
        iou_cm, _ = macro_iou_f1(collapse_binary(cm) if is_bin else cm)
        assert abs(iou_cm - m["IoU"]) <= 0.01, (exp, postproc, t, iou_cm, m["IoU"])
        ci = block_bootstrap_ci(cms, n_boot=1000, binary=is_bin)
        rows.append({"model": MODELS[exp.split("-rgb")[0]], "experiment": exp, "phase": exp[-1],
                     "postproc": postproc, "task": f"change-{t}", **m,
                     **{k: ci[k] for k in ["IoU_ci_low", "IoU_ci_high", "F1_ci_low", "F1_ci_high"]},
                     "n_tiles": ci["n_tiles"]})
    print(f"done {exp} {postproc}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-root", type=Path, required=True, help="local copy of the HabitAlp 2.0 dataset")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        jobs = [(args.hf_root, Path(tmp), e, p) for e in EXPERIMENTS for p in PRED]
        with Pool(args.workers) as pool:
            rows = [r for rs in pool.imap(run_job, jobs) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "hf_postproc_metrics.csv", index=False)
    print(f"wrote {OUT / 'hf_postproc_metrics.csv'}")

    b0_csv = OUT / "b0_dwarffix_cd_metrics.csv"
    if b0_csv.exists():
        key = ["experiment", "postproc", "task"]
        b0 = pd.read_csv(b0_csv).set_index(key)
        hf = df.set_index(key)
        cols = ["IoU", "F1", "BA", "IoU_ci_low", "IoU_ci_high", "F1_ci_low", "F1_ci_high"]
        diff = (hf[cols] - b0.loc[hf.index, [f"{c}_new" for c in cols]].to_numpy()).abs()
        worst = diff.to_numpy().max()
        print(f"largest difference to {b0_csv.name}: {worst:.2e}")
        if worst > TOL:
            sys.exit(f"mismatch: largest difference {worst:.2e} exceeds {TOL:g}")


if __name__ == "__main__":
    main()
