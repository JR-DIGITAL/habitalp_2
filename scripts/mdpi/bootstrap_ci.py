"""Spatial block bootstrap CIs for the cross-temporal change metrics, and the
change-class confusion matrices of Clay v1.0.

The test area is cut into 500 m x 500 m tiles (2500 px at 0.2 m; 88 non-empty
tiles), per-tile confusion matrices are resampled 1000 times with replacement,
and the 2.5th/97.5th percentiles of macro IoU and F1 give the 95 % CI
(src/trainers/bootstrap.py). The macro IoU recomputed from the summed confusion
matrix is checked against metrics_master.csv.

Writes outputs/mdpi/metrics_ci.csv and outputs/mdpi/confusion_matrix_change_*.csv.

Usage:
    HABITALP_DATA=/path/to/data python scripts/mdpi/bootstrap_ci.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CHANGE_CLASSES, EXPERIMENTS, OUTPUTS, REF_CHANGE_2013_2020, data_root
from src.trainers.bootstrap import (block_bootstrap_ci, collapse_binary, load_aligned_2d,
                                    macro_iou_f1, tile_confusion_matrices)

BLOCK = 2500  # px at 0.2 m = 500 m
N_BOOT = 1000
CM_MODEL = "Clay v1.0"


def export_confusion_matrices(cm) -> None:
    for name, mat, labels in (("9x9", cm, CHANGE_CLASSES),
                              ("8x8_transitions", cm[1:, 1:], CHANGE_CLASSES[1:])):
        pd.DataFrame(mat, index=labels, columns=labels).to_csv(
            OUTPUTS / f"confusion_matrix_change_{name}.csv")
        rownorm = mat / mat.sum(axis=1, keepdims=True)
        pd.DataFrame(rownorm, index=labels, columns=labels).to_csv(
            OUTPUTS / f"confusion_matrix_change_{name}_rownorm.csv")


def main() -> None:
    root = data_root()
    master = pd.read_csv(OUTPUTS / "metrics_master.csv")
    rows = []
    for model, phases in EXPERIMENTS.items():
        change_map = root / "model_output" / phases["E"] / "change_map_2013-2020.tif"
        ref2d, pred2d = load_aligned_2d(root / REF_CHANGE_2013_2020, change_map)
        tile_cms = tile_confusion_matrices(ref2d, pred2d, len(CHANGE_CLASSES), block=BLOCK)
        del ref2d, pred2d
        cm9 = tile_cms.sum(axis=0)
        for task, cm, binary in (("change-binary", collapse_binary(cm9), True),
                                 ("change-multiclass", cm9, False)):
            iou, f1 = macro_iou_f1(cm)
            ci = block_bootstrap_ci(tile_cms, n_boot=N_BOOT, binary=binary)
            q = master.query("model==@model and phase=='E' and postproc=='before' "
                             "and task==@task and metric=='macro_IoU'")["value"]
            check = ("OK" if len(q) and abs(iou - q.iloc[0]) <= 0.01
                     else f"CHECK (metrics_master: {q.iloc[0] if len(q) else 'missing'})")
            print(f"{model:10s} {task:18s} IoU {iou:.3f} [{ci['IoU_ci_low']:.3f}-"
                  f"{ci['IoU_ci_high']:.3f}]  F1 {f1:.3f} [{ci['F1_ci_low']:.3f}-"
                  f"{ci['F1_ci_high']:.3f}]  tiles={ci['n_tiles']}  {check}")
            rows.append({"model": model, "phase": "E", "task": task,
                         "macro_IoU": iou, "macro_F1": f1, **ci})
        if model == CM_MODEL:
            export_confusion_matrices(cm9)

    pd.DataFrame(rows).to_csv(OUTPUTS / "metrics_ci.csv", index=False)
    print(f"wrote {OUTPUTS / 'metrics_ci.csv'}")


if __name__ == "__main__":
    main()
