"""Collect all MDPI manuscript metrics into two tidy tables.

Reads the per-experiment metric CSVs written by evaluate.py and writes
  outputs/mdpi/metrics_master.csv    macro metrics, one row per
                                    [model, phase, task, postproc, metric]
  outputs/mdpi/metrics_perclass.csv  per-class IoU / F1 / precision / recall

Every row carries the source CSV path relative to HABITALP_DATA. The manuscript
tables are read off these files (see paper_tables.py).

Usage:
    HABITALP_DATA=/path/to/data python scripts/mdpi/build_metrics_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EXPERIMENTS, MODALITIES, MODEL_TYPE, OUTPUTS, PP_DIR, data_root

# task -> (csv basename, setting)
TASKS = {
    "seg-2013": ("classification_metrics_2013.csv", "in-domain-seg"),
    "seg-2020": ("classification_metrics_2020.csv", "cross-temporal-seg"),
    "change-binary": ("change_metrics_binary_2013-2020.csv", "cross-temporal"),
    "change-multiclass": ("change_metrics_multiclass_2013-2020.csv", "cross-temporal"),
}

# torchmetrics column -> metric name. "Accuracy" is macro-averaged, i.e. the
# balanced accuracy (mean per-class recall) reported as BA in the paper.
METRIC_COLS = {
    "Accuracy": "BA",
    "JaccardIndex": "IoU",
    "F1Score": "F1",
    "Precision": "Precision",
    "Recall": "Recall",
}


def main() -> None:
    root = data_root()
    macro_rows, class_rows = [], []
    for model, phases in EXPERIMENTS.items():
        for phase, exp in phases.items():
            base = root / "model_output" / exp
            for task, (fname, setting) in TASKS.items():
                for postproc in ("before", "after"):
                    csv_path = (base / fname if postproc == "before" else
                                base / PP_DIR / fname.replace(".csv", "_after_postprocessing.csv"))
                    if not csv_path.exists():
                        continue
                    df = pd.read_csv(csv_path, index_col=0)
                    keys = {"model": model, "type": MODEL_TYPE[model], "phase": phase,
                            "modality": MODALITIES[phase], "setting": setting, "task": task,
                            "postproc": postproc}
                    source = csv_path.relative_to(root).as_posix()
                    for col, metric in METRIC_COLS.items():
                        macro_rows.append({**keys, "metric": f"macro_{metric}",
                                           "value": float(df.loc["macro", col]),
                                           "source_csv": source})
                    per_class = df.drop(index="macro")
                    if task.startswith("seg"):
                        # row 0 is the nodata index, ignored during scoring
                        per_class = per_class.drop(index="0", errors="ignore")
                    for class_id, row in per_class.iterrows():
                        class_rows.append({**keys, "class_id": int(class_id),
                                           **{METRIC_COLS[c]: float(row[c])
                                              for c in METRIC_COLS if c != "Accuracy"},
                                           "source_csv": source})

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    master = pd.DataFrame(macro_rows)
    master.to_csv(OUTPUTS / "metrics_master.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUTPUTS / "metrics_perclass.csv", index=False)
    print(f"wrote {OUTPUTS / 'metrics_master.csv'} ({len(master)} rows)")
    print(f"wrote {OUTPUTS / 'metrics_perclass.csv'} ({len(class_rows)} rows)")

    piv = (master.query("task=='change-multiclass' and phase=='E' and postproc=='before'")
           .pivot_table(index="model", columns="metric", values="value"))
    print("\ncross-temporal multi-class change (RGB+NIR+nDSM, before post-processing):")
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
