"""Print the result tables of the MDPI manuscript from the CSVs in outputs/mdpi/.

Needs no raw data, so the published numbers can be checked against the
released metric files. Values are rounded to two decimals as in the paper;
deltas are computed from unrounded values.

Usage: python scripts/mdpi/paper_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CHANGE_CLASSES, OUTPUTS

MODELS = ["Clay v1.0", "U-Net", "Clay v1.5", "DOFA", "TerraMind", "Prithvi"]
PIXEL_HA = 0.2 * 0.2 / 10_000  # one 0.2 m reference pixel in hectares

master = pd.read_csv(OUTPUTS / "metrics_master.csv")
perclass = pd.read_csv(OUTPUTS / "metrics_perclass.csv")
ci = pd.read_csv(OUTPUTS / "metrics_ci.csv").set_index(["model", "task"])
cm = pd.read_csv(OUTPUTS / "confusion_matrix_change_9x9.csv", index_col=0)


def macro(model, task, metric, phase="E", postproc="before") -> float:
    q = master.query("model==@model and task==@task and metric==@metric "
                     "and phase==@phase and postproc==@postproc")["value"]
    return float(q.iloc[0]) if len(q) else float("nan")


def show(title: str, df: pd.DataFrame) -> None:
    print(f"\n=== {title}\n{df.round(2).to_string()}")


def transitions() -> None:
    area = cm.sum(axis=1) * PIXEL_HA
    df = pd.DataFrame({"area_ha": area.round(1), "share_pct": (100 * area / area.sum()).round(1)})
    print(f"\n=== Table 3: reference change area (evaluated test area)\n"
          f"{df.sort_values('area_ha', ascending=False).to_string()}")


def segmentation() -> None:
    df = pd.DataFrame({
        f"{yr} {m}": [macro(model, f"seg-{yr}", f"macro_{m}") for model in MODELS]
        for yr in (2013, 2020) for m in ("BA", "IoU", "F1")
    }, index=MODELS)
    show("Single-date segmentation (RGB+NIR+nDSM)", df)


def change() -> None:
    rows = {}
    for model in MODELS:
        row = {}
        for task, label in (("change-binary", "bin"), ("change-multiclass", "mc")):
            c = ci.loc[(model, task)]
            row[f"{label} BA"] = macro(model, task, "macro_BA")
            row[f"{label} IoU"] = macro(model, task, "macro_IoU")
            row[f"{label} CI"] = f"[{c.IoU_ci_low:.2f}-{c.IoU_ci_high:.2f}]"
            row[f"{label} F1"] = macro(model, task, "macro_F1")
        rows[model] = row
    df = pd.DataFrame(rows).T
    num = [c for c in df.columns if not c.endswith("CI")]
    df[num] = df[num].astype(float)
    show("Cross-temporal change detection (RGB+NIR+nDSM)", df)


def perclass_clay() -> None:
    q = perclass.query("model=='Clay v1.0' and phase=='E' and postproc=='before' "
                       "and task=='change-multiclass'").set_index("class_id")
    share = 100 * cm.sum(axis=1).values / cm.values.sum()
    df = pd.DataFrame({"area_pct": share, "IoU": q.IoU, "F1": q.F1, "Recall": q.Recall,
                       "Precision": q.Precision})
    df.index = CHANGE_CLASSES
    show("Per-class change metrics, Clay v1.0", df.sort_values("IoU", ascending=False))


def ablation() -> None:
    rows = {}
    for model in MODELS:
        row = {}
        for task, label in (("change-binary", "bin"), ("change-multiclass", "mc"),
                            ("seg-2013", "seg13")):
            f, e = macro(model, task, "macro_IoU", "F"), macro(model, task, "macro_IoU", "E")
            row |= {f"{label} RGB+NIR": f, f"{label} +nDSM": e, f"{label} d": e - f}
        rows[model] = row
    show("LiDAR ablation, macro IoU", pd.DataFrame(rows).T)


def postproc() -> None:
    rows = {}
    for model in MODELS:
        row = {}
        for task, label in (("change-binary", "bin"), ("change-multiclass", "mc")):
            b = macro(model, task, "macro_IoU")
            a = macro(model, task, "macro_IoU", postproc="after")
            row |= {f"{label} before": b, f"{label} after": a, f"{label} d": a - b}
        rows[model] = row
    show("Post-processing, macro IoU (RGB+NIR+nDSM)", pd.DataFrame(rows).T)


def perclass_all() -> None:
    q = perclass.query("phase=='E' and postproc=='before'")
    for task, title in (("seg-2013", "Per-class IoU, segmentation 2013"),
                        ("seg-2020", "Per-class IoU, segmentation 2020"),
                        ("change-multiclass", "Per-class IoU, multi-class change")):
        df = q[q.task == task].pivot(index="class_id", columns="model", values="IoU")[MODELS]
        if task == "change-multiclass":
            df.index = CHANGE_CLASSES
        show(title, df)


if __name__ == "__main__":
    transitions()
    segmentation()
    change()
    perclass_clay()
    ablation()
    postproc()
    perclass_all()
