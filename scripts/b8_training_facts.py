#!/usr/bin/env python
"""B8 (R1-M3, R1-m2): training facts of the revision runs and the LR table, from W&B.

Training facts: every finished run tagged ``revision-r1`` (runs tagged ``superseded-es`` or
``aborted`` are skipped). Per run: model, split, seed, patches per epoch, epochs run, best
epoch (0-based, as in the checkpoint names), best val mIoU, stop reason, batch size, LR,
scheduler, wall time and GPU. The val mIoU history is read from W&B, not from the summary
(the summary holds the last epoch).

LR table: every run in the benchmark project (preliminary runs of 2025 in
``semantic-segmentation-terratorch``, phases A to G and the revision in
``semantic-segmentation-terratorch-extended``) with at least one validation epoch, for
the six models of the paper. The six Phase E runs of the paper are marked.

Uses the stored W&B login (~/.netrc). Read only, nothing is written to W&B.

Run: python scripts/b8_training_facts.py [--no-lr-table]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import wandb

BASE_FOLDER = Path(__file__).resolve().parent.parent
OUT = BASE_FOLDER / "outputs/r1"
ENTITY = "university-of-graz-geo"
PROJECT = "semantic-segmentation-terratorch-extended"
PRELIM_PROJECT = "semantic-segmentation-terratorch"
SKIP_TAGS = {"superseded-es", "aborted"}
# Phase E runs of the submitted manuscript (Table 6)
PAPER_RUNS = {"dkggiaj1": "Clay v1.0", "80dh36vg": "Clay v1.5", "62ol6euh": "Prithvi",
              "bho779er": "DOFA", "ba0nku12": "TerraMind", "eih9va5p": "MiT-B2 U-Net"}
# batch size of the paper runs, not in their W&B config: from the repo Phase E configs
PAPER_BATCH_SIZE = {"MiT-B2 U-Net": 8}  # all GFMs: 4
MODEL_NAMES = [  # (pattern on backbone or run name, model)
    (r"clay15|clay[-_]?v1[._]5", "Clay v1.5"), (r"clay", "Clay v1.0"),
    (r"prithvi", "Prithvi"), (r"dofa", "DOFA"), (r"terramind", "TerraMind"),
    (r"resnet50", "ResNet-50 U-Net"), (r"mit_b2|mit-b2", "MiT-B2 U-Net"),
]


def model_of(run) -> str | None:
    c = run.config
    s = " ".join(str(x) for x in [(c.get("model_args") or {}).get("backbone"), run.name]).lower()
    for pat, name in MODEL_NAMES:
        if re.search(pat, s):
            return name
    return None


def decoder_of(run) -> str:
    c = run.config
    dec = (c.get("model_args") or {}).get("decoder")
    if dec:
        return dec
    return "UNet (SMP)" if c.get("model_factory") == "SMPModelFactory" else ""


def scheduler_of(run) -> str:
    c = run.config
    if c.get("scheduler") != "SequentialLR":
        return str(c.get("scheduler") or "none (constant)")
    sh = c.get("scheduler_hparams") or {}
    cos = (sh.get("schedulers") or {}).get("CosineAnnealingLR", {})
    warm = (sh.get("milestones") or [None])[0]
    return f"linear warm-up {warm} ep + cosine (T_max {cos.get('T_max')}, eta_min {cos.get('eta_min')})"


def val_history(run) -> pd.DataFrame:
    rows = [(r.get("epoch"), r.get("val/mIoU")) for r in
            run.scan_history(keys=["epoch", "val/mIoU"], page_size=1000)]
    df = pd.DataFrame(rows, columns=["epoch", "val_mIoU"]).dropna()
    return df.drop_duplicates("epoch", keep="last").sort_values("epoch")


def split_of(run) -> str:
    ex = run.config.get("exclusion") or {}
    if not ex:
        return "original"
    return {"mask": f"masked ({ex.get('buffer')} m buffer)",
            "drop_cells": "block drop"}.get(ex.get("mode"), str(ex))


def training_facts(api) -> pd.DataFrame:
    rows = []
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"tags": {"$in": ["revision-r1"]}}, per_page=200)
    for run in runs:
        tags = set(run.tags)
        if tags & SKIP_TAGS or run.state != "finished":
            print(f"skip {run.id} {run.name} state={run.state} tags={sorted(tags)}", flush=True)
            continue
        c = run.config
        hp = c.get("hyperparameters") or {}
        bs = hp.get("batch_size")
        h = val_history(run)
        best = h.loc[h["val_mIoU"].idxmax()]
        n_ep = int(h["epoch"].max()) + 1
        max_ep = hp.get("max_epochs")
        gstep = run.summary.get("trainer/global_step")
        md = run.metadata or {}
        row_id = next((t for t in run.tags if re.fullmatch(r"[TK]\d+", t)), "")
        rows.append({
            "runbook_id": row_id if row_id.startswith("K") and not c.get("seed") else f"{row_id}-s{c.get('seed')}",
            "wandb_id": run.id, "run_name": run.name, "model": model_of(run),
            "decoder": decoder_of(run), "factory": c.get("model_factory"),
            "split": split_of(run), "seed": c.get("seed"),
            "train_patches_per_epoch": hp.get("train_batches_per_epoch", 0) * bs,
            "val_patches_per_epoch": hp.get("val_batches_per_epoch", 0) * bs,
            "train_steps_per_epoch_logged": (gstep / n_ep) if gstep else None,
            "epochs_run": n_ep, "max_epochs": max_ep,
            "best_epoch": int(best["epoch"]), "best_val_mIoU": round(float(best["val_mIoU"]), 4),
            "last_val_mIoU": round(float(h["val_mIoU"].iloc[-1]), 4),
            "stop_reason": ("max_epochs (fixed budget, no early stopping)" if n_ep == max_ep
                            else f"stopped early at epoch {n_ep} (patience {hp.get('patience')})"),
            "batch_size": bs, "lr": c.get("lr"),
            "weight_decay": (c.get("optimizer_hparams") or {}).get("weight_decay"),
            "scheduler": scheduler_of(run),
            "final_lr_logged": run.summary.get("lr-AdamW"),
            "wall_time_h": round(run.summary.get("_runtime", 0) / 3600, 2),
            "host": md.get("host"), "gpu": md.get("gpu"),
            "started_utc": md.get("startedAt"),
        })
        print(f"{run.id} {run.name}: {n_ep} ep, best {int(best['epoch'])} "
              f"{best['val_mIoU']:.4f}", flush=True)
    return pd.DataFrame(rows).sort_values(["model", "split", "decoder", "seed", "run_name"])


def phase_of(project: str, name: str, created: str) -> str:
    if project == PRELIM_PROJECT:
        return "preliminary (2025)" if created < "2026" else "preliminary (2026)"
    if name.startswith("r1-"):
        return "revision"
    m = re.search(r"phase([A-G])", name)
    if m:
        return f"Phase {m.group(1)}"
    if name.endswith("[ref]"):
        return "2025 run re-logged as reference"
    if name.endswith("full-1"):
        return "Phase B" if "ndsm" in name else "Phase A"
    return "other"


def lr_table(api) -> pd.DataFrame:
    rows = []
    for project in [PRELIM_PROJECT, PROJECT]:
        for run in api.runs(f"{ENTITY}/{project}", per_page=200):
            if set(run.tags) & SKIP_TAGS or run.state != "finished":
                continue
            model = model_of(run)
            if model is None or run.config.get("lr") is None:
                continue
            h = val_history(run)
            if h.empty:
                continue
            best = h.loc[h["val_mIoU"].idxmax()]
            bands = (run.config.get("model_args") or {}).get("bands")
            hp = run.config.get("hyperparameters") or {}
            bs = hp.get("batch_size")
            if bs is None and run.id in PAPER_RUNS:
                bs = PAPER_BATCH_SIZE.get(model, 4)
            n_ep = int(h["epoch"].max()) + 1
            gstep = run.summary.get("trainer/global_step")
            rows.append({
                "model": model, "phase": phase_of(project, run.name, run.created_at),
                "created": run.created_at[:10], "project": project, "wandb_id": run.id,
                "run_name": run.name, "state": run.state,
                "paper_run": "yes" if run.id in PAPER_RUNS else "",
                "decoder": decoder_of(run), "bands": " ".join(map(str, bands or [])),
                "lr": run.config.get("lr"), "scheduler": scheduler_of(run),
                "batch_size": bs,
                "train_steps_per_epoch_logged": round(gstep / n_ep) if gstep else None,
                "epochs_run": n_ep, "best_epoch": int(best["epoch"]),
                "best_val_mIoU": round(float(best["val_mIoU"]), 4),
                "test_mIoU_last_epoch": run.summary.get("test/mIoU"),
            })
            print(f"lr {project[-8:]} {run.id} {run.name}", flush=True)
    return pd.DataFrame(rows).sort_values(["model", "created", "run_name"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-lr-table", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=120)
    facts = training_facts(api)
    facts.to_csv(OUT / "b8_training_facts.csv", index=False)
    print(facts.drop(columns=["scheduler", "started_utc"]).to_string(), flush=True)
    if not args.no_lr_table:
        lrs = lr_table(api)
        lrs.to_csv(OUT / "b8_lr_table.csv", index=False)
        print(lrs.groupby(["model", "lr"]).size().to_string())


if __name__ == "__main__":
    sys.exit(main())
