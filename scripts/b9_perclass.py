#!/usr/bin/env python
"""Per-class change-detection metrics for Clay v1.0, revision runs.

Masked split, best checkpoint, seeds 0 to 2. Per seed: IoU, F1, recall and precision per change
class from the seed's 9 x 9 confusion matrix (sum of the 88 tile matrices in b9_tile_cms.npz);
the table reports the mean over the three seeds, so the macro IoU equals the seed mean in
b9_summary.csv. Area share from the reference row sums (identical for all seeds).

Output: outputs/r1/b9_clay_v1_cd_perclass.csv (read by make_r1_tables.py, table b9_clay_perclass).
Run: .venv314/bin/python scripts/b9_perclass.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
R1 = BASE / "outputs/r1"
CLASSES = ["No change", "Mature tree density loss", "Old-growth density loss",
           "Forest setback / young loss", "Forest stage progression", "Forest density gain",
           "Early forest establishment", "New clearcut", "Other transition"]


def per_class(cm: np.ndarray) -> dict[str, np.ndarray]:
    tp = np.diag(cm)
    ref, pred = cm.sum(1), cm.sum(0)
    rec, prec = tp / ref, tp / pred
    return {"IoU": tp / (ref + pred - tp), "F1": 2 * prec * rec / (prec + rec),
            "recall": rec, "precision": prec}


def main() -> None:
    d = np.load(R1 / "b9_tile_cms.npz")
    cms = [d[f"r1-clay-v1-masked-s{s}__cd"].sum(0).astype(float) for s in range(3)]
    assert len({c.sum(1).tobytes() for c in cms}) == 1  # same reference in every seed
    stats = [per_class(c) for c in cms]
    mean = {k: np.mean([s[k] for s in stats], 0) for k in stats[0]}
    area = cms[0].sum(1) / cms[0].sum() * 100

    # sanity: seed-mean macro IoU equals b9_summary
    with open(R1 / "b9_summary.csv") as fh:
        ref = next(r for r in csv.DictReader(fh) if r["model_key"] == "clay-v1" and
                   r["split"] == "masked" and r["ckpt"] == "best" and r["metric"] == "cd_mc_IoU")
    assert abs(mean["IoU"].mean() - float(ref["mean"])) < 1e-6, (mean["IoU"].mean(), ref["mean"])

    out = R1 / "b9_clay_v1_cd_perclass.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "area_pct", "IoU", "F1", "recall", "precision"])
        for i, c in enumerate(CLASSES):
            w.writerow([c, area[i]] + [mean[k][i] for k in ("IoU", "F1", "recall", "precision")])
        w.writerow(["Macro average", ""] + [mean[k].mean() for k in ("IoU", "F1", "recall", "precision")])
    print(f"wrote {out}; macro IoU {mean['IoU'].mean():.4f} (b9_summary {float(ref['mean']):.4f})")


if __name__ == "__main__":
    main()
