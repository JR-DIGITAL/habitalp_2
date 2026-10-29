#!/usr/bin/env python
"""B5 (R1-M4, R2-M6): change-detection post-processing step by step.

From existing rasters of the Phase E paper runs (no retraining), for Clay v1.0 and
the MiT-B2 U-Net, stages:

  raw          model only: B0 change map (reference 2013 -> predicted 2020, dwarf-forest fix)
  rule1..rule6 the plausibility filter with one rule only
  all_rules    the filter with all six rules (pixel stage before vectorisation)
  polygon      vectorisation + canopy-cover update (IMPACT output, B0 "after" map)

The filter (physical_constraints_module in src/inference/utils.py) resets every violating
2020 prediction to the 2013 reference class, i.e. to "no change" in the change map. The rule
masks it exported (constraint_resolution/constraint_violation_<k>_*.tif) are reused here, so
a rule-k stage is the raw change map with the rule-k pixels set to 0; no rule is
reimplemented. Checks: the union of the six masks equals constraint_violation_total.tif, and
the all-rules map equals generate_change_map(2013 reference, the stored filtered 2020
prediction); raw and polygon confusion matrices equal the B0 ones.

Evaluation as B0/B7: 0.2 m, reference change map v4, prediction window read on the
intersection (nearest, load_aligned_2d), 9 classes and binary (collapse_binary).

Also:
  - rule-3 cost: area reset by rule 3, split by the reference into no change / other
    real change / true stage regression (forest stage 2013 > stage 2020 in the reference
    class rasters); setback (class 3) precision and recall before and after.
  - share of the evaluated area flagged by each rule and by all rules, all 12 runs
    (manuscript: "up to ~9% of the area").
  - polygon-level rates on connected components (8-connected, >= MIN_M2):
    filter: share of predicted change patches (raw map) the filter touches / mostly resets;
    detection: share of reference change patches per class with >= 50% of their area
    predicted as change (any class) and as the correct class, per stage.

Outputs: outputs/r1/b5_steps.csv, b5_rule3_cost.csv, b5_area_share.csv, b5_polygons.csv;
filtered change maps in <model_output>/<experiment>/r1_b5/ (never overwritten).

Run: nice -n 19 python scripts/b5_postproc_steps.py [--workers 3]
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.coords
from rasterio.windows import Window, from_bounds
from scipy import ndimage

BASE_FOLDER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_FOLDER))
from scripts.b0_dwarffix_cd import (  # noqa: E402
    MC_NAMES, MO, NEW_DIR, OUT, REF_2013, REF_2020, REF_CT,
)
from src.inference.eval import generate_change_map  # noqa: E402
from src.trainers.bootstrap import collapse_binary, macro_iou_f1  # noqa: E402

FULL = {"clay-v1-rgb-nir-ndsm-phaseE": "Clay v1.0", "unet-mit-b2-rgb-nir-ndsm-phaseE": "MiT-B2 U-Net"}
SHARE_ONLY = [f"{m}-rgb-nir-{p}" for p in ["ndsm-phaseE", "phaseF"]
              for m in ["clay-v1", "clay-v1.5", "prithvi-v2-300", "dofa-base", "terramind", "unet-mit-b2"]]
RULES = {
    1: "non_forest_to_mature_forest", 2: "young_growth_to_old_forest",
    3: "forest_setback_to_younger_stage", 4: "rock_to_grassland",
    5: "water_bodies_on_steep_slopes", 6: "alpine_vegetation_in_lowland",
}
SHORT = ["nochange", "mature_dens_loss", "old_dens_loss", "setback", "progression",
         "dens_gain", "early_estab", "new_clearcut", "other"]
B5_DIR = "r1_b5"
ROWS = 2000  # row chunk for confusion matrices and component statistics
MIN_M2 = 100.0  # minimum patch size for the polygon-level rates
# forest stage per class (as get_change_array): young 6-7, pole 8-10, mature 11-14, old 15-18
STAGE = np.zeros(256, dtype=np.uint8)
STAGE[[6, 7]], STAGE[[8, 9, 10]], STAGE[11:15], STAGE[15:19] = 1, 2, 3, 4


def intersection(a, b) -> rasterio.coords.BoundingBox:
    return rasterio.coords.BoundingBox(
        left=max(a.left, b.left), bottom=max(a.bottom, b.bottom),
        right=min(a.right, b.right), top=min(a.top, b.top))


def read_on(path: Path, inter, shape=None) -> np.ndarray:
    """Read the intersection window as load_aligned_2d does (uint8, cropped to shape)."""
    with rasterio.open(path) as s:
        a = s.read(1, window=s.window(*inter))
    if shape is not None:
        a = a[:shape[0], :shape[1]]
    return a


def aligned_pair(path: Path) -> tuple[np.ndarray, np.ndarray, object]:
    """Reference change map and a prediction on their intersection (= load_aligned_2d)."""
    with rasterio.open(REF_CT) as m, rasterio.open(path) as p:
        assert m.res == p.res
        inter = intersection(m.bounds, p.bounds)
    ref, prd = read_on(REF_CT, inter), read_on(path, inter)
    h, w = min(ref.shape[0], prd.shape[0]), min(ref.shape[1], prd.shape[1])
    return ref[:h, :w], prd[:h, :w], inter


def confusion(ref: np.ndarray, prd: np.ndarray, reset: np.ndarray | None = None, n: int = 9) -> np.ndarray:
    """9x9 CM over pixels where both are valid; pixels in `reset` count as predicted 0."""
    cm = np.zeros(n * n, dtype=np.int64)
    for r0 in range(0, ref.shape[0], ROWS):
        r = ref[r0:r0 + ROWS].ravel()
        p = prd[r0:r0 + ROWS].ravel()
        valid = (r != 255) & (p != 255)
        if reset is not None:
            p = np.where(reset[r0:r0 + ROWS].ravel(), 0, p)
        cm += np.bincount(r[valid].astype(np.int64) * n + p[valid], minlength=n * n)
    return cm.reshape(n, n)


def metrics_row(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (cm.sum(0) + cm.sum(1) - tp)
        prec = tp / cm.sum(0)
        rec = tp / cm.sum(1)
    b = collapse_binary(cm)
    btp = np.diag(b).astype(float)
    d = {"pixels_eval": int(cm.sum()),
         "mc_IoU": macro_iou_f1(cm)[0], "mc_F1": macro_iou_f1(cm)[1], "mc_BA": float(np.nanmean(rec)),
         "bin_IoU": macro_iou_f1(b)[0], "bin_F1": macro_iou_f1(b)[1],
         "bin_BA": float(np.mean(btp / b.sum(1))),
         "change_recall": btp[1] / b[1].sum(), "change_precision": btp[1] / b[:, 1].sum(),
         "pred_change_ha": None}
    for c, s in enumerate(SHORT):
        d[f"{s}_IoU"], d[f"{s}_precision"], d[f"{s}_recall"] = iou[c], prec[c], rec[c]
        d[f"{s}_omission"] = 1 - rec[c]
    return d


def patch_stats(labels: np.ndarray, n: int, masks: dict) -> dict[str, np.ndarray]:
    """Per-component pixel counts: size and count of pixels in each mask. Masks are
    functions of a row slice returning a boolean chunk (built chunk-wise to save memory)."""
    out = {k: np.zeros(n + 1, dtype=np.int64) for k in ["size", *masks]}
    for r0 in range(0, labels.shape[0], ROWS):
        sl = slice(r0, r0 + ROWS)
        lab = labels[sl].ravel()
        out["size"] += np.bincount(lab, minlength=n + 1)
        for k, f in masks.items():
            out[k] += np.bincount(lab[f(sl).ravel()], minlength=n + 1)
    return out


def share_job(exp: str) -> list[dict]:
    """Area flagged by each rule (and the union) inside the evaluated area."""
    raw_path = MO / exp / NEW_DIR / "change_map_2013-2020.tif"
    ref, raw, inter = aligned_pair(raw_path)
    valid = (ref != 255) & (raw != 255)
    del ref, raw
    px_ha = abs(rasterio.open(REF_CT).res[0]) ** 2 / 1e4
    row = {"experiment": exp, "eval_ha": valid.sum() * px_ha}
    cr = MO / exp / "constraint_resolution"
    for k, name in RULES.items():
        v = read_on(cr / f"constraint_violation_{k}_{name}.tif", inter, valid.shape) == 1
        row[f"rule{k}_ha"] = int((v & valid).sum()) * px_ha
    v = read_on(cr / "constraint_violation_total.tif", inter, valid.shape) == 1
    row["all_rules_ha"] = int((v & valid).sum()) * px_ha
    for k in [*[f"rule{i}" for i in RULES], "all_rules"]:
        row[f"{k}_pct"] = 100 * row[f"{k}_ha"] / row["eval_ha"]
    print(f"share {exp}: all rules {row['all_rules_pct']:.1f}%, rule 3 {row['rule3_pct']:.1f}%", flush=True)
    return [row]


def full_job(exp: str) -> dict:
    model = FULL[exp]
    folder = MO / exp
    cr = folder / "constraint_resolution"
    out_dir = folder / B5_DIR
    out_dir.mkdir(exist_ok=True)
    with rasterio.open(REF_CT) as s:
        px_ha = abs(s.res[0]) ** 2 / 1e4
        ct_transform, ct_bounds = s.transform, s.bounds

    # --- raw and per-rule stages (rule masks share the raw change map's grid) ---
    ref, raw, inter = aligned_pair(folder / NEW_DIR / "change_map_2013-2020.tif")
    shape = ref.shape
    valid = (ref != 255) & (raw != 255)
    cms = {"raw": confusion(ref, raw)}
    resets = {}
    union = np.zeros(shape, dtype=bool)
    for k, name in RULES.items():
        v = read_on(cr / f"constraint_violation_{k}_{name}.tif", inter, shape) == 1
        cms[f"rule{k}"] = confusion(ref, raw, reset=v)
        resets[f"rule{k}"] = int((v & valid).sum())
        union |= v
        if k == 3:
            v3 = v
        del v
    total = read_on(cr / "constraint_violation_total.tif", inter, shape) == 1
    check_union = int((total != union).sum())
    del total
    cms["all_rules"] = confusion(ref, raw, reset=union)
    resets["all_rules"] = int((union & valid).sum())

    # canonical filtered change map from the stored filtered 2020 prediction
    filt_map = out_dir / "change_map_2013-2020_filter.tif"
    if not filt_map.exists():
        generate_change_map(mask_path=REF_2013, prediction_path=cr / "prediction_2020_constraint_resolution.tif",
                            output_path=filt_map)
    filt = read_on(filt_map, inter, shape)
    composed_diff = 0
    for r0 in range(0, shape[0], ROWS):
        comp = np.where(union[r0:r0 + ROWS], 0, raw[r0:r0 + ROWS])
        f = filt[r0:r0 + ROWS]
        both = (ref[r0:r0 + ROWS] != 255) & (comp != 255)
        composed_diff += int(((comp != f) & both).sum())
    del filt

    # --- polygon stage: B0 "after" map with its own alignment (reproduces B0) ---
    pp_path = folder / NEW_DIR / "change_map_2013-2020_after_postprocessing.tif"
    ref_pp, pp_full, _ = aligned_pair(pp_path)
    cms["polygon"] = confusion(ref_pp, pp_full)
    del ref_pp, pp_full
    with rasterio.open(pp_path) as s:
        assert s.transform.a == ct_transform.a and s.bounds == ct_bounds
    pp = read_on(pp_path, inter, shape)  # same window as the raw stages, for the patch rates

    # --- B0 cross-check ---
    b0 = {}
    for st, sfx in [("raw", ""), ("polygon", "_after_postprocessing")]:
        cm_b0 = pd.read_csv(folder / NEW_DIR / f"confusion_matrix_change_9x9{sfx}.csv", index_col=0).to_numpy()
        b0[st] = int(np.abs(cm_b0 - cms[st]).sum())

    rows = []
    for st, cm in cms.items():
        d = {"model": model, "experiment": exp, "stage": st, **metrics_row(cm)}
        d["pred_change_ha"] = cm[:, 1:].sum() * px_ha
        d["reset_ha"] = resets.get(st, np.nan) * px_ha if st in resets else (0.0 if st == "raw" else np.nan)
        d["b0_cm_abs_diff"] = b0.get(st, np.nan)
        rows.append(d)
    steps = pd.DataFrame(rows)
    steps["check_union_vs_total_px"] = check_union
    steps["check_allrules_vs_stored_px"] = composed_diff

    # --- rule-3 cost ---
    with rasterio.open(REF_2013) as s13, rasterio.open(REF_2020) as s20:
        assert s20.transform == ct_transform and s20.shape == rasterio.open(REF_CT).shape
        w13 = from_bounds(*ct_bounds, transform=s13.transform).round_offsets().round_lengths()
        ct_win = rasterio.open(REF_CT).window(*inter)
        ro, co = round(ct_win.row_off), round(ct_win.col_off)
        with rasterio.open(REF_CT) as sct:  # same pixels as the aligned reference window
            assert np.array_equal(sct.read(1, window=Window(co, ro, shape[1], shape[0])), ref)
        a13 = s13.read(1, window=Window(w13.col_off + co, w13.row_off + ro, shape[1], shape[0]))
        a20 = s20.read(1, window=Window(co, ro, shape[1], shape[0]))
    true_reg = (STAGE[a13] > STAGE[a20]) & (STAGE[a20] > 0)
    del a13, a20
    m3 = v3 & valid
    ref_true_reg_ha = int((true_reg & valid).sum()) * px_ha
    reset_nochange = int((m3 & (ref == 0)).sum())
    reset_true = int((m3 & true_reg).sum())
    reset_other = int(m3.sum()) - reset_nochange - reset_true
    # true-regression pixels predicted as setback, before and after rule 3
    tr_pred3_raw = int((true_reg & valid & (raw == 3)).sum())
    tr_pred3_after = int((true_reg & valid & (raw == 3) & ~v3).sum())
    rule3 = {"model": model, "experiment": exp,
             "eval_ha": int(valid.sum()) * px_ha,
             "reset_ha": int(m3.sum()) * px_ha,
             "reset_pct_of_eval": 100 * m3.sum() / valid.sum(),
             "reset_nochange_ha": reset_nochange * px_ha,
             "reset_other_real_change_ha": reset_other * px_ha,
             "reset_true_regression_ha": reset_true * px_ha,
             "ref_setback_ha": cms["raw"][3].sum() * px_ha,
             "ref_true_regression_ha": ref_true_reg_ha,
             "true_regression_pred_setback_raw_ha": tr_pred3_raw * px_ha,
             "true_regression_pred_setback_after_rule3_ha": tr_pred3_after * px_ha}
    # what the reset pixels are in the reference, by change class
    ref_cls = np.bincount(ref[m3], minlength=256)[:9]
    for c, s in enumerate(SHORT):
        rule3[f"reset_ref_{s}_ha"] = ref_cls[c] * px_ha
    for st in ["raw", "rule3", "all_rules", "polygon"]:
        cm = cms[st]
        rule3[f"setback_precision_{st}"] = cm[3, 3] / cm[:, 3].sum()
        rule3[f"setback_recall_{st}"] = cm[3, 3] / cm[3].sum()
        rule3[f"setback_IoU_{st}"] = cm[3, 3] / (cm[:, 3].sum() + cm[3].sum() - cm[3, 3])
        rule3[f"pred_setback_ha_{st}"] = cm[:, 3].sum() * px_ha

    # --- polygon-level rates ---
    min_px = int(np.ceil(MIN_M2 / (px_ha * 1e4)))
    struct = np.ones((3, 3), dtype=bool)
    poly_rows = []
    # (a) predicted change patches in the raw map and what the filter does to them
    labels, n = ndimage.label(valid & (raw > 0) & (raw < 9), structure=struct)
    st_ = patch_stats(labels, n, {"all_rules": lambda sl: union[sl], "rule3": lambda sl: v3[sl]})
    keep = st_["size"][1:] >= min_px
    for k in ["all_rules", "rule3"]:
        frac = st_[k][1:][keep] / st_["size"][1:][keep]
        poly_rows.append({"model": model, "experiment": exp, "kind": f"filter_{k}", "class": "any change",
                          "n_patches": int(keep.sum()),
                          "share_touched": float((frac > 0).mean()),
                          "share_ge50": float((frac >= 0.5).mean()),
                          "share_removed_all": float((frac >= 1).mean()),
                          "area_share": float(st_[k][1:][keep].sum() / st_["size"][1:][keep].sum())})
    del labels
    # (b) reference change patches per class: share detected (any change / correct class) per stage
    def chg(a):
        return (a > 0) & (a < 9)
    masks = {
        "raw_det": lambda sl: chg(raw[sl]),
        "rule3_det": lambda sl: chg(raw[sl]) & ~v3[sl],
        "all_rules_det": lambda sl: chg(raw[sl]) & ~union[sl],
        "polygon_det": lambda sl: chg(pp[sl]),
    }
    for c in range(1, 9):
        cls = {
            "raw_cls": lambda sl, c=c: raw[sl] == c,
            "rule3_cls": lambda sl, c=c: (raw[sl] == c) & ~v3[sl],
            "all_rules_cls": lambda sl, c=c: (raw[sl] == c) & ~union[sl],
            "polygon_cls": lambda sl, c=c: pp[sl] == c,
        }
        labels, n = ndimage.label(valid & (ref == c), structure=struct)
        st_ = patch_stats(labels, n, {**masks, **cls})
        del labels
        keep = st_["size"][1:] >= min_px
        for st in ["raw", "rule3", "all_rules", "polygon"]:
            sz = st_["size"][1:][keep]
            fd = st_[f"{st}_det"][1:][keep] / sz
            fc = st_[f"{st}_cls"][1:][keep] / sz
            ok = bool(keep.any())
            poly_rows.append({"model": model, "experiment": exp, "kind": f"reference_{st}",
                              "class": MC_NAMES[c], "n_patches": int(keep.sum()),
                              "share_touched": float((fd > 0).mean()) if ok else np.nan,
                              "share_ge50": float((fd >= 0.5).mean()) if ok else np.nan,
                              "share_correct_ge50": float((fc >= 0.5).mean()) if ok else np.nan,
                              "area_share": float(st_[f"{st}_det"][1:][keep].sum() / sz.sum()) if ok else np.nan})
    print(f"done {exp}: union check {check_union} px, stored-filter check {composed_diff} px, "
          f"B0 CM diff {b0}", flush=True)
    return {"steps": steps, "rule3": pd.DataFrame([rule3]), "poly": pd.DataFrame(poly_rows)}


def run(job):
    kind, exp = job
    return kind, exp, (full_job(exp) if kind == "full" else share_job(exp))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [("full", e) for e in FULL] + [("share", e) for e in SHARE_ONLY]
    res = {"steps": [], "rule3": [], "poly": [], "share": []}
    with Pool(args.workers, maxtasksperchild=1) as pool:
        for kind, exp, r in pool.imap_unordered(run, jobs):
            if kind == "full":
                for k in ["steps", "rule3", "poly"]:
                    res[k].append(r[k])
            else:
                res["share"] += r
    pd.concat(res["steps"]).to_csv(OUT / "b5_steps.csv", index=False)
    pd.concat(res["rule3"]).to_csv(OUT / "b5_rule3_cost.csv", index=False)
    pd.concat(res["poly"]).to_csv(OUT / "b5_polygons.csv", index=False)
    pd.DataFrame(res["share"]).sort_values("experiment").to_csv(OUT / "b5_area_share.csv", index=False)
    print("wrote outputs/r1/b5_*.csv")


if __name__ == "__main__":
    main()
