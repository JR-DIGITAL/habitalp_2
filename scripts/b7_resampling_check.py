#!/usr/bin/env python
"""B7 (R2-M5): check the 0.5 m to 0.2 m resampling path of the evaluation on real data.

The predictions are written at 0.5 m; the reference rasters are at 0.19990 m. The
evaluation (compute_final_metrics, read_intersecting_area_of_two_rasters) reads the
reference over the intersection window, whose fractional offset GDAL rounds to whole
pixels, and resamples the prediction with nearest neighbour onto the window transform,
whose origin is the prediction origin. The prediction is therefore sampled up to half
a reference pixel away from the reference pixel centre. This script measures the effect
on the full 2020 test area for one model: it rebuilds the code-path arrays, checks them
against the stored change map, and compares with an exact nearest-neighbour sampling at
the reference pixel centres.

Run: nice -n 19 python scripts/b7_resampling_check.py [--experiment clay-v1-rgb-nir-ndsm-phaseE]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

BASE_FOLDER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_FOLDER))
from src.trainers.bootstrap import collapse_binary, macro_iou_f1
from src.trainers.utils import get_change_array

DATA_ROOT = Path(os.environ.get(
    "HABITALP_DATA_ROOT", BASE_FOLDER / "data/Originaldaten"))
REF_2013 = DATA_ROOT / "processed/mask/classes_v4_2013.tif"
REF_2020 = DATA_ROOT / "processed/mask/classes_v4_2020.tif"
REF_CT = DATA_ROOT / "habitalp_change/habitalp_change_v4_2013-2020.tif"
OUT = BASE_FOLDER / "outputs/r1"
STRIP = 512


def inter_window(a, b):
    l, bt = max(a.bounds.left, b.bounds.left), max(a.bounds.bottom, b.bounds.bottom)
    r, t = min(a.bounds.right, b.bounds.right), min(a.bounds.top, b.bounds.top)
    return a.window(l, bt, r, t), b.window(l, bt, r, t)


def pred_index(x0: float, y0: float, n_c: int, n_r: int, pr) -> tuple[np.ndarray, np.ndarray]:
    """Prediction column/row sampled by nearest neighbour at the centres of a grid with
    origin (x0, y0) and the reference pixel size."""
    res = REF_CT_RES
    cols = np.floor((x0 + (np.arange(n_c) + 0.5) * res - pr.bounds.left) / pr.res[0]).astype(int)
    rows = np.floor((pr.bounds.top - (y0 - (np.arange(n_r) + 0.5) * res)) / pr.res[1]).astype(int)
    return cols, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="clay-v1-rgb-nir-ndsm-phaseE")
    args = ap.parse_args()
    folder = DATA_ROOT / "model_output" / args.experiment
    pred_path = folder / f"2020_{args.experiment}.tif"
    cm_path = folder / "r1_dwarffix/change_map_2013-2020.tif"

    global REF_CT_RES
    with (rasterio.open(REF_CT) as ct, rasterio.open(REF_2020) as r20,
          rasterio.open(REF_2013) as r13, rasterio.open(pred_path) as pr,
          rasterio.open(cm_path) as cm):
        REF_CT_RES = ct.res[0]
        assert ct.transform == r20.transform and ct.shape == r20.shape
        res = ct.res[0]
        info = {"pred_res": pr.res[0], "ref_res": res, "pred_nodata": pr.nodata,
                "ref2020_nodata": r20.nodata, "refct_nodata": ct.nodata}
        # 2013 class raster on the 2020 grid (integer offset, checked)
        off13 = ((ct.bounds.left - r13.bounds.left) / res, (r13.bounds.top - ct.bounds.top) / res)
        assert all(abs(o - round(o)) < 1e-6 for o in off13), off13
        c13, r13o = (int(round(o)) for o in off13)

        # code path, segmentation 2020: compute_final_metrics(REF_2020, pred)
        wref_seg, _ = inter_window(r20, pr)
        # code path, change map: generate_change_map(REF_2013, pred), then
        # compute_final_metrics(REF_CT, change map)
        wref_cd, wcm = inter_window(ct, cm)
        info.update({"seg_ref_window_off": (wref_seg.col_off, wref_seg.row_off),
                     "cd_ref_window_off": (wref_cd.col_off, wref_cd.row_off),
                     "cd_changemap_window_off": (wcm.col_off, wcm.row_off)})
        cs, rs = int(round(wref_seg.col_off)), int(round(wref_seg.row_off))
        cc, rc = int(round(wref_cd.col_off)), int(round(wref_cd.row_off))
        cmc, cmr = int(round(wcm.col_off)), int(round(wcm.row_off))
        # sampling grids: origin of each destination grid, in reference pixel units
        seg_x0, seg_y0 = max(ct.bounds.left, pr.bounds.left), min(ct.bounds.top, pr.bounds.top)
        cdx0, cdy0 = cm.bounds.left, cm.bounds.top
        info["seg_shift_px"] = ((seg_x0 - ct.bounds.left) / res - cs,
                                (ct.bounds.top - seg_y0) / res - rs)
        info["cd_shift_px"] = ((cdx0 - ct.bounds.left) / res - (cc - cmc),
                               (ct.bounds.top - cdy0) / res - (rc - cmr))

        W = min(ct.width - max(cs, cc), cm.width)
        c0 = max(cs, cc)
        # exact: prediction at the reference pixel centres
        ex_cols, _ = pred_index(ct.bounds.left + c0 * res, 0, W, 0, pr)
        sg_cols, _ = pred_index(seg_x0 + (c0 - cs) * res, 0, W, 0, pr)
        cd_cols, _ = pred_index(cdx0 + (c0 - cc + cmc) * res, 0, W, 0, pr)

        n_seg, n_cd = 24, 9
        cms = {k: np.zeros((n, n), np.int64) for k, n in
               [("seg_code", n_seg), ("seg_exact", n_seg), ("cd_code", n_cd), ("cd_exact", n_cd),
                ("cd_stored", n_cd)]}
        stats = {"cd_code_vs_stored_px_differ": 0, "cd_valid_px": 0,
                 "cd_px_differ_code_vs_exact": 0, "seg_valid_px": 0,
                 "seg_px_differ_code_vs_exact": 0, "ref_valid_pred_nodata_px": 0}
        H = min(ct.height, cm.height - cmr + rc)
        for r0 in range(0, H, STRIP):
            h = min(STRIP, H - r0)
            ref_ct = ct.read(1, window=Window(c0, r0, W, h))
            if not (ref_ct != 255).any():
                continue
            ref20 = r20.read(1, window=Window(c0, r0, W, h))
            a13 = r13.read(1, window=Window(c13 + c0, r13o + r0, W, h))
            rows_ex = np.floor((pr.bounds.top - (ct.bounds.top - (r0 + np.arange(h) + 0.5) * res))
                               / pr.res[1]).astype(int)
            rows_sg = np.floor((pr.bounds.top - (seg_y0 - (r0 - rs + np.arange(h) + 0.5) * res))
                               / pr.res[1]).astype(int)
            rows_cd = np.floor((pr.bounds.top - (cdy0 - (r0 - rc + cmr + np.arange(h) + 0.5) * res))
                               / pr.res[1]).astype(int)
            rmin = max(0, min(rows_ex.min(), rows_sg.min(), rows_cd.min()))
            rmax = min(pr.height - 1, max(rows_ex.max(), rows_sg.max(), rows_cd.max()))
            cmin = max(0, min(ex_cols.min(), sg_cols.min(), cd_cols.min()))
            cmax = min(pr.width - 1, max(ex_cols.max(), sg_cols.max(), cd_cols.max()))
            pblk = pr.read(1, window=Window(cmin, rmin, cmax - cmin + 1, rmax - rmin + 1))

            def sample(rr, cols):
                ok_r = (rr >= rmin) & (rr <= rmax)
                ok_c = (cols >= cmin) & (cols <= cmax)
                out = np.zeros((len(rr), len(cols)), np.uint8)
                out[np.ix_(ok_r, ok_c)] = pblk[np.ix_(rr[ok_r] - rmin, cols[ok_c] - cmin)]
                return out

            p_ex, p_sg, p_cd = sample(rows_ex, ex_cols), sample(rows_sg, sg_cols), sample(rows_cd, cd_cols)
            # segmentation 2020 (ignore_index 0 on both)
            v = (ref20 != 0) & (p_ex != 0) & (p_sg != 0)
            for k, p in [("seg_code", p_sg), ("seg_exact", p_ex)]:
                cms[k] += np.bincount(ref20[v].astype(np.int64) * n_seg + p[v],
                                      minlength=n_seg * n_seg).reshape(n_seg, n_seg)
            stats["seg_valid_px"] += int(v.sum())
            stats["seg_px_differ_code_vs_exact"] += int((p_sg[v] != p_ex[v]).sum())
            stats["ref_valid_pred_nodata_px"] += int(((ref20 != 0) & (p_ex == 0)).sum())
            # change detection (ignore_index 255 on both)
            ch_code = get_change_array(a13, p_cd, in_nodata=0, out_nodata=255)
            ch_ex = get_change_array(a13, p_ex, in_nodata=0, out_nodata=255)
            stored = cm.read(1, window=Window(c0 - cc + cmc, r0 - rc + cmr, W, h))
            vv = (ref_ct != 255) & (ch_code != 255) & (ch_ex != 255)
            stats["cd_valid_px"] += int(vv.sum())
            stats["cd_px_differ_code_vs_exact"] += int((ch_code[vv] != ch_ex[vv]).sum())
            stats["cd_code_vs_stored_px_differ"] += int((ch_code[vv] != stored[vv]).sum())
            for k, p in [("cd_code", ch_code), ("cd_exact", ch_ex), ("cd_stored", stored)]:
                cms[k] += np.bincount(ref_ct[vv].astype(np.int64) * n_cd + p[vv],
                                      minlength=n_cd * n_cd).reshape(n_cd, n_cd)

    rows = []
    for k, cmx in cms.items():
        c = cmx[1:, 1:] if k.startswith("seg") else cmx
        iou, f1 = macro_iou_f1(c)
        row = {"what": k, "macro_IoU": iou, "macro_F1": f1,
               "pixel_agreement": np.trace(c) / c.sum()}
        if k.startswith("cd"):
            row["binary_macro_IoU"] = macro_iou_f1(collapse_binary(cmx))[0]
        rows.append(row)
    df = pd.DataFrame(rows)
    print(info)
    print(stats)
    print(df.to_string())
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "b7_resampling_check.csv", index=False)
    pd.DataFrame([{**{k: str(v) for k, v in info.items()}, **stats}]).to_csv(
        OUT / "b7_resampling_check_info.csv", index=False)


if __name__ == "__main__":
    main()
