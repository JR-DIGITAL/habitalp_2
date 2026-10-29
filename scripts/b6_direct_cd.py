#!/usr/bin/env python
"""B6 (R2-M4): score the direct change-detection maps under the current evaluation.

Maps in <model_output>/direct_change/2013_2020/ (2025 runs, RGB image pairs 2013/2020):

  - ChangeViT, binary (values 0/1)
  - U-Net Change, MiT-B2 encoder, binary (values 0/1)
  - U-Net Change, EfficientNet-B4 encoder, 9 change classes (values 0..8)

All three store nodata as 0, so 0 is both "no change" and "no prediction". We separate it
with the test-area mask: only pixels where the reference change map
(habitalp_change_v4_2013-2020.tif) is valid are scored, and there 0 means "no change".
The two binary maps are on a grid shifted by about 0.32 px (6 cm) against the reference
and have a slightly different pixel size, so they are resampled to the reference grid with
nearest neighbour, as compute_final_metrics does for any resolution mismatch.

The 9-class direct model was trained on the 2003-2013 labels, whose class codes 2..8 are a
permutation of the v4 codes (same 8 transition classes, the "old class order" of
habitalp_change_2013_2020_old_class_order.tif). Its map is therefore recoded to the v4 order
(OLD_TO_V4) before scoring; this is the same as recoding the v4 reference to the old order,
and it leaves the binary (change/no change) scores unchanged.

The AGIT v2 Table 2 direct rows reproduce to four decimals against the earlier reference
habitalp_change_2013_2020.tif (smaller change share; its post-classification rows reproduce
against v4), and its "U-Net Change binary" row is the EfficientNet-B4 9-class map collapsed
to binary, not the MiT-B2 binary model (ChangeViT 0.3626, EfficientNet-B4
collapsed 0.3822, 9 classes against the old-order reference 0.1037).

Same metric code as B0: compute_final_metrics (torchmetrics, ignore_index 255) for the
point values, written to <model_output>/direct_change/2013_2020/r1_b6/; tile confusion
matrices on the B0 block grid (2500 px) for the block-bootstrap CIs (B0 method and the
fixed-class-set method of B2). Results are compared with AGIT v2 Table 2
(outputs/AGIT_metrics.xlsx, sheet "Table 2 - Cross-temporal").

Run: nice -n 19 python scripts/b6_direct_cd.py [--workers 3]
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds

BASE_FOLDER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_FOLDER))
from scripts.b0_dwarffix_cd import BLOCK, MO, OUT, REF_CT, write_metrics_csv
from src.trainers.bootstrap import (
    block_bootstrap_ci, block_bootstrap_ci_fixed, collapse_binary, macro_iou_f1,
    tile_confusion_matrices,
)

DIRECT = MO / "direct_change" / "2013_2020"
NEW_DIR = DIRECT / "r1_b6"
MAPS = {
    "ChangeViT": "change_prediction_2013_2020_best_model_changevit_long_training_changemap_CLIP.tif",
    "U-Net Change MiT-B2": "change_prediction_2013_2020_binary_change-sweep-OneCycleLR_mit_b2_p256-"
                           "epoch=05-step=4170-val_jaccard=0.5140_CLIP.tif",
    "U-Net Change EfficientNet-B4": "change_prediction_multiclass_2013_2020_multiclass-sweep-9cls-"
                                    "OneCycleLR_efficientnet-b4_p256-epoch=13-step=14588-"
                                    "val_jaccard=0.2096_CLIP.tif",
}
MULTICLASS = {"U-Net Change EfficientNet-B4"}
# class code of the direct 9-class model (2003-2013 label order) -> v4 class code
OLD_TO_V4 = {0: 0, 1: 1, 2: 7, 3: 5, 4: 8, 5: 4, 6: 6, 7: 3, 8: 2}
# AGIT v2 Table 2 (direct CD rows): binary BA ("OA" there is macro recall), IoU, F1; multi-class.
# AGIT's "U-Net Change binary" row is the EfficientNet-B4 map collapsed to binary.
AGIT = {
    "U-Net Change EfficientNet-B4": {"binary": (0.52, 0.38, 0.51), "multiclass": (0.20, 0.10, 0.15)},
    "ChangeViT": {"binary": (0.50, 0.36, 0.47)},
}


def read_on_reference_grid(pred_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Reference change map and the prediction resampled (nearest) to the reference grid.
    Also returns pixel counts that describe the 0 = nodata/no-change ambiguity."""
    with rasterio.open(REF_CT) as r, rasterio.open(pred_path) as p:
        assert r.crs == p.crs
        ref = r.read(1)
        win0 = from_bounds(*r.bounds, transform=p.transform)
        win = win0
        # one pixel margin so the nearest neighbour at the border is inside the read
        win = rasterio.windows.Window(int(np.floor(win.col_off)) - 1, int(np.floor(win.row_off)) - 1,
                                      int(np.ceil(win.width)) + 3, int(np.ceil(win.height)) + 3)
        src = p.read(1, window=win, boundless=True, fill_value=0)
        dst = np.zeros(ref.shape, dtype=np.uint8)
        reproject(source=src, destination=dst, src_transform=p.window_transform(win),
                  src_crs=p.crs, dst_transform=r.transform, dst_crs=r.crs,
                  src_nodata=None, dst_nodata=None, resampling=Resampling.nearest)
        info = {"grid_shift_col_px": float(win0.col_off % 1), "grid_shift_row_px": float(win0.row_off % 1),
                "res_ref": r.res[0], "res_pred": p.res[0]}
    test = ref != 255
    info.update({
        "test_area_px": int(test.sum()),
        "pred_zero_in_test_area_px": int(((dst == 0) & test).sum()),
        "pred_values_in_test_area": str(dict(zip(*[x.tolist() for x in np.unique(dst[test], return_counts=True)]))),
    })
    return ref, dst, info


def recode_to_v4(path: Path) -> Path:
    """Write the 9-class direct map recoded from the old class order to the v4 order."""
    out = NEW_DIR / (path.stem + "_v4_class_order.tif")
    if out.exists():
        return out
    lut = np.arange(256, dtype=np.uint8)
    for old, new in OLD_TO_V4.items():
        lut[old] = new
    with rasterio.open(path) as src:
        prof = src.profile
        prof.update(compress="deflate", tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
        with rasterio.open(out, "w", **prof) as dst:
            for _, win in src.block_windows(1):
                dst.write(lut[src.read(1, window=win)], 1, window=win)
    return out


def run_job(name: str) -> list[dict]:
    import torch
    torch.set_num_threads(3)
    NEW_DIR.mkdir(exist_ok=True)
    path = DIRECT / MAPS[name]
    is_mc = name in MULTICLASS
    if is_mc:
        path = recode_to_v4(path)
    tasks = ["binary", "multiclass"] if is_mc else ["binary"]
    stem = name.replace(" ", "_")
    ref, prd, info = read_on_reference_grid(path)
    cms = tile_confusion_matrices(ref, prd, 9, block=BLOCK, ignore_index=255)
    del ref, prd
    np.save(NEW_DIR / f"{stem}_tile_cms.npy", cms)
    rows = []
    for t in tasks:
        is_bin = t == "binary"
        csv = NEW_DIR / f"{stem}_change_metrics_{t}_2013-2020.csv"
        write_metrics_csv(path, csv, binary=is_bin)
        df = pd.read_csv(csv, index_col=0)
        df.index = df.index.astype(str)
        cm = cms.sum(axis=0)
        cm = collapse_binary(cm) if is_bin else cm
        iou_cm, f1_cm = macro_iou_f1(cm)
        ci = block_bootstrap_ci(cms, n_boot=1000, binary=is_bin)
        cif = block_bootstrap_ci_fixed(cms, n_boot=1000, binary=is_bin)
        row = {"method": name, "task": f"change-{t}", "IoU": df.loc["macro", "JaccardIndex"],
               "F1": df.loc["macro", "F1Score"], "BA": df.loc["macro", "Recall"],
               "IoU_cm": iou_cm, "F1_cm": f1_cm,
               "IoU_ci_low": ci["IoU_ci_low"], "IoU_ci_high": ci["IoU_ci_high"],
               "IoU_ci_low_fixed": cif["IoU_ci_low"], "IoU_ci_high_fixed": cif["IoU_ci_high"],
               "redraw_rate_fixed": cif["redraw_rate"], "n_tiles": ci["n_tiles"]}
        if not is_bin:
            for c in range(9):
                row[f"c{c}_IoU"] = df.loc[str(c), "JaccardIndex"]
        agit = AGIT.get(name, {}).get(t)
        if agit:
            row["AGIT_BA"], row["AGIT_IoU"], row["AGIT_F1"] = agit
        row.update(info)
        rows.append(row)
    print(f"done {name}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(min(args.workers, len(MAPS))) as pool:
        rows = [r for rs in pool.imap(run_job, list(MAPS)) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "b6_direct_cd_metrics.csv", index=False)
    cols = ["method", "task", "BA", "IoU", "F1", "IoU_ci_low_fixed", "IoU_ci_high_fixed",
            "AGIT_BA", "AGIT_IoU", "AGIT_F1"]
    print(df[[c for c in cols if c in df]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
