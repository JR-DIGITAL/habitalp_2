"""C2: labelled 2013 training+validation area lost by the masked split and by the block-drop split.

Usage: python scripts/c2_split_label_loss.py [data_folder]
       (default data_folder: $HABITALP_DATA_ROOT, else data/Originaldaten)
Writes outputs/r1/c2_label_loss.csv and outputs/r1/c2_note.txt.

Rebuilds the grid split exactly as HabitAlp2DataModule.setup does (HabitAlp2 dataset of the
Phase E / revision configs, roi_split with roi/habitalp_2013_boundary.gpkg,
random_grid_cell_assignment([0.7, 0.15, 0.15], grid_size=12, seed 0)) and the train/val
sampling area of the three splits with HabitAlp2DataModule._exclude_from_roi:
  orig      no exclusion (paper runs, K2)
  masked    roi/training_rois_2020.gpkg + 128 m buffer removed (T1-T10)
  blockdrop every train/val grid cell intersecting the 2020 test patches removed (K1)
The test cells are checked against processed/mask/classes_v4_2013_test_cells.tif.

Per class (2013 label raster, read at the 0.5 m training resolution with nearest
resampling, as the RasterDataset does), two measures for each split:
  area    labelled pixels of the train+val cells inside the sampling area
  reach   labelled pixels that at least one admissible 256 x 256 px training/validation
          patch can contain (patch inside the cell's bounding box and inside the sampling
          area, as in RandomGeoSamplerWithinRoi)
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from torch import Generator
from torchgeo.datasets import random_grid_cell_assignment
from torchgeo.datasets.splits import roi_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.datamodules.habitalp import HabitAlp2DataModule  # noqa: E402
from src.data.datasets import HabitAlp2  # noqa: E402

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("HABITALP_DATA_ROOT", ROOT / "data" / "Originaldaten"))
CFG = ROOT / "src/trainers/terratorch/revision/config-r1-unet-mit-b2-masked-s0.yaml"
OUT = ROOT / "outputs" / "r1"
RES = 0.5
PATCH = 256


def build_split():
    cfg = yaml.safe_load(CFG.read_text())
    year = cfg["training_years"]["value"][0]
    ds = HabitAlp2(image_paths={k: DATA / v for k, v in year["image_data_paths"].items()},
                   mask_path=DATA / year["mask_path"], res=RES)
    roi = gpd.read_file(DATA / cfg["roi_shape_path"]["value"]).geometry.union_all()
    ds = roi_split(ds, [roi])[0]
    train, val, test = random_grid_cell_assignment(ds, [0.7, 0.15, 0.15], grid_size=12,
                                                   generator=Generator().manual_seed(0))
    excl = cfg["exclusion"]["value"]
    names = [str(c) for c in cfg["class_names"]["value"]]
    return roi, train, val, test, DATA / excl["shape_path"], excl["buffer"], names


def sampler_roi(roi, train, val, shape_path, buffer, mode):
    if mode is None:
        return roi, 0
    shim = SimpleNamespace(exclusion_shape_path=shape_path, exclusion_buffer=buffer,
                           exclusion_mode=mode, train_dataset=train, val_dataset=val)
    sroi = HabitAlp2DataModule._exclude_from_roi(shim, roi)
    excluded = gpd.read_file(shape_path).geometry.union_all()
    n_drop = sum(g.intersects(excluded) for d in (train, val) for g in d.index.geometry)
    return sroi, n_drop


def main():
    roi, train, val, test, shape_path, buffer, names = build_split()
    trainval = [g for d in (train, val) for g in d.index.geometry]
    print(f"cells: train {len(train.index)}, val {len(val.index)}, test {len(test.index)}")

    with rasterio.open(DATA / "processed/mask/classes_v4_2013.tif") as src:
        b = src.bounds
        w = int(round((b.right - b.left) / RES))
        h = int(round((b.top - b.bottom) / RES))
        lab = src.read(1, out_shape=(h, w), resampling=Resampling.nearest)
    transform = from_origin(b.left, b.top, RES, RES)
    n_cls = len(names)
    print(f"label grid {lab.shape} at {RES} m")

    # check the reconstructed test cells against the stored test-cell raster
    with rasterio.open(DATA / "processed/mask/classes_v4_2013_test_cells.tif") as src:
        tc = src.read(1, out_shape=(h, w), resampling=Resampling.nearest)
    test_r = rasterize([(g, 1) for g in test.index.geometry], out_shape=(h, w), transform=transform,
                       fill=0, dtype="uint8")
    stored = (tc > 0) & (lab > 0)
    mine = (test_r > 0) & (lab > 0)
    test_check = (int(stored.sum()), int(mine.sum()), int((stored ^ mine).sum()))
    print(f"test-cell check (labelled px stored / rebuilt / disagreeing): {test_check}")
    del tc, test_r, stored, mine

    rows, summary = [], {}
    for split, mode in (("orig", None), ("masked", "mask"), ("blockdrop", "drop_cells")):
        sroi, n_drop = sampler_roi(roi, train, val, shape_path, buffer, mode)
        sroi_r = rasterize([(sroi, 1)], out_shape=(h, w), transform=transform, fill=0,
                           dtype="uint8", all_touched=False).astype(bool)
        area = np.zeros(n_cls + 1, np.int64)
        reach = np.zeros(n_cls + 1, np.int64)
        cells_used = 0
        for g in trainval:
            minx, miny, maxx, maxy = g.bounds
            c0, r0 = int(np.floor((minx - b.left) / RES)), int(np.floor((b.top - maxy) / RES))
            c1, r1 = int(np.ceil((maxx - b.left) / RES)), int(np.ceil((b.top - miny) / RES))
            r0, c0 = max(r0, 0), max(c0, 0)
            sub_lab = lab[r0:r1, c0:c1]
            cell = rasterize([(g, 1)], out_shape=sub_lab.shape,
                             transform=from_origin(b.left + c0 * RES, b.top - r0 * RES, RES, RES),
                             fill=0, dtype="uint8").astype(bool)
            ok = sroi_r[r0:r1, c0:c1]
            inside = cell & ok
            area += np.bincount(sub_lab[inside], minlength=n_cls + 1)[: n_cls + 1]
            # admissible patch origins: the 256 x 256 window lies entirely in the sampling area
            # (and, by construction of the window, in the cell's bounding box)
            hh, ww = ok.shape
            if hh < PATCH or ww < PATCH:
                continue
            ii = np.zeros((hh + 1, ww + 1), np.int64)
            ii[1:, 1:] = ok.cumsum(0).cumsum(1)
            full = (ii[PATCH:, PATCH:] - ii[:-PATCH, PATCH:] - ii[PATCH:, :-PATCH]
                    + ii[:-PATCH, :-PATCH]) == PATCH * PATCH
            if not full.any():
                continue
            cells_used += 1
            origin = np.zeros_like(ok)
            origin[: full.shape[0], : full.shape[1]] = full
            # a pixel is reachable if an admissible origin lies within PATCH-1 above/left of it
            io = np.zeros((hh + PATCH, ww + PATCH), np.int64)
            io[PATCH:, PATCH:] = origin.cumsum(0).cumsum(1)
            # io[i + PATCH, j + PATCH] = sum of origins in [0..i] x [0..j]
            cov = (io[PATCH:, PATCH:] - io[:-PATCH, PATCH:] - io[PATCH:, :-PATCH]
                   + io[:-PATCH, :-PATCH]) > 0
            reach += np.bincount(sub_lab[cov], minlength=n_cls + 1)[: n_cls + 1]
        summary[split] = dict(n_drop=n_drop, cells_with_patches=cells_used,
                              sroi_km2=sroi.area / 1e6)
        for k in range(1, n_cls + 1):
            rows.append(dict(split=split, class_id=k, class_name=names[k - 1],
                             area_px=int(area[k]), reach_px=int(reach[k])))
        print(split, summary[split], int(area[1:].sum()), int(reach[1:].sum()))

    df = pd.DataFrame(rows)
    wide = df.pivot(index=["class_id", "class_name"], columns="split", values=["area_px", "reach_px"])
    out = pd.DataFrame(index=wide.index)
    px_ha = RES * RES / 1e4
    for m in ("area", "reach"):
        for s in ("orig", "masked", "blockdrop"):
            out[f"{m}_{s}_ha"] = wide[(f"{m}_px", s)] * px_ha
        for s in ("masked", "blockdrop"):
            out[f"{m}_lost_{s}_pct"] = 100 * (1 - wide[(f"{m}_px", s)] / wide[(f"{m}_px", "orig")])
    out = out.reset_index()
    tot = {"class_id": 0, "class_name": "all classes"}
    for c in out.columns[2:]:
        if c.endswith("_ha"):
            tot[c] = out[c].sum()
    for m in ("area", "reach"):
        for s in ("masked", "blockdrop"):
            tot[f"{m}_lost_{s}_pct"] = 100 * (1 - tot[f"{m}_{s}_ha"] / tot[f"{m}_orig_ha"])
    out = pd.concat([pd.DataFrame([tot]), out], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(OUT / "c2_label_loss.csv", index=False)

    n_tv = len(trainval)
    lines = [
        "C2: 2013 train+val labels lost by the masked split and the block-drop split",
        "Script: scripts/c2_split_label_loss.py; per class: c2_label_loss.csv",
        f"Grid split rebuilt from the datamodule code (grid 12, seed 0): train {len(train.index)}, "
        f"val {len(val.index)}, test {len(test.index)} cells (ROI-clipped grid cells = 'blocks').",
        f"Test-cell check vs classes_v4_2013_test_cells.tif (labelled px at 0.5 m, stored / rebuilt / "
        f"disagreeing): {test_check[0]} / {test_check[1]} / {test_check[2]}",
        f"Train+val blocks: {n_tv}; block drop removes {summary['blockdrop']['n_drop']} "
        f"(masked split: {summary['masked']['n_drop']} blocks touched by the patches, none removed).",
        "Sampling area (km2): " + ", ".join(f"{s} {v['sroi_km2']:.1f}" for s, v in summary.items()),
        "area = labelled px of train+val blocks inside the sampling area; reach = labelled px that",
        "at least one admissible 256 px patch can contain (inside block bbox and sampling area).",
        "",
        "Kept / lost share of all train+val labels:",
    ]
    t = out.iloc[0]
    for m in ("area", "reach"):
        lines.append(f"  {m}: masked lost {t[f'{m}_lost_masked_pct']:.1f}%, block drop lost "
                     f"{t[f'{m}_lost_blockdrop_pct']:.1f}% (orig {t[f'{m}_orig_ha']:.0f} ha)")
    lines += ["", "Per class, % lost (area | reach), masked vs block drop:"]
    for _, r in out.iloc[1:].iterrows():
        lines.append(f"  {r.class_id:2d} {r.class_name:38s} orig {r.area_orig_ha:7.1f} ha  masked "
                     f"{r.area_lost_masked_pct:5.1f} | {r.reach_lost_masked_pct:5.1f}  block drop "
                     f"{r.area_lost_blockdrop_pct:5.1f} | {r.reach_lost_blockdrop_pct:5.1f}")
    (OUT / "c2_note.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
