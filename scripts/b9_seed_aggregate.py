#!/usr/bin/env python
"""B9 (R1-M1, R2-M2): test-set metrics of all revision runs, aggregated over seeds.

For every revision run (model_output/r1-*/, best val checkpoint) and every last-epoch
raster (model_output/r1-*-last/) this computes, with the evaluation path of the paper:

  seg2020  2020 segmentation vs processed/mask/classes_v4_2020.tif (24 classes, ignore 0
           in reference and prediction). The reference is read over the intersection
           window and the 0.5 m prediction is warped onto that window with
           rasterio.warp.reproject(Resampling.nearest), exactly as compute_final_metrics
           does (B7); here the warp runs in row strips of the same destination grid and
           the confusion matrix is counted with bincount instead of torchmetrics.
           mIoU = mean IoU over classes 1..23, BA = mean recall over classes 1..23 (the
           torchmetrics macro values in classification_metrics_*.csv).
  seg2013  2013 in-domain test cells vs processed/mask/classes_v4_2013_test_cells.tif,
           same path (best checkpoint only; there are no last-epoch 2013 rasters).
  cd       raw change detection: generate_change_map(classes_v4_2013, predicted 2020)
           with the B0 dwarf-forest rule (get_change_array), then the change map vs
           habitalp_change_v4_2013-2020.tif (ignore 255) through load_aligned_2d, as in
           B0/B2. Multi-class (9) and binary (collapse_binary) macro IoU, F1, BA.
           Change maps go to <model_output>/<run>/r1_b9/change_map_2013-2020.tif.

Post-processed CD metrics are not computed: they need the IMPACT polygon step
(proprietary), which does not exist for the revision rasters.

Sanity check (job "check"): the same pipeline on the Phase E MiT-B2 U-Net raster must
reproduce classification_metrics_2020.csv / _2013.csv and the B0 rows (IoU_new, F1_new,
BA_new), and its CD tile matrices must equal b2_tile_cms.npz bit for bit.

Tile confusion matrices, outputs/r1/b9_tile_cms.npz (for the paired bootstrap B1):
  The tile grid is that of B0/B2 (scripts/mdpi/bootstrap_ci.py): 2500 x 2500 reference pixels
  (0.19990 m, about 500 m) from the top-left corner of the intersection window of the
  reference change map, row-major; of the 17 x 19 grid tiles the 88 with valid pixels are
  kept, in the same order as tile_confusion_matrices and b2_tile_cms.npz.
  Keys:
    "tile_rc"                int64 (88, 2): grid row and column of each kept tile
    "<dir>__cd"              int64 (88, 9, 9): change map CMs, rows = reference change
                             class 0..8, columns = predicted (B2 layout; binary =
                             collapse_binary of the tile sum)
    "<dir>__seg2020"         int64 (88, 24, 24): 2020 segmentation CMs on the same tiles,
                             rows = reference class 0..23, columns = predicted; row and
                             column 0 (nodata) are always 0, macro over classes 1..23
  <dir> is the model_output folder name: r1-<model>-<split>-s<seed> (best checkpoint) or
  r1-<model>-<split>-s<seed>-last (last epoch). The seg2020 tile grid starts at the same
  reference pixel as the cd grid (asserted), so tile i is the same 500 m block in both.

Outputs: outputs/r1/b9_per_run.csv, b9_summary.csv, b9_tile_cms.npz, b9_note.txt.
Per-job caches in <model_output>/<dir>/r1_b9/ (delete to recompute), written atomically.
A cache with "placeholder": true in b9_row.json reserves a job for another machine (its
real cache is copied in later); the aggregation refuses placeholders.

Run (env set before python starts):
  LD_PRELOAD=/usr/lib64/libjpeg.so.8 GDAL_CACHEMAX=512 OMP_NUM_THREADS=3 \
  nice -n 10 .venv314/bin/python scripts/b9_seed_aggregate.py [--workers 6]
  Only some folders, no aggregation (e.g. on the h200):
  ... scripts/b9_seed_aggregate.py --workers 4 --runs r1-clay-v1-masked-s0-last ...
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.warp import Resampling, reproject

BASE_FOLDER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_FOLDER))
from scripts.b0_dwarffix_cd import BLOCK, MO, OUT, REF_2013, REF_2020, REF_CT, metrics_from_csv
from src.inference.eval import generate_change_map
from src.trainers.bootstrap import collapse_binary, load_aligned_2d, tile_confusion_matrices

DATA_ROOT = MO.parent
REF_2013_TC = DATA_ROOT / "processed/mask/classes_v4_2013_test_cells.tif"
JOB_DIR = "r1_b9"
CHECK_EXP = "unet-mit-b2-rgb-nir-ndsm-phaseE"
N_SEG, N_CD = 24, 9
RUN_RE = re.compile(r"^r1-(.+)-(masked|orig|blockdrop)-s(\d)(-last)?$")
LABELS = {
    "clay-v1": "Clay v1.0", "clay-v1.5": "Clay v1.5", "unet-mit-b2": "MiT-B2 U-Net",
    "unet-resnet50": "ResNet-50 U-Net", "prithvi-v2-300": "Prithvi", "dofa-base": "DOFA",
    "terramind": "TerraMind", "clay-v1-unetdec": "Clay v1.0 + U-Net dec",
    "prithvi-v2-300-unetdec": "Prithvi + U-Net dec", "terramind-lr5e5": "TerraMind LR 5e-5",
}
ORDER = list(LABELS)


# ---------------------------------------------------------------- confusion matrices
def strip_tile_cms(ref: np.ndarray, pred: np.ndarray, n: int, block: int,
                   ignore: int) -> np.ndarray:
    """CMs of the grid tiles of one strip (height <= block), one per column tile, with the
    validity rule of tile_confusion_matrices (ignore in reference or prediction)."""
    out = []
    for c0 in range(0, ref.shape[1], block):
        r = ref[:, c0:c0 + block].ravel()
        p = pred[:, c0:c0 + block].ravel()
        v = (r != ignore) & (p != ignore)
        cm = np.bincount(r[v].astype(np.int64) * n + p[v], minlength=n * n)
        out.append(cm.reshape(n, n))
    return np.stack(out)


def seg_grid_cms(ref_path: Path, pred_path: Path, block: int) -> tuple[np.ndarray, tuple]:
    """Segmentation CMs on the full tile grid (n_rows, n_cols, 24, 24), aligned as in
    compute_final_metrics: reference read over the intersection window, prediction warped
    (nearest) onto window_transform(window), here strip by strip. Returns the grid and the
    rounded window offset of the reference."""
    with rasterio.open(ref_path) as m, rasterio.open(pred_path) as p:
        bm, bp = m.bounds, p.bounds
        inter = rasterio.coords.BoundingBox(
            left=max(bm.left, bp.left), bottom=max(bm.bottom, bp.bottom),
            right=min(bm.right, bp.right), top=min(bm.top, bp.top))
        win = m.window(*inter)
        ref = m.read(1, window=win)
        tr = m.window_transform(win)
        assert m.res != p.res  # predictions at 0.5 m, references at 0.2 m
        rows = []
        for r0 in range(0, ref.shape[0], block):
            rs = ref[r0:r0 + block]
            pr = np.empty(rs.shape, dtype=p.dtypes[0])
            reproject(source=rasterio.band(p, 1), destination=pr, src_transform=p.transform,
                      src_crs=p.crs, dst_transform=tr * Affine.translation(0, r0),
                      dst_crs=m.crs, resampling=Resampling.nearest)
            rows.append(strip_tile_cms(rs, pr, N_SEG, block, ignore=0))
        offset = (int(round(win.row_off)), int(round(win.col_off)))
    return np.stack(rows), offset


def cd_grid_cms(change_map: Path, block: int) -> tuple[np.ndarray, tuple]:
    """Change CMs on the full tile grid via load_aligned_2d (as B0/B2); also asserts that
    the kept tiles equal tile_confusion_matrices."""
    ref2d, prd2d = load_aligned_2d(REF_CT, change_map)
    rows = [strip_tile_cms(ref2d[r0:r0 + block], prd2d[r0:r0 + block], N_CD, block, 255)
            for r0 in range(0, ref2d.shape[0], block)]
    grid = np.stack(rows)
    tcm = tile_confusion_matrices(ref2d, prd2d, N_CD, block=block, ignore_index=255)
    flat = grid.reshape(-1, N_CD, N_CD)
    assert np.array_equal(flat[flat.sum(axis=(1, 2)) > 0], tcm)
    with rasterio.open(REF_CT) as m, rasterio.open(change_map) as p:
        bm, bp = m.bounds, p.bounds
        win = m.window(max(bm.left, bp.left), max(bm.bottom, bp.bottom),
                       min(bm.right, bp.right), min(bm.top, bp.top))
    return grid, (int(round(win.row_off)), int(round(win.col_off)))


# ---------------------------------------------------------------- metrics
def seg_metrics(cm: np.ndarray, prefix: str) -> dict:
    """torchmetrics macro values with ignore_index 0: mean over classes 1..23 that occur."""
    c = cm[1:, 1:].astype(np.float64)  # pixels with 0 on either side are never counted
    tp = np.diag(c)
    fp, fn = c.sum(axis=0) - tp, c.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (tp + fp + fn)
        rec = tp / (tp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
    d = {f"{prefix}_mIoU": np.nanmean(iou), f"{prefix}_BA": np.nanmean(rec),
         f"{prefix}_F1": np.nanmean(f1), f"{prefix}_OA": tp.sum() / c.sum(),
         f"{prefix}_px": int(c.sum())}
    if prefix == "seg2020":
        d.update({f"seg2020_IoU_c{k + 1}": iou[k] for k in range(len(iou))})
    return d


def cd_metrics(cm9: np.ndarray) -> dict:
    d = {}
    for name, cm in [("cd_mc", cm9), ("cd_bin", collapse_binary(cm9))]:
        c = cm.astype(np.float64)
        tp = np.diag(c)
        fp, fn = c.sum(axis=0) - tp, c.sum(axis=1) - tp
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = tp / (tp + fp + fn)
            d[f"{name}_IoU"] = np.nanmean(iou)
            d[f"{name}_F1"] = np.nanmean(2 * tp / (2 * tp + fp + fn))
            d[f"{name}_BA"] = np.nanmean(tp / (tp + fn))
        if name == "cd_mc":
            d.update({f"cd_IoU_c{k}": iou[k] for k in range(N_CD)})
    d["cd_px"] = int(cm9.sum())
    return d


# ---------------------------------------------------------------- jobs
def run_job(folder_name: str) -> dict:
    import torch
    torch.set_num_threads(3)
    is_check = folder_name == CHECK_EXP
    folder = MO / folder_name
    jd = folder / JOB_DIR
    jd.mkdir(exist_ok=True)
    cache_json, cache_npz = jd / "b9_row.json", jd / "b9_tiles.npz"
    if cache_json.exists() and cache_npz.exists():
        return {"row": json.loads(cache_json.read_text()), "tiles": dict(np.load(cache_npz))}

    pred20 = folder / f"2020_{folder_name}.tif"
    pred13 = folder / f"2013_{folder_name}.tif"
    row = {"dir": folder_name}

    seg_grid, seg_off = seg_grid_cms(REF_2020, pred20, BLOCK)
    change_map = jd / "change_map_2013-2020.tif"
    if not change_map.exists():
        tmp = jd / f"change_map_2013-2020.tmp{os.getpid()}.tif"  # renamed when complete
        generate_change_map(mask_path=REF_2013, prediction_path=pred20, output_path=tmp)
        tmp.rename(change_map)
    cd_grid, cd_off = cd_grid_cms(change_map, BLOCK)
    row["seg_window_off"], row["cd_window_off"] = list(seg_off), list(cd_off)
    row["seg_grid_shape"], row["cd_grid_shape"] = list(seg_grid.shape[:2]), list(cd_grid.shape[:2])
    row.update(seg_metrics(seg_grid.sum(axis=(0, 1)), "seg2020"))
    row.update(cd_metrics(cd_grid.sum(axis=(0, 1))))
    if pred13.exists():
        seg13, _ = seg_grid_cms(REF_2013_TC, pred13, BLOCK)
        row.update(seg_metrics(seg13.sum(axis=(0, 1)), "seg2013"))
    row = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in row.items()}

    tiles = {"seg_grid": seg_grid, "cd_grid": cd_grid}
    tmp_npz, tmp_json = jd / f"b9_tiles.tmp{os.getpid()}.npz", jd / f"b9_row.tmp{os.getpid()}.json"
    np.savez_compressed(tmp_npz, **tiles)
    tmp_json.write_text(json.dumps(row))
    tmp_npz.rename(cache_npz)
    tmp_json.rename(cache_json)
    print(f"done {folder_name}: seg2020 {row['seg2020_mIoU']:.4f} cd_mc {row['cd_mc_IoU']:.4f}"
          f" cd_bin {row['cd_bin_IoU']:.4f}", flush=True)
    return {"row": row, "tiles": tiles}


def check_reproduction(res: dict) -> list[str]:
    """Phase E MiT-B2 U-Net: compare with the stored paper / B0 / B2 values."""
    row, lines = res["row"], []
    for y in [2020, 2013]:
        df = pd.read_csv(MO / CHECK_EXP / f"classification_metrics_{y}.csv", index_col=0)
        for m, col in [("mIoU", "JaccardIndex"), ("BA", "Recall"), ("F1", "F1Score")]:
            a, b = row[f"seg{y}_{m}"], float(df.loc["macro", col])
            lines.append(f"seg{y} {m}: b9 {a:.6f} stored {b:.6f} diff {a - b:+.1e}")
    b0 = pd.read_csv(OUT / "b0_dwarffix_cd_metrics.csv").query(
        "experiment == @CHECK_EXP and postproc == 'before'")
    for task, pre in [("change-multiclass", "cd_mc"), ("change-binary", "cd_bin")]:
        r = b0[b0.task == task].iloc[0]
        for m in ["IoU", "F1", "BA"]:
            a, b = row[f"{pre}_{m}"], float(r[f"{m}_new"])
            lines.append(f"{pre} {m}: b9 {a:.6f} B0 {b:.6f} diff {a - b:+.1e}")
    return lines


def parse(dir_name: str) -> dict:
    m = RUN_RE.match(dir_name)
    model, split, seed, last = m.groups()
    return {"model_key": model, "model": LABELS[model], "split": split, "seed": int(seed),
            "ckpt": "last" if last else "best",
            "run": dir_name.removesuffix("-last")}


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--runs", nargs="+", metavar="DIR",
                    help="only compute the per-job caches of these model_output folders "
                         "(e.g. on another machine), skip those already cached, no aggregation")
    ap.add_argument("--unstarted-reverse", action="store_true",
                    help="helper instance: only jobs whose r1_b9 folder does not exist yet "
                         "(not started by another instance), in reverse order, no aggregation")
    ap.add_argument("--last-only-reverse", action="store_true",
                    help="helper instance: only fill the per-job caches of the last-epoch "
                         "rasters, in reverse order, no aggregation")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    dirs = sorted(d.name for d in MO.glob("r1-*") if d.is_dir() and RUN_RE.match(d.name))
    dirs = [d for d in dirs if (MO / d / f"2020_{d}.tif").exists()]
    jobs = [CHECK_EXP] + sorted(dirs, key=lambda d: d.endswith("-last"))  # 2013 jobs first
    if args.runs:
        missing = [d for d in args.runs if not (MO / d / f"2020_{d}.tif").exists()]
        assert not missing, f"no 2020 raster for {missing}"
        with Pool(args.workers) as pool:
            list(pool.imap_unordered(run_job, args.runs))
        return
    if args.unstarted_reverse:
        todo = [d for d in reversed(jobs[1:]) if not (MO / d / JOB_DIR).exists()]
        print("unstarted:", todo, flush=True)
        with Pool(args.workers) as pool:
            list(pool.imap_unordered(run_job, todo))
        return
    if args.last_only_reverse:
        with Pool(args.workers) as pool:
            list(pool.imap_unordered(run_job, [d for d in reversed(dirs) if d.endswith("-last")]))
        return
    with Pool(args.workers) as pool:
        results = dict(zip(jobs, pool.imap(run_job, jobs)))

    # --- reproduction check and tile alignment
    ph = [d for d, r in results.items() if r["row"].get("placeholder")]
    assert not ph, f"placeholder caches not replaced by real results: {ph}"
    check = results.pop(CHECK_EXP)
    note = ["B9: revision runs, test-set metrics aggregated over seeds",
            "Script: scripts/b9_seed_aggregate.py; per run: b9_per_run.csv; aggregated: "
            "b9_summary.csv; tile CMs: b9_tile_cms.npz", "",
            f"Reproduction check on {CHECK_EXP}:"] + ["  " + s for s in check_reproduction(check)]
    b2 = np.load(OUT / "b2_tile_cms.npz")[f"{CHECK_EXP}__before"]
    cgrid = check["tiles"]["cd_grid"].reshape(-1, N_CD, N_CD)
    keep = cgrid.sum(axis=(1, 2)) > 0
    note.append(f"  cd tile CMs equal to b2_tile_cms.npz: {np.array_equal(cgrid[keep], b2)}"
                f" ({int(keep.sum())} tiles)")
    grid_shape = check["tiles"]["cd_grid"].shape[:2]
    ref_keep = keep.copy()

    tile_store, same_tiles, ref_dev, seg_out, offs = {}, [], 0, [], set()
    for d, res in results.items():
        r, t = res["row"], res["tiles"]
        assert t["cd_grid"].shape[:2] == grid_shape and t["seg_grid"].shape[:2] == grid_shape, d
        assert r["seg_window_off"] == r["cd_window_off"], (d, r["seg_window_off"], r["cd_window_off"])
        offs.add(tuple(r["cd_window_off"]))
        cd = t["cd_grid"].reshape(-1, N_CD, N_CD)
        seg = t["seg_grid"].reshape(-1, N_SEG, N_SEG)
        k = cd.sum(axis=(1, 2)) > 0
        same_tiles.append(np.array_equal(k, ref_keep))
        # reference pixels per tile and class vs the Phase E map of B2 (edge pixels may differ)
        ref_dev = max(ref_dev, int(np.abs(cd[ref_keep].sum(axis=2) - b2.sum(axis=2)).max()))
        seg_out.append(int(seg[~ref_keep].sum()))
        tile_store[f"{d}__cd"] = cd[ref_keep]
        tile_store[f"{d}__seg2020"] = seg[ref_keep]
    rr, cc = np.divmod(np.flatnonzero(ref_keep), grid_shape[1])
    tile_store["tile_rc"] = np.stack([rr, cc], axis=1)
    np.savez_compressed(OUT / "b9_tile_cms.npz", **tile_store)
    n_maps = len(results)
    note += ["", "Tile alignment (88-tile 500 m grid of B0/B2):",
             f"  {sum(same_tiles)} of {n_maps} rasters give the same {int(ref_keep.sum())} non-empty "
             f"cd tiles of the {grid_shape[0]} x {grid_shape[1]} grid as B2; reference window offset "
             f"{sorted(offs)} for all, seg2020 grid starts at the same reference pixel",
             f"  max |reference pixels per tile and class - B2| over all maps: {ref_dev} px",
             f"  seg2020 pixels outside the 88 tiles: max {max(seg_out)} px"]

    # --- per-run table
    rows = [{**parse(d), **{k: v for k, v in res["row"].items()
                             if not k.endswith(("_off", "_shape"))}} for d, res in results.items()]
    per = pd.DataFrame(rows)
    per["model_order"] = per.model_key.map(ORDER.index)
    per = per.sort_values(["model_order", "split", "ckpt", "seed"]).drop(columns="model_order")
    per.to_csv(OUT / "b9_per_run.csv", index=False)

    # --- summary over seeds (long format)
    metrics = [c for c in per.columns if c.startswith(("seg2020_", "seg2013_", "cd_"))
               and not c.endswith("_px")]
    srows = []
    for (mk, split, ckpt), g in per.groupby(["model_key", "split", "ckpt"], sort=False):
        for m in metrics:
            v = g[["seed", m]].dropna()
            if v.empty:
                continue
            srows.append({"model": LABELS[mk], "model_key": mk, "split": split, "ckpt": ckpt,
                          "metric": m, "n_seeds": len(v), "mean": v[m].mean(),
                          "sd": v[m].std(ddof=1) if len(v) > 1 else np.nan,
                          "min": v[m].min(), "max": v[m].max(),
                          "seeds": ";".join(str(s) for s in v.seed),
                          "values": ";".join(f"{x:.4f}" for x in v[m])})
    summ = pd.DataFrame(srows)
    summ["model_order"] = summ.model_key.map(ORDER.index)
    summ = summ.sort_values(["model_order", "split", "ckpt"], kind="stable").drop(columns="model_order")
    # best minus last, paired per seed (runs that have both)
    pair = per.pivot_table(index=["model_key", "split", "seed"], columns="ckpt",
                           values=["seg2020_mIoU", "cd_mc_IoU", "cd_bin_IoU"])
    drows = []
    for m in ["seg2020_mIoU", "cd_mc_IoU", "cd_bin_IoU"]:
        dd = (pair[(m, "best")] - pair[(m, "last")]).dropna()
        for (mk, split), g in dd.groupby(level=[0, 1], sort=False):
            drows.append({"model": LABELS[mk], "model_key": mk, "split": split,
                          "ckpt": "best-last", "metric": m, "n_seeds": len(g),
                          "mean": g.mean(), "sd": g.std(ddof=1) if len(g) > 1 else np.nan,
                          "min": g.min(), "max": g.max(),
                          "seeds": ";".join(str(s) for s in g.index.get_level_values(2)),
                          "values": ";".join(f"{x:+.4f}" for x in g)})
    summ = pd.concat([summ, pd.DataFrame(drows)], ignore_index=True)
    summ.to_csv(OUT / "b9_summary.csv", index=False)

    # --- note: headline table
    def ms(mk, split, ckpt, m, vals=False):
        q = summ[(summ.model_key == mk) & (summ.split == split) & (summ.ckpt == ckpt)
                 & (summ.metric == m)]
        if q.empty:
            return "-"
        r = q.iloc[0]
        s = f"{r['mean']:.3f}" + (f" ± {r['sd']:.3f}" if r.n_seeds > 1 else "")
        return s + (f" ({r['values']})" if vals else "")

    note += ["", "Headline (best val checkpoint unless stated; mean ± SD over seeds, SD with ddof 1):",
             "model | split | n | 2020 mIoU (per seed) | 2020 BA | 2013 mIoU | CD multi IoU | "
             "CD binary IoU | 2020 mIoU last epoch (n) | best - last"]
    for (mk, split), g in per[per.ckpt == "best"].groupby(["model_key", "split"], sort=False):
        nl = int(((per.model_key == mk) & (per.split == split) & (per.ckpt == "last")).sum())
        note.append(" | ".join([
            LABELS[mk], split, str(len(g)), ms(mk, split, "best", "seg2020_mIoU", True),
            ms(mk, split, "best", "seg2020_BA"), ms(mk, split, "best", "seg2013_mIoU"),
            ms(mk, split, "best", "cd_mc_IoU"), ms(mk, split, "best", "cd_bin_IoU"),
            f"{ms(mk, split, 'last', 'seg2020_mIoU')} ({nl})",
            ms(mk, split, "best-last", "seg2020_mIoU")]))
    note += ["", "MiT-B2 U-Net by training split (masked = T3, original = K2, block drop = K1); "
             "mean ± SD (per seed):"]
    for m in ["seg2020_mIoU", "seg2020_BA", "seg2013_mIoU", "cd_mc_IoU", "cd_bin_IoU"]:
        note.append(f"  {m}: " + "; ".join(
            f"{sp} {ms('unet-mit-b2', sp, 'best', m, True)}"
            for sp in ["masked", "orig", "blockdrop"]))
    note += ["", "Post-processed CD: not computed (needs the IMPACT polygon step)."]
    (OUT / "b9_note.txt").write_text("\n".join(note) + "\n")
    print("\n".join(note))


if __name__ == "__main__":
    main()
