"""Write the inference configs of the MDPI revision (round 1) for the trained runs in models/.

Usage: python scripts/make_r1_infer_configs.py [--data-root DIR] [--models-dir DIR] [--out DIR]
                                               [--roi-2013 test-area|boundary]
For every models/r1-*/ folder with a best checkpoint (and, if it was trained with scripts/r1_queue.sh,
a .status file with rc=0 in models/r1_queue/) it writes
<out>/infer-<run>-<year>.yaml for the best val checkpoint (2020 and 2013) and
<out>/infer-<run>-2020-last.yaml for the last epoch (latest.ckpt). Best-checkpoint
rasters go to model_output/<run>/, last-epoch rasters to their own folder model_output/<run>-last/
so that downstream globs on 2020_*.tif never mix the two. Same inputs and tiling as the Phase E
configs; the 2013 run covers by default only the area that is evaluated: the 2020 test patches
(change maps, post-processing) and the 2013 test cells (in-domain test), about a fifth of the full
2013 boundary. That ROI is written once to <data root>/roi/r1_infer_2013_roi.gpkg; --roi-2013
boundary restores the Phase E extent. Paths are machine specific, so the configs are generated on each machine and
not committed; the default output folder models/r1_infer/configs/ is outside git. The job list for
the queue is printed to stdout: best checkpoint 2020, best 2013, then last epoch 2020.
"""
import argparse
import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

YEARS = {
    2013: {
        "input_data": {
            "rgb": "processed/orthophoto_gis_stmk/flug_2013_2015_rgb.tif",
            "cir": "processed/orthophoto_gis_stmk/falschfarben_2013_2015.tif",
            "ndsm": "processed/elevation_gis_stmk_2010-2012/ndsm.tif",
        },
        "mask_path": "processed/mask/classes_v4_2013.tif",
        "roi_path": "roi/r1_infer_2013_roi.gpkg",  # Phase E: roi/habitalp_2013_boundary.gpkg
    },
    2020: {
        "input_data": {
            "rgb": "processed/orthophoto_gis_stmk/flug_2019_2021_rgb.tif",
            "cir": "processed/orthophoto_gis_stmk/falschfarben_2019_2021.tif",
            "ndsm": "processed/elevation_fefimapla_2020/ndsm.tif",
        },
        "mask_path": "processed/mask/classes_v4_2020.tif",
        "roi_path": "roi/training_rois_2020.gpkg",
    },
}
# same tiling as the Phase E inference configs
INFERENCE = {"patch_size": 256, "res": 0.5, "overlap": 64, "delta": 8, "batch_size": 4, "n_classes": 23}


def write_2013_roi(data_root, buffer=10.0, step=10):
    """2020 test patches plus the 2013 test cells (non-zero pixels of classes_v4_2013_test_cells.tif,
    traced at `step` times the raster resolution and buffered by `buffer` m so no edge pixel is lost)."""
    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape
    from shapely.ops import unary_union

    rois = gpd.read_file(data_root / "roi" / "training_rois_2020.gpkg")
    with rasterio.open(data_root / "processed" / "mask" / "classes_v4_2013_test_cells.tif") as src:
        cells = src.read(1, out_shape=(src.height // step, src.width // step)) > 0
        transform = src.transform * src.transform.scale(src.width / cells.shape[1], src.height / cells.shape[0])
        crs = src.crs
    polys = [shape(g) for g, v in shapes(cells.astype(np.uint8), mask=cells, transform=transform) if v]
    union = unary_union(polys + list(rois.to_crs(crs).geometry)).buffer(buffer)
    gpd.GeoDataFrame(geometry=[union], crs=crs).to_file(data_root / YEARS[2013]["roi_path"])


def best_checkpoint(run_dir):
    """The checkpoint with the highest val/mIoU in its name (ModelCheckpoint writes it under val/)."""
    ckpts = [(float(m.group(1)), c) for c in run_dir.rglob("*.ckpt")
             if (m := re.search(r"mIoU=([0-9]+\.[0-9]+)(-v[0-9]+)?\.ckpt$", c.name))]
    return max(ckpts)[1] if ckpts else None


def finished(state, run):
    status = state / f"{run}.status"
    if not status.exists():  # trained outside scripts/r1_queue.sh: trust the checkpoints
        return True
    return status.read_text().startswith("done") and "rc=0" in status.read_text()


def config(data_root, year, checkpoint, out_dir):
    y = YEARS[year]
    return {
        "model_type": "terratorch",
        "input_data": {k: str(data_root / v) for k, v in y["input_data"].items()},
        "mask_path": str(data_root / y["mask_path"]),
        "roi_path": str(data_root / y["roi_path"]),
        "models": {"terratorch": {"checkpoint": str(checkpoint)}},
        "inference": dict(INFERENCE),
        "output": {"dir": str(out_dir)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("HABITALP_DATA_ROOT", ROOT / "data" / "Originaldaten")))
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "r1_infer" / "configs")
    parser.add_argument("--roi-2013", choices=["test-area", "boundary"], default="test-area")
    args = parser.parse_args()
    if args.roi_2013 == "boundary":
        YEARS[2013]["roi_path"] = "roi/habitalp_2013_boundary.gpkg"
    elif not (args.data_root / YEARS[2013]["roi_path"]).exists():
        write_2013_roi(args.data_root)
    args.out.mkdir(parents=True, exist_ok=True)
    state = args.models_dir / "r1_queue"

    best2020, best2013, last = [], [], []
    for run_dir in sorted(args.models_dir.glob("r1-*")):
        run = run_dir.name
        if not finished(state, run):
            continue
        ckpt = best_checkpoint(run_dir)
        if ckpt is None:
            print(f"warning: no best checkpoint found for {run}, skipped", file=sys.stderr)
            continue
        out_dir = args.data_root / "model_output" / run
        for year in (2020, 2013):
            name = f"{run}-{year}"
            (args.out / f"infer-{name}.yaml").write_text(
                yaml.safe_dump(config(args.data_root, year, ckpt.resolve(), out_dir), sort_keys=False))
            (best2020 if year == 2020 else best2013).append(name)
        if (run_dir / "latest.ckpt").exists():
            name = f"{run}-2020-last"
            (args.out / f"infer-{name}.yaml").write_text(yaml.safe_dump(
                config(args.data_root, 2020, (run_dir / "latest.ckpt").resolve(),
                       args.data_root / "model_output" / f"{run}-last"), sort_keys=False))
            last.append(name)
    print("\n".join(best2020 + best2013 + last))


if __name__ == "__main__":
    main()
