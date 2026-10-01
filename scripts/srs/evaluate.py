"""Segmentation and change-detection metrics for every SRS experiment.

For each experiment folder model_output/<experiment>/ this script
  1. scores the 2013 prediction on the in-domain test cells and the 2020
     prediction on the cross-temporal test patches (23 habitat classes),
  2. builds the 2013->2020 change map (2013 reference x 2020 prediction),
  3. scores the change map against the reference change map, as 9-class
     multi-class change and as binary change/no change.

With --stage after it does the same for the post-processed polygons in
<experiment>/constraint_resolution_+_polygonfix_cc_updated/ (see postprocess.py),
which are first re-rasterised onto the 0.2 m reference grid.

Metric CSVs are written next to the predictions; build_metrics_table.py collects them.

Usage:
    HABITALP_DATA=/path/to/data python scripts/srs/evaluate.py --stage before
    HABITALP_DATA=/path/to/data python scripts/srs/evaluate.py --stage after
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (BINARY_CLASSES, CHANGE_CLASSES, PP_DIR, REF_2013, REF_2013_TEST_CELLS,
                    REF_2020, REF_CHANGE_2013_2020, all_experiments, data_root)
from src.inference.eval import generate_change_map
from src.trainers.utils import compute_final_metrics


def save_metrics(out_csv: Path, **kwargs) -> None:
    macro, each_label, _figs = compute_final_metrics(**kwargs)
    each_label.pop("MulticlassConfusionMatrix")
    df = pd.concat([
        pd.DataFrame([macro], index=["macro"]).astype(float),
        pd.DataFrame(each_label),
    ])
    df.to_csv(out_csv)
    print(f"    {out_csv.name}: macro IoU {float(macro['JaccardIndex']):.3f}")


def rerasterize(gpkg: Path, out_tif: Path, template: Path) -> None:
    """Burn the polygon 'Class' attribute onto the grid of the template raster."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    with rasterio.open(template) as ref:
        profile = ref.profile
        shape, transform, nodata = (ref.height, ref.width), ref.transform, ref.nodata
    polys = gpd.read_file(gpkg)
    arr = rasterize(
        zip(polys.geometry, polys["Class"]), out_shape=shape, transform=transform,
        fill=nodata if nodata is not None else 0, dtype="uint8",
    )
    profile.update(dtype="uint8", count=1, nodata=nodata, compress="lzw", tiled=True,
                   predictor=2, bigtiff=True)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(arr, 1)


def find_prediction(folder: Path, year: int, after: bool) -> Path | None:
    if after:
        p = folder / f"{year}_re-rasterized.tif"
        return p if p.exists() else None
    hits = sorted(folder.glob(f"{year}_*.tif"))
    if len(hits) > 1:
        raise RuntimeError(f"expected one {year}_*.tif in {folder}, found {len(hits)}")
    return hits[0] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["before", "after"], default="before",
                    help="score raw model output or post-processed output")
    ap.add_argument("--experiments", nargs="*", default=None,
                    help="experiment folders to evaluate (default: all in config.py)")
    args = ap.parse_args()

    root = data_root()
    after = args.stage == "after"
    suffix = "_after_postprocessing" if after else ""
    refs = {2013: root / REF_2013_TEST_CELLS, 2020: root / REF_2020}

    for exp in args.experiments or all_experiments():
        folder = root / "model_output" / exp
        if after:
            folder = folder / PP_DIR
        if not folder.exists():
            print(f"!! missing {folder}, skipping")
            continue
        print(f"--- {exp} ({args.stage} post-processing)")

        if after:
            for year in (2013, 2020):
                gpkg = folder / f"prediction_{year}_constraint_resolution_cc_updated.gpkg"
                if gpkg.exists():
                    rerasterize(gpkg, folder / f"{year}_re-rasterized.tif", refs[2020])

        # 1. single-date segmentation (24 = 23 classes + nodata index 0)
        for year, ref in refs.items():
            pred = find_prediction(folder, year, after)
            if pred is None:
                print(f"    no {year} prediction")
                continue
            save_metrics(folder / f"classification_metrics_{year}{suffix}.csv",
                         reference_file_path=ref, prediction_file_path=pred,
                         num_classes=24, ignore_index=0)

        # 2. change map: 2013 reference x 2020 prediction
        pred_2020 = find_prediction(folder, 2020, after)
        if pred_2020 is None:
            continue
        change_map = folder / "change_map_2013-2020.tif"
        generate_change_map(mask_path=root / REF_2013, prediction_path=pred_2020,
                            output_path=change_map)

        # 3. change metrics
        for task, names in (("multiclass", CHANGE_CLASSES), ("binary", BINARY_CLASSES)):
            save_metrics(folder / f"change_metrics_{task}_2013-2020{suffix}.csv",
                         reference_file_path=root / REF_CHANGE_2013_2020,
                         prediction_file_path=change_map, num_classes=len(names),
                         class_names=names, ignore_index=255)


if __name__ == "__main__":
    main()
