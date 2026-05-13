import numpy as np
from contextlib import ExitStack
from pathlib import Path
import rasterio
import rasterio.coords
import rasterio.windows
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy,
    F1Score,
    JaccardIndex,
    Precision,
    Recall,
)
from torchmetrics.classification import (
    ConfusionMatrix as TorchMetricsConfusionMatrix,
)


def physical_constraints_module(
    mask_path: Path,
    prediction_path: Path,
    dtm_path: Path,
    slope_path: Path,
    experiment_dir: Path,
    output_name: str,
    export_mask_for_each_constraint: bool = False,
):
    """Calculates physical constraints based on the provided ground truth mask and model predictions.

    Args:
        mask_path (Path): Path to label mask image.
        prediction_path (Path): Path to the prediction file.
        dtm_path (Path): Path to the digital terrain model (DTM) file.
        slope_path (Path): Path to the slope raster file.
        experiment_dir (Path): Directory to save the constraint violation masks.
        output_name (str): Name of the resulting prediction raster file.
        export_mask_for_each_constraint (bool, optional): Exports violation constraint masks if true. Defaults to False.
    """
    experiment_dir.mkdir(parents=True, exist_ok=True)

    non_forest_classes = [1, 2, 3, 4, 5, 19, 21, 22, 23]
    alpine_vegetation = [20, 22]
    grassland = [21, 22]
    young_growth = [6, 7]
    pole_wood = [8, 9, 10]
    mature_forest = [11, 12, 13, 14]
    old_forest = [15, 16, 17, 18]

    with (
        rasterio.open(mask_path) as mask_src,
        rasterio.open(prediction_path) as pred_src,
        rasterio.open(slope_path) as slope_src,
        rasterio.open(dtm_path) as dtm_src,
    ):
        assert mask_src.crs == pred_src.crs

        # Compute intersection bounding box of mask and prediction
        intersection = rasterio.coords.BoundingBox(
            left=max(mask_src.bounds.left, pred_src.bounds.left),
            bottom=max(mask_src.bounds.bottom, pred_src.bounds.bottom),
            right=min(mask_src.bounds.right, pred_src.bounds.right),
            top=min(mask_src.bounds.top, pred_src.bounds.top),
        )
        if intersection.left >= intersection.right or intersection.bottom >= intersection.top:
            raise ValueError("No overlapping area between rasters.")

        mask_win = mask_src.window(*intersection)
        orig_nodata = mask_src.nodata

        output_meta = mask_src.meta.copy()
        output_meta.update(
            {
                "transform": mask_src.window_transform(mask_win),
                "height": round(mask_win.height),
                "width": round(mask_win.width),
                "compress": "lzw",
                "nodata": 255,
                "tiled": True,
                "predictor": 2,
                "bigtiff": True,
                "dtype": "uint8",
            }
        )

        constraint_filenames = {
            1: "constraint_violation_1_non_forest_to_mature_forest.tif",
            2: "constraint_violation_2_young_growth_to_old_forest.tif",
            3: "constraint_violation_3_forest_setback_to_younger_stage.tif",
            4: "constraint_violation_4_rock_to_grassland.tif",
            5: "constraint_violation_5_water_bodies_on_steep_slopes.tif",
            6: "constraint_violation_6_alpine_vegetation_in_lowland.tif",
            "total": "constraint_violation_total.tif",
        }

        violation_counts = [0] * 6

        with ExitStack() as stack:
            dst = stack.enter_context(
                rasterio.open(experiment_dir / f"{output_name}.tif", "w", **output_meta)
            )
            if export_mask_for_each_constraint:
                c_dsts = {
                    k: stack.enter_context(
                        rasterio.open(experiment_dir / v, "w", **output_meta)
                    )
                    for k, v in constraint_filenames.items()
                }

            # Process one block at a time — only the current tile is held in memory
            for _, out_block in dst.block_windows(1):
                block_bounds = rasterio.windows.bounds(out_block, output_meta["transform"])
                tile_h = round(out_block.height)
                tile_w = round(out_block.width)

                mask_tile = mask_src.read(
                    1,
                    window=mask_src.window(*block_bounds),
                    out_shape=(tile_h, tile_w),
                    resampling=rasterio.enums.Resampling.nearest,
                )
                pred_tile = pred_src.read(
                    1,
                    window=pred_src.window(*block_bounds),
                    out_shape=(tile_h, tile_w),
                    resampling=rasterio.enums.Resampling.nearest,
                )
                slope_tile = slope_src.read(
                    1,
                    window=slope_src.window(*block_bounds),
                    out_shape=(tile_h, tile_w),
                    resampling=rasterio.enums.Resampling.bilinear,
                )
                dtm_tile = dtm_src.read(
                    1,
                    window=dtm_src.window(*block_bounds),
                    out_shape=(tile_h, tile_w),
                    resampling=rasterio.enums.Resampling.bilinear,
                )

                nodata_cells = (mask_tile == orig_nodata) | (pred_tile == orig_nodata)

                # --- Evaluate all 6 constraints ---
                c1 = np.isin(mask_tile, non_forest_classes) & np.isin(pred_tile, mature_forest + old_forest)
                c2 = np.isin(mask_tile, young_growth) & np.isin(pred_tile, old_forest)
                c3 = (
                    (np.isin(mask_tile, pole_wood) & np.isin(pred_tile, young_growth))
                    | (np.isin(mask_tile, mature_forest) & np.isin(pred_tile, young_growth + pole_wood))
                    | (np.isin(mask_tile, old_forest) & np.isin(pred_tile, young_growth + pole_wood + mature_forest))
                )
                c4 = (mask_tile == 5) & np.isin(pred_tile, grassland)
                c5 = (slope_tile > 35.0) & (mask_tile != 1) & (pred_tile == 1)
                c6 = (dtm_tile < 650.0) & ~np.isin(mask_tile, alpine_vegetation) & np.isin(pred_tile, alpine_vegetation)

                violation_counts[0] += int(np.count_nonzero(c1))
                violation_counts[1] += int(np.count_nonzero(c2))
                violation_counts[2] += int(np.count_nonzero(c3))
                violation_counts[3] += int(np.count_nonzero(c4))
                violation_counts[4] += int(np.count_nonzero(c5))
                violation_counts[5] += int(np.count_nonzero(c6))

                total_violations = c1 | c2 | c3 | c4 | c5 | c6

                # Reset violating predictions to ground truth
                result = np.where(total_violations & ~nodata_cells, mask_tile, pred_tile).astype(np.uint8)
                dst.write(result, 1, window=out_block)

                if export_mask_for_each_constraint:
                    for i, c in enumerate([c1, c2, c3, c4, c5, c6], 1):
                        tile = np.where(nodata_cells, 255, c).astype(np.uint8)
                        c_dsts[i].write(tile, 1, window=out_block)
                    total_tile = np.where(nodata_cells, 255, total_violations).astype(np.uint8)
                    c_dsts["total"].write(total_tile, 1, window=out_block)

    constraint_labels = [
        "Non-forest to mature/old forest",
        "Young growth to old forest",
        "Forest setback to younger stage",
        "Rock to grassland",
        "Water bodies on steep slopes (> 35°)",
        "Alpine vegetation in lowland (< 650m)",
    ]
    for i, (count, label) in enumerate(zip(violation_counts, constraint_labels), 1):
        print(f"{count} pixels violate constraint {i}: {label}")
