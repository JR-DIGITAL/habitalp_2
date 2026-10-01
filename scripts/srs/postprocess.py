"""Operational post-processing of the model predictions (SRS manuscript, RQ4).

The pipeline has four steps:
  1. Physical-plausibility filtering (this script, --step constraints): pixels
     with one of six implausible transitions relative to the previous reference
     map are reset to the reference class. Implemented in
     src/inference/utils.py::physical_constraints_module.
  2. Vectorisation onto the HabitAlp polygon geometry (majority class per
     polygon, >=90 % agreement, 5 m simplification, 0.1 ha minimum mapping unit).
  3. LiDAR crown cover per polygon (share of nDSM above 1.3 m), stored in the
     polygon attribute CCD.
  4. Canopy-cover correction (this script, --step canopy-cover): forest classes
     are set to their CC<80 or CC>=80 variant from the polygon's LiDAR crown cover.

Steps 2 and 3 were run outside this repository with IMPACT Tools. Their output,
one GeoPackage per experiment and year named
    <polygonfix-dir>/<experiment>/prediction_<year>_constraint_resolution_polygonfix_fp.gpkg
with the attributes Class and CCD, is the input of step 4.

Afterwards run `evaluate.py --stage after`.

Usage:
    HABITALP_DATA=/path/to/data python scripts/srs/postprocess.py --step constraints
    HABITALP_DATA=/path/to/data python scripts/srs/postprocess.py --step canopy-cover \
        --polygonfix-dir /path/to/polygonfix_output
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DTM, PP_DIR, REF_2013, SLOPE, all_experiments, data_root

# The 2013 prediction is checked against the 2003 reference, the 2020 prediction
# against the 2013 reference.
PREVIOUS_REFERENCE = {2013: "processed/mask/classes_v3_2003.tif", 2020: REF_2013}

# forest class -> (CC<80 id, CC>=80 id)
CANOPY_COVER_PAIRS = {
    "Conif. pole timber": (8, 9),
    "Conif. mature forest": (11, 12),
    "Broadl. mature forest": (13, 14),
    "Old conif. forest, multilayered": (15, 16),
    "Old broadl. forest, multilayered": (17, 18),
}


def step_constraints(root: Path, experiments: list[str]) -> None:
    from src.inference.utils import physical_constraints_module

    for exp in experiments:
        folder = root / "model_output" / exp
        for year, ref in PREVIOUS_REFERENCE.items():
            preds = sorted(folder.glob(f"{year}_*.tif"))
            if len(preds) != 1:
                print(f"!! {exp}: expected one {year}_*.tif, found {len(preds)}; skipping")
                continue
            print(f"--- {exp} {year}: {preds[0].name}")
            physical_constraints_module(
                mask_path=root / ref,
                prediction_path=preds[0],
                dtm_path=root / DTM,
                slope_path=root / SLOPE,
                experiment_dir=folder / "constraint_resolution",
                output_name=f"prediction_{year}_constraint_resolution",
                export_mask_for_each_constraint=False,
            )


def step_canopy_cover(root: Path, experiments: list[str], polygonfix_dir: Path) -> None:
    import geopandas as gpd

    for exp in experiments:
        out_folder = root / "model_output" / exp / PP_DIR
        out_folder.mkdir(parents=True, exist_ok=True)
        for year in (2013, 2020):
            src = polygonfix_dir / exp / f"prediction_{year}_constraint_resolution_polygonfix_fp.gpkg"
            if not src.exists():
                print(f"!! missing {src}; skipping")
                continue
            polys = gpd.read_file(src).set_index("OBJECTID")
            crown_cover = polys["CCD"].replace(-99999, 0)
            for low, high in CANOPY_COVER_PAIRS.values():
                forest = polys["Class"].isin([low, high])
                polys.loc[forest & (crown_cover < 80), "Class"] = low
                polys.loc[forest & (crown_cover >= 80), "Class"] = high
            out = out_folder / f"prediction_{year}_constraint_resolution_cc_updated.gpkg"
            polys.to_file(out)
            print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--step", choices=["constraints", "canopy-cover"], required=True)
    ap.add_argument("--polygonfix-dir", type=Path,
                    help="output of steps 2-3, one subfolder per experiment")
    ap.add_argument("--experiments", nargs="*", default=None,
                    help="experiment folders (default: all in config.py)")
    args = ap.parse_args()

    root = data_root()
    experiments = args.experiments or all_experiments()
    if args.step == "constraints":
        step_constraints(root, experiments)
    else:
        if args.polygonfix_dir is None:
            ap.error("--step canopy-cover needs --polygonfix-dir")
        step_canopy_cover(root, experiments, args.polygonfix_dir)


if __name__ == "__main__":
    main()
