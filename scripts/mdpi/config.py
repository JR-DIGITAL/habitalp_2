"""Shared paths and experiment registry for the MDPI manuscript scripts.

Set HABITALP_DATA to a local copy of the HabitAlp 2.0 dataset, version v4
(https://huggingface.co/datasets/JR-DIGITAL/habitalp2.0, folders labels/,
data_2013/, data_2020/, splits/). Model predictions are expected under
$HABITALP_DATA/model_output/<experiment>/; point output.dir of the inference
configs in src/inference/configs/ there.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs" / "mdpi"


def data_root() -> Path:
    root = os.environ.get("HABITALP_DATA")
    if not root:
        raise SystemExit("Set HABITALP_DATA to the HabitAlp 2.0 data root.")
    return Path(root)


# Dataset files, relative to the data root (HuggingFace layout, v4 labels)
REF_2003 = "labels/classes_2003.tif"
REF_2013 = "labels/classes_2013.tif"
REF_2013_TEST_CELLS = "labels/classes_2013_test_cells.tif"
REF_2020 = "labels/classes_2020.tif"
REF_CHANGE_2013_2020 = "labels/habitalp_change_2013_2020.tif"
DTM = "data_2013/dtm.tif"
SLOPE = "data_2013/slope.tif"
RGB_2013 = "data_2013/aerial_rgb_2013_2015.tif"
RGB_2020 = "data_2020/aerial_rgb_2019_2021.tif"
OUTLINES = "splits/outlines.gpkg"  # study area (year 2013) and test patches (year 2020)

# Subfolder of model_output/<experiment>/ that holds the post-processed results
PP_DIR = "constraint_resolution_+_polygonfix_cc_updated"

# Input modalities used in the MDPI paper. The experiment folder names keep the
# internal phase suffix of the inference configs (F = RGB+NIR, E = RGB+NIR+nDSM;
# the RGB-only phase G belongs to the AGIT paper).
MODALITIES = {"F": "RGB+NIR", "E": "RGB+NIR+nDSM"}

# model display name -> {modality phase: experiment folder}
EXPERIMENTS: dict[str, dict[str, str]] = {
    "Clay v1.0": {
        "F": "clay-v1-rgb-nir-phaseF",
        "E": "clay-v1-rgb-nir-ndsm-phaseE",
    },
    "Clay v1.5": {
        "F": "clay-v1.5-rgb-nir-phaseF",
        "E": "clay-v1.5-rgb-nir-ndsm-phaseE",
    },
    "Prithvi": {
        "F": "prithvi-v2-300-rgb-nir-phaseF",
        "E": "prithvi-v2-300-rgb-nir-ndsm-phaseE",
    },
    "DOFA": {
        "F": "dofa-base-rgb-nir-phaseF",
        "E": "dofa-base-rgb-nir-ndsm-phaseE",
    },
    "TerraMind": {
        "F": "terramind-rgb-nir-phaseF",
        "E": "terramind-rgb-nir-ndsm-phaseE",
    },
    "U-Net": {
        "F": "unet-mit-b2-rgb-nir-phaseF",
        "E": "unet-mit-b2-rgb-nir-ndsm-phaseE",
    },
}
MODEL_TYPE = {m: ("CNN" if m == "U-Net" else "GFM") for m in EXPERIMENTS}


def all_experiments() -> list[str]:
    return [exp for phases in EXPERIMENTS.values() for exp in phases.values()]


# Change classes, ids as produced by src/trainers/utils.py::get_change_array
CHANGE_CLASSES = [
    "No change", "Mature Tree Density Loss", "Old Growth Density Loss",
    "Forest Setback YoungLoss", "Forest Stage Progression", "Forest Density Gain",
    "Early Forest Establishment", "Clearcut Loss", "Other Transition",
]
BINARY_CLASSES = ["No change", "Change"]
