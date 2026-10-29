# HabitAlp 2.0

Official code repository for the paper **"Habitat and Land Cover Change Detection in Alpine Protected Areas: A Comparison of AI Architectures"**.

**Paper:** [arXiv:2511.00073](https://arxiv.org/abs/2511.00073)
**Dataset:** [huggingface.co/datasets/JR-DIGITAL/habitalp2.0](https://huggingface.co/datasets/JR-DIGITAL/habitalp2.0)

## Overview

This repository contains the code and notebooks used to develop deep learning models for habitat mapping in alpine environments. The project explores both change detection and semantic segmentation approaches using multi-temporal aerial imagery and foundation models.

## Repository Structure

```
.
├── notebooks/              # Jupyter notebooks for training and data processing
│   ├── data-*.ipynb       # Data download and preprocessing
│   └── train-*.ipynb      # Model training notebooks
├── src/
│   ├── data/              # Data handling and processing
│   │   ├── datamodules/   # PyTorch Lightning data modules
│   │   ├── datasets/      # Custom dataset implementations
│   │   ├── download/      # Data download utilities
│   │   ├── processing/    # Data preprocessing tools
│   │   └── samplers/      # Custom samplers for spatial data
│   ├── inference/         # Inference scripts and configs
│   │   ├── predict_*.py   # Prediction scripts for change detection
│   │   └── configs/       # Inference configuration files
│   └── trainers/          # Training tasks and utilities
│       ├── terratorch/    # Foundation-model and U-Net training configs
│       │   └── revision/  # Training configs of the MDPI revision runs
│       └── unet_segmentation/ # U-Net training code
├── scripts/
│   ├── run_*.sh           # Training runs per input set
│   ├── mdpi/              # Evaluation, post-processing, tables and figures (MDPI manuscript)
│   └── b*.py, make_r1_*.py, r1_*.sh  # Analyses, tables and run queues of the MDPI revision
├── outputs/
│   ├── AGIT_metrics.xlsx  # Metrics of the arXiv/AGIT paper
│   ├── mdpi/              # Metric tables, CIs and confusion matrices (MDPI manuscript)
│   └── r1/                # Analysis outputs of the MDPI revision
└── environment/           # uv environment (setup script and pinned requirements)

```

## Getting Started

### Prerequisites

- Linux with an NVIDIA GPU and CUDA (the pinned requirements contain CUDA wheels)
- [uv](https://docs.astral.sh/uv/)

### Installation

```bash
git clone https://github.com/JR-DIGITAL/habitalp_2.git
cd habitalp_2
environment/setup_env.sh .venv314   # the r1_*.sh run scripts default to .venv314/bin/python; override with PY=...
```

`environment/setup_env.sh` recreates the environment of the foundation-model benchmark and its revision: Python 3.14.3, torch 2.10.0, terratorch 1.2.13, torchgeo 0.9.0 and lightning 2.6.1, pinned in `environment/requirements-train.lock.txt`. The benchmark runs used terratorch 1.2.4rc1 with a Clay v1.5 fix that is upstream from 1.2.13; all six models build with identical weights and outputs in both versions (see the script header). The GDAL Python bindings are optional and only needed for the data preparation in `src/data/processing/` and `src/data/download/`. The older notebooks of the arXiv/AGIT paper were written for an earlier environment (torchgeo 0.7) and may need small changes.

### Data

Download the HabitAlp 2.0 dataset from HuggingFace:
- [huggingface.co/datasets/JR-DIGITAL/habitalp2.0](https://huggingface.co/datasets/JR-DIGITAL/habitalp2.0)

The dataset includes:
- Multi-temporal aerial orthophotos (RGB, CIR)
- Digital elevation models and derived terrain features
- Habitat classification labels
- Change detection annotations

## Usage

### Training

The `notebooks/` directory contains training notebooks for different models:

- `train-01-unet-post-classification-change.ipynb` - Post-classification change detection with U-Net
- `train-02-binary-change-unet.ipynb` - Binary change detection
- `train-03-multiclass-change-unet.ipynb` - Multi-class change detection
- `train-04-terratorch-CLAY.ipynb` - Training with CLAY foundation model

**Note:** File paths in the scripts are examples from our setup. Update them to match your local data paths.

### Inference

#### Semantic Segmentation (Post-Classification)

The main inference pipeline supports both Clay (terratorch) and U-Net models with weighted blending for large images:

```bash
# Run inference with config file (recommended)
python src/inference/inference.py --config src/inference/configs/clay_data_2020_full_roi.yaml

# Override checkpoint
python src/inference/inference.py --config src/inference/configs/clay_data_2020_full_roi.yaml \
    --checkpoint path/to/your_model.ckpt

# Override ROI
python src/inference/inference.py --config src/inference/configs/clay_data_2020_full_roi.yaml \
    --roi path/to/different_roi.gpkg

# Full extent inference (no ROI clipping)
python src/inference/inference.py --config src/inference/configs/clay_data_2020_full_roi.yaml --roi none
```

Features:
- Weighted tile blending for seamless predictions
- Support for both Clay foundation models and U-Net
- Configurable via YAML files
- ROI-based inference or full extent

#### Direct Change Detection

For binary and multi-class change detection between time periods:

```bash
# Binary change prediction
python -m src.inference.predict_binary_change

# Multi-class change prediction
python -m src.inference.predict_multiclass_change
```

**Note:** Update the file paths in these scripts before running.

### WandB Integration

The training scripts use Weights & Biases for experiment tracking. Provide your API key through the environment, not the config files:

```bash
export WANDB_API_KEY=<your key>   # or run `wandb login` once
```

`src/trainers/train_terratorch.py` reads `WANDB_API_KEY` first; the `wandb_key` entry in the configs is empty.

## Reproducing the foundation-model benchmark (Remote Sensing / MDPI manuscript)

This section covers the manuscript *"Benchmarking Geospatial Foundation Models Against a Supervised Baseline for Fine-Grained Alpine Habitat Change Detection with Very High-Resolution Imagery and LiDAR"*, a manuscript for *Remote Sensing* (MDPI). It benchmarks five geospatial foundation models (Clay v1.0, Clay v1.5, Prithvi-EO-2.0, DOFA, TerraMind) against a U-Net baseline for post-classification change detection, with RGB+NIR and RGB+NIR+nDSM inputs.

This section and `outputs/mdpi/` belong to the **submitted version** of the manuscript. For the revised version, see [Revised manuscript](#revised-manuscript-review-round-1) below: it retrains all models under one protocol and fixes the change rule for dwarf forest, so with the current code `scripts/mdpi/evaluate.py` no longer reproduces the change-detection numbers in `outputs/mdpi/`.

The released metric files in `outputs/mdpi/` are enough to rebuild every result table of the submitted version and the chart figures without the raw data:

```bash
python scripts/mdpi/paper_tables.py   # prints all result tables of the submitted version
python scripts/mdpi/make_figures.py   # benchmark bar chart and confusion matrix
```

To recompute everything from the data, download version v4 (or v4.1, which adds the model outputs) of the [HabitAlp 2.0 dataset](https://huggingface.co/datasets/JR-DIGITAL/habitalp2.0) and set `HABITALP_DATA` to its folder; the evaluation scripts read all labels from there. The paper uses the v4 labels, whose canopy-cover classes are derived from LiDAR (see the dataset card); version v3 belongs to the arXiv preprint. Data paths in the training and inference configs are examples from our setup and need to be adapted.

| Step | Command | Output |
|------|---------|--------|
| 1. Train | `bash scripts/run_rgb_nir_ndsm.sh` and `bash scripts/run_rgb_nir.sh` | checkpoints for the 12 configurations (6 models × 2 input sets) |
| 2. Predict 2013 and 2020 | `python src/inference/inference.py --config src/inference/configs/infer-<model>-<inputs>-<year>.yaml` | `model_output/<experiment>/<year>_*.tif` |
| 3. Evaluate | `python scripts/mdpi/evaluate.py --stage before` | segmentation and change metrics per experiment, change maps |
| 4. Post-process | `python scripts/mdpi/postprocess.py --step constraints`, then vectorisation and crown cover (see below), then `python scripts/mdpi/postprocess.py --step canopy-cover --polygonfix-dir <dir>` | post-processed polygons |
| 5. Evaluate post-processed | `python scripts/mdpi/evaluate.py --stage after` | metrics after post-processing |
| 6. Collect metrics | `python scripts/mdpi/build_metrics_table.py` | `outputs/mdpi/metrics_master.csv`, `metrics_perclass.csv` |
| 7. Confidence intervals | `python scripts/mdpi/bootstrap_ci.py` | `outputs/mdpi/metrics_ci.csv`, change confusion matrices |
| 8. Tables and figures | `python scripts/mdpi/paper_tables.py`, `make_figures.py`, `make_study_area_map.py`, `make_visual_comparison.py` | `outputs/mdpi/figures/` |

Notes:

- **Experiment names.** Experiment folders keep the internal phase suffix of the inference configs: `phaseF` = RGB+NIR, `phaseE` = RGB+NIR+nDSM. `scripts/mdpi/config.py` maps them to models and input sets.
- **Change maps** are built from the 2013 reference and the 2020 prediction (`src/inference/eval.py::generate_change_map`). The transition rules are in `src/trainers/utils.py::get_change_array`.
- **Confidence intervals** come from a spatial block bootstrap over 500 m × 500 m tiles (88 tiles, 1000 resamples; `src/trainers/bootstrap.py`).
- **Post-processing** has four steps. Step 1, physical-plausibility filtering, is in `src/inference/utils.py::physical_constraints_module`. Steps 2 and 3 were run outside this repository with IMPACT Tools: vectorisation onto the HabitAlp polygons (majority class, ≥90 % agreement, 5 m simplification, 0.1 ha minimum mapping unit) and LiDAR crown cover per polygon (share of nDSM above 1.3 m, attribute `CCD`). Step 4, the canopy-cover correction, is in `postprocess.py`.
- **Figures.** `make_study_area_map.py` additionally needs `NE_DIR` with the Natural Earth admin-1 10m shapefile.

### Revised manuscript (review round 1)

For the revision every model was retrained under one protocol, with three seeds per model. The settings are in the configs in `src/trainers/terratorch/revision/`, which `scripts/make_r1_configs.py` generates from the RGB+NIR+nDSM configs in `src/trainers/terratorch/`:

- **Masked split:** the 2020 test patches plus a 128 m buffer are kept out of training and validation sampling (`exclusion` in the configs, `RandomGeoSamplerWithinRoi`).
- **Training:** batch size 4, a fixed budget of 200 epochs (no early stopping), 10-epoch linear warm-up followed by a cosine schedule with T_max 190, fixed training seeds 0 to 2; evaluation uses the best validation checkpoint.
- **Runs:** six models × three seeds on the masked split, a ResNet-50 U-Net, U-Net decoder checks for Clay v1.0 and Prithvi, TerraMind at LR 5e-5, and two MiT-B2 controls (block-drop split, original split). `scripts/r1_queue_all.txt` lists all 30 runs.
- **Change rule:** dwarf forest (class 20) is no longer in the non-forest/shrubland group of `src/trainers/utils.py::get_change_array`; it is its own group, so dwarf forest to forest-stage transitions count as "Other transition".

The analysis outputs in `outputs/r1/` are enough to rebuild every revision table without the raw data; the result is identical to the tables in the revised manuscript:

```bash
python scripts/make_r1_tables.py    # LaTeX tables -> paper/srs-mdpi/revision/tables/
python scripts/make_r1_figures.py   # Figures 3 and 4 -> paper/srs-mdpi/
python scripts/b1_paired_bootstrap.py   # paired hierarchical bootstrap (tiles and seeds) from outputs/r1/b9_tile_cms.npz
```

**Post-processing table from the released dataset.** The post-processing results (Table 11 of the revised manuscript, "Effect of the full operational post-processing ... on cross-temporal change-detection macro IoU") can be recomputed from the [HabitAlp 2.0 dataset](https://huggingface.co/datasets/JR-DIGITAL/habitalp2.0) v4.1 alone, which contains the raw and post-processed 2020 predictions of the twelve original runs in `model_outputs/`:

```bash
hf download JR-DIGITAL/habitalp2.0 --repo-type dataset --revision v4.1 --local-dir habitalp2.0 \
    --include "labels/classes_2013.tif" "labels/habitalp_change_2013_2020.tif" \
    "model_outputs/*/prediction_2020.tif" "model_outputs/*/prediction_2020_postprocessed.tif"   # about 600 MB
python scripts/r1_postproc_from_hf.py --hf-root habitalp2.0
```

It writes `outputs/r1/hf_postproc_metrics.csv`, compares it with the released `outputs/r1/b0_dwarffix_cd_metrics.csv` and exits with an error if any value differs.

**Full pipeline.** The other revision scripts read the folder layout of our data store under `HABITALP_DATA_ROOT` (default `data/Originaldaten`; not to be confused with `HABITALP_DATA`, which the submitted-version scripts in `scripts/mdpi/` use for the Hugging Face layout; paths at the top of each script, e.g. `processed/mask/classes_v4_2013.tif`, `habitalp_change/habitalp_change_v4_2013-2020.tif`, `model_output/<experiment>/2020_<experiment>.tif`). With the Hugging Face dataset, point these paths at the corresponding files (`labels/classes_2013.tif`, `labels/habitalp_change_2013_2020.tif`, ...). The rasters of the revision runs are not released; they follow from steps 3 and 4. Run the steps in this order, since later scripts read earlier outputs:

| Step | Command | Output |
|------|---------|--------|
| 1. Original runs | steps 1 to 5 of the submitted-version table above (training, prediction, `scripts/mdpi/evaluate.py`, post-processing) | predictions, `classification_metrics_<year>.csv`, `constraint_resolution/` and the post-processed rasters per experiment, read by B0, B4, B5 and B9. Without IMPACT Tools, copy the released `model_outputs/<experiment>/prediction_2020_postprocessed.tif` to `model_output/<experiment>/constraint_resolution_+_polygonfix_cc_updated/2020_re-rasterized.tif` before `evaluate.py --stage after` |
| 2. Dwarf-forest fix | `python scripts/b0_dwarffix_cd.py` | change maps in `model_output/<experiment>/r1_dwarffix/`, `outputs/r1/b0_dwarffix_cd_metrics.csv` |
| 3. Train revision runs | `scripts/r1_queue.sh scripts/r1_queue_all.txt 2` (or `python -m src.trainers.train_terratorch --config src/trainers/terratorch/revision/config-<run>.yaml` per run) | checkpoints in `models/<run>/` |
| 4. Predict 2013 and 2020 | `scripts/r1_infer_queue.sh 2` (writes configs with `scripts/make_r1_infer_configs.py` for every run in `models/` that has a best checkpoint and no failed queue status) | `model_output/<run>/<year>_<run>.tif` |
| 5. Bootstrap tiles | `python scripts/b2_bootstrap_dropout.py` | `outputs/r1/b2_*.csv`, `b2_tile_cms.npz` (needs step 2) |
| 6. Test metrics over seeds | `python scripts/b9_seed_aggregate.py`, then `python scripts/b9_perclass.py` | `outputs/r1/b9_*.csv`, `b9_tile_cms.npz` (needs steps 1, 2, 4, 5) |
| 7. Paired comparison | `python scripts/b1_paired_bootstrap.py` | `outputs/r1/b1_paired.csv` (needs step 6) |
| 8. Further analyses | `scripts/b3_*.py` to `scripts/b8_*.py`, `scripts/b10_*.py`, `scripts/c2_split_label_loss.py` | `outputs/r1/b*_*.csv`, `c2_*.csv` |
| 9. Tables and figures | `python scripts/make_r1_tables.py`, `python scripts/make_r1_figures.py` | LaTeX tables, figures |

`b8_training_facts.py` and the `wandb` stage of `b10_efficiency.py` read the training logs from our W&B project, which is not public; their outputs are released in `outputs/r1/`.

What each analysis covers (the docstring of each script has the details):

| Script | Content |
|--------|---------|
| `b0_dwarffix_cd.py` | change maps and CD metrics of the original runs with the dwarf-forest fix |
| `b1_paired_bootstrap.py` | paired bootstrap of macro IoU differences between models |
| `b2_bootstrap_dropout.py` | class drop-out in the block bootstrap |
| `b3_cc_collapsed_cd.py` | change detection with the LiDAR canopy-cover classes collapsed |
| `b4_ndsm_ablation_2020.py` | nDSM ablation for single-date segmentation |
| `b5_postproc_steps.py` | post-processing step by step, cost of the stage-regression rule |
| `b6_direct_cd.py` | direct change-detection models under the current evaluation |
| `b7_resampling_check.py` | 0.5 m to 0.2 m resampling of the evaluation |
| `b8_training_facts.py` | training facts and learning rates of the revision runs (from W&B) |
| `b9_seed_aggregate.py`, `b9_perclass.py` | test metrics of all revision runs over seeds, per class |
| `b10_efficiency.py` | parameters, GFLOPs, inference throughput and training cost per model |
| `c2_split_label_loss.py` | labels lost by the alternative exclusion splits |
| `r1_lr_find.py` | learning-rate range test for a revision config |
| `r1_postproc_from_hf.py` | post-processing table from the released dataset |
| `check_change_rule.py` | check of the dwarf-forest change rule |

The post-processing results (B5, Figure 5) use the original runs, one per model on the original split, as stated in the manuscript.

## Key Features

- **Direct CD**: Binary and multi-class change detection between time periods
- **Post-CLassifcation CD (Semantic Segmentation)**: Direct habitat classification using foundation models (CLAY, Prithvi)
- **Geospatial Data Handling**: Custom samplers and datasets for working with large rasters
- **Multi-Modal Input**: Support for RGB, CIR, and terrain features
- **Production-Ready Inference**: Scripts for predicting on large images with tiling

## Citation

If you use this code or dataset in your research, please cite our paper:

```bibtex
@misc{kristen2025habitatlandcoverchange,
      title={Habitat and Land Cover Change Detection in Alpine Protected Areas: A Comparison of AI Architectures}, 
      author={Harald Kristen and Daniel Kulmer and Manuela Hirschmugl},
      year={2025},
      eprint={2511.00073},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.00073}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was conducted as part of the HabitAlp 2.0 project for alpine habitat monitoring and conservation.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
