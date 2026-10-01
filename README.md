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
│       └── unet_segmentation/ # U-Net training code
├── scripts/
│   ├── run_*.sh           # Training runs per input set
│   └── srs/               # Evaluation, post-processing, tables and figures (SRS manuscript)
├── outputs/
│   ├── AGIT_metrics.xlsx  # Metrics of the arXiv/AGIT paper
│   └── srs/               # Metric tables, CIs and confusion matrices (SRS manuscript)
└── env.yml                # Conda environment specification

```

## Getting Started

### Prerequisites

- Conda or Mamba package manager
- CUDA-compatible GPU (recommended)

### Installation

**Option 1: Using Docker (Recommended for Reproducibility)**

The easiest way to reproduce our research environment is using Docker:

```bash
# Pull the pre-built image from Docker Hub
docker pull hkristen/habitalp_2:v1

# Run with GPU support
docker run --gpus all -it hkristen/habitalp_2:v1
```

See [DOCKER.md](DOCKER.md) for detailed Docker setup instructions, including building locally, using docker-compose, and running Jupyter notebooks.

**Option 2: Local Installation with Conda**

1. Clone this repository:
```bash
git clone https://github.com/hkristen/habitalp_2.git
cd habitalp_2
```

2. Create the conda environment:
```bash
conda env create -f env.yml
conda activate habitalp_2
```

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

The training scripts use Weights & Biases for experiment tracking. Set your API key:

```python
wandb_key = "YOUR_WANDB_API_KEY_HERE"  # Replace with your WandB API key
```

## Reproducing the foundation-model benchmark (SRS manuscript)

This section covers the manuscript *"Benchmarking Geospatial Foundation Models Against a Supervised Baseline for Fine-Grained Alpine Habitat Change Detection with Very High-Resolution Imagery and LiDAR"* (under review). It benchmarks five geospatial foundation models (Clay v1.0, Clay v1.5, Prithvi-EO-2.0, DOFA, TerraMind) against a U-Net baseline for post-classification change detection, with RGB+NIR and RGB+NIR+nDSM inputs.

The released metric files in `outputs/srs/` are enough to rebuild every result table and the chart figures without the raw data:

```bash
python scripts/srs/paper_tables.py   # prints all result tables of the paper
python scripts/srs/make_figures.py   # benchmark bar chart and confusion matrix
```

To recompute everything from the data, run these steps in order. Data paths in the training and inference configs are examples from our setup and need to be adapted. The evaluation scripts read the data root from `HABITALP_DATA`.

| Step | Command | Output |
|------|---------|--------|
| 1. Train | `bash scripts/run_rgb_nir_ndsm.sh` and `bash scripts/run_rgb_nir.sh` | checkpoints for the 12 configurations (6 models × 2 input sets) |
| 2. Predict 2013 and 2020 | `python src/inference/inference.py --config src/inference/configs/infer-<model>-<inputs>-<year>.yaml` | `model_output/<experiment>/<year>_*.tif` |
| 3. Evaluate | `python scripts/srs/evaluate.py --stage before` | segmentation and change metrics per experiment, change maps |
| 4. Post-process | `python scripts/srs/postprocess.py --step constraints`, then vectorisation and crown cover (see below), then `python scripts/srs/postprocess.py --step canopy-cover --polygonfix-dir <dir>` | post-processed polygons |
| 5. Evaluate post-processed | `python scripts/srs/evaluate.py --stage after` | metrics after post-processing |
| 6. Collect metrics | `python scripts/srs/build_metrics_table.py` | `outputs/srs/metrics_master.csv`, `metrics_perclass.csv` |
| 7. Confidence intervals | `python scripts/srs/bootstrap_ci.py` | `outputs/srs/metrics_ci.csv`, change confusion matrices |
| 8. Tables and figures | `python scripts/srs/paper_tables.py`, `make_figures.py`, `make_study_area_map.py`, `make_visual_comparison.py` | `outputs/srs/figures/` |

Notes:

- **Experiment names.** Experiment folders keep the internal phase suffix of the inference configs: `phaseF` = RGB+NIR, `phaseE` = RGB+NIR+nDSM. `scripts/srs/config.py` maps them to models and input sets.
- **Change maps** are built from the 2013 reference and the 2020 prediction (`src/inference/eval.py::generate_change_map`). The transition rules are in `src/trainers/utils.py::get_change_array`.
- **Confidence intervals** come from a spatial block bootstrap over 500 m × 500 m tiles (88 tiles, 1000 resamples; `src/trainers/bootstrap.py`).
- **Post-processing** has four steps. Step 1, physical-plausibility filtering, is in `src/inference/utils.py::physical_constraints_module`. Steps 2 and 3 were run outside this repository with IMPACT Tools: vectorisation onto the HabitAlp polygons (majority class, ≥90 % agreement, 5 m simplification, 0.1 ha minimum mapping unit) and LiDAR crown cover per polygon (share of nDSM above 1.3 m, attribute `CCD`). Step 4, the canopy-cover correction, is in `postprocess.py`.
- **Figures.** `make_study_area_map.py` additionally needs `NE_DIR` with the Natural Earth admin-1 10m shapefile.

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
