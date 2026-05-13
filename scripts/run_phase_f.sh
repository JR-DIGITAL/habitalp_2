#!/usr/bin/env bash
# Phase F: RGB+NIR ablation (no nDSM), all 6 models
# Usage: bash scripts/run_phase_f.sh
set -euo pipefail

mkdir -p logs/phase_f

configs=(
  "src/trainers/terratorch/config-clay-v1-rgb-nir-phaseF.yaml"
  "src/trainers/terratorch/config-clay-v1.5-rgb-nir-phaseF.yaml"
  "src/trainers/terratorch/config-dofa-base-rgb-nir-phaseF.yaml"
  "src/trainers/terratorch/config-prithvi-v2-300-rgb-nir-phaseF.yaml"
  "src/trainers/terratorch/config-terramind-rgb-nir-phaseF.yaml"
)

for cfg in "${configs[@]}"; do
  name=$(basename "$cfg" .yaml)
  echo "====== Starting: $name ======"
  python -m src.trainers.train_terratorch --config "$cfg" 2>&1 | tee "logs/phase_f/${name}.log"
  echo "====== Finished: $name ======"
done

echo "Phase F complete."
