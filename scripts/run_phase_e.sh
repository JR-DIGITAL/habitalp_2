#!/usr/bin/env bash
# Phase E: RGB+NIR+nDSM training, all 6 models
# Usage: bash scripts/run_phase_e.sh
set -euo pipefail

mkdir -p logs/phase_e

configs=(
  "src/trainers/terratorch/config-clay-v1-rgb-nir-ndsm-phaseE.yaml"
  "src/trainers/terratorch/config-clay-v1.5-rgb-nir-ndsm-phaseE.yaml"
  "src/trainers/terratorch/config-dofa-base-rgb-nir-ndsm-phaseE.yaml"
  "src/trainers/terratorch/config-prithvi-v2-300-rgb-nir-ndsm-phaseE.yaml"
  "src/trainers/terratorch/config-terramind-rgb-nir-ndsm-phaseE.yaml"
)

for cfg in "${configs[@]}"; do
  name=$(basename "$cfg" .yaml)
  echo "====== Starting: $name ======"
  python -m src.trainers.train_terratorch --config "$cfg" 2>&1 | tee "logs/phase_e/${name}.log"
  echo "====== Finished: $name ======"
done

echo "Phase E complete."
