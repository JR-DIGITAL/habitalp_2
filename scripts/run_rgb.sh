#!/usr/bin/env bash
# RGB-only training (Clay v1.5, DOFA, TerraMind)
# Usage: bash scripts/run_rgb.sh
set -euo pipefail

mkdir -p logs/phase_g

configs=(
  "src/trainers/terratorch/config-clay-v1.5-rgb.yaml"
  "src/trainers/terratorch/config-dofa-base-rgb.yaml"
  "src/trainers/terratorch/config-terramind-rgb.yaml"
)

for cfg in "${configs[@]}"; do
  name=$(basename "$cfg" .yaml)
  echo "====== Starting: $name ======"
  python -m src.trainers.train_terratorch --config "$cfg" 2>&1 | tee "logs/phase_g/${name}.log"
  echo "====== Finished: $name ======"
done

echo "Phase G complete."
echo ""
echo "Next steps:"
echo "  1. Note the W&B run IDs from the logs above."
echo "  2. Run predict-04-terratorch-infer_on_whole_image.ipynb for year=2013 and year=2020"
echo "     for each of: clay-v1.5-rgb, dofa-base-rgb-phaseG, terramind-rgb-phaseG"
echo "  3. Add the 3 new experiments + W&B run IDs to evaluate-06-batch-wise_evaluation.ipynb"
echo "  4. Run evaluate-06 to generate classification_metrics_*.csv and change_metrics_*.csv"
echo "  5. Update Table 1, Table 2, and Table 3 in the AGIT paper."
