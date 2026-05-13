#!/usr/bin/env bash
# Phase G inference: RGB-only predictions for Clay v1.5, DOFA, TerraMind
# Runs 2013 and 2020 for each model, outputs to model_output/<experiment>/
# Run inside tmux: bash scripts/run_phase_g_inference.sh
set -euo pipefail

REPO=/home/hkristen/habitalp2
CONFIGS=$REPO/src/inference/configs

run() {
    local config=$1
    local experiment=$2
    local year=$3
    echo "====== ${experiment} (${year}) ======"
    conda run --no-capture-output -n habitalp2 python "$REPO/src/inference/inference.py" \
        --config "$CONFIGS/$config" \
        --output "${year}_${experiment}"
    echo "====== Done: ${experiment} (${year}) ======"
}

run infer-clay-v1.5-rgb-2013.yaml   clay-v1.5-rgb   2013
run infer-clay-v1.5-rgb-2020.yaml   clay-v1.5-rgb   2020
run infer-dofa-base-rgb-2013.yaml   dofa-base-rgb   2013
run infer-dofa-base-rgb-2020.yaml   dofa-base-rgb   2020
run infer-terramind-rgb-2013.yaml   terramind-rgb   2013
run infer-terramind-rgb-2020.yaml   terramind-rgb   2020

echo ""
echo "Phase G inference complete."
echo "Next: run evaluate-06-batch-wise_evaluation.ipynb (add phaseG entries first)"
