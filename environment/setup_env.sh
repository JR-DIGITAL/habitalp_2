#!/usr/bin/env bash
# Recreate the training environment of the paper runs (W&B Phase C-F, March 2026) with uv.
#
# Versions come from the W&B requirements.txt of the Phase D training runs:
# Python 3.14.3, torch 2.10.0, terratorch 1.2.4rc1, torchgeo 0.9.0, lightning 2.6.1.
# The paper runs used terratorch 1.2.4rc1 with the Clay v1.5 factory fix
# (branch fix/clay-1.5-segmentation-factory), which is upstream from 1.2.13. We pin 1.2.13:
# all six Phase E models build with bit-identical weights and outputs in both versions, and
# the paper Clay v1.5 checkpoint (W&B model-80dh36vg) loads with 0 missing / 0 unexpected keys.
#
# Usage: environment/setup_env.sh [venv_dir]   (default: .venv)
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${1:-.venv}
PY=$VENV/bin/python

uv venv --python 3.14.3 "$VENV"
uv pip install --python "$PY" -r environment/requirements-train.lock.txt

# GDAL Python bindings (osgeo) are optional: only the data preparation scripts use them,
# training and inference run without. Built against the system GDAL if one is installed.
if command -v gdal-config >/dev/null; then
    uv pip install --python "$PY" setuptools wheel
    uv pip install --python "$PY" --no-build-isolation "gdal==$(gdal-config --version)"
    # system GDAL links the system libjpeg, which must load before the copies bundled in wheels
    for lib in /usr/lib64/libjpeg.so.8 /usr/lib/x86_64-linux-gnu/libjpeg.so.8; do
        if [ -e "$lib" ]; then
            echo "export LD_PRELOAD=$lib" >> "$VENV/bin/activate"
            echo "LD_PRELOAD=$lib added to $VENV/bin/activate (also set it for non-activated runs)"
            break
        fi
    done
else
    echo "no system GDAL found: skipping the optional osgeo bindings (not needed for training or inference)"
fi

LD_PRELOAD=${LD_PRELOAD:-} "$PY" -c "import torch, terratorch, torchgeo; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| torchgeo', torchgeo.__version__)"
