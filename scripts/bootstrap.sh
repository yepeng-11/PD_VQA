#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
# Point PD_VQA_BASE_PYTHON at an existing Python with the desired CUDA PyTorch.
BASE_PYTHON="${PD_VQA_BASE_PYTHON:-python3}"
if [ ! -x .venv/bin/python ]; then
  "$BASE_PYTHON" -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install -e '.[dev,models]'
