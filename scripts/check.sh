#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
PYTHON="${PD_VQA_PYTHON:-.venv/bin/python}"
"$PYTHON" -m pytest -q
"$PYTHON" -m ruff check .
