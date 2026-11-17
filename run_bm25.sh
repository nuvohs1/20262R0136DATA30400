#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" scripts/prepare_data.py
"$PYTHON_BIN" -m unittest discover -s tests -v
"$PYTHON_BIN" scripts/run_bm25.py "$@"
