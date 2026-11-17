#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python}"
BM25_STUDENT_ID="${1:-2024320020}"
BM25_WORKERS="${2:-8}"
"$PYTHON_BIN" scripts/prepare_data.py
"$PYTHON_BIN" -m unittest discover -s tests -v
"$PYTHON_BIN" scripts/tune_bm25.py --workers "$BM25_WORKERS"
"$PYTHON_BIN" scripts/finish_bm25.py --predict-test --student-id "$BM25_STUDENT_ID" --workers "$BM25_WORKERS"
