#!/usr/bin/env bash
# Multi-stage ranking (after run_gpu.sh and run_dense_eval.sh): first-stage candidates from
# BM25 v2 + LTR BERT, then a BERT cross-encoder reranker. Needs a GPU for the cross-encoder.
#
# Usage: bash run_rerank.sh [STUDENT_ID] [WORKERS] [all|candidates|dev|full|eval]
# Finished steps are skipped, so the script can be re-run after an interruption.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python}"
STUDENT_ID="${1:-2024320020}"
WORKERS="${2:-8}"
STAGE="${3:-all}"
R=outputs/rerank

completed() {  # completed <train_log.json>
  [ -f "$1" ] && "$PYTHON_BIN" -c "import json,sys; sys.exit(0 if json.load(open('$1')).get('completed') else 1)"
}

if [ "$STAGE" = all ] || [ "$STAGE" = candidates ]; then
  echo "== Stage 1 candidates (CPU) =="
  [ -f $R/candidates_test.json ] || "$PYTHON_BIN" scripts/make_candidates.py --workers "$WORKERS"
fi

if [ "$STAGE" = all ] || [ "$STAGE" = dev ]; then
  echo "== Cross-encoder model A: 12,000 queries, epoch chosen on the tuning slice; validation scores =="
  completed $R/ce-dev/train_log.json || "$PYTHON_BIN" scripts/train_cross_encoder.py \
    --negatives outputs/ltr/negatives_dev.json --tune-candidates $R/candidates_tune.json --output-dir $R/ce-dev
  [ -f $R/ce-dev/scores_validation.npy ] || "$PYTHON_BIN" scripts/score_cross_encoder.py \
    --checkpoint $R/ce-dev/checkpoint --candidates $R/candidates_validation.json
fi

if [ "$STAGE" = all ] || [ "$STAGE" = full ]; then
  echo "== Cross-encoder model B: all 20,000 queries, stopped at model A's epoch; test scores =="
  K=$("$PYTHON_BIN" -c "import json; print(json.load(open('$R/ce-dev/train_log.json'))['selected_epoch'])")
  completed $R/ce-full/train_log.json || "$PYTHON_BIN" scripts/train_cross_encoder.py \
    --negatives outputs/ltr/negatives_full.json --output-dir $R/ce-full --stop-after-epoch "$K"
  [ -f $R/ce-full/scores_test.npy ] || "$PYTHON_BIN" scripts/score_cross_encoder.py \
    --checkpoint $R/ce-full/checkpoint --candidates $R/candidates_test.json
fi

if [ "$STAGE" = all ] || [ "$STAGE" = eval ]; then
  echo "== Fusion weight on the tuning slice, validation once, Kaggle file =="
  "$PYTHON_BIN" scripts/run_rerank_eval.py --student-id "$STUDENT_ID"
  "$PYTHON_BIN" scripts/validate_submission.py "$R/submission/${STUDENT_ID}_final.csv"
fi
echo "Rerank stage '$STAGE' finished."
