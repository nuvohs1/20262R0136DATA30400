#!/usr/bin/env bash
# CPU stage after run_gpu.sh: evaluate dense retrieval (zero-shot BERT) and the learning-to-rank
# fine-tuned BERT, alone and fused with BM25 v2, and write validated Kaggle-format files.
#
# Usage: bash run_dense_eval.sh [STUDENT_ID] [WORKERS] [all|zeroshot|ltr]
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python}"
STUDENT_ID="${1:-2024320020}"
WORKERS="${2:-8}"
STAGE="${3:-all}"

if [ "$STAGE" = all ] || [ "$STAGE" = zeroshot ]; then
  echo "== Dense retrieval: zero-shot bert-base-uncased =="
  "$PYTHON_BIN" scripts/run_dense.py --embeddings outputs/dense/bert-base-uncased \
    --output-dir outputs/dense_eval/bert-base-uncased --predict-test --student-id "$STUDENT_ID" --workers "$WORKERS"
  for f in dense_only hybrid_bm25_expansion_dense; do
    "$PYTHON_BIN" scripts/validate_submission.py "outputs/dense_eval/bert-base-uncased/$f/${STUDENT_ID}_final.csv"
  done
fi

if [ "$STAGE" = all ] || [ "$STAGE" = ltr ]; then
  echo "== Learning to rank: fine-tuned BERT (model A for validation, model B for test) =="
  "$PYTHON_BIN" scripts/run_dense.py --embeddings outputs/dense/bert-ltr-dev --test-embeddings outputs/dense/bert-ltr-full \
    --output-dir outputs/dense_eval/bert-ltr --predict-test --student-id "$STUDENT_ID" --workers "$WORKERS"
  for f in dense_only hybrid_bm25_expansion_dense; do
    "$PYTHON_BIN" scripts/validate_submission.py "outputs/dense_eval/bert-ltr/$f/${STUDENT_ID}_final.csv"
  done
fi
echo "Dense evaluation stage '$STAGE' finished."
