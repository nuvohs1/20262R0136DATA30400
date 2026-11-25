#!/usr/bin/env bash
# Category labels as auxiliary information (after run_rerank.sh). A label-GCN classifier, trained
# multi-task on queries and products, adds a category-agreement score to the step-4b reranker.
#   encode : training-review embeddings from LTR model B (GPU, ~2 min; needed by the full model)
#   dev    : classifier on 12,000 development-training queries, epoch on the tuning slice
#   full   : classifier on all 20,000 training queries, stopped at the dev epoch
#   eval   : (w, beta) on the tuning slice, validation once, Kaggle file
#
# Usage: bash run_category.sh [STUDENT_ID] [all|encode|dev|full|eval]
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python}"
STUDENT_ID="${1:-2024320020}"
STAGE="${2:-all}"
C=outputs/category

completed() {  # completed <train_log.json>
  [ -f "$1" ] && "$PYTHON_BIN" -c "import json,sys; sys.exit(0 if json.load(open('$1')).get('completed') else 1)"
}

if [ "$STAGE" = all ] || [ "$STAGE" = encode ]; then
  echo "== Training-review embeddings from LTR model B =="
  [ -f outputs/dense/bert-ltr-full/train_reviews.npy ] || "$PYTHON_BIN" scripts/encode_dense.py \
    --model outputs/ltr/bert-ltr-full/checkpoint --tag bert-ltr-full --parts train
fi

if [ "$STAGE" = all ] || [ "$STAGE" = dev ]; then
  echo "== Category model A: 12,000 queries, epoch chosen on the tuning slice =="
  completed $C/dev/train_log.json || "$PYTHON_BIN" scripts/train_category.py --set dev
fi

if [ "$STAGE" = all ] || [ "$STAGE" = full ]; then
  echo "== Category model B: all 20,000 queries, stopped at model A's epoch =="
  K=$("$PYTHON_BIN" -c "import json; print(json.load(open('$C/dev/train_log.json'))['selected_epoch'])")
  completed $C/full/train_log.json || "$PYTHON_BIN" scripts/train_category.py --set full --stop-after-epoch "$K"
fi

if [ "$STAGE" = all ] || [ "$STAGE" = eval ]; then
  echo "== (w, beta) on the tuning slice, validation once, Kaggle file =="
  "$PYTHON_BIN" scripts/run_category_eval.py --student-id "$STUDENT_ID"
  "$PYTHON_BIN" scripts/validate_submission.py "$C/submission/${STUDENT_ID}_final.csv"
fi
echo "Category stage '$STAGE' finished."
