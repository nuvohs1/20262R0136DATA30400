#!/usr/bin/env bash
# Cross-encoder with hybrid hard negatives (after run_rerank.sh and run_category.sh encode).
# Same reranker as step 4b, but training negatives come from the hybrid BM25 v2 + LTR BERT
# first stage, i.e. from the same distribution as the 50 candidates it reranks.
#   mine : hybrid negatives for the 12,000 dev and 20,000 full training queries (CPU)
#   dev  : cross-encoder A on 12,000 queries, epoch chosen on the tuning slice; validation scores
#   full : cross-encoder B on all 20,000 queries, stopped at A's epoch; test scores
#   eval : fusion weight on the tuning slice, validation once, Kaggle file
#
# Usage: bash run_hardneg.sh [STUDENT_ID] [WORKERS] [all|mine|dev|full|eval]
# Finished steps are skipped, so the script can be re-run after an interruption.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python}"
STUDENT_ID="${1:-2024320020}"
WORKERS="${2:-8}"
STAGE="${3:-all}"
R=outputs/rerank
H=outputs/hardneg

completed() {  # completed <train_log.json>
  [ -f "$1" ] && "$PYTHON_BIN" -c "import json,sys; sys.exit(0 if json.load(open('$1')).get('completed') else 1)"
}

for f in $R/candidates_tune.json $R/candidates_validation.json $R/candidates_test.json \
         outputs/dense/bert-ltr-dev/train_reviews.npy outputs/dense/bert-ltr-full/train_reviews.npy; do
  [ -f "$f" ] || { echo "Missing $f: run run_rerank.sh and run_category.sh encode first"; exit 1; }
done

if [ "$STAGE" = all ] || [ "$STAGE" = mine ]; then
  echo "== Hybrid hard negatives (CPU) =="
  [ -f $H/negatives_full.json ] || "$PYTHON_BIN" scripts/mine_hybrid_negatives.py --workers "$WORKERS"
fi

if [ "$STAGE" = all ] || [ "$STAGE" = dev ]; then
  echo "== Cross-encoder A with hybrid negatives: 12,000 queries; validation scores =="
  completed $H/ce-dev/train_log.json || "$PYTHON_BIN" scripts/train_cross_encoder.py \
    --negatives $H/negatives_dev.json --tune-candidates $R/candidates_tune.json --output-dir $H/ce-dev
  [ -f $H/ce-dev/scores_validation.npy ] || "$PYTHON_BIN" scripts/score_cross_encoder.py \
    --checkpoint $H/ce-dev/checkpoint --candidates $R/candidates_validation.json
fi

if [ "$STAGE" = all ] || [ "$STAGE" = full ]; then
  echo "== Cross-encoder B with hybrid negatives: all 20,000 queries, stopped at A's epoch; test scores =="
  K=$("$PYTHON_BIN" -c "import json; print(json.load(open('$H/ce-dev/train_log.json'))['selected_epoch'])")
  completed $H/ce-full/train_log.json || "$PYTHON_BIN" scripts/train_cross_encoder.py \
    --negatives $H/negatives_full.json --output-dir $H/ce-full --stop-after-epoch "$K"
  [ -f $H/ce-full/scores_test.npy ] || "$PYTHON_BIN" scripts/score_cross_encoder.py \
    --checkpoint $H/ce-full/checkpoint --candidates $R/candidates_test.json
fi

if [ "$STAGE" = all ] || [ "$STAGE" = eval ]; then
  echo "== Fusion weight on the tuning slice, validation once, Kaggle file =="
  "$PYTHON_BIN" scripts/run_rerank_eval.py --student-id "$STUDENT_ID" \
    --dev-model $H/ce-dev --full-model $H/ce-full --output-dir $H/submission \
    --method "multi-stage: BM25 v2 + LTR BERT candidates -> BERT cross-encoder trained with hybrid hard negatives"
  "$PYTHON_BIN" scripts/validate_submission.py "$H/submission/${STUDENT_ID}_final.csv"
fi
echo "Hard-negative stage '$STAGE' finished."
