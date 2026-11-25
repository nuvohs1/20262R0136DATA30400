#!/usr/bin/env bash
# Self-training with pseudo-labels (after run_rerank.sh). Confident predictions of the
# multi-stage ranker on unlabeled queries become extra training pairs for the cross-encoder.
#   labels-dev : pseudo-label the tuning slice + validation queries with model A (threshold chosen
#                on the tuning slice; CPU)
#   dev        : retrain cross-encoder A with them, epoch chosen on the tuning slice; validation scores
#   labels-full: pseudo-label the test queries with model B (same threshold and w; CPU)
#   full       : retrain cross-encoder B with them, stopped at A's epoch; test scores
#   eval       : fusion weight on the tuning slice, validation once, Kaggle file
#
# Usage: bash run_pseudo.sh [STUDENT_ID] [all|labels-dev|dev|labels-full|full|eval]
# Finished steps are skipped, so the script can be re-run after an interruption.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python}"
STUDENT_ID="${1:-2024320020}"
STAGE="${2:-all}"
R=outputs/rerank
P=outputs/pseudo

completed() {  # completed <train_log.json>
  [ -f "$1" ] && "$PYTHON_BIN" -c "import json,sys; sys.exit(0 if json.load(open('$1')).get('completed') else 1)"
}

for f in $R/candidates_tune.json $R/candidates_validation.json $R/candidates_test.json \
         $R/ce-dev/scores_tune.npy $R/ce-dev/scores_validation.npy $R/ce-full/scores_test.npy $R/submission/metrics.json; do
  [ -f "$f" ] || { echo "Missing $f: run run_rerank.sh first"; exit 1; }
done

if [ "$STAGE" = all ] || [ "$STAGE" = labels-dev ]; then
  echo "== Pseudo-labels for the tuning slice + validation queries (model A) =="
  [ -f $P/pseudo_dev.json ] || "$PYTHON_BIN" scripts/make_pseudo_labels.py --stage dev
fi

if [ "$STAGE" = all ] || [ "$STAGE" = dev ]; then
  echo "== Self-trained cross-encoder A: 12,000 labelled + pseudo-labelled queries; validation scores =="
  completed $P/ce-dev/train_log.json || "$PYTHON_BIN" scripts/train_cross_encoder.py \
    --negatives outputs/ltr/negatives_dev.json --pseudo $P/pseudo_dev.json \
    --tune-candidates $R/candidates_tune.json --output-dir $P/ce-dev
  [ -f $P/ce-dev/scores_validation.npy ] || "$PYTHON_BIN" scripts/score_cross_encoder.py \
    --checkpoint $P/ce-dev/checkpoint --candidates $R/candidates_validation.json
fi

if [ "$STAGE" = all ] || [ "$STAGE" = labels-full ]; then
  echo "== Pseudo-labels for the test queries (model B) =="
  [ -f $P/pseudo_full.json ] || "$PYTHON_BIN" scripts/make_pseudo_labels.py --stage full
fi

if [ "$STAGE" = all ] || [ "$STAGE" = full ]; then
  echo "== Self-trained cross-encoder B: 20,000 labelled + pseudo-labelled test queries; test scores =="
  K=$("$PYTHON_BIN" -c "import json; print(json.load(open('$P/ce-dev/train_log.json'))['selected_epoch'])")
  completed $P/ce-full/train_log.json || "$PYTHON_BIN" scripts/train_cross_encoder.py \
    --negatives outputs/ltr/negatives_full.json --pseudo $P/pseudo_full.json \
    --output-dir $P/ce-full --stop-after-epoch "$K"
  [ -f $P/ce-full/scores_test.npy ] || "$PYTHON_BIN" scripts/score_cross_encoder.py \
    --checkpoint $P/ce-full/checkpoint --candidates $R/candidates_test.json
fi

if [ "$STAGE" = all ] || [ "$STAGE" = eval ]; then
  echo "== Fusion weight on the tuning slice, validation once, Kaggle file =="
  "$PYTHON_BIN" scripts/run_rerank_eval.py --student-id "$STUDENT_ID" \
    --dev-model $P/ce-dev --full-model $P/ce-full --output-dir $P/submission \
    --pseudo-dev $P/pseudo_dev_summary.json \
    --method "multi-stage with a self-trained cross-encoder (pseudo-labels on unlabeled course queries)"
  "$PYTHON_BIN" scripts/validate_submission.py "$P/submission/${STUDENT_ID}_final.csv"
fi
echo "Pseudo-label stage '$STAGE' finished."
