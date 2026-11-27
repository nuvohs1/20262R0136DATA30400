#!/usr/bin/env bash
# Augmented cross-encoder (after run_rerank.sh, run_hardneg.sh and run_memory.sh candidates).
# Same base model and listwise loss as the earlier cross-encoders, with three input/training changes:
#   - the five reviews share one 320-token budget instead of 64 tokens each (same total length);
#   - training shuffles the reviews and keeps only 3-4 of them in half of the draws (seeded);
#   - each group mixes 4 hybrid negatives and 3 BM25 negatives.
# Stages:
#   dev   : model A on 12,000 queries, epoch chosen on the tuning slice
#   full  : model B on all 20,000 queries, stopped at A's epoch
#   score : A scores validation and the proxy splits, B scores test (GPU)
#   eval  : old-protocol validation file + proxy selection including the 3-model ensemble (CPU)
#
# Usage: bash run_augce.sh [STUDENT_ID] [all|dev|full|score|eval]
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python}"
STUDENT_ID="${1:-2024320020}"
STAGE="${2:-all}"
R=outputs/rerank
A=outputs/augce
M=outputs/memory
OPTS=(--review-budget 320 --augment --extra-count 3)

completed() {  # completed <train_log.json>
  [ -f "$1" ] && "$PYTHON_BIN" -c "import json,sys; sys.exit(0 if json.load(open('$1')).get('completed') else 1)"
}

for f in $R/candidates_tune.json $R/candidates_validation.json outputs/ltr/negatives_dev.json outputs/ltr/negatives_full.json \
         outputs/hardneg/negatives_dev.json outputs/hardneg/negatives_full.json \
         $M/candidates_exp_proxy_tune.json $M/candidates_exp_proxy_validation.json $M/candidates_exp_test.json; do
  [ -f "$f" ] || { echo "Missing $f: run run_rerank.sh, run_hardneg.sh and run_memory.sh candidates first"; exit 1; }
done

if [ "$STAGE" = all ] || [ "$STAGE" = dev ]; then
  echo "== Augmented cross-encoder A: 12,000 queries, epoch chosen on the tuning slice =="
  completed $A/ce-dev/train_log.json || "$PYTHON_BIN" scripts/train_cross_encoder.py "${OPTS[@]}" \
    --negatives outputs/hardneg/negatives_dev.json --extra-negatives outputs/ltr/negatives_dev.json \
    --tune-candidates $R/candidates_tune.json --output-dir $A/ce-dev
fi

if [ "$STAGE" = all ] || [ "$STAGE" = full ]; then
  echo "== Augmented cross-encoder B: all 20,000 queries, stopped at A's epoch =="
  K=$("$PYTHON_BIN" -c "import json; print(json.load(open('$A/ce-dev/train_log.json'))['selected_epoch'])")
  completed $A/ce-full/train_log.json || "$PYTHON_BIN" scripts/train_cross_encoder.py "${OPTS[@]}" \
    --negatives outputs/hardneg/negatives_full.json --extra-negatives outputs/ltr/negatives_full.json \
    --output-dir $A/ce-full --stop-after-epoch "$K"
fi

if [ "$STAGE" = all ] || [ "$STAGE" = score ]; then
  echo "== Scores: model A on validation and proxy splits, model B on test (GPU) =="
  [ -f $A/ce-dev/scores_validation.npy ] || "$PYTHON_BIN" scripts/score_cross_encoder.py \
    --checkpoint $A/ce-dev/checkpoint --candidates $R/candidates_validation.json
  for split in proxy_tune proxy_validation; do
    [ -f $M/scores/aug_exp_$split.npy ] || "$PYTHON_BIN" scripts/score_cross_encoder.py \
      --checkpoint $A/ce-dev/checkpoint --candidates $M/candidates_exp_$split.json --output $M/scores/aug_exp_$split.npy
  done
  [ -f $M/scores/aug_exp_test.npy ] || "$PYTHON_BIN" scripts/score_cross_encoder.py \
    --checkpoint $A/ce-full/checkpoint --candidates $M/candidates_exp_test.json --output $M/scores/aug_exp_test.npy
  [ -f $A/ce-full/scores_test.npy ] || cp $M/scores/aug_exp_test.npy $A/ce-full/scores_test.npy
fi

if [ "$STAGE" = all ] || [ "$STAGE" = eval ]; then
  echo "== Old-protocol validation (single model) and proxy selection with the 3-model ensemble =="
  "$PYTHON_BIN" scripts/run_rerank_eval.py --student-id "$STUDENT_ID" \
    --dev-model $A/ce-dev --full-model $A/ce-full --output-dir $A/submission \
    --method "multi-stage: BM25 v2 + LTR BERT candidates -> augmented BERT cross-encoder (shared review budget, review shuffling, mixed negatives)"
  "$PYTHON_BIN" scripts/run_memory_eval.py --student-id "$STUDENT_ID"
  "$PYTHON_BIN" scripts/validate_submission.py "$M/submission/exp_ens3/${STUDENT_ID}_final.csv"
fi
echo "Augmented cross-encoder stage '$STAGE' finished."
