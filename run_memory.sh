#!/usr/bin/env bash
# Test-like proxy validation with labelled-review memory (after run_rerank.sh, run_hardneg.sh and
# run_category.sh encode). No training: existing cross-encoders are re-scored on new candidates.
#   candidates : proxy splits + memory first stage, (mu, tau) chosen on the proxy tuning slice (CPU)
#   score      : cross-encoder scores for every candidate set (GPU, scoring only)
#   eval       : w and the (first stage, cross-encoder) pair on the proxy tuning slice,
#                proxy validation once, Kaggle files
#
# Usage: bash run_memory.sh [STUDENT_ID] [WORKERS] [all|candidates|score|eval]
# Finished steps are skipped, so the script can be re-run after an interruption.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python}"
STUDENT_ID="${1:-2024320020}"
WORKERS="${2:-8}"
STAGE="${3:-all}"
M=outputs/memory

for f in outputs/rerank/ce-dev/checkpoint outputs/rerank/ce-full/checkpoint \
         outputs/hardneg/ce-dev/checkpoint outputs/hardneg/ce-full/checkpoint \
         outputs/dense/bert-ltr-dev/train_reviews.npy outputs/dense/bert-ltr-full/train_reviews.npy \
         outputs/dense/bert-ltr-full/test_reviews.npy; do
  [ -e "$f" ] || { echo "Missing $f: run run_rerank.sh, run_hardneg.sh and run_category.sh encode first"; exit 1; }
done

if [ "$STAGE" = all ] || [ "$STAGE" = candidates ]; then
  echo "== Proxy splits and memory first stage (CPU) =="
  [ -f $M/candidates_mem_test.json ] || "$PYTHON_BIN" scripts/make_memory_candidates.py --workers "$WORKERS"
fi

if [ "$STAGE" = all ] || [ "$STAGE" = score ]; then
  echo "== Cross-encoder scores (GPU) =="
  for model in rerank hardneg; do
    for variant in exp mem; do
      for split in proxy_tune proxy_validation test; do
        set_name=dev; [ "$split" = test ] && set_name=full
        out=$M/scores/${model}_${variant}_${split}.npy
        [ -f "$out" ] || "$PYTHON_BIN" scripts/score_cross_encoder.py \
          --checkpoint outputs/$model/ce-$set_name/checkpoint --candidates $M/candidates_${variant}_${split}.json --output "$out"
      done
    done
  done
fi

if [ "$STAGE" = all ] || [ "$STAGE" = eval ]; then
  echo "== Proxy tuning-slice selection, proxy validation once, Kaggle files =="
  "$PYTHON_BIN" scripts/run_memory_eval.py --student-id "$STUDENT_ID"
  "$PYTHON_BIN" scripts/validate_submission.py "$M/submission/${STUDENT_ID}_final.csv"
fi
echo "Memory stage '$STAGE' finished."
