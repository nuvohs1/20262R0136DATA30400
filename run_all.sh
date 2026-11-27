#!/usr/bin/env bash
# One-command reproduction of the final submission (docs/reproduction.md).
#
#   quick : download the three verified model-B checkpoints (scripts/fetch_models.py), then run
#           test inference only: test embeddings, stage-1 candidates, two cross-encoder scores,
#           final file. Settings (alpha, w) come from the committed records in artifacts/.
#           GPU needed for encoding/scoring (about 30-40 minutes on one GPU).
#   full  : retrain everything from bert-base-uncased (LTR model A/B, both cross-encoders A/B),
#           re-select every setting on the tuning slice / proxy tuning slice, final file.
#           GPU, about 10-12 hours.
#
# Usage: bash run_all.sh [quick|full] [STUDENT_ID] [WORKERS]
#        MODELS_URL=<link> bash run_all.sh quick     # or put models_<id>.zip in the project root
# Output: outputs/final/<STUDENT_ID>_final.csv (19,858 rows, 10 distinct catalog IDs per row).
# Every random operation uses seed 42; finished steps are skipped on re-runs.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-quick}"
STUDENT_ID="${2:-2024320020}"
WORKERS="${3:-8}"
export PYTHON_BIN

echo "== Data check and unit tests =="
"$PYTHON_BIN" scripts/prepare_data.py
"$PYTHON_BIN" -m unittest discover -s tests

case "$MODE" in
  quick)
    echo "== Model-B checkpoints (verified against artifacts/model_manifest.json) =="
    if [ ! -f outputs/hardneg/ce-full/checkpoint/provenance.json ] || [ ! -f outputs/rerank/ce-full/checkpoint/provenance.json ] \
       || [ ! -f outputs/ltr/bert-ltr-full/checkpoint/provenance.json ]; then
      "$PYTHON_BIN" scripts/fetch_models.py ${MODELS_URL:+--url "$MODELS_URL"}
    fi
    echo "== Test embeddings from LTR model B (GPU) =="
    [ -f outputs/dense/bert-ltr-full/test_reviews.npy ] || "$PYTHON_BIN" scripts/encode_dense.py \
      --model outputs/ltr/bert-ltr-full/checkpoint --tag bert-ltr-full --parts catalog,test
    echo "== Stage-1 candidates for test (CPU; alpha from the committed tuning record) =="
    [ -f outputs/rerank/candidates_test.json ] || "$PYTHON_BIN" scripts/make_candidates.py \
      --splits test --alpha-from artifacts/experiments/ltr_bert.json --workers "$WORKERS"
    echo "== Cross-encoder scores for test (GPU) =="
    for model in rerank hardneg; do
      [ -f outputs/$model/ce-full/scores_test.npy ] || "$PYTHON_BIN" scripts/score_cross_encoder.py \
        --checkpoint outputs/$model/ce-full/checkpoint --candidates outputs/rerank/candidates_test.json
    done
    SELECTION=artifacts/experiments/memory_proxy.json
    ;;
  full)
    bash run_gpu.sh ltr-dev
    bash run_gpu.sh ltr-full
    bash run_dense_eval.sh "$STUDENT_ID" "$WORKERS" ltr       # alpha on the tuning slice
    bash run_category.sh "$STUDENT_ID" encode                 # training-review embeddings, model B
    bash run_rerank.sh "$STUDENT_ID" "$WORKERS"               # candidates + cross-encoder (BM25 negatives)
    bash run_hardneg.sh "$STUDENT_ID" "$WORKERS"              # cross-encoder (hybrid negatives)
    bash run_memory.sh "$STUDENT_ID" "$WORKERS"               # proxy tuning slice: w for exp+ens
    SELECTION=outputs/memory/submission/metrics.json
    ;;
  *)
    echo "Usage: bash run_all.sh [quick|full] [STUDENT_ID] [WORKERS]"; exit 2 ;;
esac

echo "== Final file =="
"$PYTHON_BIN" scripts/make_final.py --student-id "$STUDENT_ID" --selection "$SELECTION"
"$PYTHON_BIN" scripts/validate_submission.py "outputs/final/${STUDENT_ID}_final.csv"
echo "Done: outputs/final/${STUDENT_ID}_final.csv"
