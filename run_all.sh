#!/usr/bin/env bash
# One-command reproduction of the final submission (docs/reproduction.md).
#
#   quick : download the three verified model-B checkpoints (scripts/fetch_models.py), then run
#           test inference only: test embeddings, stage-1 candidates, two cross-encoder scores,
#           final file. Settings (alpha, w) come from the committed records in artifacts/.
#           GPU needed for encoding/scoring (about 2.5 hours on one NVIDIA L4).
#   full  : retrain everything from bert-base-uncased (LTR model A/B, both cross-encoders A/B),
#           re-select every setting on the tuning slice / proxy tuning slice, final file.
#           GPU, about 10-12 hours.
#
# Usage: bash run_all.sh [quick|full] [STUDENT_ID] [WORKERS]
#        The model pack is downloaded from MODELS_URL (default: the submitted Google Drive link);
#        or put models_<id>.zip in the project root.
# Output: outputs/final/<STUDENT_ID>_final.csv (19,858 rows, 10 distinct catalog IDs per row).
# Every random operation uses seed 42; finished steps are skipped on re-runs.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-quick}"
STUDENT_ID="${2:-2024320020}"
WORKERS="${3:-8}"
MODELS_URL="${MODELS_URL:-https://drive.google.com/file/d/1BM16spl36PTRRuj8Tl8XxWP2iO6fwucc/view?usp=sharing}"
export PYTHON_BIN

for f in catalog.json train_gold.json test_queries.json category_taxonomy.txt; do
  if [ ! -f "project_release/$f" ]; then
    echo "Missing project_release/$f. Copy the four course data files (catalog.json, train_gold.json,"
    echo "test_queries.json, category_taxonomy.txt) into project_release/ first; they are not in the repository."
    exit 1
  fi
done

echo "== Environment =="
"$PYTHON_BIN" - <<'PY'
import sys, numpy, torch, transformers
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
print(f"python {sys.version.split()[0]} ({sys.executable}), numpy {numpy.__version__}, torch {torch.__version__}, "
      f"transformers {transformers.__version__}, GPU {gpu}")
if not torch.cuda.is_available():
    sys.exit("No CUDA GPU is visible to this Python; the encoding and scoring steps need one.")
if (torch.__version__.split("+")[0], numpy.__version__, transformers.__version__) != ("2.10.0", "2.2.6", "5.20.2"):
    print("WARNING: the verified run used torch 2.10.0, numpy 2.2.6 and transformers 5.20.2 "
          "(python -m pip install -r requirements.txt). Other versions give a near-identical, "
          "not byte-identical, file.")
PY

echo "== Data check and unit tests =="
"$PYTHON_BIN" scripts/prepare_data.py
"$PYTHON_BIN" -m unittest discover -s tests

case "$MODE" in
  quick)
    echo "== Model-B checkpoints (verified against artifacts/model_manifest.json) =="
    if [ ! -f outputs/hardneg/ce-full/checkpoint/provenance.json ] || [ ! -f outputs/rerank/ce-full/checkpoint/provenance.json ] \
       || [ ! -f outputs/ltr/bert-ltr-full/checkpoint/provenance.json ]; then
      "$PYTHON_BIN" scripts/fetch_models.py --url "$MODELS_URL"
    fi
    touch outputs/.quick_mode  # full mode must not reuse these downloaded models and test-only caches
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
    MATCH=--require-match
    ;;
  full)
    if [ -f outputs/.quick_mode ]; then
      echo "This folder already holds a quick-mode run. Run full mode in a fresh clone (or delete outputs/ first)."
      exit 1
    fi
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
"$PYTHON_BIN" scripts/make_final.py --student-id "$STUDENT_ID" --selection "$SELECTION" ${MATCH:-}
"$PYTHON_BIN" scripts/validate_submission.py "outputs/final/${STUDENT_ID}_final.csv"
if [ "$MODE" = quick ]; then
  echo "Done: outputs/final/${STUDENT_ID}_final.csv is identical to the submitted file."
else
  echo "Done: outputs/final/${STUDENT_ID}_final.csv (format checked; see the SHA-256 line above for whether it is identical)."
fi
