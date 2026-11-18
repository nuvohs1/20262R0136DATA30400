#!/usr/bin/env bash
# GPU stage (e.g. Colab T4): dense embeddings and learning-to-rank fine-tuning of BERT.
# Only google-bert/bert-base-uncased at its verified revision (or checkpoints fine-tuned from it
# on train_gold.json by this pipeline) can be loaded; see docs/model_eligibility.md.
#
# Usage: bash run_gpu.sh [all|zeroshot|ltr-dev|ltr-full]
# Finished steps are skipped, so the script can be re-run after a Colab disconnect.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE="${1:-all}"

completed() {  # completed <train_log.json>
  [ -f "$1" ] && "$PYTHON_BIN" -c "import json,sys; sys.exit(0 if json.load(open('$1')).get('completed') else 1)"
}

if [ ! -f outputs/ltr/negatives_dev.json ] || [ ! -f outputs/ltr/negatives_full.json ]; then
  echo "== Mining BM25 hard negatives (CPU, NumPy) =="
  "$PYTHON_BIN" scripts/mine_negatives.py
fi

if [ "$STAGE" = all ] || [ "$STAGE" = zeroshot ]; then
  echo "== Zero-shot BERT embeddings =="
  [ -f outputs/dense/bert-base-uncased/test_reviews.npy ] || "$PYTHON_BIN" scripts/encode_dense.py --tag bert-base-uncased
fi

if [ "$STAGE" = all ] || [ "$STAGE" = ltr-dev ]; then
  echo "== LTR model A: 12,000 development-training queries, epoch chosen on the tuning slice =="
  completed outputs/ltr/bert-ltr-dev/train_log.json || \
    "$PYTHON_BIN" scripts/train_dense_ltr.py --negatives outputs/ltr/negatives_dev.json --output-dir outputs/ltr/bert-ltr-dev
  [ -f outputs/dense/bert-ltr-dev/train_reviews.npy ] || \
    "$PYTHON_BIN" scripts/encode_dense.py --model outputs/ltr/bert-ltr-dev/checkpoint --tag bert-ltr-dev --parts catalog,train
fi

if [ "$STAGE" = all ] || [ "$STAGE" = ltr-full ]; then
  echo "== LTR model B: all 20,000 training queries, stopped at model A's epoch =="
  K=$("$PYTHON_BIN" -c "import json; print(json.load(open('outputs/ltr/bert-ltr-dev/train_log.json'))['selected_epoch'])")
  completed outputs/ltr/bert-ltr-full/train_log.json || \
    "$PYTHON_BIN" scripts/train_dense_ltr.py --negatives outputs/ltr/negatives_full.json --output-dir outputs/ltr/bert-ltr-full --stop-after-epoch "$K"
  [ -f outputs/dense/bert-ltr-full/test_reviews.npy ] || \
    "$PYTHON_BIN" scripts/encode_dense.py --model outputs/ltr/bert-ltr-full/checkpoint --tag bert-ltr-full --parts catalog,test
fi
echo "GPU stage '$STAGE' finished."
