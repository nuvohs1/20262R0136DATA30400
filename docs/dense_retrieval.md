# Dense retrieval (BERT bi-encoder)

## Model

`google-bert/bert-base-uncased` at revision `86b5e093…`, the only allowed checkpoint ([model eligibility](model_eligibility.md)). It is used zero-shot here; the fine-tuned version is described in [learning-to-rank fine-tuning](ltr_finetuning.md).

## Method

- Product passage: `Title: … | Store: … | Features: … | Description: …` after the shared text cleaning (`src/review_retrieval/dense_text.py`), truncated to 256 tokens.
- Reviews: each of the five reviews is encoded separately (up to 256 tokens). No instruction prefixes are used.
- Pooling: mean over the attention mask, then L2 normalization; stored as float16.
- Score: the mean of the five review–product cosine similarities, which equals the dot product with the mean review embedding. The maximum over reviews is reported as an ablation.
- Hybrid: `z(BM25 v2) + α · z(dense)`, with α chosen from a fixed grid on the 4,000-query tuning slice. BM25 RSJ weights for tuning come from the other 12,000 development-training queries. Validation is evaluated once with RSJ from the 16,000 development-training queries. For test, the BM25 part is the Kaggle-best variant with labelled-review expansion, learned from all 20,000 training queries.

## How to run

1. GPU (Colab T4): `notebooks/gpu_colab.ipynb`, step 1 (`bash run_gpu.sh zeroshot`) writes `outputs/dense/bert-base-uncased/`.
2. Locally (NumPy only):

```bash
python scripts/run_dense.py --embeddings outputs/dense/bert-base-uncased \
    --output-dir outputs/dense_eval/bert-base-uncased --predict-test --student-id 2024320020 --workers 8
```

## Results

Pending the Colab run.
