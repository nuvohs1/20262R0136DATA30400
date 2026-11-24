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

Embeddings were computed on the course AWS SageMaker instance (NVIDIA L4; Python 3.10.20, torch 2.10.0+cu128, transformers 5.20.2): about 100 s for the catalog and about 85 s for each review set. Evaluation ran with NumPy 2.2.6. Record: `artifacts/experiments/dense_bert_zero_shot.json`.

Local validation (4,000 held-out queries, full catalog):

| Method | R@1 | R@5 | R@10 | Kaggle R@10 |
| --- | --- | --- | --- | --- |
| BM25 v2 (metadata only, reference) | 27.65% | 47.60% | 57.83% | 0.57538 |
| Zero-shot BERT, mean of five reviews | 0.27% | 0.92% | 1.70% | pending |
| Zero-shot BERT, max over reviews | 0.35% | 1.47% | 2.38% | — |
| BM25 v2 + α · zero-shot BERT (α = 0 selected) | 27.65% | 47.60% | 57.83% | = BM25 v2 + expansion, 0.60761 |

Raw bert-base-uncased embeddings are nearly useless for retrieval: it was trained only with MLM/NSP, never for sentence similarity, and mean-pooled BERT vectors are known to be highly anisotropic. On the tuning slice the hybrid only got worse as α grew (55.80% at α = 0, 54.70% at α = 1, 29.68% at α = 10), so α = 0 was selected. The resulting test file is byte-identical to the BM25 v2 + expansion submission. The identical file, produced on Linux (AWS) and Windows, also confirms that the BM25 stage reproduces exactly across platforms. Learning-to-rank fine-tuning turns the same model into a useful retriever ([learning-to-rank fine-tuning](ltr_finetuning.md)).
