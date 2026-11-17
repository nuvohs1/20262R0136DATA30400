# Dense retrieval (zero-shot E5)

## Model eligibility

`intfloat/e5-base-v2` (MIT license, 110M parameters, mean pooling, `query: ` / `passage: ` prefixes). According to Wang et al., *Text Embeddings by Weakly-Supervised Contrastive Pre-training* (arXiv:2212.03533), which the model card cites for training details:

- Contrastive pre-training on CCPairs (Table 9): Wikipedia, Reddit, Common Crawl, StackExchange, S2ORC, News, and "Others" (SimpleWiki, GooAQ, WikiHow, Yahoo Answers).
- Supervised fine-tuning (Sec. 4.2 and 5.1): MS-MARCO passage ranking, Natural Questions, and NLI.
- Amazon datasets (AmazonPolarity, AmazonReviews, AmazonCounterfactual) appear only as MTEB evaluation tasks, not as training data.

We therefore treat the model as not fine-tuned on Amazon product data. The exact model revision used is recorded in `outputs/dense/e5-base-v2/manifest.json` (`model_revision`). Commonly used `sentence-transformers/all-*` models were not chosen because their 1B-pair training mixture is broad and harder to verify.

## Method

- Products: `passage: Title: … | Store: … | Features: … | Description: …`, truncated to 256 tokens (description is truncated first).
- Reviews: each of the five reviews is encoded separately as `query: <review>` (256 tokens). Product score = mean of the five cosine similarities (`dense_mean`), or the maximum (`dense_max`).
- Hybrid: `z(BM25 v2) + alpha * z(dense_mean)`; alpha chosen from {0, 0.25, …, 3} on the 4,000-query tuning slice. The test hybrid adds the dense term on top of the Kaggle-best BM25 (with review expansion); alpha = 0 reproduces that BM25 file byte for byte.
- No labels are used by the encoder. The leakage protocol is the same as BM25 v2 (see `docs/bm25_v2.md`).

## How to run

1. Colab (GPU): open `notebooks/dense_encode_colab.ipynb`, following its instructions. This produces `outputs/dense/e5-base-v2/{catalog,train_reviews,test_reviews}.npy` and `manifest.json` (about 400 MB).
2. Download that folder into the local `outputs/dense/e5-base-v2/`.
3. Locally (NumPy only):

```bash
python scripts/run_dense.py --predict-test --student-id 2024320020 --workers 8
```

This writes `outputs/dense_eval/e5-base-v2/dense_only/2024320020_final.csv`, `outputs/dense_eval/e5-base-v2/hybrid_bm25_expansion_dense/2024320020_final.csv`, and `metrics.json`.

## Results

Pending: the embeddings have not been computed yet.
