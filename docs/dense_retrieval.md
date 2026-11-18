# Dense retrieval (zero-shot E5)

## Model eligibility

`intfloat/e5-base-v2` (MIT license, 110M parameters, mean pooling, `query: ` / `passage: ` prefixes). According to Wang et al., *Text Embeddings by Weakly-Supervised Contrastive Pre-training* (arXiv:2212.03533), which the model card cites for training details:

- Contrastive pre-training on CCPairs (Table 9): Wikipedia, Reddit, Common Crawl, StackExchange, S2ORC, News, and "Others" (SimpleWiki, GooAQ, WikiHow, Yahoo Answers).
- Supervised fine-tuning (Sec. 4.2 and 5.1): MS-MARCO passage ranking, Natural Questions, and NLI.
- Amazon datasets (AmazonPolarity, AmazonReviews, AmazonCounterfactual) appear only as MTEB evaluation tasks, not as training data.

We therefore treat the model as not fine-tuned on Amazon product data. `scripts/encode_dense.py` pins Hugging Face revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`. The recorded Colab run (2026-11-18) did not log the revision, because transformers 5 does not expose it. The repository's last commit is dated 2025-02-17, before that run, so the run used this revision. Commonly used `sentence-transformers/all-*` models were not chosen because their 1B-pair training mixture is broad and harder to verify.

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

Local results are on the 4,000 held-out validation queries, searching all 50,000 products. They are not Kaggle scores.

| Method | Recall@1 | Recall@5 | Recall@10 | Kaggle |
| --- | ---: | ---: | ---: | ---: |
| E5 zero-shot, mean of five review similarities | 18.33% | 36.78% | 46.23% | pending |
| E5 zero-shot, max of five review similarities | 18.65% | 34.95% | 43.93% | not submitted |
| BM25 v2, metadata only (alpha = 0) | 27.65% | 47.60% | 57.83% | 0.57538 |
| **BM25 v2 + E5 hybrid (alpha = 3)** | **30.18%** | **51.73%** | **61.95%** | pending (test file includes BM25 review expansion) |

Tuning-slice Recall@10 by alpha: 0 → 55.80%, 0.5 → 57.90%, 1 → 59.23%, 2 → 60.50%, **3 → 60.68%**, 4 → 60.22%, 5 → 59.92%, 7 → 58.40%, 10 → 56.83%. The first grid stopped at 3, its upper edge, so the grid was extended on the tuning slice and the run repeated. The selection and every output file were byte-identical, and validation was never used for selection.

Observations:
- Zero-shot dense retrieval alone is weaker than the tuned BM25 (46.2% vs 57.8%), but the two make different errors: fusing them adds 4.1 points of validation Recall@10.
- Averaging the five review similarities beats taking the maximum by 2.3 points of Recall@10: evidence from several reviews is more reliable than the single best-matching review.
- Two complete runs produced byte-identical submissions.

Embedding cost on a Colab T4: about 258 s for the catalog and about 227 s each for train and test reviews. Local fusion and evaluation take about 10 minutes with 12 CPU workers.
