# Ranking metrics

`src/review_retrieval/metrics.py` implements Recall@N, Precision@N, NDCG@N and MRR@N for queries with exactly one relevant product. The definitions are the standard graded ones (DCG = Σ g_i / log2(i + 1), nDCG = DCG / IDCG, as in the course retrieval-evaluation exercise), specialised to one relevant item with gain 1. If the correct product is at rank k:

- Recall@N = 1 if k ≤ N, else 0. Recall@10 is the Kaggle metric.
- Precision@N = Recall@N / N, because at most one of the N results can be relevant.
- NDCG@N = 1 / log2(k + 1) if k ≤ N, else 0 (the ideal ranking puts the item at rank 1).
- MRR@N = 1 / k if k ≤ N, else 0.

`python scripts/evaluate_rankings.py` recomputes all metrics from the saved validation rankings (4,000 held-out queries, full catalog, top 10) and writes `artifacts/experiments/ranking_metrics.json`. Nothing is tuned on these numbers.

| Method | R@1 | R@5 | R@10 | P@5 | P@10 | NDCG@5 | NDCG@10 | MRR@10 | Kaggle R@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 (original baseline) | 0.1320 | 0.2802 | 0.3625 | 0.0561 | 0.0362 | 0.2083 | 0.2348 | 0.1955 | 0.38120 |
| BM25 v2 (metadata only) | 0.2765 | 0.4760 | 0.5783 | 0.0952 | 0.0578 | 0.3816 | 0.4145 | 0.3638 | 0.57538 |
| Dense: zero-shot BERT | 0.0027 | 0.0092 | 0.0170 | 0.0019 | 0.0017 | 0.0062 | 0.0086 | 0.0061 | 0.02306 |
| Dense: LTR fine-tuned BERT | 0.1600 | 0.3483 | 0.4567 | 0.0697 | 0.0457 | 0.2580 | 0.2931 | 0.2426 | 0.56058 |
| Hybrid: BM25 v2 + LTR BERT (α = 3) | 0.3295 | 0.5727 | 0.6673 | 0.1145 | 0.0667 | 0.4578 | 0.4883 | 0.4322 | 0.72333¹ |
| Multi-stage: hybrid top 50 + BERT cross-encoder (w = 0.5) | 0.3767 | 0.6130 | 0.7073 | 0.1226 | 0.0707 | 0.5040 | 0.5344 | 0.4801 | 0.76916¹ |
| Multi-stage + self-trained cross-encoder (pseudo-labels) | 0.3845 | 0.6140 | 0.7075 | 0.1228 | 0.0707 | 0.5080 | 0.5383 | 0.4852 | 0.76936¹ |

¹ The test file uses BM25 v2 with labelled-review expansion (Kaggle 0.60761 on its own). Expansion cannot be measured on the local validation split, because validation products never appear in training.

The hybrid improves every metric. Its NDCG@10 gain over BM25 v2 (+0.074) is a little smaller than its Recall@10 gain (+0.089). Part of the gain therefore comes from moving the correct product into the top 10 from lower ranks, not only from reordering within it.

The cross-encoder reranker adds +0.046 NDCG@10 and +0.047 Recall@1 on top of the hybrid, slightly more than its Recall@10 gain (+0.040). Most of its effect is reordering within the candidate list: the correct product moves toward rank 1 ([multi-stage ranking](reranking.md)).
