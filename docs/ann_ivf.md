# Approximate nearest-neighbour search with an IVF index

## Idea

Dense retrieval currently scores all 50,000 products for every query (exact search). That is affordable here, but it grows linearly with the catalog. An inverted-file (IVF) index, a course technique for approximate nearest-neighbour (ANN) search, scores only a small part of the catalog:

1. **Build.** Spherical k-means (seed 42, 20 iterations) splits the L2-normalized product embeddings into `nlist` = 256 cells. Each product is stored in the list of its nearest centroid (`src/review_retrieval/ivf.py`, NumPy).
2. **Search.** A query is compared with the 256 centroids. Only the products in the `nprobe` closest lists are scored exactly, and the best 10 (or 50) of those are returned. `nprobe` = 256 is exact search.

The bi-encoder's dense score is the mean of the five review–product inner products. That equals a single inner product with the mean review embedding, so each query is one vector and IVF applies directly.

## Protocol

- **Embeddings.** LTR model A (`outputs/dense/bert-ltr-dev`), which was never trained on the tuning or validation queries. k-means uses product embeddings only, with no labels.
- **Metrics** for each `nprobe` ∈ {1, 2, 4, 8, 16, 32, 64, 128}:
  - ANN recall@10: the share of the exact top 10 that IVF also returns.
  - Recall@10 and Recall@50 of the correct product. Recall@50 matters because stage 1 passes 50 candidates to the reranker.
  - Products scored per query.
  - Milliseconds per query, single process. Timings are machine-dependent.
- **Selection.** `nprobe` is the smallest value whose tuning-slice ANN recall@10 is at least 0.95 (fixed in advance). Validation is reported once.
- **Scope.** This step studies the efficiency of the dense stage only and does not change the submission.

## How to run

On the course AWS notebook (step 4e), CPU only:

```bash
python scripts/run_ann.py
```

Results are written to `outputs/ann/metrics.json`.

## Results

Run on the course AWS notebook (CPU, 8 cores, NumPy 2.2.6). Full record: `artifacts/experiments/ann_ivf.json`.

**Index.** Building the index takes 9.4 s. The k-means mean cosine to the assigned centroid rises from 0.544 to 0.707 over 20 iterations. List sizes are uneven: min 49, median 177, max 723.

**Validation** (4,000 queries, model A, dense stage only; nprobe = 8 was selected on the tuning slice, where it is the smallest value reaching ANN recall@10 0.95, at 0.960):

| nprobe | Products scored / query | ms / query | Speedup | ANN recall@10 | Recall@10 | Recall@50 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 235 | 0.15 | 27.9× | 0.709 | 0.3202 | 0.5282 |
| 2 | 444 | 0.24 | 18.0× | 0.844 | 0.3832 | 0.6250 |
| 4 | 837 | 0.40 | 10.8× | 0.920 | 0.4198 | 0.6783 |
| **8** | **1,603** | **0.71** | **6.0×** | **0.963** | **0.4400** | **0.7090** |
| 16 | 3,120 | 1.32 | 3.3× | 0.984 | 0.4507 | 0.7210 |
| 32 | 6,126 | 2.56 | 1.7× | 0.994 | 0.4547 | 0.7285 |
| 64 | 12,191 | 5.39 | 0.8× | 0.999 | 0.4560 | 0.7305 |
| 128 | 24,537 | 18.53 | 0.2× | 1.000 | 0.4567 | 0.7315 |
| exact | 50,000 | 4.29 | 1.0× | 1.000 | 0.4567 | 0.7312 |

The exact-search Recall@10 (0.4567) equals the earlier LTR dense result, which confirms the setup. The tuning slice shows the same pattern (full table in the JSON).

**Findings.**

- **nprobe = 8** scores 3.2% of the catalog and is 6.0× faster than exact search. It keeps 96.3% of the exact top 10. The cost in task quality is 1.7 points of Recall@10 (0.4567 → 0.4400) and 2.2 points of Recall@50 (0.7312 → 0.7090).
- **nprobe = 16–32** is nearly lossless (Recall@10 within 0.6 points, Recall@50 within 1 point) and still 1.7–3.3× faster.
- **Large nprobe is slower than exact search** in this implementation. Exact search is one dense matrix–vector product that NumPy runs efficiently. IVF gathers the rows of many lists into a new array first, and that copying costs more than the arithmetic it saves once a large share of the catalog is probed. IVF pays off when only a small share is probed, and the savings grow with catalog size.
- At nprobe = 128, Recall@50 is 0.0003 above exact search (one query). A product tied in score at the cutoff can be ordered differently, because the two searches visit rows in a different order.
- **Scope.** At 50,000 products, exact search takes 4.3 ms per query, so the submission keeps exact dense retrieval and does not trade recall for speed. The IVF results show the trade-off a larger catalog would require. Stage 1 passes 50 candidates to the reranker, so Recall@50 is the number to watch: with nprobe = 16, it loses only 1 point.
