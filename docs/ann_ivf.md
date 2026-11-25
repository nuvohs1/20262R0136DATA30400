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

Pending the AWS run.
