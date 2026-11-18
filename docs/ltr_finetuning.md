# Fine-tuning E5 with a listwise learning-to-rank loss

## Objective

For a batch of B training queries, each query's candidate list contains its correct product, the other B−1 correct products in the batch, and h = 3 hard negatives per query (shared across the batch, so 4B candidates in total). The query vector is the re-normalized mean of the five review embeddings; product vectors use the same `passage:` text as zero-shot retrieval. The loss is the listwise softmax cross-entropy (ListNet top-1 probability, equivalent to InfoNCE):

`L = −log exp(q·p⁺/τ) / Σ_j exp(q·p_j/τ)`, with τ = 0.02.

Copies of a query's correct product elsewhere in the batch are masked out. Gradients flow through all five reviews.

## Hard negatives

`scripts/mine_negatives.py` ranks the catalog for every training query with the validated first stage: BM25 v2 (metadata only) + 3 · z(zero-shot E5). It keeps the top 50 products other than the correct one, as item IDs only. During training, 3 negatives per query are drawn without replacement from the top 30, using `random.Random(42 + epoch)`. Review expansion is disabled during mining, because a training query's own reviews would otherwise be in the review index.

## Leakage protocol

| Model | Trains on | RSJ weights for mining | Epoch selection | Used for |
| --- | --- | --- | --- | --- |
| A (`e5-ltr-dev`) | 12,000 development-training queries | the same 12,000 | dense Recall@10 on the 4,000-query tuning slice | α tuning and the single validation evaluation |
| B (`e5-ltr-full`) | all 20,000 training queries | the same 20,000 | none: same schedule, stops at A's selected epoch | test predictions |

Validation queries are never trained on by model A or used for any choice. First-stage Recall@10 on the mining queries: 60.93% (dev) and 61.11% (full), which is close to validation (61.95%).

## Settings

e5-base-v2 at revision `f52bf8ec…`; AdamW with lr 2e-5 and weight decay 0.01; linear warmup (10%) and decay over 3 epochs; batch 32 queries (160 reviews + 128 products); fp16 autocast with gradient checkpointing; gradient clipping at 1.0; review and passage lengths 192 and 256 tokens for training (256 / 256 for encoding). Seeds: Python, NumPy, and PyTorch (CPU and CUDA) are all set to 42; `torch.use_deterministic_algorithms(True, warn_only=True)`; `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

## How to run

1. Locally: `python scripts/mine_negatives.py --workers 8` creates `outputs/ltr/negatives_{dev,full}.json`.
2. Colab (T4): `notebooks/ltr_finetune_colab.ipynb` trains A, encodes with A (catalog and train), trains B, and encodes with B (catalog and test).
3. Locally:

```bash
python scripts/run_dense.py --embeddings outputs/dense/e5-ltr-dev --test-embeddings outputs/dense/e5-ltr-full \
    --output-dir outputs/dense_eval/e5-ltr --predict-test --student-id 2024320020 --workers 8
```

## Results

Pending the Colab run.
