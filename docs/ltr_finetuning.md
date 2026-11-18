# Fine-tuning BERT with a listwise learning-to-rank loss

## Objective

The base model is `google-bert/bert-base-uncased` at the verified revision ([model eligibility](model_eligibility.md)), trained as a bi-encoder with shared weights for reviews and products.

For a batch of B training queries, each query's candidate list contains:

- its correct product,
- the other B−1 correct products in the batch,
- h = 3 hard negatives per query, shared across the batch.

That gives 4B candidates in total. The query vector is the re-normalized mean of the five review embeddings, and product vectors use the same passage text as zero-shot retrieval. The loss is the listwise softmax cross-entropy (ListNet top-1 probability, equivalent to InfoNCE):

`L = −log exp(q·p⁺/τ) / Σ_j exp(q·p_j/τ)`, with τ = 0.05.

If a query's correct product also appears elsewhere in the candidate list (another query's hard negative, for example), that copy is masked out. Gradients flow through all five reviews.

## Hard negatives

`scripts/mine_negatives.py` ranks the catalog for every training query with BM25 v2 (metadata only, no pretrained model). It keeps the top 50 products other than the correct one, as item IDs only. During training, 3 negatives per query are drawn without replacement from the top 30, using `random.Random(42 + epoch)`. Review expansion is disabled during mining, because otherwise the review index would contain the training query's own reviews.

## Leakage protocol

| Model | Trains on | RSJ weights for mining | Epoch selection | Used for |
| --- | --- | --- | --- | --- |
| A (`bert-ltr-dev`) | 12,000 development-training queries | the same 12,000 | dense Recall@10 on the 4,000-query tuning slice | α tuning and the single validation evaluation |
| B (`bert-ltr-full`) | all 20,000 training queries | the same 20,000 | none: same schedule, stops at A's selected epoch | test predictions |

Model A never trains on the validation queries, and they are not used for any choice. `scripts/run_dense.py` refuses validation embeddings from a checkpoint trained on all 20,000 queries.

## Settings

- Optimizer: AdamW, lr 2e-5, weight decay 0.01.
- Schedule: linear warmup (10%) and decay over 4 epochs.
- Batch: 32 queries (160 reviews + 128 products).
- fp16 autocast with gradient checkpointing; gradient clipping at 1.0.
- Lengths: 192 tokens for reviews and 256 for passages during training; 256 / 256 for encoding.
- Seeds: Python, NumPy, and PyTorch (CPU and CUDA) are all set to 42; `torch.use_deterministic_algorithms(True, warn_only=True)`; `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Each saved checkpoint contains `provenance.json` (base model and revision, course-data-only statement, SHA-256 of `train_gold.json` and of the negatives file).

## How to run

1. Locally: `python scripts/mine_negatives.py --workers 8` creates `outputs/ltr/negatives_{dev,full}.json`.
2. GPU (Colab T4): `notebooks/gpu_colab.ipynb`, steps 2 and 3 (`bash run_gpu.sh ltr-dev`, `bash run_gpu.sh ltr-full`). These train A, encode with A (catalog and train), train B, and encode with B (catalog and test).
3. Locally:

```bash
python scripts/run_dense.py --embeddings outputs/dense/bert-ltr-dev --test-embeddings outputs/dense/bert-ltr-full \
    --output-dir outputs/dense_eval/bert-ltr --predict-test --student-id 2024320020 --workers 8
```

## Results

Pending the Colab run.
