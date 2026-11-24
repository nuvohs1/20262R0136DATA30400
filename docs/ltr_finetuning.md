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
2. GPU: `notebooks/aws_sagemaker.ipynb` (course AWS L4) or `notebooks/gpu_colab.ipynb` (Colab), steps 2 and 3 (`bash run_gpu.sh ltr-dev`, `bash run_gpu.sh ltr-full`). These train A, encode with A (catalog and train), train B, and encode with B (catalog and test).
3. Locally:

```bash
python scripts/run_dense.py --embeddings outputs/dense/bert-ltr-dev --test-embeddings outputs/dense/bert-ltr-full \
    --output-dir outputs/dense_eval/bert-ltr --predict-test --student-id 2024320020 --workers 8
```

## Results

Trained on the course AWS SageMaker instance (NVIDIA L4; Python 3.10.20, torch 2.10.0+cu128, transformers 5.20.2). An epoch took about 815 s for model A and 1,363 s for model B. Record: `artifacts/experiments/ltr_bert.json`.

Model A, dense-only retrieval on the 4,000-query tuning slice after each epoch (epoch selection):

| Epoch | Mean loss | R@1 | R@10 |
| --- | --- | --- | --- |
| 1 | 1.628 | 12.35% | 36.10% |
| 2 | 0.916 | 13.90% | 42.58% |
| **3 (selected)** | 0.672 | 15.35% | **44.48%** |
| 4 | 0.552 | 15.53% | 44.33% |

Model B was trained on all 20,000 queries for the same schedule and stopped after epoch 3 (mean loss 1.496, 0.859, 0.643).

Local validation (4,000 held-out queries, full catalog; model A embeddings, α tuned on the tuning slice):

| Method | R@1 | R@5 | R@10 | Kaggle R@10 |
| --- | --- | --- | --- | --- |
| BM25 v2 (metadata only) | 27.65% | 47.60% | 57.83% | 0.57538 |
| BM25 v2 + review expansion | —¹ | —¹ | —¹ | 0.60761 |
| Zero-shot BERT (dense only) | 0.27% | 0.92% | 1.70% | 0.02306 |
| LTR fine-tuned BERT (dense only, mean of reviews) | 16.00% | 34.83% | 45.67% | 0.56058 |
| LTR fine-tuned BERT (max over reviews) | 12.15% | 27.17% | 36.48% | — |
| BM25 v2 + 3 · LTR BERT (hybrid) | 32.95% | 57.27% | **66.73%** | **0.72333** (test uses BM25 with expansion) |

¹ Not measurable locally: the development split keeps every product in one partition, so validation products never appear among the labelled training reviews, whereas about half of the test products do ([BM25 v2](bm25_v2.md)).

Fine-tuning raises dense-only Recall@10 from 1.70% to 45.67%. The hybrid adds 8.9 points over BM25 alone. The tuning-slice curve peaks at α = 3 (65.58%) and is flat between 2 and 4. On validation the same region is best (66.37–66.73%), so the choice transfers. On test, the hybrid's top-1 agrees with the BM25 + expansion submission for 56.8% of queries, and the two top-10 lists share 6.0 items on average.

On Kaggle the hybrid scores 0.72333, 11.6 points above the best BM25 file (0.60761), and is the current best submission. Kaggle scores are higher than local validation for the dense and hybrid runs (0.56058 vs 45.67%, 0.72333 vs 66.73%). This is expected: model B is trained on all 20,000 training queries, and about half of the test products occur in training, whereas no validation product occurs in model A's training data.
