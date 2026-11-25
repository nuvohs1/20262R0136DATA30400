# Multi-stage ranking with a BERT cross-encoder

## Pipeline

1. **Candidate generation (stage 1).** BM25 v2 + α · fine-tuned BERT bi-encoder (the hybrid from [learning-to-rank fine-tuning](ltr_finetuning.md), α = 3 as selected there) scores the full catalog. The top K = 50 products per query are kept with their scores (`scripts/make_candidates.py`, NumPy, CPU). The bi-encoder works as an efficient first stage: one dot product per product.
2. **Reranking (stage 2).** A BERT cross-encoder reads the reviews and one candidate product together and outputs a relevance score (`scripts/train_cross_encoder.py`, `scripts/score_cross_encoder.py`, GPU). It sees review and product tokens jointly in every layer. That makes it more accurate than the bi-encoder, but it is too expensive to run on all 50,000 products.
3. **Fusion.** `final = (1 − w) · z(stage 1) + w · z(cross-encoder)`, with z-scores taken within each query's K candidates. w = 0 keeps the stage-1 order and w = 1 is the pure cross-encoder order (`src/review_retrieval/rerank.py`).

## Cross-encoder

- **Base model:** `google-bert/bert-base-uncased` at the verified revision ([model eligibility](model_eligibility.md)), with a one-logit classification head. Fine-tuning uses `train_gold.json` only.
- **Input:** `[CLS] review_1 [SEP] … review_5 [SEP] passage [SEP]`, at most 512 tokens. Each review is cut to 64 WordPiece tokens (segment A). The product passage (Title | Store | Features | Description) fills the remaining space (segment B).
- **Objective (listwise learning to rank):** each training query forms a group of G = 8 candidates: the correct product and 7 negatives. The negatives are sampled without replacement from the top 30 BM25 results (`outputs/ltr/negatives_*.json`) with `random.Random(42 + epoch)`. The loss is softmax cross-entropy over the group's 8 scores.
- **Settings:** batch of 8 queries (64 pairs); AdamW with lr 2e-5 and weight decay 0.01; linear warmup (10%) and decay over 2 epochs; fp16 autocast with gradient checkpointing; gradient clipping at 1.0. All seeds are 42.

## Leakage protocol

| Item | Trained / computed on | Used for |
| --- | --- | --- |
| Tuning-slice candidates | RSJ from the other 12,000 development-training queries; bi-encoder model A | epoch selection and w |
| Cross-encoder A (`ce-dev`) | 12,000 development-training queries | tuning-slice and validation scores |
| Validation candidates and scores | RSJ from 16,000 development-training queries; model A | single final evaluation |
| Cross-encoder B (`ce-full`) | all 20,000 training queries, stopped at A's epoch | test scores |
| Test candidates | RSJ and review expansion from all 20,000 training queries; bi-encoder model B | Kaggle file |

The scripts refuse to score tuning or validation candidates with a model trained on all training queries. The top 10 of the test candidates must equal the stage-1 hybrid submission; this is checked automatically.

## How to run

On a GPU machine (course AWS notebook, step 4b), after `run_gpu.sh` and `run_dense_eval.sh`:

```bash
bash run_rerank.sh 2024320020 8   # candidates -> model A -> validation scores -> model B -> test scores -> evaluation
```

The final file is `outputs/rerank/submission/2024320020_final.csv`, and metrics are in `outputs/rerank/submission/metrics.json`.

## Results

Run on the course AWS notebook (NVIDIA L4, Python 3.10.20, torch 2.10.0+cu128, transformers 5.20.2). Full record: `artifacts/experiments/rerank_bert.json`.

**Candidate recall of stage 1** (share of queries whose correct product is in the top N candidates):

| Split | @10 | @20 | @30 | @50 |
| --- | --- | --- | --- | --- |
| Tuning slice (4,000) | 0.6558 | 0.7560 | 0.8143 | 0.8755 |
| Validation (4,000) | 0.6673 | 0.7668 | 0.8158 | 0.8790 |

With K = 50, reranking can reach at most 87.9% Recall@10 on validation.

**Model A training** (12,000 queries, 1,500 steps per epoch, about 33 minutes per epoch). Mean loss: 1.451 (epoch 1), 0.962 (epoch 2). Tuning-slice Recall@10 at the best w: 0.6940 (epoch 1) and 0.6950 (epoch 2), so epoch 2 is selected. Model B (20,000 queries, 2,500 steps per epoch) stops after epoch 2, with losses of 1.385 and 0.925.

**Fusion weight on the tuning slice** (model A, epoch 2):

| w | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | **0.5** | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R@10 | 0.6558 | 0.6643 | 0.6750 | 0.6853 | 0.6923 | **0.6950** | 0.6915 | 0.6875 | 0.6765 | 0.6605 | 0.6400 |

**Validation** (4,000 queries, evaluated once, w = 0.5 fixed beforehand):

| Ranking | R@1 | R@5 | R@10 | NDCG@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- |
| Stage 1 only (w = 0) | 0.3295 | 0.5728 | 0.6673 | 0.4883 | 0.4322 |
| Cross-encoder only (w = 1) | 0.3018 | 0.5360 | 0.6380 | 0.4592 | 0.4033 |
| **Fused (w = 0.5)** | **0.3768** | **0.6130** | **0.7073** | **0.5344** | **0.4801** |

Observations:

- On its own, the cross-encoder ranks the 50 candidates slightly worse than stage 1 (−2.9 points of Recall@10). The two scores are complementary, though. Fusing them gains +4.0 points of Recall@10 and +4.7 points of Recall@1 over stage 1, and the gain is largest at the top of the list.
- The tuning-slice curve is smooth and peaks at w = 0.5. The validation gain (+4.0) matches the tuning-slice gain (+3.9), so the choice of w did not overfit the tuning slice.
- The selected epoch differs little from epoch 1 (0.6950 vs 0.6940). Longer training is unlikely to help much.

**Test file:** `outputs/rerank/submission/2024320020_final.csv` (19,858 rows, ten distinct valid IDs per row, sha256 `e16dacbe…`). The test candidates' top 10 equal the stage-1 hybrid submission (Kaggle 0.72333), as checked by `run_rerank_eval.py`. Reranking changes the top-1 product for 6,744 of the 19,858 test queries. Kaggle Recall@10: **0.76916** (stage-1 hybrid: 0.72333, +4.6 points).
