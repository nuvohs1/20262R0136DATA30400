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

Pending the AWS run.
