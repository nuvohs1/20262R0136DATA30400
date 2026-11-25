# Self-training with pseudo-labels

## Idea

The training file labels 20,000 queries, but the 19,858 test queries are unlabeled. Self-training uses the current model's confident predictions on unlabeled data as extra labels and trains again (course technique; compare the self-training example with a 0.9 confidence threshold). Here the "model" is the multi-stage ranker from [multi-stage ranking](reranking.md), and the extra labels train its cross-encoder.

Only course files are used: reviews from `train_gold.json` / `test_queries.json` and products from `catalog.json`. No external data, model or API is involved. The base model is still `google-bert/bert-base-uncased` at the verified revision. The provenance file of a self-trained checkpoint records `fine_tuning_data = "course train_gold.json + pseudo-labelled course queries (self-training)"` and the SHA-256 of the pseudo-label file. `model_policy.py` refuses such a checkpoint without that fingerprint.

## Method

1. **Confidence.** For an unlabeled query, the step-4b ranker computes `final = (1 − w) · z(stage 1) + w · z(cross-encoder)` over its 50 candidates (w = 0.5, chosen on the tuning slice). The pseudo-label is the top product. The confidence is the margin between the best and second-best `final` scores.
2. **Threshold.** The margin threshold is the smallest value on a grid (0.00, 0.05, …, 4.00) whose pseudo-label precision on the **labelled tuning slice** reaches 0.9. The 0.9 target is fixed in advance, not tuned. Validation labels are not read.
3. **Hard negatives.** Each pseudo-labelled query gets up to 30 negatives: its stage-1 candidates in stage-1 order, without the pseudo-labelled product.
4. **Retraining.** The cross-encoder is trained again from the base BERT checkpoint on labelled + pseudo-labelled queries. Settings are identical to step 4b: listwise loss over 1 positive + 7 negatives, 2-epoch schedule, lr 2e-5, seed 42. Fusion with stage 1 and the choice of w on the tuning slice are also unchanged.

## Leakage protocol

| Run | Labelled queries | Pseudo-labelled pool (reviews only) | Pseudo-labels from | Used for |
| --- | --- | --- | --- | --- |
| Self-trained model A (`outputs/pseudo/ce-dev`) | 12,000 development-training | tuning slice + validation (8,000) | step-4b model A | epoch and w on the tuning slice; validation once |
| Self-trained model B (`outputs/pseudo/ce-full`) | all 20,000 | test (19,858) | step-4b model B | Kaggle file, stopped at A's epoch |

The validation experiment mirrors the test setting. Test queries are pseudo-labelled and then reranked, so validation queries are likewise pseudo-labelled (from their reviews only) before they are reranked. This is transductive self-training. Validation labels are compared only once, at the final evaluation, together with the precision of the validation pseudo-labels. The tuning-slice labels select the margin threshold, the epoch and w, as in all earlier steps. Model A never sees test reviews. Model B never scores tuning or validation candidates; `score_cross_encoder.py` refuses this.

## How to run

On the course AWS notebook (step 4c), after `run_rerank.sh`:

```bash
bash run_pseudo.sh 2024320020   # labels-dev -> dev -> labels-full -> full -> eval
```

The final file is `outputs/pseudo/submission/2024320020_final.csv`. Metrics are in `outputs/pseudo/submission/metrics.json`. The pseudo-label summaries are in `outputs/pseudo/pseudo_{dev,full}_summary.json`.

## Results

Run on the course AWS notebook (NVIDIA L4). Full record: `artifacts/experiments/pseudo_bert.json`.

**Pseudo-labels.** On the tuning slice, precision rises steadily with the margin (0.36 for all queries, 0.76 at margin 1.0, 0.90 at 1.65, about 0.98 above 2.6). The selected threshold is **1.65**: tuning-slice precision 0.901 at 13.4% coverage.

| Pool | Queries | Pseudo-labelled | Precision |
| --- | --- | --- | --- |
| Tuning slice | 4,000 | 537 | 0.901 (used to pick the threshold) |
| Validation | 4,000 | 542 | **0.915** (checked once, after the final evaluation) |
| Test | 19,858 | 2,762 (13.9%) | unknown |

The threshold transfers: validation precision (0.915) matches the 0.9 target, and test coverage (13.9%) matches the tuning slice (13.4%).

**Training.** Self-trained model A: 12,000 + 1,079 queries, 1,634 steps per epoch, about 36 minutes per epoch. Its tuning-slice Recall@10 at the best w was 0.6945 (epoch 1) and 0.6950 (epoch 2), so epoch 2 and w = 0.5 are selected, the same as without self-training. Self-trained model B: 20,000 + 2,762 queries, 2 epochs, about 62 minutes per epoch.

**Validation** (4,000 queries, evaluated once):

| Ranking | R@1 | R@10 | NDCG@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| Stage 1 only | 0.3295 | 0.6673 | 0.4883 | 0.4322 |
| Cross-encoder only, step 4b | 0.3018 | 0.6380 | 0.4592 | 0.4033 |
| Cross-encoder only, self-trained | 0.3258 | 0.6413 | 0.4728 | 0.4203 |
| Fused, step 4b | 0.3768 | 0.7073 | 0.5344 | 0.4801 |
| **Fused, self-trained** | **0.3845** | **0.7075** | **0.5383** | **0.4852** |

Paired comparison of the two fused rankings on the same validation queries: Recall@10 has 75 wins and 74 losses (no change). Recall@1 has 117 wins and 86 losses (+0.8 points; sign test p ≈ 0.03).

Observations:

- Self-training improves the order at the top of the list. The cross-encoder alone gains 2.4 points of Recall@1, and the fused ranking gains 0.8 points of Recall@1 and 0.4 points of NDCG@10. It does not change Recall@10, the competition metric.
- This is expected from the pseudo-labels that pass a 0.9 precision threshold. They are the queries the ranker already gets right with a clear margin, so they add few new correct products to the top 10. They mainly sharpen the score of products the model already ranks highly.
- Lowering the threshold would add more queries at lower precision (for example 0.76 precision at 24% coverage). Testing that would need another tuning loop of GPU runs, and it was not pursued.

**Test file:** `outputs/pseudo/submission/2024320020_final.csv` (valid; the candidates are identical to step 4b). Compared with the step-4b file, the top-1 product differs for 3,352 test queries, and the top-10 set differs for 16,742. Kaggle Recall@10: pending.

**Selection.** Validation Recall@10 is equal to step 4b, so self-training is not adopted as the final method on local evidence. The step-4b reranker (Kaggle 0.76916) remains the selected pipeline, because it is simpler at equal Recall@10. A Kaggle score for this file is recorded for completeness and does not change the selection.
