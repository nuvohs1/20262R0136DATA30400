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

Pending the AWS run.
