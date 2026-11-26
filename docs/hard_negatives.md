# Cross-encoder trained with hybrid hard negatives

## Idea

The step-4b cross-encoder reranks the top 50 candidates of the hybrid first stage (BM25 v2 + 3 · z(LTR BERT)). It was trained with negatives sampled from the top 30 of **BM25 v2 alone** (`outputs/ltr/negatives_*.json`, mined before any dense model existed). Training and inference therefore see different negatives. BM25 negatives share words with the reviews. Hybrid candidates are also semantically close, because the bi-encoder adds products that match the reviews' meaning without sharing their words. A reranker learns to separate the kind of negatives it is trained on, so negatives from its own first stage should match the test situation better. This is the standard practice of mining hard negatives from the retriever that produces the candidates.

Only the source of the negatives changes. Model, input format, loss (listwise softmax over 1 positive + 7 negatives), negative pool (top 30), schedule, and seeds are the same as in step 4b (`docs/reranking.md`).

## Mining (`scripts/mine_hybrid_negatives.py`, CPU)

For each training query, the hybrid ranks the full catalog. The top 50 products other than the correct one are kept, as item IDs only.

| Set | Training queries | BM25 RSJ from | Dense embeddings |
| --- | --- | --- | --- |
| `dev` (`outputs/hardneg/negatives_dev.json`) | 12,000 development-training (excluding the tuning slice) | the same 12,000 | LTR model A (trained on these 12,000) |
| `full` (`outputs/hardneg/negatives_full.json`) | all 20,000 | all 20,000 | LTR model B |

- **No held-out data.** The tuning slice and the validation queries are not mined and not used. The script and `train_cross_encoder.py` both check that the training queries are disjoint from them.
- **No review expansion.** A training query's own reviews would otherwise be in the review index and reveal the correct product.
- **Only our own models.** The dense part must be the LTR `bert-base-uncased` checkpoint trained on the same query set, verified by `model_policy.py` from the embedding manifest. `train_cross_encoder.py` refuses hybrid negatives without this record. The checkpoint provenance records the negative source.
- **Same α as stage 1** (3, selected on the tuning slice in the dense step).
- **In-sample bi-encoder.** On its own training queries, the bi-encoder ranks the correct product higher than it would for new queries. That does not matter for mining, because the correct product is removed. The remaining top products are the ones the model still confuses with it, which is what hard negatives should be.

## Protocol

Unchanged from step 4b:

- **Model A:** cross-encoder on the 12,000 `dev` queries. The epoch is selected by tuning-slice Recall@10 at the best fusion weight.
- **Fusion weight:** w is chosen on the tuning slice with the same step-4b candidates. Validation is scored once.
- **Model B:** cross-encoder on all 20,000 queries, stopped at A's epoch. It scores the step-4b test candidates.

The step-4b candidates are reused, so stage-1 recall is identical and any change comes from the reranker. The new method replaces step 4b only if its validation Recall@10 is higher.

## How to run

On the course AWS notebook (step 4f), after `run_rerank.sh` and `run_category.sh 2024320020 encode`. The encode step provides model B's training-review embeddings.

```bash
bash run_hardneg.sh 2024320020 8   # mine -> dev -> full -> eval
```

The final file is `outputs/hardneg/submission/2024320020_final.csv`.

## Results

Run on the course AWS notebook (NVIDIA L4). Full record: `artifacts/experiments/hardneg_bert.json`.

**Negatives.** On its own training queries, the hybrid places the correct product in the top 10 for 78.5% (dev) and 79.1% (full) of queries. BM25 alone did so for 57.1%. The negatives that remain after removing the correct product are closer to it than BM25 negatives were. The training loss is higher than in step 4b (epoch 2: 1.24 vs 0.96 for model A), as expected for a harder task.

**Model A, tuning slice** (Recall@10 at the best w):

| Epoch | Step 4b (BM25 negatives) | Hybrid negatives |
| --- | --- | --- |
| 1 | 0.6940 | 0.6955 (w 0.5) |
| 2 | 0.6950 | **0.7020** (w 0.6) |

Epoch 2 and w = 0.6 are selected. Model B stops after epoch 2 (losses 1.57 and 1.20).

**Validation** (4,000 queries, same step-4b candidates, evaluated once):

| Ranker | R@1 | R@5 | R@10 | NDCG@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- |
| Stage 1 (hybrid) | 0.3295 | 0.5728 | 0.6673 | 0.4883 | 0.4322 |
| Step 4b cross-encoder only | 0.3018 | 0.5360 | 0.6380 | 0.4592 | 0.4033 |
| Hybrid-negative cross-encoder only | 0.3408 | 0.5700 | 0.6675 | 0.4947 | 0.4406 |
| Step 4b fused (w 0.5) | 0.3768 | 0.6130 | 0.7073 | 0.5344 | 0.4801 |
| **Hybrid-negative fused (w 0.6)** | **0.3853** | **0.6233** | **0.7145** | **0.5409** | **0.4864** |

**Paired comparison with step 4b** on validation (queries where only one ranker has the correct product within the cutoff; two-sided sign test):

| Cutoff | Hybrid negatives better | Step 4b better | p |
| --- | --- | --- | --- |
| @1 | 208 | 174 | 0.091 |
| @5 | 195 | 154 | 0.032 |
| @10 | 149 | 120 | 0.088 |

**Interpretation.**

- **The cross-encoder itself improves the most.** Alone, it gains 3 points of Recall@10 (0.638 → 0.668) and now matches the hybrid first stage. This is the effect the change targets: it has learned to separate the semantically close candidates that the bi-encoder brings in. It also earns a higher fusion weight (0.6 vs 0.5).
- **The fused gain is smaller (+0.7 points) but consistent.** The tuning slice improved first (0.695 → 0.702) and validation confirmed it (0.7073 → 0.7145). Every cutoff and every metric moves in the same direction. The individual sign tests are borderline (p 0.03–0.09), so the size of the gain is uncertain, but its direction is supported by two independent query sets.
- **Selection rule.** Validation Recall@10 is higher than step 4b, so this ranker replaces step 4b as the final method (`docs/final_selection.md`).

On test, the top 1 changes for 5,986 queries compared with step 4b. The top-10 lists share 7.7 items on average. The test CSV (SHA-256 `5853f6cb…`) passes the format check.
