# Test-like proxy validation and labelled-review memory

## Why the earlier validation is not enough

The cross-encoder trained with hybrid negatives (`docs/hard_negatives.md`) improved local validation Recall@10 from 0.7073 to 0.7145. On Kaggle it scored 0.75929, against 0.76916 for step 4b. The two test files differ in the top 1 for about 6,000 queries, so a gap of one point is far beyond noise. The cause is a known difference between validation and test (`docs/bm25_v2.md`, review expansion):

- **Test.** About half of the test products also have a labelled training query, i.e. five labelled reviews in `train_gold.json`. The test first stage uses them through BM25 review expansion. On Kaggle, review expansion added 3.2 points to BM25.
- **Validation.** Every product has exactly one query, so no validation product ever has labelled reviews. Review expansion cannot help locally, and the candidates were built without it.

The cross-encoder never sees labelled reviews. When validation favours a larger cross-encoder weight, the test ranking loses part of the labelled-review signal. A label-free check is consistent with this. Among the test top-1 products, the share that have labelled reviews is 58.8% for the hybrid, 55.0% for step 4b and 50.6% for the hybrid-negative reranker. That ordering matches the Kaggle results in reverse.

This check reads only our own test predictions and the list of labelled training products. It uses no test labels, and nothing was tuned on it. The Kaggle score was used once, to notice that the validation did not represent test. It is not used to choose any setting below.

## Proxy (`src/review_retrieval/memory.py`)

The proxy recreates the test situation inside the tuning slice and the validation set:

- **Memory half.** Half of the queries are chosen with `random.Random(42)`. For each, 2 of its 5 reviews become labelled memory of its own product, and the query keeps the other 3 reviews.
- **Other half.** These queries also keep 3 reviews, and the other 2 are dropped. This keeps query length the same across both halves.
- **No overlap.** Memory reviews and query reviews never overlap. The query and its memory are built from the same product's reviews, exactly as a test query and a training query of the same product would be.
- **Share of 0.5 is a neutral prior** (either case equally likely). It is not fitted to test data. It happens to agree with the rough earlier observation from BM25 v2, but no setting is tuned to it. Results are also reported separately for each half, so the reader can see how a different real share would change them.
- **Labelled pool.** The tuning slice uses the other 12,000 development-training queries plus the proxy memory. Validation uses the 16,000 development-training queries plus the proxy memory. This mirrors test, which uses all 20,000 training queries.

The proxy is weaker than test: queries have 3 reviews instead of 5, and memory has 2 reviews instead of 5. It is meant to rank methods, not to predict the Kaggle score.

## Memory in the first stage (`scripts/make_memory_candidates.py`)

| Variant | First stage |
| --- | --- |
| `exp` | BM25 v2 with review expansion + 3 · z(LTR BERT). This is the step-4b test first stage, now also used on the proxy splits. |
| `mem` | `exp` + a dense memory bonus `μ · max(0, z(memory) − τ)`. The memory score is the inner product between the query's mean review embedding and the mean embedding of the product's labelled reviews. z is taken over products with memory, so only matches that stand out count, as in BM25 review expansion. |

- **Choosing μ and τ.** (μ, τ) is chosen from {0, 0.5, 1, 2, 3, 4} × {0, 1, 2, 3} by first-stage Recall@10 on the proxy tuning slice. On ties, the smaller μ wins, then the larger τ.
- **RSJ weights** are still learned from the full labelled queries only.
- **Embeddings.** The proxy splits use LTR model A (never trained on these queries). Test uses model B and all 20,000 training queries.

## Reranking and selection (`scripts/run_memory_eval.py`)

No model is retrained. Both existing cross-encoders are scored on the candidates of both variants:

- `rerank`: step 4b, trained with BM25 negatives;
- `hardneg`: trained with hybrid negatives.

The proxy splits are scored with model A, and test with model B. The proxy queries give the cross-encoder only their 3 kept reviews.

**Selection rule (fixed before the run).**

- **Fusion weight.** For each of the four pairs (variant × cross-encoder), w is chosen on the proxy tuning slice.
- **Pair.** The pair with the best proxy tuning-slice Recall@10 is selected. Ties go to `exp` before `mem`, then `rerank` before `hardneg`.
- **Validation.** Proxy validation is reported once for every pair: overall, and for the memory and no-memory halves.
- **Earlier weights.** The submitted weights (rerank w 0.5, hardneg w 0.6) are also scored on the proxy. If the proxy reproduces the Kaggle ordering (step 4b above hardneg), that supports it as a better stand-in for test than the old validation.

All four test files are written to `outputs/memory/submission/<variant>_<model>/`. The selected one is `outputs/memory/submission/2024320020_final.csv`.

## How to run

On the course AWS notebook (step 4g), after steps 4b, 4d (`encode`) and 4f. There is no training, only scoring.

```bash
bash run_memory.sh 2024320020 8   # candidates (CPU) -> score (GPU) -> eval
```

## Results

Pending the AWS run.
