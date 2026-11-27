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

Run on the course AWS notebook. Full record: `artifacts/experiments/memory_proxy.json`.

**Memory bonus (proxy tuning slice, first stage).** Recall@10 rises from 0.5470 without memory to 0.5690 with μ = 1, τ = 3 (selected). Neighbouring settings are close (μ = 1, τ = 2: 0.5678; μ = 0.5, τ = 1: 0.5653), so the choice is not a lucky spike.

**Fusion weight and pair (proxy tuning slice, Recall@10 at the selected w):**

| First stage | rerank (step 4b) | hardneg |
| --- | --- | --- |
| exp | 0.5908 (w 0.5) | 0.5853 (w 0.5) |
| mem | **0.6025** (w 0.5) | 0.6025 (w 0.4) |

The tie goes to `rerank` by the fixed rule, so the selected pair is **mem + rerank**.

**Proxy validation (Recall@10, read once):**

| Ranker | All | Memory half | No-memory half |
| --- | --- | --- | --- |
| exp first stage | 0.5610 | 0.5735 | 0.5485 |
| exp + rerank, w 0.5 (= step 4b as submitted) | 0.5995 | 0.6125 | 0.5865 |
| exp + hardneg, w 0.6 (as submitted) | 0.5955 | 0.6020 | 0.5890 |
| exp + hardneg, w 0.5 (proxy-tuned) | 0.5965 | 0.6040 | 0.5890 |
| mem first stage | 0.5643 | 0.6285 | 0.5000 |
| **mem + rerank, w 0.5 (selected)** | **0.6000** | **0.6545** | 0.5455 |
| mem + hardneg, w 0.4 | 0.6015 | 0.6485 | 0.5545 |

**Findings.**

- **The proxy reproduces what Kaggle showed, unlike the old validation.** On the proxy, step 4b is above the hybrid-negative reranker (0.5995 vs 0.5955). The whole gap comes from the memory half (0.6125 vs 0.6020); on the no-memory half the hybrid-negative reranker is slightly better (0.5890 vs 0.5865), as on the old validation. This is the mechanism proposed above: the hybrid-negative reranker overrides the labelled-review signal more often.
- **The memory bonus moves the ranking toward products with labelled reviews.** For queries whose product has memory, it adds 4.2 points (0.6125 → 0.6545). For queries whose product has none, it costs 4.1 points (0.5865 → 0.5455), because other products with memory are pushed up. At the proxy's 50/50 share the two effects cancel (0.5995 → 0.6000). The real effect on test depends on how many test products have labelled reviews, which this project does not estimate from test data. Test memory is also stronger than in the proxy (5 labelled reviews instead of 2, and 5-review queries).
- **Tuning slice vs validation.** The memory gain is larger on the tuning slice (+1.2 points after fusion) than on validation (+0.05). The validation memory pool is larger (18,000 vs 14,000 products), so more non-matching products can receive a bonus. The test pool (20,000) is larger still. This is a reason for caution.
- **Reproducibility check.** The `exp + rerank` test file, rebuilt from scratch on this run, equals the step-4b submission in 19,857 of 19,858 rows. The remaining row swaps two adjacent items whose scores are tied to within floating-point rounding.

On test, the top 1 is a product with labelled reviews for 67.7% of queries with mem + rerank, against 55.0% for step 4b.

## Cross-encoder ensemble (added after the first proxy run)

`rerank` and `hardneg` were trained on different negatives, and the proxy shows their strengths differ: `rerank` is better on the memory half, and `hardneg` on the no-memory half. Averaging the two scores, `(z(rerank) + z(hardneg)) / 2` within each query's candidates, needs no extra training and is more diverse than seed variants of one model.

The ensemble is added to `run_memory_eval.py` as a third cross-encoder (`ens`) with the same rule. w is chosen on the proxy tuning slice, and a pair is selected by proxy tuning-slice Recall@10. Ties go to the simpler pair (`rerank`, then `hardneg`, then `ens`). Proxy validation is reported once.

**Results (same run, `eval` stage only).**

Proxy tuning slice, Recall@10 at the selected w:

| First stage | rerank | hardneg | ens |
| --- | --- | --- | --- |
| exp | 0.5908 (w 0.5) | 0.5853 (w 0.5) | 0.5923 (w 0.5) |
| mem | 0.6025 (w 0.5) | 0.6025 (w 0.4) | **0.6068** (w 0.4) |

By the fixed rule, the selected pair becomes **mem + ens** (`outputs/memory/submission/2024320020_final.csv`, SHA-256 `f949a1d4…`).

Proxy validation (Recall@10, read once):

| Ranker | All | Memory half | No-memory half | MRR@10 |
| --- | --- | --- | --- | --- |
| exp + rerank, w 0.5 (= step 4b) | 0.5995 | 0.6125 | 0.5865 | 0.3821 |
| **exp + ens, w 0.5** | **0.6043** | 0.6125 | **0.5960** | 0.3889 |
| mem + rerank, w 0.5 | 0.6000 | **0.6545** | 0.5455 | 0.3891 |
| **mem + ens, w 0.4 (selected)** | 0.6010 | 0.6525 | 0.5495 | **0.3927** |

- **The ensemble helps with both first stages.** With `exp`, it keeps step 4b's memory-half result (0.6125) and adds about one point on the no-memory half (0.5865 → 0.5960). With `mem`, it raises MRR@10 to the best value of all pairs. The two cross-encoders make different errors, as expected from their different training negatives.
- **Memory is still a trade-off.** `mem + ens` is selected on the tuning slice, but on proxy validation `exp + ens` has the higher Recall@10 (0.6043 vs 0.6010). As above, the gap between the two first stages depends on the unknown test share of products with labelled reviews.

**Kaggle (public, recorded after selection; not used to tune any setting).** `mem + rerank` scored 0.76865, against 0.76916 for step 4b (`exp + rerank`). The −0.05-point difference matches the proxy-validation tie (0.6000 vs 0.5995) and suggests the memory bonus does not add to review expansion on test.

| File | SHA-256 (prefix) | Kaggle |
| --- | --- | --- |
| `exp_rerank` (≈ step 4b) | `5900d189` | 0.76916 (step-4b file) |
| `mem_rerank` | `52cce880` | 0.76865 |
| `exp_ens` | `31a1c7f2` | **0.77188** |
| `mem_ens` (selected by the proxy rule) | `f949a1d4` | 0.76795 |

The Kaggle results agree with proxy validation, not with the proxy tuning slice: `exp + ens` has the best proxy-validation Recall@10 and the best Kaggle score, and every `mem` file is below its `exp` counterpart. The memory bonus is therefore not used in the final file (`docs/final_selection.md`).
