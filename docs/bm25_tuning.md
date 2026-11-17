# BM25 comparison results

## Decision

Retain the original equal-field BM25 with five concatenated reviews, up to 64 selected query terms, and binary query frequency. It achieves 36.25% local Recall@10 and the student reported a Kaggle score of 0.38120 for its submission. All six additional configurations performed worse on the same fixed development split. No replacement Kaggle submission is needed for this stage.

The selected CSV under `outputs/bm25_tuning/2024320020_final.csv` is byte-for-byte identical to the original baseline submission. It was revalidated for all 19,858 test rows. The original baseline files are preserved.

## Controlled local comparisons

All rows use the same 4,000 validation queries and search all 50,000 catalog products. Product metadata supplies the entire index; validation reviews, targets, and category labels are not used to enrich it. Targets are used only for scoring and method selection.

| Configuration | Recall@1 | Recall@5 | Recall@10 | Hits@10 |
| --- | ---: | ---: | ---: | ---: |
| Original BM25: equal fields, 64 terms, binary query frequency | 13.200% | 28.025% | **36.250%** | **1,450** |
| Field weights: title/store/features/description = 4/2/2/1 | 13.125% | 26.500% | 34.950% | 1,398 |
| Title emphasis: weights 8/3/2/1 | 13.375% | 26.975% | 35.700% | 1,428 |
| Review-only reciprocal rank fusion | 7.350% | 19.125% | 27.725% | 1,109 |
| Hybrid concatenated-query and review fusion | 8.175% | 23.225% | 33.525% | 1,341 |
| Equal fields, up to 128 terms | 11.450% | 24.650% | 32.300% | 1,292 |
| Equal fields, 64 terms, quantized logarithmic query frequency | 12.250% | 26.025% | 34.425% | 1,377 |

The original method's Kaggle score is student-reported, not independently checked; the leaderboard partition was not specified. The six alternatives were not submitted. Local scores above must not be presented as Kaggle scores.

Selection used Recall@10, then Recall@1, then the earlier configuration on ties. The reported Kaggle score was recorded but was not used to select configurations. The additional two query comparisons were specified after observing that field weighting and review fusion underperformed, and used the same development split. This is validation-based model selection, not a new independent held-out evaluation.

## Method details

The field implementation separates title, store, features, and description into FTS5 columns. Column weights multiply term frequency before BM25 saturation; document length remains the unweighted sum of all four fields. This is not independent field length normalization. With all weights equal to one, twenty actual full-catalog validation predictions were checked against the original concatenated implementation and matched exactly. Synthetic tests also check numerical score equivalence.

The two fusion variants use the best concatenated-query configuration, which remained the equal-field baseline. Each individual review retains up to 24 selected terms, and each view retrieves its first 100 products from the full catalog. Reciprocal rank fusion uses a constant of 60. Review-only fusion gives all five review views weight one. Hybrid fusion adds the concatenated query, with up to 64 terms, at weight two. These are comparisons of complete aggregation configurations: the per-review term budget changes alongside the aggregation mechanism.

The 128-term comparison changes only the maximum number of selected distinct query terms. The query-frequency comparison keeps 64 distinct terms and assigns each selected term integer weight `min(4, 1 + floor(log2(query_count)))`. Repeated FTS5 OR phrases implement these weights; automated tests verify the resulting score multiplication. Both keep equal metadata weights. All variants retain the original catalog-derived maximum document frequency of 20%, text normalization, BM25 `k1=1.2`, and `b=0.75`.

## What the results support

Emphasizing titles is not uniformly beneficial for the official metric. The strongest title weights slightly raised Recall@1 from 13.200% to 13.375%, but reduced Recall@10 from 36.250% to 35.700%. Selection therefore follows Recall@10.

Review-only fusion recovered 218 queries missed by the baseline but lost 559 baseline hits. Hybrid fusion recovered 190 and lost 299. Their different errors may be useful for future experiments, but these measured combinations lose more hits than they gain. Simply including more ranked views did not improve this validation set.

Increasing query coverage to 128 terms recovered 164 baseline misses but lost 322 hits. Query-frequency weighting recovered 178 but lost 251. These results support retaining the tested 64-term binary-frequency setting. They do not identify the precise textual cause of each failure; that requires local case inspection. No review text or rewritten reviews were exported for this analysis.

## Reproduction and verification

```bash
python scripts/tune_bm25.py --workers 8
python scripts/finish_bm25.py --predict-test --student-id 2024320020 --workers 16
python scripts/validate_submission.py outputs/bm25_tuning/2024320020_final.csv
```

Or run `bash run_bm25_tuning.sh 2024320020 8` for the full data audit, tests, comparison, and selected prediction. The course data must be supplied locally as described in the README. The SQLite pipeline needs no third-party packages. The scripts store validation checkpoints and can resume only when source, code, split, configuration, and prediction fingerprints match. Baseline predictions can be reused only after verifying their provenance and checksums.

The recorded field/fusion experiments used eight workers; the query comparisons used sixteen. The two fusion records share one six-view retrieval pass, so their recorded wall times must not be added as if they were separate runs. The selected baseline test file was reused after checksum validation rather than recomputed.

An optional deterministic NumPy core scorer precomputes document-term contributions. Its full 4,000-query validation CSVs match SQLite exactly for the original baseline, 128-term variant, and query-frequency variant: 12,000 rankings in total. Queries with close scores use SQLite to preserve order, including 211, 271, and 230 reference fallbacks respectively. This is a validation equivalence check; no additional accelerated test submission was generated because the original baseline remains selected. NumPy random generators and SciPy are not used. Eighteen tests passed with the optional NumPy dependency installed.

Aggregate records are available in [the seven-method experiment record](../artifacts/experiments/bm25_tuning.json) and [the optional array verification](../artifacts/experiments/bm25_numpy_verification.json). They include input and implementation fingerprints. Raw reviews and full prediction files remain local. No external LLM API calls were made by these pipelines, and Kaggle uploads remain the student's responsibility.

The next project stage is to select an eligible pretrained dense retrieval model, verify its training-data restrictions, and evaluate it against this retained BM25 reference.
