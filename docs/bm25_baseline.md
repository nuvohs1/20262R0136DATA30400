# Initial BM25 baseline

## Measured local results

The fixed development split contains 4,000 validation queries. Every query searches the entire 50,000-product catalog. The index uses only product metadata, with no training-review enrichment, category labels, or validation labels as features.

| Method | Recall@1 | Recall@5 | Recall@10 | Kaggle Recall@10 |
| --- | ---: | ---: | ---: | --- |
| BM25 with catalog-derived query filtering | 13.20% | 28.025% | 36.25% | Not submitted |

The correct product is among the first ten candidates for 1,450 of 4,000 validation queries. There are 528 rank-one hits and 1,121 top-five hits. The first twenty full-catalog parallel predictions exactly match the earlier serial smoke-check predictions. Eight automated checks cover numerical BM25 correctness, preprocessing, common-term scoring, deterministic ties, query filtering, submission rejection, zero-match behavior, and serial/parallel equivalence.

These are local development scores. No leaderboard score has been observed, and the baseline has not been selected as the final solution. Dense retrieval and fine-tuning with learning-to-rank have not yet been evaluated.

## Configuration

- Normalize text with Unicode NFKC and HTML unescaping, strip HTML and media placeholders, lowercase, and retain ASCII letter/digit tokens. Hyphens separate tokens; numbers are retained.
- Concatenate title, store, features, and description with equal weight. Missing fields contribute no tokens.
- Concatenate all five reviews. Use review text only.
- Exclude query terms occurring in more than 20% of catalog documents. Choose up to 64 remaining terms by positive-IDF times `1 + log(query frequency)`, using token order to break selection ties.
- Search the selected terms with binary query frequency. SQLite FTS5 BM25 uses `k1=1.2`, `b=0.75`, and Robertson IDF, replacing non-positive IDF with `1e-6`. Positive IDF is used only for query term selection.
- Break ranking ties by product ID. If fewer than ten products match, fill the remaining positions with non-duplicate catalog IDs in sorted order.
- Eight local worker processes use the same read-only product index. Python's random seed is 42; the scoring and ranking procedure uses no random sampling. NumPy and PyTorch are not used.

The query filters were introduced to reduce the cost of long review queries on CPU. They are part of this method's configuration. Their effect on matching quality has not been isolated in a full ablation.

## Reproduce

Use the original course data and fixed split, following the README setup instructions:

```bash
python scripts/run_bm25.py --predict-test --student-id 2024320020 --workers 8
python scripts/validate_submission.py outputs/bm25_baseline/2024320020_final.csv
```

The script rebuilds the index from the original catalog, runs validation, and generates the test CSV in its original query order. It does not submit to Kaggle. No external API calls are made. All review processing stays on the local machine. The generated SQLite index contains product metadata only and can be recreated from the course catalog.

Aggregate execution metadata is stored in [the experiment record](../artifacts/experiments/bm25_baseline.json). Local prediction files, category metrics, ID-only case records, and the index remain under `outputs/bm25_baseline/` and are excluded from Git.

The recorded eight-worker CPU run took about 10 seconds to build and save the index, 124 seconds for full validation, and 598 seconds for test retrieval and file generation (736 seconds total including loading and bookkeeping). These timings depend on the machine and load. All 19,858 test rows passed both the built-in check and the standalone submission validator. Every validation and test query retained at least one matching term after filtering.

## Next comparisons

The next BM25 experiments should compare metadata field weighting, query term selection, and aggregating review-level scores against this fixed baseline. Keep the development split and all 50,000 candidate products unchanged. Do not treat category-level differences as stable estimates: the 500 categories contain only 4 to 38 validation queries each.

Select one successful and one failed query for a later local case study using `validation_cases.json`. Inspect the original reviews locally; that file intentionally stores IDs and ranks rather than review text.
