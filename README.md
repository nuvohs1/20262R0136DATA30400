# DATA304 Final Project: Review-to-Product Retrieval

Match five reviews of one product to 10 distinct candidates from the full 50,000-product catalog. The official metric is Recall@10.

## Current status

The data audit and fixed development split are complete. A BM25 baseline, shared text preprocessing, and strict submission validation are implemented. No Kaggle scores are available yet. Dense retrieval and learning-to-rank are future steps.

## Data setup

Obtain the original files from the course distribution and place them under `project_release/`:

```text
project_release/
  catalog.json
  train_gold.json
  test_queries.json
  category_taxonomy.txt
  dummy_baseline.ipynb
```

The provided data and assignment PDF are excluded from Git. Review contents must remain local and must not be sent to external LLM services, including as summaries or rewrites. Only aggregate audit statistics and split query IDs are saved by the script below.

## Reproduce Step 1

Python 3.10 or later; no third-party dependencies are required.

```bash
python scripts/prepare_data.py
```

This validates the original inputs, checks ID and label consistency, audits missing fields and duplicate review texts, and writes:

- `artifacts/data_audit.json`: counts, validation checks, and SHA-256 input fingerprints.
- `artifacts/dev_split.json`: fixed training and validation query IDs (seed 42).

Override paths with `--data-dir` and `--output-dir`. Do not move or edit the original data to run the script.

## Run the BM25 baseline

Python 3.10 or later with SQLite FTS5 support is sufficient. No package installation, GPU, model download, or external service is required. The recorded run uses Python 3.15.2 and SQLite 3.53.4. Check your interpreter with:

```bash
python -c "import sqlite3; sqlite3.connect(':memory:').execute('CREATE VIRTUAL TABLE t USING fts5(text)'); print(sqlite3.sqlite_version)"
python -m unittest discover -s tests -v
python scripts/run_bm25.py
```

The default command evaluates all 4,000 validation queries against all 50,000 products. It writes metrics, per-category results, validation rankings, and ID-only success/failure records under `outputs/bm25_baseline/`. That directory stays local. Metrics include code and input fingerprints, Python/SQLite versions, timings, and the exact configuration.

The preprocessing applies Unicode NFKC normalization, HTML unescaping, tag/media-placeholder removal, lowercasing, and ASCII letter/digit tokenization. Numbers are retained; hyphens split tokens. All four metadata fields have equal weight. All five review texts are concatenated; ratings, helpfulness, timestamps, verified-purchase flags, IDs, and category labels are not retrieval features.

For a practical CPU baseline, terms appearing in more than 20% of catalog products are excluded from queries, using the catalog itself rather than an external stopword list. Up to 64 remaining terms are selected by `IDF * (1 + log(query term frequency))`; selected terms each contribute once to the BM25 query. Query filtering is part of this baseline, and does not reduce the candidate catalog. FTS5 fixes `k1=1.2`, `b=0.75`, and uses Robertson IDF, replacing non-positive IDF with `1e-6`. It differs from the positive `log(1 + ...)` IDF used by some libraries. The positive IDF variant is used only to select query terms. No supervised training or parameter search is performed in this step. Ties use item ID order.

To disable both query filters for a separate ablation, pass `--max-query-terms 0 --max-document-frequency 1 --output-dir outputs/bm25_unfiltered`. Unfiltered long review queries can be considerably slower. `--validation-limit 20` is available for a smoke check only; a limited run is not a full validation result.

To generate a Kaggle-format test file, replace `YOUR_STUDENT_ID` with your own student ID:

```bash
python scripts/run_bm25.py --predict-test --student-id YOUR_STUDENT_ID
python scripts/validate_submission.py outputs/bm25_baseline/YOUR_STUDENT_ID_final.csv
```

Test generation also runs full validation first. The CSV has 19,858 rows in the original test order, exactly ten distinct valid candidates per query, and the required header. The script validates it automatically. This creates a local file; submit it on Kaggle to obtain an actual leaderboard score.

On Bash-compatible systems, the audit, checks, and full baseline can be reproduced together with:

```bash
bash run_bm25.sh
# Or include test prediction:
bash run_bm25.sh --predict-test --student-id YOUR_STUDENT_ID
```

Set `PYTHON_BIN` to select another interpreter. On Windows PowerShell, the individual Python commands above work directly.

## Evaluation protocol

All validation retrieval searches must use the entire 50,000-product catalog. Validation labels, category labels, and reviews must be withheld from supervised training and from any review-enriched product index. Exact duplicate review texts across the split are counted by the audit and should be considered when interpreting results. Test labels are unavailable; do not infer a one-to-one matching constraint from file sizes or query IDs.

For experiments, use the same development split and report local Recall@10 separately from actual Kaggle leaderboard Recall@10. Required approaches are BM25, dense retrieval, and fine-tuning with learning-to-rank. Later steps will add their dependencies and exact execution commands. This BM25 baseline is an initial comparison method; the final method has not been selected.

See [the project plan](docs/project_plan.md) for requirements and planned milestones.

