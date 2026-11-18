# DATA304 Final Project: Review-to-Product Retrieval

Match five reviews of one product to 10 distinct candidates from the full 50,000-product catalog. The official metric is Recall@10.

## Current status

The improved BM25 (BM25F, plural folding, and supervised Robertson–Spärck Jones query weights) reaches **57.83% local Recall@10** on the 4,000 held-out validation queries against all 50,000 products, and **0.57538** on Kaggle. Adding a thresholded bonus from labelled training reviews (review expansion) gives the best Kaggle score so far, **0.60761**. The original BM25 reached 36.25% locally (Kaggle 0.38120). All hyperparameters were chosen on a tuning slice of the development-training partition; validation was evaluated once. See [the BM25 v2 record](docs/bm25_v2.md). The earlier SQLite comparisons are in [the BM25 comparison](docs/bm25_tuning.md) and [the original baseline](docs/bm25_baseline.md). Dense retrieval and learning-to-rank fine-tuning use only `google-bert/bert-base-uncased`, whose training data (BookCorpus and English Wikipedia, MLM/NSP) is publicly documented; the code refuses any other checkpoint ([model eligibility](docs/model_eligibility.md)). Their results are pending.

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

The default command evaluates all 4,000 validation queries against all 50,000 products. It uses up to eight CPU worker processes reading the same local SQLite index; pass `--workers 1` for serial execution. It writes metrics, per-category results, validation rankings, and ID-only success/failure records under `outputs/bm25_baseline/`. That directory, including the generated catalog index, stays local. The index contains only product metadata. Metrics include code and input fingerprints, Python/SQLite versions, timings, and the exact configuration. Tests verify that serial and parallel rankings match.

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

## Run the improved BM25 (v2)

Requires NumPy (`pip install -r requirements-fast.txt`); SciPy is not used.

```bash
python scripts/run_bm25_v2.py --predict-test --student-id 2024320020 --workers 8
python scripts/validate_submission.py outputs/bm25_v2/2024320020_final.csv
# Final BM25 submission (best Kaggle score, 0.60761): adds labelled-review expansion
python scripts/run_bm25_v2.py --predict-test --student-id 2024320020 --workers 8 --review-expansion
python scripts/validate_submission.py outputs/bm25_v2_expansion/2024320020_final.csv
```

`bash run_bm25_v2.sh 2024320020 8` runs the data audit, tests, both v2 variants, and submission checks. The final file is `outputs/bm25_v2_expansion/2024320020_final.csv`. The script reports held-out validation for the metadata model with and without review expansion, plus an expansion proxy. It then writes `outputs/bm25_v2/2024320020_final.csv` (or `outputs/bm25_v2_expansion/` with the flag). Label-derived statistics come only from the development-training partition for validation, and from all of `train_gold.json` for test. Details are in [docs/bm25_v2.md](docs/bm25_v2.md).

## Dense retrieval and learning-to-rank fine-tuning (BERT)

Only `google-bert/bert-base-uncased` at a pinned revision may be loaded. `src/review_retrieval/model_policy.py` enforces this in every script; evidence is in [docs/model_eligibility.md](docs/model_eligibility.md).

1. Locally: `python scripts/mine_negatives.py --workers 8` mines BM25 hard negatives (item IDs only) into `outputs/ltr/`.
2. On a GPU (Colab T4): `notebooks/gpu_colab.ipynb` runs `bash run_gpu.sh` (`requirements-dense.txt`). It computes zero-shot BERT embeddings, fine-tunes BERT with a listwise softmax ranking loss (model A on 12,000 development-training queries, model B on all 20,000), and encodes with both models. Finished steps are skipped on re-runs.
3. Locally, NumPy only:

```bash
# Zero-shot dense retrieval and BM25 + dense hybrid
python scripts/run_dense.py --embeddings outputs/dense/bert-base-uncased --output-dir outputs/dense_eval/bert-base-uncased --predict-test --student-id 2024320020 --workers 8
# Fine-tuned (learning to rank)
python scripts/run_dense.py --embeddings outputs/dense/bert-ltr-dev --test-embeddings outputs/dense/bert-ltr-full --output-dir outputs/dense_eval/bert-ltr --predict-test --student-id 2024320020 --workers 8
```

Protocols and results: [docs/dense_retrieval.md](docs/dense_retrieval.md), [docs/ltr_finetuning.md](docs/ltr_finetuning.md).

## Compare BM25 weighting and review fusion

```bash
python scripts/tune_bm25.py --workers 8
python scripts/finish_bm25.py --predict-test --student-id 2024320020 --workers 16
```

This compares the original equal-field baseline, field weights `(4, 2, 2, 1)`, and title-emphasis weights `(8, 3, 2, 1)` in title/store/features/description order. On the best concatenated-query configuration, it compares equal-weight reciprocal rank fusion of the five individual review lists with a hybrid fusion that also includes the concatenated query at weight two. All retrieval searches the full catalog; fusion uses the first 100 matches of each view with RRF constant 60. Individual review queries retain up to 24 selected terms; the concatenated view retains up to 64. This is a comparison of complete aggregation configurations, not an isolated test of fusion alone.

The separate-field implementation uses SQLite BM25 column weights on term frequency before saturation; document length remains the unweighted sum of all fields. This is not independent per-field length normalization. A rank-sorted cursor with explicit cutoff-tie expansion preserves deterministic product-ID tie-breaking.

The second command adds two controlled query comparisons with equal field weights: up to 128 distinct terms with binary frequency, and up to 64 distinct terms with query count weight `min(4, 1 + floor(log2(count)))`. FTS5 repeated OR phrases implement the integer query weights; automated tests check the score multiplication. This is a quantized logarithmic query weight, not an exact floating-point log-frequency factor.

Results are saved incrementally under `outputs/bm25_tuning/`, separately from the original baseline. The method is selected by local Recall@10, then Recall@1, then the earlier configuration on ties, before accessing test predictions. Only that method generates `outputs/bm25_tuning/2024320020_final.csv`. Add `--resume` to either command to reuse validation checkpoints only when source, code, split, configuration, and prediction fingerprints match. If verified baseline predictions are already present, the first script reuses them after checking twenty actual queries for equal-field equivalence; otherwise it recomputes the baseline. If the baseline remains selected, the second script reuses the verified original test file. Neither command submits to Kaggle.

The full audit, tests, seven-method comparison, and selected test prediction can also be reproduced with `bash run_bm25_tuning.sh 2024320020 8`. The last argument controls local worker count; reduce it for a machine with fewer CPU cores. The recorded run used eight workers for the field/fusion comparisons and sixteen for the two query comparisons.

An optional NumPy core scorer is provided to avoid repeatedly calculating the same document-term BM25 contributions. After the SQLite comparisons are complete, install `requirements-fast.txt` and run `python scripts/fast_bm25_prediction.py --predict-test --workers 8`. It checks all 4,000 validation rankings for the baseline and the two query variants against the SQLite CSVs, and requires an exact full-validation match for the selected method before generating `outputs/bm25_tuning/fast/2024320020_final.csv`. Close-score cases use the SQLite reference to preserve tie order. It uses no NumPy random sampling or SciPy. The package-free SQLite pipeline above remains sufficient for reproduction.

## Evaluation protocol

All validation retrieval searches must use the entire 50,000-product catalog. Validation labels, category labels, and reviews must be withheld from supervised training and from any review-enriched product index. Exact duplicate review texts across the split are counted by the audit and should be considered when interpreting results. Test labels are unavailable; do not infer a one-to-one matching constraint from file sizes or query IDs.

For experiments, use the same development split and report local Recall@10 separately from actual Kaggle leaderboard Recall@10. Required approaches are BM25, dense retrieval, and fine-tuning with learning-to-rank. Later steps will add their dependencies and exact execution commands. This BM25 baseline is an initial comparison method; the final method has not been selected.

See [the project plan](docs/project_plan.md) for requirements and planned milestones.

