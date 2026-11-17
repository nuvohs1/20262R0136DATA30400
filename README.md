# DATA304 Final Project: Review-to-Product Retrieval

Match five reviews of one product to 10 distinct candidates from the full 50,000-product catalog. The official metric is Recall@10.

## Current status

Step 1 prepares the project requirements, validates the supplied data locally, and creates a deterministic, category-stratified product-level development split. No retrieval model has been trained and no Kaggle scores are available yet.

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

## Evaluation protocol

All validation retrieval searches must use the entire 50,000-product catalog. Validation labels, category labels, and reviews must be withheld from supervised training and from any review-enriched product index. Exact duplicate review texts across the split are counted by the audit and should be considered when interpreting results. Test labels are unavailable; do not infer a one-to-one matching constraint from file sizes or query IDs.

For experiments, use the same development split and report local Recall@10 separately from actual Kaggle leaderboard Recall@10. Required approaches are BM25, dense retrieval, and fine-tuning with learning-to-rank. Later steps will add their dependencies and exact execution commands. The current repository does not yet reproduce final predictions.

See [the project plan](docs/project_plan.md) for requirements and planned milestones.

