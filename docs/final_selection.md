# Final submission selection

## Rule

The final file is chosen by **local validation** (4,000 held-out development queries, full catalog, evaluated once per method), not by the Kaggle public leaderboard. The project metric is Recall@10. If two methods tie on Recall@10, the simpler pipeline wins, because it is cheaper and less fragile to reproduce.

## Candidates (local validation)

From `python scripts/evaluate_rankings.py` (`artifacts/experiments/ranking_metrics.json`):

| Method | R@1 | R@10 | NDCG@10 | MRR@10 | Kaggle (public) |
| --- | --- | --- | --- | --- | --- |
| Hybrid: BM25 v2 + 3·z(LTR BERT) | 0.3295 | 0.6673 | 0.4883 | 0.4322 | 0.72333 |
| **Multi-stage: hybrid top 50 + BERT cross-encoder (w 0.5)** | 0.3767 | **0.7073** | 0.5344 | 0.4801 | 0.76916 |
| Multi-stage + self-trained cross-encoder | 0.3845 | 0.7075 | 0.5383 | 0.4852 | 0.76936 |
| Multi-stage + category auxiliary (β = 0 selected) | identical to the cross-encoder row | | | | same file |
| ANN (IVF) | efficiency study of the dense stage; no new submission | | | | – |

## Decision

**Final file: the multi-stage ranker, `outputs/rerank/submission/2024320020_final.csv`** (produced by `run_rerank.sh`).

- **Recall@10 is tied.** The cross-encoder ranker scores 0.70725 and self-training scores 0.7075, a difference of one query in 4,000. In the paired comparison, self-training wins 75 queries and loses 74 (`docs/pseudo_labeling.md`). The Kaggle difference (0.76936 vs 0.76916) is equally small and is not used for selection.
- **Simplicity breaks the tie.** Self-training improves Recall@1 (0.3845 vs 0.3767), but the project metric is Recall@10. It also adds a second labelling round and two more cross-encoder trainings. The multi-stage ranker reaches the same Recall@10 with fewer moving parts.
- **No overfitting to the leaderboard.** Every weight, epoch and threshold was chosen on the tuning slice of the training data. Validation was read once per method. Test labels and Kaggle scores were never used to choose anything.

## Kaggle final submissions

If two final submissions may be selected on Kaggle, select:

1. the multi-stage ranker (`outputs/rerank/submission/2024320020_final.csv`, Kaggle 0.76916): the selected final file;
2. the self-trained variant (`outputs/pseudo/submission/2024320020_final.csv`, Kaggle 0.76936): tied on local Recall@10.

The same multi-stage file is uploaded to the LMS as `2024320020_final.csv`.

## Format check

`python scripts/validate_submission.py <file>` checks the header `query_id,item_id_1,…,item_id_10`, the 19,858 test query IDs in their original order, 10 distinct IDs per row, and that every ID is in the 50,000-product catalog.

| File | Rows | Valid | SHA-256 |
| --- | --- | --- | --- |
| `outputs/rerank/submission/2024320020_final.csv` (final) | 19,858 | yes | `e16dacbeaf32059ad97f410fde33672f582a9c00e4935d06799fcc6296b37380` |
| `outputs/pseudo/submission/2024320020_final.csv` | 19,858 | yes | `501bbfa8c12258f34612fb038db27f4759a5fac8e5a136a6ef8ec1438ca48ee6` |

The CSV files are generated outputs and are not committed (`outputs/` is in `.gitignore`). The SHA-256 above identifies the submitted file.
