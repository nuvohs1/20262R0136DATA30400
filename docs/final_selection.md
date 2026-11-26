# Final submission selection

## Rule

The final file is chosen by **local validation** (4,000 held-out development queries, full catalog, evaluated once per method), not by the Kaggle public leaderboard. The project metric is Recall@10. If two methods tie on Recall@10, the simpler pipeline wins, because it is cheaper and less fragile to reproduce.

## Candidates (local validation)

From `python scripts/evaluate_rankings.py` (`artifacts/experiments/ranking_metrics.json`):

| Method | R@1 | R@10 | NDCG@10 | MRR@10 | Kaggle (public) |
| --- | --- | --- | --- | --- | --- |
| Hybrid: BM25 v2 + 3·z(LTR BERT) | 0.3295 | 0.6673 | 0.4883 | 0.4322 | 0.72333 |
| Multi-stage: hybrid top 50 + BERT cross-encoder (w 0.5) | 0.3767 | 0.7073 | 0.5344 | 0.4801 | 0.76916 |
| Multi-stage + self-trained cross-encoder | 0.3845 | 0.7075 | 0.5383 | 0.4852 | 0.76936 |
| Multi-stage + category auxiliary (β = 0 selected) | identical to the cross-encoder row | | | | same file |
| Multi-stage, cross-encoder trained with hybrid hard negatives (w 0.6) | 0.3852 | 0.7145 | 0.5409 | 0.4864 | 0.75929 |
| ANN (IVF) | efficiency study of the dense stage; no new submission | | | | – |

## Decision (provisional)

**By the proxy rule fixed in advance, the selected pipeline is memory first stage + step-4b cross-encoder (w 0.5): `outputs/memory/submission/2024320020_final.csv`** (`run_memory.sh`).

- **Proxy evidence.** Proxy tuning-slice Recall@10 is 0.6025, against 0.5908 for step 4b's pipeline. Proxy validation is 0.6000 vs 0.5995, a tie: +4.2 points for products with labelled reviews, −4.1 for products without (`docs/memory_proxy.md`).
- **Why the old validation was replaced.** The hybrid-negative reranker had the best old-validation Recall@10 (0.7145) but scored 0.75929 on Kaggle, below step 4b (0.76916). The proxy reproduces that ordering; the old validation did not.
- **Step 4b remains the safe alternative** (Kaggle 0.76916). Its pipeline is identical to `exp + rerank` on the proxy.
- **Still open.** The cross-encoder ensemble (`ens`) is evaluated with the same rule, and the selection will be updated if it wins on the proxy tuning slice.
- **No overfitting to the leaderboard.** No weight, epoch, threshold or proxy setting is fitted to Kaggle scores or test data. The Kaggle result was used only to notice that the old validation did not match test.
- **Self-training is excluded from the final choice.** It used the test queries' reviews (without labels) as extra training data. That is transductive learning, and the final model should not be trained on test inputs.

## Kaggle final submissions

The student chooses the final Kaggle submissions. The candidates are the proxy-selected file and step 4b (0.76916).

## Format check

`python scripts/validate_submission.py <file>` checks the header `query_id,item_id_1,…,item_id_10`, the 19,858 test query IDs in their original order, 10 distinct IDs per row, and that every ID is in the 50,000-product catalog.

| File | Rows | Valid | SHA-256 |
| --- | --- | --- | --- |
| `outputs/hardneg/submission/2024320020_final.csv` | 19,858 | yes | `5853f6cb53f4a5c5b5ae857dd0acbdec90193eec70f6ec498732764edb401e73` |
| `outputs/rerank/submission/2024320020_final.csv` (current final) | 19,858 | yes | `e16dacbeaf32059ad97f410fde33672f582a9c00e4935d06799fcc6296b37380` |
| `outputs/pseudo/submission/2024320020_final.csv` | 19,858 | yes | `501bbfa8c12258f34612fb038db27f4759a5fac8e5a136a6ef8ec1438ca48ee6` |

The CSV files are generated outputs and are not committed (`outputs/` is in `.gitignore`). The SHA-256 above identifies the submitted file.
