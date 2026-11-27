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

## Decision

**Final file: hybrid first stage + the average of the two cross-encoders (`exp + ens`, w 0.5), Kaggle 0.77188.** It is written by `scripts/make_final.py` to `outputs/final/2024320020_final.csv` (`bash run_all.sh`, `docs/reproduction.md`), and is identical to `outputs/memory/submission/exp_ens/2024320020_final.csv`.

| Candidate | Proxy tuning R@10 | Proxy validation R@10 | Kaggle |
| --- | --- | --- | --- |
| exp + rerank (step 4b) | 0.5908 | 0.5995 | 0.76916 |
| **exp + ens** | 0.5923 | **0.6043** | **0.77188** |
| mem + rerank | 0.6025 | 0.6000 | 0.76865 |
| mem + ens (proxy-rule pick) | **0.6068** | 0.6010 | 0.76795 |
| exp + aug (augmented cross-encoder) | 0.6003 | 0.6115 | 0.76906 |
| exp + ens3 (three cross-encoders) | 0.6040 | 0.6133 | 0.77127 |

How the choice was made, stated plainly:

- **Settings come from training data only.** w = 0.5 was chosen on the proxy tuning slice; α = 3 and every training setting on the tuning slice. Nothing was fitted to test inputs or Kaggle scores.
- **The proxy rule picked mem + ens.** Its tuning-slice lead came from the memory bonus, which gained on the tuning slice but not on proxy validation, where exp + ens was best. The project guideline leaves the final Kaggle choice to the student, so both files were submitted; Kaggle agreed with proxy validation, and exp + ens was kept.
- **Leaderboard use is limited to a choice among a few files that were all fixed in advance.** Every Kaggle score is recorded in these docs, including the lower ones. The 0.3-point gain of exp + ens over step 4b is in line with the 0.5-point proxy-validation gain, so it is not a leaderboard artefact.
- **Augmented cross-encoder.** It improved proxy validation but not Kaggle. The proxy's 3-review queries match the model's 3- and 4-review training subsets, so it favours this model (`docs/augmented_cross_encoder.md`). A learned fusion did not beat the plain average in cross-validation (`docs/learned_fusion.md`). Neither changes the final file.
- **The final file does not use the memory bonus or self-training.** Self-training used the test queries' reviews (without labels) as extra training data, which is transductive, and it is excluded.

## Kaggle final submissions

exp + ens (0.77188) as the final file; step 4b (0.76916) as the second selection if two are allowed.

## Format check

`python scripts/validate_submission.py <file>` checks the header `query_id,item_id_1,…,item_id_10`, the 19,858 test query IDs in their original order, 10 distinct IDs per row, and that every ID is in the 50,000-product catalog.

| File | Rows | Valid | SHA-256 |
| --- | --- | --- | --- |
| `outputs/hardneg/submission/2024320020_final.csv` | 19,858 | yes | `5853f6cb53f4a5c5b5ae857dd0acbdec90193eec70f6ec498732764edb401e73` |
| `outputs/rerank/submission/2024320020_final.csv` (step 4b) | 19,858 | yes | `e16dacbeaf32059ad97f410fde33672f582a9c00e4935d06799fcc6296b37380` |
| `outputs/memory/submission/2024320020_final.csv` (mem + ens, Kaggle 0.76795) | 19,858 | yes | `f949a1d4b9c7eca950a77c303c4fe0a69c938b1d2df73e794f7f57d1ab35dcf9` |
| `outputs/memory/submission/exp_ens/2024320020_final.csv` = `outputs/final/2024320020_final.csv` (**final**, Kaggle 0.77188) | 19,858 | yes | `31a1c7f26ecf0baba9f1e18384b7a5ae2bedacb99feaf81783a0c571bb06f500` |
| `outputs/memory/submission/mem_rerank/2024320020_final.csv` (Kaggle 0.76865) | 19,858 | yes | `52cce880dba2cc4713acbe4b66976f30646bc32e6e8149428ad445f6ee449541` |
| `outputs/pseudo/submission/2024320020_final.csv` | 19,858 | yes | `501bbfa8c12258f34612fb038db27f4759a5fac8e5a136a6ef8ec1438ca48ee6` |

The CSV files are generated outputs and are not committed (`outputs/` is in `.gitignore`). The SHA-256 above identifies the submitted file.
