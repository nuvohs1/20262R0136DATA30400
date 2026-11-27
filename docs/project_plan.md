# Requirements and project plan

## Required deliverables

- Deadline: December 19, 2026, 23:59 KST.
- Kaggle: `studentID_final.csv`, containing `query_id,item_id_1,...,item_id_10`; 19,858 rows in the supplied test order; 10 distinct valid catalog item IDs per row, ranked by likelihood.
- Set the Kaggle team name to the assigned User ID in `User_IDs.xlsx`. The student ID and assigned User ID have not been supplied here.
- LMS: an English report PDF of at most 8 pages using the linked course template, plus the GitHub repository URL.
- Code: reproduce the best Kaggle submission without manual code edits; include complete README instructions, fixed random seeds, and a `.sh` runner if the final pipeline requires multiple steps. Document downloads for large intermediate artifacts or checkpoints.
- Grading: performance 50%, report quality 50%; at least 10 GitHub commits for 2% extra credit. Commit meaningful completed milestones before the deadline.

## Data and objective

- Catalog: 50,000 products; item ID, title, store, features, description. Some fields can be empty. No catalog reviews or category labels are supplied.
- Training: 20,000 queries, five reviews each, correct item ID and category path.
- Test: 19,858 queries, five reviews each; item and category labels are hidden.
- Taxonomy: 500 category paths.
- Each query has exactly one correct item. Return 10 distinct catalog IDs. Recall@10 is the fraction of queries whose correct item is in that list.
- `query_id` identifies the query, not the product. The problem does not impose unique predictions across queries.
- All supplied data, including test reviews, may be used during training. Keep development validation independent when comparing methods.

## Required experiments and report evidence

1. BM25 retrieval.
2. Dense retrieval.
3. Fine-tuning with learning-to-rank.
4. Meaningful comparisons or ablations and a final selected method.

Report actual Kaggle Recall@10 for all three required approaches in an experimental table. Local validation scores supplement those results and do not replace them. Do not invent scores or successful runs.

The report must explain the final matching method and design rationale; include a figure of the full prediction framework; describe preprocessing, use of all five reviews, training objectives, and relevant training strategies; and analyze at least one successful and one failed matching example. Report discussion will stay focused on retrieval evidence and errors, following the user's explicit scope instruction.

## Data and model restrictions

- Use only the supplied dataset. Do not collect outside product information or hidden labels.
- Public pretrained checkpoints are permitted only if their training data is publicly documented and shows no Amazon data used for downstream tasks (QA, retrieval). Unverifiable checkpoints must not be used (TA ruling). The only allowed checkpoint is `google-bert/bert-base-uncased` at a pinned revision; see [model eligibility](model_eligibility.md). The code enforces this allow-list.
- The stated external LLM API policy permits only GPT-4o mini, at most 1,000 calls total, with all prompts and outputs submitted separately. No external LLM API calls are planned for the pipeline.
- Never transmit review text, summaries, or rewrites to external LLM services. Run review processing locally and inspect aggregate results.
- LLM assistance with report wording is allowed; the student must review and take responsibility for all submitted content.
- Fix Python, NumPy, and PyTorch random seeds to 42 when those libraries are used, including CUDA seeds. Record relevant package, model, hardware, and training settings.

## Development protocol

- Verify the supplied data before modifying or indexing it; retain source fingerprints.
- Group development queries by item ID, then split products within categories with seed 42. Keep every query for a product in a single partition.
- Search all 50,000 catalog products during validation.
- Withhold validation reviews, product labels, and category labels from supervised fitting, hard-negative training, and any review-enriched catalog index. Original catalog metadata remains available for all products.
- Track duplicate review texts as an additional source of dependence across partitions.
- Record each experiment's configuration, split fingerprint, local metric, runtime, and actual Kaggle score when available.

## Planned milestones and commit checkpoints

1. Requirements, data audit, and development split (complete).
2. Shared text preprocessing and submission validation (implemented and checked).
3. BM25 baseline and full-catalog validation (complete: local Recall@10 36.25%; user-reported Kaggle score 0.38120).
4. BM25 field weighting, review aggregation, and query comparisons (complete: seven configurations; original baseline retained).
   - BM25 v2 (complete: BM25F k1 2.5 / b 1.0, plural folding, supervised RSJ query weights; held-out local Recall@10 57.83%; Kaggle 0.57538. With labelled-review expansion: Kaggle 0.60761, current best).
5. Eligible dense model selection and embedding pipeline (bert-base-uncased, revision pinned, allow-list enforced in code).
6. Dense baseline validation and Kaggle prediction (complete locally: zero-shot BERT 1.70% local Recall@10; hybrid selects α = 0; Kaggle 0.02306).
7. Learning-to-rank training with BM25 hard negatives (complete locally on AWS L4: dense 45.67%, BM25 v2 + LTR BERT hybrid 66.73% local Recall@10; Kaggle dense 0.56058, hybrid 0.72333).
8. Multi-stage ranking with a BERT cross-encoder reranker (complete on AWS L4: top-50 candidates, fused w = 0.5; local Recall@10 66.73% → 70.73%, NDCG@10 0.4883 → 0.5344; Kaggle 0.76916 = current best).
9. Self-training with pseudo-labels for the cross-encoder (complete on AWS L4: threshold margin 1.65 at 0.9 tuning precision, validation pseudo-label precision 0.915; local Recall@10 70.73% → 70.75% (no change), Recall@1 37.68% → 38.45%; Kaggle 0.76936 vs 0.76916, within noise; not adopted on local evidence).
10. Category labels as auxiliary information: label-GCN classifier trained multi-task on queries and products, category agreement as a third reranking score (complete: validation query accuracy 0.509@1 / 0.795@5 over 500 leaves, but every β > 0 lowers tuning-slice Recall@10, so β = 0 is selected and the ranking equals step 4b; documented negative result).
11. ANN with an IVF index for dense search (complete: nprobe 8 selected on the tuning slice scores 3.2% of the catalog, 6.0× faster than exact, ANN recall@10 0.963, validation Recall@10 0.4400 vs 0.4567 exact; efficiency study only, submission unchanged; docs/ann_ivf.md). Also: Precision@N, Recall@N, NDCG@N evaluation (implemented: docs/evaluation_metrics.md).
12. Cross-encoder trained with hybrid hard negatives, mined from its own first stage instead of BM25 (complete: tuning-slice R@10 0.695 → 0.702; validation R@10 0.7073 → 0.7145, every metric up; but Kaggle 0.75929 vs 0.76916, see item 13; docs/hard_negatives.md).
13. Test-like proxy validation with labelled-review memory: the hybrid-negative reranker scored 0.75929 on Kaggle (vs 0.76916) despite a better validation, because validation products never have labelled reviews while about half of the test products do. The proxy gives half of the tuning/validation products 2 of their own reviews as memory, adds a dense memory bonus to the first stage, and re-selects on the proxy tuning slice (complete: the proxy reproduces the Kaggle ordering of step 4b over hardneg; memory first stage + step-4b cross-encoder selected on the proxy tuning slice (0.6025 vs 0.5908); proxy validation tie 0.6000 vs 0.5995 (+4.2 memory half, −4.1 no-memory half); mem + rerank Kaggle 0.76865; averaging the two cross-encoders (ens) selected with memory on the proxy tuning slice (0.6068), proxy validation exp + ens 0.6043 / mem + ens 0.6010; Kaggle exp + ens 0.77188 (final), mem + ens 0.76795; docs/memory_proxy.md).
14. Final selection, reproducibility runner, and submission checks (selection complete: hybrid first stage + mean of the two cross-encoders, Kaggle 0.77188, docs/final_selection.md; runner `run_all.sh` quick|full with a SHA-256-verified model pack, docs/reproduction.md; AWS quick-path check and the model link pending).
15. Augmented cross-encoder (shared review budget, review augmentation, mixed negatives) and a three-model ensemble (complete: validation fused R@10 0.7147, cross-encoder alone 0.6765; proxy validation exp + ens3 0.6133 vs exp + ens 0.6043; docs/augmented_cross_encoder.md). Learned listwise fusion vs ens3 by 5-fold CV on the proxy tuning slice: ens3 kept (0.6040 vs 0.6028; docs/learned_fusion.md).
16. Case studies, report, and final reproduction documentation.

These are substantive checkpoints, not a reason to create empty or artificial commits. Steps may be adjusted based on measured results. Student ID: 2024320020. Course template, Kaggle competition URL, assigned User ID, and compute resources for neural models still need to be confirmed before their dependent tasks. The student handles Kaggle uploads.

