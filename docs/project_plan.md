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
6. Dense baseline validation and Kaggle prediction (complete locally: zero-shot BERT 1.70% local Recall@10; hybrid selects α = 0; Kaggle pending).
7. Learning-to-rank training with BM25 hard negatives (complete locally on AWS L4: dense 45.67%, BM25 v2 + LTR BERT hybrid 66.73% local Recall@10; Kaggle pending).
8. Further class techniques, each with course data and the allowed checkpoint only: ANN index (IVF) for dense search; multi-stage ranking with a BERT cross-encoder reranker; pseudo-labeling / self-training on unlabeled test reviews; category labels as auxiliary information (multi-task learning); Precision@N, Recall@N, NDCG@N evaluation.
9. Final selection, reproducibility runner, and submission checks.
10. Case studies, report, and final reproduction documentation.

These are substantive checkpoints, not a reason to create empty or artificial commits. Steps may be adjusted based on measured results. Student ID: 2024320020. Course template, Kaggle competition URL, assigned User ID, and compute resources for neural models still need to be confirmed before their dependent tasks. The student handles Kaggle uploads.

