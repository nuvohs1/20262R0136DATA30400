# Pretrained model eligibility

## Rule

Public pretrained checkpoints may be used only if their training data is publicly documented well enough to verify that they were not trained on Amazon data for downstream tasks such as QA or retrieval (TA ruling). Amazon text inside generic language-model pretraining (for example MLM) would be acceptable, but each exact checkpoint must be verified from its official documentation. A model whose training data cannot be verified must not be used. A model family name alone does not make a checkpoint eligible.

## Allowed checkpoint

| Checkpoint | Revision | License | Training data | Objectives | Amazon data |
| --- | --- | --- | --- | --- | --- |
| `google-bert/bert-base-uncased` | `86b5e0934494bd15c9632b12f734a8a67f723594` | Apache-2.0 | BookCorpus (11,038 unpublished books), English Wikipedia (excluding lists, tables, headers) | masked language modeling, next sentence prediction | none |

Evidence:

- Official model card, "Training data" section, at the pinned revision: <https://huggingface.co/google-bert/bert-base-uncased/blob/86b5e0934494bd15c9632b12f734a8a67f723594/README.md#training-data>
- Devlin et al., 2019, *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, arXiv:1810.04805, Section 3.1 (pre-training data: BooksCorpus and English Wikipedia).
- The checkpoint is a raw self-supervised model; it has no downstream (QA, retrieval, similarity) training stage. BERT is also named in the course honor code (tutorial slides, p.13).

All further training of this model in this project uses only the course file `train_gold.json` (learning-to-rank fine-tuning in `scripts/train_dense_ltr.py`).

## Enforcement in code

`src/review_retrieval/model_policy.py` holds the allow-list above.

- `scripts/encode_dense.py` and `scripts/train_dense_ltr.py` refuse to load any checkpoint that is not in the list, or a listed checkpoint at a different revision.
- A fine-tuned checkpoint is accepted only with the `provenance.json` written by `scripts/train_dense_ltr.py`. That file records the allowed base model and revision, the statement that fine-tuning used only `train_gold.json`, and that file's SHA-256.
- Every embedding manifest stores this eligibility record. `scripts/run_dense.py` refuses embeddings without a valid record.
- Hard negatives for fine-tuning come from BM25 only (`scripts/mine_negatives.py`), so no other model influences training.

Adding a checkpoint requires the same evidence (exact training datasets and objectives from official documentation) to be recorded here and in `ALLOWED_CHECKPOINTS`.
