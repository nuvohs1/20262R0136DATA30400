# Review-to-Product Retrieval

DATA304 final project (student ID 2024320020).

Given 5 reviews of a product, find that product among the 50,000 products in the catalog and return a top-10 list. The metric is Recall@10.

| | |
| --- | --- |
| Final file | `outputs/final/2024320020_final.csv` |
| Kaggle public Recall@10 | **0.77188** |
| Trained models | [models_2024320020.zip on Google Drive](https://drive.google.com/file/d/1BM16spl36PTRRuj8Tl8XxWP2iO6fwucc/view?usp=sharing) (1.3 GB) |
| Only pretrained model used | `google-bert/bert-base-uncased` (revision `86b5e09`) |

## How to run

The command below reproduces the submitted file exactly from the trained models. We tested it from a fresh clone on an AWS SageMaker notebook with an NVIDIA L4 GPU. The output was byte-identical to the submitted file.

**1. Requirements**

- Linux or macOS with `bash`. On Windows, use WSL.
- An NVIDIA GPU with CUDA.
- Python 3.10 or later.
- About 5 GB of free disk space.

```bash
pip install -r requirements.txt     # numpy, torch, transformers, gdown
```

**2. Course data.** Copy these four files into `project_release/`. They are not in the repository.

```text
project_release/catalog.json
project_release/train_gold.json
project_release/test_queries.json
project_release/category_taxonomy.txt
```

**3. Run**

```bash
bash run_all.sh quick
```

This takes about 2.5 hours on one L4, and it runs these steps in order:

1. Checks the data and runs the unit tests.
2. Downloads the model zip from the Google Drive link above and checks the SHA-256 of every file.
3. Encodes the catalog and the test reviews with the fine-tuned BERT.
4. Builds 50 candidates per query (BM25 + BERT).
5. Scores the candidates with the two cross-encoders.
6. Writes `outputs/final/2024320020_final.csv` and checks its format.

At the end the script prints `Matches the submitted file (Kaggle 0.77188).` when the result has the same SHA-256 as the submitted file.

You can also download the model zip yourself. Put `models_2024320020.zip` in the project root before running, and the script will use it instead of downloading. If `python` is not the right interpreter, set it: `PYTHON_BIN=python3 bash run_all.sh quick`.

**Retraining from scratch (optional).** `bash run_all.sh full` retrains every model from `bert-base-uncased` and selects every setting again. It takes about 12 hours on one L4. GPU training is not bit-exact across hardware, so this mode reproduces the method and the scores closely but not byte for byte. Details: [docs/reproduction.md](docs/reproduction.md).

## Method

```text
5 reviews ──► Stage 1: candidate retrieval (top 50 of 50,000)
               BM25 v2  +  3 × fine-tuned BERT bi-encoder
                 │
                 ▼
              Stage 2: reranking
               cross-encoder A (BM25 negatives)  ┐ averaged
               cross-encoder B (hard negatives)  ┘
                 │
                 ▼
              final score = 0.5 × stage 1 + 0.5 × stage 2  ──► top 10
```

All scores are z-normalised within each query's 50 candidates before they are combined.

- **BM25 v2.** BM25F over title, store, features and description. It folds plurals and learns query-term weights from the training pairs (RSJ weights). A second BM25 index is built over the labelled training reviews. When a query matches a product's training reviews much better than usual, that product gets a bonus (labelled-review expansion).
- **Bi-encoder.** BERT fine-tuned with a listwise softmax loss on the 20,000 training queries, using BM25 hard negatives.
- **Cross-encoders.** Each one reads `[CLS] reviews [SEP] product [SEP]` (max 512 tokens) and is trained with a listwise loss. Cross-encoder A uses negatives from BM25; cross-encoder B uses negatives from the stage-1 hybrid.

**How settings were chosen.** We held out 4,000 of the 20,000 training queries as validation (seed 42). Every setting (α = 3, the epochs, w = 0.5) was chosen on training data only. Each model was trained twice:

- model A on the development queries, where its epoch was chosen;
- model B on all 20,000 queries, for the same number of epochs.

Model B is the one used for test.

## Results

Recall@10 on the 4,000 validation queries (full catalog) and on the Kaggle public leaderboard:

| Step | Validation | Kaggle |
| --- | --- | --- |
| BM25 baseline (SQLite FTS5) | 0.3625 | 0.38120 |
| BM25 v2 | 0.5783 | 0.57538 |
| BM25 v2 + labelled-review expansion | –¹ | 0.60761 |
| BERT zero-shot (dense) | 0.0170 | 0.02306 |
| BERT fine-tuned with learning to rank (dense) | 0.4568 | 0.56058 |
| Hybrid: BM25 v2 + fine-tuned BERT | 0.6673 | 0.72333 |
| + cross-encoder A reranking | 0.7073 | 0.76916 |
| **+ average of cross-encoders A and B (final)** | –² | **0.77188** |

¹ Validation products never have labelled reviews, so the expansion cannot be measured on validation.
² Chosen on a test-like proxy validation instead (0.6043 vs 0.5995 for cross-encoder A alone); see below.

Precision, NDCG and MRR at 1/5/10 for every step are in [docs/evaluation_metrics.md](docs/evaluation_metrics.md).

**Why a proxy validation.** About half of the test products have labelled training reviews, but no validation product does. Because of this gap, cross-encoder B looked better on validation (0.7145) yet scored lower on Kaggle (0.75929). The proxy fixes the gap. Half of the validation products get 2 of their own reviews as "labelled" reviews, and the queries keep the other 3. On the proxy the models rank in the same order as on Kaggle. Details: [docs/memory_proxy.md](docs/memory_proxy.md).

### Tried but not used

| Idea | Outcome | Why not used |
| --- | --- | --- |
| BM25 field weights, review-level fusion | below BM25 v2 | [docs/bm25_tuning.md](docs/bm25_tuning.md) |
| Category classifier as an extra feature | selected weight 0, no change | [docs/category_auxiliary.md](docs/category_auxiliary.md) |
| Approximate search (IVF) | 6× faster, Recall@10 0.4567 → 0.4400 | kept exact search; [docs/ann_ivf.md](docs/ann_ivf.md) |
| Self-training with pseudo-labels | Kaggle 0.76936, no validation gain | trains on test reviews, so excluded; [docs/pseudo_labeling.md](docs/pseudo_labeling.md) |
| Cross-encoder B alone | Kaggle 0.75929 | worse than A alone; [docs/hard_negatives.md](docs/hard_negatives.md) |
| Dense "labelled-review memory" bonus | Kaggle 0.76865 / 0.76795 | helped only on the tuning slice; [docs/memory_proxy.md](docs/memory_proxy.md) |
| Third cross-encoder (review augmentation) | Kaggle 0.77127 (3-model average) | below the final file; [docs/augmented_cross_encoder.md](docs/augmented_cross_encoder.md) |
| Learned fusion weights | lost to the plain average in 5-fold CV | [docs/learned_fusion.md](docs/learned_fusion.md) |

## Rules followed

- **Data.** Only the course files are used: no external data, no web crawling and no LLM APIs. Review text never leaves the machine.
- **Models.** Only `bert-base-uncased` at a pinned revision. `src/review_retrieval/model_policy.py` refuses any other checkpoint ([docs/model_eligibility.md](docs/model_eligibility.md)).
- **Test data.** Test queries are used only for the final prediction. Nothing is trained, tuned or estimated on them.
- **Reproducibility.** Every random step uses seed 42, and the model zip is verified against `artifacts/model_manifest.json`.
- **Repository contents.** Data, outputs and checkpoints are in `.gitignore`.

## Repository layout

```text
run_all.sh                 one-command reproduction (quick | full)
run_*.sh                   individual stages (BM25, GPU training, reranking, ...)
scripts/                   entry points: training, scoring, evaluation, make_final.py
src/review_retrieval/      shared code: BM25, metrics, reranking, model policy
tests/                     unit tests (python -m unittest discover -s tests)
artifacts/                 data audit, dev split, selected settings and metrics (small JSON)
docs/                      one page per experiment, plus reproduction and final selection
notebooks/                 AWS SageMaker / Colab notebooks used to run the GPU steps
```

Two other documents may be useful: [docs/final_selection.md](docs/final_selection.md) explains how the final file was chosen, and [docs/project_plan.md](docs/project_plan.md) lists the project requirements and their status.
