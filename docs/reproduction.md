# Reproducing the final submission

The final file is `outputs/final/2024320020_final.csv` (Kaggle public Recall@10 0.77188). It is made by one command, `bash run_all.sh`, in one of two modes.

## Final pipeline

1. **Stage 1 (candidates).** BM25 v2 (BM25F, plural folding, supervised RSJ query weights) with labelled-review expansion, plus 3 · z(LTR-fine-tuned BERT bi-encoder). The top 50 products per query are kept (`scripts/make_candidates.py`).
2. **Stage 2 (reranking).** Two BERT cross-encoders, both fine-tuned from `google-bert/bert-base-uncased` on `train_gold.json` only:
   - one trained with BM25 negatives (`run_rerank.sh`);
   - one trained with hybrid hard negatives (`run_hardneg.sh`).

   Their scores are z-normalised within each query's 50 candidates and averaged.
3. **Fusion.** The final score is `0.5 · z(stage 1) + 0.5 · z(stage 2)` (`scripts/make_final.py`). w = 0.5 was chosen on the proxy tuning slice (`docs/memory_proxy.md`).

Every model used for test (model B) was trained on all 20,000 training queries and stopped at the epoch chosen for its development twin (model A) on the tuning slice. Model selection is in `docs/final_selection.md`.

## Requirements

- **Software.** Python 3.10+ and `pip install -r requirements.txt` (NumPy, torch, transformers, gdown). Tested with Python 3.10.20, NumPy 2.2.6, torch 2.10.0+cu128 and transformers 5.20.2.
- **Data.** The course files go in `project_release/`: `catalog.json`, `train_gold.json` and `test_queries.json`. `scripts/prepare_data.py` checks their fingerprints.
- **GPU.** One CUDA GPU is needed for encoding and cross-encoder scoring. The reference runs used an NVIDIA L4 on the course AWS SageMaker notebook.

## Quick mode: inference with the trained checkpoints

```bash
bash run_all.sh quick 2024320020 8   # downloads the model pack from the link below (needs gdown)
# or download models_2024320020.zip by hand into the project root first; it is then used as is
```

**Model pack.** `models_2024320020.zip` (about 1.3 GB) contains three model-B checkpoints:

- the LTR bi-encoder;
- the cross-encoder trained with BM25 negatives;
- the cross-encoder trained with hybrid negatives.

Each checkpoint carries a `provenance.json` that records the base model and revision, and confirms that only the course training data was used.

**Verification.** `scripts/fetch_models.py` extracts only the paths listed in the committed `artifacts/model_manifest.json`. It refuses the archive if the archive or any file does not match its SHA-256.

**Steps** (about 2.5 hours on one NVIDIA L4; scoring 992,900 pairs takes about one hour per cross-encoder):

1. Encode the test reviews and the catalog with the LTR model.
2. Build the stage-1 candidates. α comes from `artifacts/experiments/ltr_bert.json`.
3. Score the candidates with both cross-encoders.
4. Fuse the scores with w from `artifacts/experiments/memory_proxy.json`.

`scripts/make_final.py` compares the result with the SHA-256 of the submitted file and reports whether they match.

**Model link:** [models_2024320020.zip](https://drive.google.com/file/d/1BM16spl36PTRRuj8Tl8XxWP2iO6fwucc/view?usp=sharing) (Google Drive, anyone with the link can view). Archive SHA-256: `d5322179e97b4db89cd4def4ee65964f1e952fa07f6358bcb3844776c08c93ab`.

**Checked run.** On 2026-11-27 the quick mode was run in an empty copy of the repository on the course AWS SageMaker notebook (NVIDIA L4, Python 3.10.20, NumPy 2.2.6). It verified all 16 files of the pack and wrote a file with the reference SHA-256 below, byte-identical to the submitted file.

## Full mode: retrain everything

```bash
bash run_all.sh full 2024320020 8
```

**Steps:**

1. Mine BM25 negatives.
2. Train LTR models A and B, and encode the catalog and reviews.
3. Choose α on the tuning slice.
4. Train both cross-encoders (A and B for each).
5. Run the proxy split, which chooses w on the proxy tuning slice.
6. Write the final file.

All seeds are 42, and finished steps are skipped on a re-run. Expect about 11–13 hours on one NVIDIA L4.

GPU training is not bit-for-bit deterministic across hardware and library versions, because cuDNN and cuBLAS kernels can differ. A retrained model can therefore move near-tied candidates. The quick mode exists so the submitted file can be reproduced exactly from the trained weights.

## Checking a reproduced file

```bash
python scripts/validate_submission.py outputs/final/2024320020_final.csv
python scripts/compare_submissions.py outputs/final/2024320020_final.csv <submitted file>
```

The first command checks:

- the header;
- the 19,858 test query IDs in their original order;
- 10 distinct IDs per row;
- that every ID is in the 50,000-product catalog.

The second command counts the rows whose top 10 has the same order, the same set, and the same top 1.

**Reference SHA-256 of the submitted file:** `31a1c7f26ecf0baba9f1e18384b7a5ae2bedacb99feaf81783a0c571bb06f500`.

## What is not in the repository

`outputs/`, model checkpoints, embeddings and the course data are listed in `.gitignore`. The repository holds only source code, run scripts, tests, configuration, and the small JSON records in `artifacts/` (selected settings, metrics and the model manifest).
