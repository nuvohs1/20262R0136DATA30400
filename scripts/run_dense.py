"""Evaluate dense retrieval and a BM25 + dense hybrid locally (NumPy only).

Requires embeddings from scripts/encode_dense.py (see notebooks/gpu_colab.ipynb): either
the pretrained bert-base-uncased (zero-shot) or the learning-to-rank fine-tuned checkpoints.
Embedding manifests are checked against review_retrieval.model_policy, so embeddings of a
non-allowed model are refused.

Query representation: each of the five reviews is embedded separately ("query: "
prefix); product scores are the mean (or max) of the five review-product cosine
similarities. The mean equals the dot product with the average review embedding,
so all five reviews contribute equally regardless of their length.

Hybrid: z-score of BM25 v2 + alpha * z-score of the dense mean score.

Leakage controls (same protocol as BM25 v2):
  * Zero-shot encoder: no labels touch it. Fine-tuned encoder: validation embeddings must
    come from model A (trained on the 12,000 development-training queries outside the
    tuning slice); test embeddings may come from model B (all 20,000 training queries).
  * alpha is chosen on the 4,000-query tuning slice of development-training (seed 42),
    with BM25 RSJ weights learned from the other 12,000 development-training queries.
  * Validation (4,000 queries) is evaluated once with all choices fixed; its RSJ
    weights come from the 16,000 development-training queries only.
  * Test predictions use RSJ weights (and the BM25 review index) from all 20,000
    training queries. Test reviews are only used as queries.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from review_retrieval.bm25_v2 import Retriever, zscore
from review_retrieval.common import read_json, sha256, write_json, write_submission
from review_retrieval.model_policy import check_embedding_manifest
from review_retrieval.sparse_bm25 import product_field_tokens, review_token_lists, top_k
from run_bm25_v2 import CONFIG as BM25_CONFIG, with_expansion, without_expansion

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0)

_W = {}


def _init(retriever, catalog_embeddings):
    _W["retriever"] = retriever
    _W["catalog"] = catalog_embeddings.astype(np.float32)


def _rank_batch(batch):
    """batch: list of (review_tokens, review_embeddings[5 x d]); returns rankings per method."""
    out = []
    retriever, catalog = _W["retriever"], _W["catalog"]
    for reviews, embeddings in batch:
        per_review = catalog @ embeddings.astype(np.float32).T  # products x 5
        dense_mean = per_review.mean(axis=1)
        result = {"dense_mean": top_k(dense_mean, 10).tolist(),
                  "dense_max": top_k(per_review.max(axis=1), 10).tolist()}
        if retriever is not None:
            bm25 = retriever.scores(reviews)
            dense_z = zscore(dense_mean)
            for alpha in ALPHAS:
                result[f"hybrid_{alpha}"] = top_k(bm25 + alpha * dense_z, 10).tolist()
        out.append(result)
    return out


def rank(retriever, catalog_embeddings, reviews, embeddings, workers):
    items = list(zip(reviews, embeddings))
    batches = [items[i:i + 64] for i in range(0, len(items), 64)]
    results = []
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(retriever, catalog_embeddings)) as ex:
        for part in ex.map(_rank_batch, batches):
            results.extend(part)
            if len(results) % 1024 < 64 or len(results) == len(items):
                print(f"  ranked {len(results):,}/{len(items):,}", flush=True)
    return results


def recall(results, targets, method):
    ranked = [r[method] for r in results]
    return {f"recall_at_{k}": float(np.mean([t in r[:k] for r, t in zip(ranked, targets)])) for k in (1, 5, 10)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "project_release")
    parser.add_argument("--split", type=Path, default=ROOT / "artifacts/dev_split.json")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "outputs/dense/bert-base-uncased")
    parser.add_argument("--test-embeddings", type=Path,
                        help="Use a different embedding directory (e.g. the model trained on all training "
                             "queries) for test predictions; validation always uses --embeddings")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/dense_eval/bert-base-uncased")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--alpha", type=float, help="Skip tuning and use this hybrid weight")
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--student-id")
    args = parser.parse_args()
    if args.predict_test and (not args.student_id or not args.student_id.isalnum()):
        parser.error("--predict-test requires an alphanumeric --student-id")
    random.seed(42)
    np.seterr(all="ignore")
    start = time.perf_counter()

    catalog = sorted(read_json(args.data_dir / "catalog.json"), key=lambda p: p["item_id"])
    train = read_json(args.data_dir / "train_gold.json")
    split = read_json(args.split)
    if sha256(args.data_dir / "train_gold.json") != split["source_sha256"]:
        raise ValueError("Training data does not match the fixed split fingerprint")
    manifest = json.loads((args.embeddings / "manifest.json").read_text(encoding="utf-8"))
    for name in ("catalog.json", "train_gold.json"):
        if manifest["source_sha256"][name] != sha256(args.data_dir / name):
            raise ValueError(f"Embeddings were computed from a different {name}")
    eligibility = check_embedding_manifest(manifest)
    if eligibility["kind"] == "fine-tuned checkpoint" and eligibility["provenance"].get("negatives_set") != "dev":
        raise ValueError("Validation embeddings must come from the development model (negatives_dev.json); "
                         "a model trained on all training queries has seen the validation queries")
    catalog_emb = np.load(args.embeddings / "catalog.npy")
    train_emb = np.load(args.embeddings / "train_reviews.npy").reshape(len(train), 5, -1)
    if catalog_emb.shape[0] != len(catalog):
        raise ValueError("Catalog embedding rows do not match the catalog")

    item_ids = [p["item_id"] for p in catalog]
    position = {item: n for n, item in enumerate(item_ids)}
    row = {q["query_id"]: n for n, q in enumerate(train)}
    by_id = {q["query_id"]: q for q in train}
    fit_ids, valid_ids = split["train_query_ids"], split["validation_query_ids"]
    tune_ids = sorted(random.Random(42).sample(sorted(fit_ids), 4000))
    rest_ids = sorted(set(fit_ids) - set(tune_ids))
    if set(fit_ids) & set(valid_ids) or {by_id[q]["item_id"] for q in fit_ids} & {by_id[q]["item_id"] for q in valid_ids}:
        raise ValueError("Development split leakage")

    stem = BM25_CONFIG["stem"]
    print("Tokenizing...", flush=True)
    fields = [product_field_tokens(p, stem) for p in catalog]
    tokens = {q["query_id"]: review_token_lists(q, stem) for q in train}
    labelled = lambda ids: [(by_id[q]["item_id"], tokens[q]) for q in ids]
    targets = lambda ids: [position[by_id[q]["item_id"]] for q in ids]
    embeddings = lambda ids: [train_emb[row[q]] for q in ids]
    results, timing = {}, {}

    retriever = Retriever(with_expansion(BM25_CONFIG), item_ids, fields)
    retriever.config = without_expansion(BM25_CONFIG)

    # 1) Tuning slice: choose alpha.
    if args.alpha is None:
        t = time.perf_counter()
        print("Tuning slice (alpha selection)...", flush=True)
        retriever.learn(labelled(rest_ids))
        tuned = rank(retriever, catalog_emb, [tokens[q] for q in tune_ids], embeddings(tune_ids), args.workers)
        tune_scores = {m: recall(tuned, targets(tune_ids), m) for m in tuned[0]}
        results["tuning_slice"] = tune_scores
        best = max(ALPHAS, key=lambda a: (tune_scores[f"hybrid_{a}"]["recall_at_10"], -a))
        timing["tuning"] = time.perf_counter() - t
        print(json.dumps({m: round(v["recall_at_10"], 4) for m, v in tune_scores.items()}), flush=True)
    else:
        best = args.alpha
    print(f"Selected alpha = {best}", flush=True)

    # 2) Held-out validation, evaluated once.
    t = time.perf_counter()
    print("Validation...", flush=True)
    retriever.learn(labelled(fit_ids))
    validated = rank(retriever, catalog_emb, [tokens[q] for q in valid_ids], embeddings(valid_ids), args.workers)
    valid_targets = targets(valid_ids)
    results["validation"] = {m: recall(validated, valid_targets, m) for m in validated[0]}
    timing["validation"] = time.perf_counter() - t
    for m in ("dense_mean", "dense_max", "hybrid_0.0", f"hybrid_{best}"):
        print(f"  {m}: {results['validation'][m]}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for m in ("dense_mean", f"hybrid_{best}"):
        write_submission(args.output_dir / f"validation_{m}.csv", valid_ids,
                         [[item_ids[i] for i in r[m]] for r in validated], item_ids)

    metrics = {
        "method": f"dense retrieval ({eligibility['kind']} of {eligibility['base_model']}) and BM25 v2 + dense hybrid",
        "evaluation": "local development validation; not a Kaggle score",
        "embeddings_manifest": manifest, "selected_alpha": best, "alpha_grid": list(ALPHAS),
        "bm25_config": BM25_CONFIG, "results": results,
        "leakage_controls": {
            "encoder": ("pretrained, zero-shot; no labels used" if eligibility["kind"] == "pretrained checkpoint"
                        else "fine-tuned on the 12,000 development-training queries outside the tuning slice"),
            "alpha_selection": "4,000-query tuning slice of development-training; BM25 RSJ from the other 12,000",
            "validation": "evaluated once; BM25 RSJ from 16,000 development-training queries",
            "test": "BM25 RSJ and review index from all 20,000 training queries",
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "split_sha256": sha256(args.split), "kaggle_recall_at_10": {},
    }

    # 3) Test predictions: dense only, and hybrid on top of the Kaggle-best BM25 (with review expansion).
    if args.predict_test:
        test = read_json(args.data_dir / "test_queries.json")
        test_dir = args.test_embeddings or args.embeddings
        test_manifest = json.loads((test_dir / "manifest.json").read_text(encoding="utf-8"))
        for name in ("catalog.json", "test_queries.json"):
            if test_manifest["source_sha256"][name] != sha256(args.data_dir / name):
                raise ValueError(f"Test embeddings were computed from a different {name}")
        check_embedding_manifest(test_manifest)
        test_emb = np.load(test_dir / "test_reviews.npy").reshape(len(test), 5, -1)
        test_catalog = np.load(test_dir / "catalog.npy") if args.test_embeddings else catalog_emb
        metrics["test_embeddings_manifest"] = test_manifest
        t = time.perf_counter()
        print("Test predictions...", flush=True)
        retriever.config = with_expansion(BM25_CONFIG)
        retriever.learn([(q["item_id"], tokens[q["query_id"]]) for q in train])
        ranked = rank(retriever, test_catalog, [review_token_lists(q, stem) for q in test], list(test_emb), args.workers)
        query_ids = [q["query_id"] for q in test]
        files = {"dense_only": "dense_mean", "hybrid_bm25_expansion_dense": f"hybrid_{best}"}
        metrics["test_csv"] = {}
        for folder, m in files.items():
            path = args.output_dir / folder / f"{args.student_id}_final.csv"
            metrics["test_csv"][folder] = write_submission(path, query_ids, [[item_ids[i] for i in r[m]] for r in ranked], item_ids)
            print(f"Wrote {path}", flush=True)
        timing["test"] = time.perf_counter() - t
    timing["total"] = time.perf_counter() - start
    metrics["timing_seconds"] = timing
    write_json(args.output_dir / "metrics.json", metrics)
    print(f"Saved outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
