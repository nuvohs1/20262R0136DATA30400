"""Improved BM25: BM25F + plural folding + supervised RSJ query weights, entirely local.

The default submission uses the metadata index only. `--review-expansion` produces an
optional variant that adds a thresholded bonus for products whose labelled training
reviews resemble the query; on held-out validation it is reported but not selected.

Hyperparameters below were chosen on a tuning slice of 4,000 queries sampled (seed 42)
from the development-training partition, with RSJ statistics and the review index
built from the remaining 12,000 development-training queries. The 4,000 validation
queries were not used for any choice and are evaluated once here.

Leakage controls:
  * Validation: RSJ weights and the review-expansion index use only the 16,000
    development-training queries. Validation reviews, targets, and categories are
    used only to compute the reported metrics.
  * Test: RSJ weights and the review-expansion index use all 20,000 labelled
    training queries. Test reviews are only ever used as queries.
  * The catalog index contains product metadata only; category labels are unused.
"""

import argparse
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_retrieval.bm25_v2 import Retriever, rank_all
from review_retrieval.common import PREPROCESSING_VERSION, TEXT_FIELDS, read_json, sha256, write_json, write_submission
from review_retrieval.sparse_bm25 import product_field_tokens, review_token_lists

CONFIG = {
    "stem": True,
    "metadata": {"k1": 2.5, "b": 1.0, "field_weights": [5.0, 1.0, 1.0, 0.5]},
    "query": {"mode": "log", "max_document_frequency": 1.0, "max_terms": 128,
              "term_weight_power": 0.75, "select_by": "weight"},
    "rsj_prior_strength": 10.0,
    "review_expansion": {"enabled": False, "k1": 1.2, "b": 0.75,
                         "query": {"mode": "log", "max_terms": 128, "idf_power": 1.0},
                         "weight": 0.3, "threshold": 6.0},
    "seed": 42,
}


def recall(rankings, targets):
    hits = [[target in ranked[:k] for ranked, target in zip(rankings, targets)] for k in (1, 5, 10)]
    return {f"recall_at_{k}": float(np.mean(h)) for k, h in zip((1, 5, 10), hits)}


def without_expansion(config):
    return {**config, "review_expansion": {**config["review_expansion"], "enabled": False}}


def with_expansion(config):
    return {**config, "review_expansion": {**config["review_expansion"], "enabled": True}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "project_release")
    parser.add_argument("--split", type=Path, default=ROOT / "artifacts/dev_split.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/bm25_v2")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--validation-limit", type=int, help="Smoke check only")
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--student-id", help="Required with --predict-test; used only for the CSV filename")
    parser.add_argument("--review-expansion", action="store_true",
                        help="Optional variant: add the thresholded labelled-review bonus to test predictions "
                             "(writes to outputs/bm25_v2_expansion by default)")
    args = parser.parse_args()
    if args.predict_test and (not args.student_id or not args.student_id.isalnum()):
        parser.error("--predict-test requires an alphanumeric --student-id")
    if args.workers < 1:
        parser.error("--workers must be positive")
    random.seed(42)
    np.seterr(all="ignore")
    config = with_expansion(CONFIG) if args.review_expansion else CONFIG
    if args.review_expansion and args.output_dir == ROOT / "outputs/bm25_v2":
        args.output_dir = ROOT / "outputs/bm25_v2_expansion"
    start = time.perf_counter()

    catalog = sorted(read_json(args.data_dir / "catalog.json"), key=lambda p: p["item_id"])
    train = read_json(args.data_dir / "train_gold.json")
    split = read_json(args.split)
    if len(catalog) != 50_000:
        raise ValueError("Evaluation requires the full 50,000-product catalog")
    if sha256(args.data_dir / "train_gold.json") != split["source_sha256"]:
        raise ValueError("Training data does not match the fixed split fingerprint")
    by_id = {q["query_id"]: q for q in train}
    fit_ids, valid_ids = split["train_query_ids"], split["validation_query_ids"]
    if set(fit_ids) & set(valid_ids) or set(fit_ids) | set(valid_ids) != set(by_id):
        raise ValueError("Split must partition the training queries")
    fit = [by_id[q] for q in fit_ids]
    validation = [by_id[q] for q in valid_ids]
    if {q["item_id"] for q in fit} & {q["item_id"] for q in validation}:
        raise ValueError("Product leakage across the development split")
    if args.validation_limit:
        validation = validation[:args.validation_limit]

    stem = CONFIG["stem"]
    print("Tokenizing catalog and reviews...", flush=True)
    item_ids = [p["item_id"] for p in catalog]
    fields = [product_field_tokens(p, stem) for p in catalog]
    fit_labelled = [(q["item_id"], review_token_lists(q, stem)) for q in fit]
    valid_reviews = [review_token_lists(q, stem) for q in validation]

    def progress(done, total):
        if done == total or done % 2048 < 64:
            print(f"  ranked {done:,}/{total:,}", flush=True)

    results, timing = {}, {}
    print("Building the metadata index...", flush=True)
    t = time.perf_counter()
    base = Retriever(with_expansion(CONFIG), item_ids, fields)  # builds the review index for both variants
    timing["index_build"] = time.perf_counter() - t
    position = base.position
    valid_targets = [position[q["item_id"]] for q in validation]

    # 1) Held-out validation, labels from the development-training partition only.
    t = time.perf_counter()
    base.learn(fit_labelled)
    final_name = "metadata_rsj_review_expansion" if args.review_expansion else "metadata_rsj"
    for name, variant in (("metadata_rsj", without_expansion(CONFIG)), ("metadata_rsj_review_expansion", with_expansion(CONFIG))):
        base.config = variant
        print(f"Validation: {name}", flush=True)
        ranked = rank_all(base, valid_reviews, args.workers, progress=progress)
        results[name] = recall(ranked, valid_targets)
        print(f"  {results[name]}", flush=True)
        if name == final_name:
            final_validation = ranked
    base.config = config
    timing["validation"] = time.perf_counter() - t

    # 2) Expansion proxy: half of the validation products receive two of their own
    #    reviews in the review index; every proxy query uses the other three reviews.
    t = time.perf_counter()
    expanded = set(random.Random(7).sample(range(len(validation)), len(validation) // 2))
    proxy_labelled = fit_labelled + [(validation[i]["item_id"], valid_reviews[i][3:]) for i in sorted(expanded)]
    proxy_queries = [reviews[:3] for reviews in valid_reviews]
    proxy = Retriever(with_expansion(CONFIG), item_ids, fields)
    proxy.learn(fit_labelled)  # RSJ weights from development-training pairs only
    rsj = proxy.term_weights
    proxy.learn(proxy_labelled)
    proxy.term_weights = rsj
    mask = np.array([i in expanded for i in range(len(validation))])
    for name, variant in (("proxy_without_expansion", without_expansion(CONFIG)), ("proxy_with_expansion", with_expansion(CONFIG))):
        proxy.config = variant
        print(f"Validation proxy: {name}", flush=True)
        ranked = rank_all(proxy, proxy_queries, args.workers, progress=progress)
        hits = np.array([target in r for r, target in zip(ranked, valid_targets)])
        results[name] = {"recall_at_10_all": float(hits.mean()),
                         "recall_at_10_expanded_half": float(hits[mask].mean()),
                         "recall_at_10_unexpanded_half": float(hits[~mask].mean())}
        print(f"  {results[name]}", flush=True)
    del proxy
    timing["expansion_proxy"] = time.perf_counter() - t

    cases = [{"query_id": q["query_id"], "target_item_id": q["item_id"],
              "rank_at_10": (r.index(position[q["item_id"]]) + 1) if position[q["item_id"]] in r else None,
              "predicted_item_ids": [item_ids[i] for i in r]} for q, r in zip(validation, final_validation)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_submission(args.output_dir / "validation_predictions.csv", [q["query_id"] for q in validation],
                     [[item_ids[i] for i in r] for r in final_validation], item_ids)
    write_json(args.output_dir / "validation_cases.json", cases)

    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    metrics = {
        "method": "BM25F + supervised RSJ query weights + thresholded review expansion",
        "evaluation": "local development validation; not a Kaggle score",
        "configuration": config, "selected_validation_result": final_name, "preprocessing": PREPROCESSING_VERSION, "fields": list(TEXT_FIELDS),
        "validation_queries": len(validation), "catalog_products": len(catalog),
        "leakage_controls": {
            "validation_labels_used_for": "metrics only",
            "rsj_and_expansion_source_for_validation": "16,000 development-training queries",
            "rsj_and_expansion_source_for_test": "all 20,000 training queries",
            "hyperparameter_selection": "4,000-query tuning slice of development-training (seed 42); validation untouched",
        },
        "results": results,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
                        "git_head_at_run": git.stdout.strip() if git.returncode == 0 else None},
        "source_sha256": {name: sha256(args.data_dir / name) for name in ("catalog.json", "train_gold.json")},
        "split_sha256": sha256(args.split),
        "kaggle_recall_at_10": None,
    }

    # 3) Test prediction with all labelled training queries.
    if args.predict_test:
        test = read_json(args.data_dir / "test_queries.json")
        if len(test) != 19_858 or any("item_id" in q or "category_path" in q for q in test):
            raise ValueError("Expected 19,858 unlabeled test queries")
        t = time.perf_counter()
        base.learn([(q["item_id"], review_token_lists(q, stem)) for q in train])
        print("Ranking test queries...", flush=True)
        ranked = rank_all(base, [review_token_lists(q, stem) for q in test], args.workers, progress=progress)
        submission = args.output_dir / f"{args.student_id}_final.csv"
        metrics["test_csv"] = write_submission(submission, [q["query_id"] for q in test],
                                               [[item_ids[i] for i in r] for r in ranked], item_ids)
        metrics["source_sha256"]["test_queries.json"] = sha256(args.data_dir / "test_queries.json")
        timing["test"] = time.perf_counter() - t
        print(f"Wrote {submission}", flush=True)
    timing["total"] = time.perf_counter() - start
    metrics["timing_seconds"] = timing
    write_json(args.output_dir / "metrics.json", metrics)
    print(f"Saved outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
