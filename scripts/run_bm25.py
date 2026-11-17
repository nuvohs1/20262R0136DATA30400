"""Validate BM25 and optionally produce test predictions, entirely locally."""

import argparse
from collections import defaultdict
import platform
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_retrieval.bm25 import BM25
from review_retrieval.common import PREPROCESSING_VERSION, TEXT_FIELDS, read_json, sha256, write_json, write_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "project_release")
    parser.add_argument("--split", type=Path, default=ROOT / "artifacts/dev_split.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/bm25_baseline")
    parser.add_argument("--validation-limit", type=int, help="Smoke check only; omit for the full 4,000-query evaluation")
    parser.add_argument("--max-query-terms", type=int, default=64, help="0 disables term-count pruning")
    parser.add_argument("--max-document-frequency", type=float, default=0.2)
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--student-id", help="Required with --predict-test; used only for the CSV filename")
    args = parser.parse_args()
    if args.validation_limit is not None and args.validation_limit < 1:
        parser.error("--validation-limit must be positive")
    if args.predict_test and (not args.student_id or not args.student_id.isalnum()):
        parser.error("--predict-test requires an alphanumeric --student-id")
    random.seed(42)
    start = time.perf_counter()
    catalog = read_json(args.data_dir / "catalog.json")
    train = read_json(args.data_dir / "train_gold.json")
    split = read_json(args.split)
    if sha256(args.data_dir / "train_gold.json") != split["source_sha256"]:
        raise ValueError("Training data does not match the fixed split fingerprint")
    train_by_id = {query["query_id"]: query for query in train}
    fit_ids, valid_ids = split["train_query_ids"], split["validation_query_ids"]
    if (len(train_by_id) != len(train) or len(set(fit_ids)) != len(fit_ids)
            or len(set(valid_ids)) != len(valid_ids) or set(fit_ids) & set(valid_ids)
            or set(fit_ids) | set(valid_ids) != set(train_by_id) or not valid_ids):
        raise ValueError("Split must partition training queries without duplicates")
    validation = [train_by_id[query_id] for query_id in valid_ids]
    if {train_by_id[q]["item_id"] for q in fit_ids} & {q["item_id"] for q in validation}:
        raise ValueError("Product leakage across the development split")
    if len(catalog) != 50_000:
        raise ValueError("Evaluation requires the full 50,000-product catalog")
    if args.validation_limit is not None:
        validation = validation[:args.validation_limit]
    valid_ids = [q["query_id"] for q in validation]
    catalog_ids = {p["item_id"] for p in catalog}
    if any(q["item_id"] not in catalog_ids for q in validation):
        raise ValueError("Unknown validation target")
    print("Building full-catalog BM25 index...", flush=True)
    build_start = time.perf_counter()
    model = BM25(catalog, args.max_query_terms, args.max_document_frequency)
    build_seconds = time.perf_counter() - build_start

    def progress(done, total):
        if done == total or done % 200 == 0:
            print(f"Retrieved {done:,}/{total:,} queries", flush=True)

    query_start = time.perf_counter()
    predictions, diagnostics = model.search(validation, progress=progress)
    retrieval_seconds = time.perf_counter() - query_start
    validation_check = write_submission(args.output_dir / "validation_predictions.csv",
                                        valid_ids, predictions, model.item_ids)
    ranks, categories, cases = [], defaultdict(list), []
    for query, items in zip(validation, predictions):
        rank = items.index(query["item_id"]) + 1 if query["item_id"] in items else None
        ranks.append(rank)
        categories[query["category_path"]].append(rank)
        cases.append({"query_id": query["query_id"], "target_item_id": query["item_id"], "rank_at_10": rank,
                      "predicted_item_ids": items})
    recall = {f"recall_at_{k}": sum(rank is not None and rank <= k for rank in ranks) / len(ranks)
              for k in (1, 5, 10)}
    git_revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    metrics = {
        "method": "BM25", "evaluation": "local development validation; not a Kaggle score",
        "configuration": {"k1": 1.2, "b": 0.75, "query_tf": "binary", "seed": 42,
                          "backend": "SQLite FTS5; in-memory index",
                          "preprocessing": PREPROCESSING_VERSION, "fields": list(TEXT_FIELDS),
                          "field_weights": "all 1", "five_reviews": "concatenated",
                          "max_query_terms": args.max_query_terms,
                          "max_document_frequency": args.max_document_frequency,
                          "query_term_selection": "descending log(1+(N-df+0.5)/(df+0.5)) * (1+log(query_tf)); ties by token",
                          "formula": "idf=ln((N-df+0.5)/(df+0.5)); replace idf<=0 with 1e-6; tf*(k1+1)/(tf+k1*(1-b+b*dl/avgdl))"},
        "environment": {"python": platform.python_version(), "sqlite": sqlite3.sqlite_version,
                        "platform": platform.platform(), "processor": platform.processor(),
                        "git_head_at_run": git_revision.stdout.strip() if git_revision.returncode == 0 else None,
                        "working_tree_dirty_at_run": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None},
        "source_sha256": {name: sha256(args.data_dir / name) for name in ("catalog.json", "train_gold.json")},
        "code_sha256": {str(path.relative_to(ROOT)).replace(chr(92), "/"): sha256(path)
                        for path in [Path(__file__).resolve(), *sorted((ROOT / "src/review_retrieval").glob("*.py"))]},
        "split_sha256": sha256(args.split), "catalog_products": len(catalog), "validation_queries": len(validation),
        "vocabulary_size": len(model.document_frequency), "average_document_length": model.average_document_length,
        "empty_catalog_documents": model.empty_documents,
        "metrics": recall, "diagnostics": diagnostics,
        "timing_seconds": {"index_build": build_seconds, "validation_retrieval": retrieval_seconds},
        "validation_csv": validation_check, "kaggle_recall_at_10": None,
    }
    write_json(args.output_dir / "validation_cases.json", cases)
    write_json(args.output_dir / "category_metrics.json", {
        category: {"queries": len(values), "recall_at_10": sum(v is not None for v in values) / len(values)}
        for category, values in sorted(categories.items())})
    if args.predict_test:
        test = read_json(args.data_dir / "test_queries.json")
        if len(test) != 19_858 or any("item_id" in q or "category_path" in q for q in test):
            raise ValueError("Expected 19,858 unlabeled test queries")
        test_start = time.perf_counter()
        test_predictions, test_diagnostics = model.search(test, progress=progress)
        submission = args.output_dir / f"{args.student_id}_final.csv"
        metrics["test_csv"] = write_submission(submission, [q["query_id"] for q in test], test_predictions, model.item_ids)
        metrics["test_diagnostics"] = test_diagnostics
        metrics["source_sha256"]["test_queries.json"] = sha256(args.data_dir / "test_queries.json")
        metrics["timing_seconds"]["test_retrieval"] = time.perf_counter() - test_start
    metrics["timing_seconds"]["total"] = time.perf_counter() - start
    write_json(args.output_dir / "metrics.json", metrics)
    model.close()
    print(recall, flush=True)
    print(f"Saved validated outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
