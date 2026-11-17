"""Verify optional NumPy scoring against full SQLite validation before use."""

import argparse
import csv
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from review_retrieval.bm25_numpy import NumpyBM25
from review_retrieval.bm25_query_variants import QueryBM25
from review_retrieval.common import read_json, sha256, validate_submission, write_json, write_submission


def read_predictions(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return [[row[f"item_id_{rank}"] for rank in range(1, 11)] for row in csv.DictReader(stream)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "project_release")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/bm25_tuning")
    parser.add_argument("--workers", type=int, default=16, help="SQLite fallback workers for close scores")
    parser.add_argument("--student-id", default="2024320020")
    parser.add_argument("--predict-test", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or not args.student_id.isalnum():
        parser.error("Require positive workers and an alphanumeric student ID")
    start = time.perf_counter()
    preceding = read_json(args.run_dir / "comparison.json")
    if any(sha256(args.data_dir / name) != value for name, value in preceding["source_sha256"].items()):
        raise ValueError("Data fingerprints differ from SQLite experiments")
    split_path = ROOT / "artifacts/dev_split.json"
    if sha256(split_path) != preceding["split_sha256"]:
        raise ValueError("Development split differs from SQLite experiments")
    split = read_json(split_path)
    train = {query["query_id"]: query for query in read_json(args.data_dir / "train_gold.json")}
    validation = [train[query_id] for query_id in split["validation_query_ids"]]
    catalog = read_json(args.data_dir / "catalog.json")
    if len(catalog) != 50_000 or len(validation) != 4000:
        raise ValueError("Use the entire catalog and complete validation split")
    print("Building optional array scorer...", flush=True)
    model = NumpyBM25(catalog, field_weights=(1, 1, 1, 1))
    reference_path = args.run_dir / "catalog_query.sqlite3"
    reference = QueryBM25.open_index(reference_path, field_weights=(1, 1, 1, 1))
    output = args.run_dir / "fast"
    output.mkdir(parents=True, exist_ok=True)
    records = []

    def retrieve(queries, configuration, label):
        model.configure(**configuration)
        reference.configure(**configuration)
        predictions, ambiguous = [], []
        stage = time.perf_counter()
        for index, query in enumerate(queries):
            result, _ = model.search_one(query)
            predictions.append(result["concat"])
            if model.last_near_tie:
                ambiguous.append(index)
            if (index + 1) % 640 == 0 or index + 1 == len(queries):
                print(f"{label} arrays: {index + 1:,}/{len(queries):,}", flush=True)
        # Near-tied scores use the SQLite reference, avoiding platform-specific
        # floating-point tie order. Review strings stay in local memory.
        if ambiguous:
            print(f"{label}: checking {len(ambiguous):,} close-score queries with SQLite", flush=True)
            corrected, _ = reference.search_parallel([queries[i] for i in ambiguous], reference_path, workers=args.workers)
            for index, row in zip(ambiguous, corrected["concat"]):
                predictions[index] = row
        return predictions, dict(seconds=time.perf_counter() - stage, sqlite_near_tie_fallback_queries=len(ambiguous))

    for name, term_limit, weighting in (("baseline", 64, "binary"), ("query_128", 128, "binary"),
                                        ("query_log_tf", 64, "log_count")):
        configuration = dict(field_weights=[1, 1, 1, 1], max_query_terms=term_limit, query_tf=weighting)
        predictions, timing = retrieve(validation, configuration, name)
        csv_path = output / f"{name}_validation.csv"
        check = write_submission(csv_path, split["validation_query_ids"], predictions, model.item_ids)
        reference_csv = args.run_dir / name / "validation_predictions.csv"
        matches = check["sha256"] == sha256(reference_csv) if reference_csv.exists() else None
        record = dict(name=name, configuration=configuration, validation_csv=check,
                      full_validation_equal_to_sqlite=matches, **timing)
        records.append(record)
        write_json(output / "verification.json", dict(numpy=np.__version__, checks=records))
        print(name, "full validation matches SQLite:", matches, flush=True)
        if matches is False:
            raise ValueError("Array predictions differ from SQLite; use the reference pipeline")
    summary_path = args.run_dir / "final_comparison.json"
    if args.predict_test:
        summary = read_json(summary_path)
        selected = next(record for record in summary["runs"] if record["name"] == summary["selected"])
        if selected["name"] not in {record["name"] for record in records}:
            raise ValueError("Selected method is not supported by this acceleration check")
        verified = next(record for record in records if record["name"] == selected["name"])
        if verified["validation_csv"]["sha256"] != selected["validation_csv"]["sha256"]:
            raise ValueError("Full validation must match the selected SQLite method before test retrieval")
        test = read_json(args.data_dir / "test_queries.json")
        if len(test) != 19_858 or any("item_id" in query or "category_path" in query for query in test):
            raise ValueError("Expected the complete unlabeled test set")
        predictions, timing = retrieve(test, selected["configuration"], "test")
        path = output / f"{args.student_id}_final.csv"
        check = write_submission(path, [query["query_id"] for query in test], predictions, model.item_ids)
        write_json(output / "prediction.json", dict(selected=selected["name"], configuration=selected["configuration"],
                                                     numpy=np.__version__, validation_csv=verified["validation_csv"],
                                                     test_csv=check, timing=timing, total_seconds=time.perf_counter() - start,
                                                     source_sha256=summary["source_sha256"], split_sha256=summary["split_sha256"],
                                                     code_sha256={str(p.relative_to(ROOT)).replace(chr(92), "/"): sha256(p)
                                                                  for p in [Path(__file__).resolve(), ROOT / "src/review_retrieval/bm25_numpy.py"]}))
        print("Saved validated accelerated submission:", path, flush=True)
    model.close()
    reference.close()


if __name__ == "__main__":
    main()
