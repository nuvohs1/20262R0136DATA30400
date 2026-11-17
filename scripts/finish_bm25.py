"""Finish BM25 with two query ablations and generate only the selected file."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from review_retrieval.bm25_query_variants import QueryBM25
from review_retrieval.common import read_json, sha256, validate_submission, write_json, write_submission


def read_predictions(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return [[row[f"item_id_{rank}"] for rank in range(1, 11)] for row in csv.DictReader(stream)]


def metrics_for(queries, predictions):
    ranks = [row.index(query["item_id"]) + 1 if query["item_id"] in row else None
             for query, row in zip(queries, predictions)]
    return {f"recall_at_{k}": sum(rank is not None and rank <= k for rank in ranks) / len(queries)
            for k in (1, 5, 10)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "project_release")
    parser.add_argument("--split", type=Path, default=ROOT / "artifacts/dev_split.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/bm25_tuning")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--student-id", default="2024320020")
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or not args.student_id.isalnum():
        parser.error("Require positive workers and an alphanumeric student ID")
    random.seed(42)
    started = time.perf_counter()
    previous = read_json(args.output_dir / "comparison.json")
    if len(previous["runs"]) != 5 or previous["validation_queries"] != 4000 or "selected" not in previous:
        raise ValueError("Complete the five-method weighting/fusion comparison first")
    if any(sha256(args.data_dir / name) != digest for name, digest in previous["source_sha256"].items()):
        raise ValueError("Source data changed since the preceding comparison")
    if any(sha256(ROOT / name) != digest for name, digest in previous["code_sha256"].items()):
        raise ValueError("Preceding comparison implementation differs from its recorded fingerprints")
    if sha256(args.split) != previous["split_sha256"]:
        raise ValueError("Development split differs from the preceding comparison")
    catalog = read_json(args.data_dir / "catalog.json")
    train = read_json(args.data_dir / "train_gold.json")
    split = read_json(args.split)
    by_id = {query["query_id"]: query for query in train}
    valid_ids = split["validation_query_ids"]
    validation = [by_id[query_id] for query_id in valid_ids]
    if len(catalog) != 50_000 or len(validation) != 4000:
        raise ValueError("Expected the complete fixed validation and full catalog")
    catalog_ids = [product["item_id"] for product in catalog]
    code_paths = [Path(__file__).resolve(), ROOT / "src/review_retrieval/bm25_query_variants.py",
                  ROOT / "src/review_retrieval/bm25_variants.py", ROOT / "src/review_retrieval/common.py"]
    code = {str(path.relative_to(ROOT)).replace(chr(92), "/"): sha256(path) for path in code_paths}
    runs, rankings = [], {}
    for old in previous["runs"]:
        path = args.output_dir / old["name"] / "validation_predictions.csv"
        check = validate_submission(path, catalog_ids, valid_ids)
        if check["sha256"] != old["validation_csv"]["sha256"]:
            raise ValueError("Prior predictions differ from the recorded checksum")
        predictions = read_predictions(path)
        if metrics_for(validation, predictions) != old["metrics"]:
            raise ValueError("Prior metrics do not match saved predictions")
        runs.append(dict(old, implementation_stage="field_weighting_and_fusion"))
        rankings[old["name"]] = predictions
    baseline_record = read_json(ROOT / "artifacts/experiments/bm25_baseline.json")
    runs[0]["kaggle_recall_at_10"] = baseline_record["kaggle_recall_at_10"]
    model = QueryBM25(catalog, field_weights=(1, 1, 1, 1))
    index_path = args.output_dir / "catalog_query.sqlite3"
    model.save_index(index_path)
    metadata = dict(source_sha256=previous["source_sha256"], split_sha256=previous["split_sha256"],
                    preceding_stage_code_sha256=previous["code_sha256"], additional_stage_code_sha256=code,
                    preceding_stage_environment=previous["environment"],
                    additional_stage_environment=dict(python=platform.python_version(), sqlite=sqlite3.sqlite_version,
                                                      workers=args.workers, platform=platform.platform()),
                    catalog_products=50_000, validation_queries=4000, pipeline_external_llm_api_calls=0,
                    selection_criterion=previous["selection_criterion"], runs=runs)

    def progress(name):
        def report(done, total):
            if done == total or done % 640 == 0:
                print(f"{name}: {done:,}/{total:,}", flush=True)
        return report

    for name, term_limit, weighting in (("query_128", 128, "binary"), ("query_log_tf", 64, "log_count")):
        model.configure(field_weights=(1, 1, 1, 1), max_query_terms=term_limit, query_tf=weighting)
        configuration = dict(model.configuration)
        signature = hashlib.sha256(json.dumps(dict(configuration=configuration, code=code,
                                                    sources=metadata["source_sha256"],
                                                    split=metadata["split_sha256"]), sort_keys=True).encode()).hexdigest()
        folder = args.output_dir / name
        use_cache = False
        if args.resume and (folder / "metrics.json").exists() and (folder / "validation_predictions.csv").exists():
            cached = read_json(folder / "metrics.json")
            use_cache = cached["signature"] == signature and sha256(folder / "validation_predictions.csv") == cached["validation_csv"]["sha256"]
        if use_cache:
            predictions = read_predictions(folder / "validation_predictions.csv")
            elapsed, zero_matches = cached["retrieval_seconds"], cached["zero_match_queries"]
        else:
            stage = time.perf_counter()
            result, diagnostics = model.search_parallel(validation, index_path, workers=args.workers, progress=progress(name))
            predictions = result["concat"]
            elapsed, zero_matches = time.perf_counter() - stage, diagnostics["concat"]
        check = write_submission(folder / "validation_predictions.csv", valid_ids, predictions, catalog_ids)
        record = dict(name=name, mode="concat", configuration=configuration, metrics=metrics_for(validation, predictions),
                      retrieval_seconds=elapsed, zero_match_queries=zero_matches, validation_csv=check,
                      signature=signature, kaggle_recall_at_10=None, implementation_stage="query_length_and_frequency")
        write_json(folder / "metrics.json", record)
        runs.append(record)
        rankings[name] = predictions
        write_json(args.output_dir / "final_comparison.json", metadata)
        print(name, record["metrics"], flush=True)
    baseline_hits = {q["query_id"] for q, row in zip(validation, rankings["baseline"]) if q["item_id"] in row}
    for record in runs:
        hits = {q["query_id"] for q, row in zip(validation, rankings[record["name"]]) if q["item_id"] in row}
        record.update(hits_at_10=len(hits), gained_vs_baseline=len(hits - baseline_hits), lost_vs_baseline=len(baseline_hits - hits))
    selected = max(runs, key=lambda record: (record["metrics"]["recall_at_10"], record["metrics"]["recall_at_1"]))
    metadata.update(selected=selected["name"], selection_completed_before_test_retrieval=True,
                    baseline_kaggle_recall_at_10=baseline_record["kaggle_recall_at_10"],
                    baseline_kaggle_score_provenance="student-reported; not used for method selection")
    cases = [{"query_id": q["query_id"], "target_item_id": q["item_id"],
              "rank_at_10": row.index(q["item_id"]) + 1 if q["item_id"] in row else None,
              "predicted_item_ids": row} for q, row in zip(validation, rankings[selected["name"]])]
    write_json(args.output_dir / "selected_validation_cases.json", cases)
    write_json(args.output_dir / "final_comparison.json", metadata)
    print("Selected", selected["name"], selected["metrics"], flush=True)
    if args.predict_test:
        test = read_json(args.data_dir / "test_queries.json")
        if len(test) != 19_858 or any("item_id" in query or "category_path" in query for query in test):
            raise ValueError("Expected all 19,858 unlabeled test queries")
        destination = args.output_dir / f"{args.student_id}_final.csv"
        original = ROOT / "outputs/bm25_baseline" / f"{args.student_id}_final.csv"
        if selected["name"] == "baseline" and original.exists() and sha256(original) == baseline_record["test_csv"]["sha256"]:
            shutil.copyfile(original, destination)
            metadata["test_prediction_provenance"] = "reused verified original baseline file; selected method unchanged"
        else:
            model.configure(**selected["configuration"])
            stage = time.perf_counter()
            result, diagnostics = model.search_parallel(test, index_path, modes=(selected["mode"],),
                                                        workers=args.workers, progress=progress("test"))
            write_submission(destination, [query["query_id"] for query in test], result[selected["mode"]], catalog_ids)
            metadata.update(test_retrieval_seconds=time.perf_counter() - stage,
                            test_zero_match_queries=diagnostics[selected["mode"]], test_prediction_provenance="new test retrieval")
        metadata["test_csv"] = validate_submission(destination, catalog_ids, [query["query_id"] for query in test])
    metadata["additional_stage_total_seconds"] = time.perf_counter() - started
    write_json(args.output_dir / "final_comparison.json", metadata)
    model.close()
    print("Saved final BM25 comparison and validated file:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
