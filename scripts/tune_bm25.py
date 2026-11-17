"""Compare field weighting and review fusion, then predict with local best."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_retrieval.bm25_variants import FieldBM25
from review_retrieval.common import read_json, sha256, validate_submission, write_json, write_submission


def score_predictions(queries, predictions):
    ranks = [items.index(query["item_id"]) + 1 if query["item_id"] in items else None
             for query, items in zip(queries, predictions)]
    return {f"recall_at_{k}": sum(rank is not None and rank <= k for rank in ranks) / len(queries)
            for k in (1, 5, 10)}


def load_predictions(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return [[row[f"item_id_{rank}"] for rank in range(1, 11)] for row in csv.DictReader(stream)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "project_release")
    parser.add_argument("--split", type=Path, default=ROOT / "artifacts/dev_split.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/bm25_tuning")
    parser.add_argument("--baseline-run-dir", type=Path, default=ROOT / "outputs/bm25_baseline")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--student-id", default="2024320020")
    parser.add_argument("--resume", action="store_true", help="Reuse only fingerprint-matching validation checkpoints")
    args = parser.parse_args()
    if args.workers < 1 or not args.student_id.isalnum():
        parser.error("Require positive --workers and an alphanumeric student ID")
    random.seed(42)
    started = time.perf_counter()
    sources = {name: sha256(args.data_dir / name) for name in ("catalog.json", "train_gold.json", "test_queries.json")}
    code = {str(path.relative_to(ROOT)).replace(chr(92), "/"): sha256(path) for path in (
        Path(__file__).resolve(), ROOT / "src/review_retrieval/bm25_variants.py",
        ROOT / "src/review_retrieval/common.py")}
    catalog = read_json(args.data_dir / "catalog.json")
    train = read_json(args.data_dir / "train_gold.json")
    split = read_json(args.split)
    if sources["train_gold.json"] != split["source_sha256"]:
        raise ValueError("Training data differs from the fixed split")
    by_id = {query["query_id"]: query for query in train}
    fit_ids, valid_ids = split["train_query_ids"], split["validation_query_ids"]
    if (len(by_id) != len(train) or len(set(fit_ids)) != len(fit_ids)
            or len(set(valid_ids)) != len(valid_ids) or set(fit_ids) & set(valid_ids)
            or set(fit_ids) | set(valid_ids) != set(by_id) or len(valid_ids) != 4000):
        raise ValueError("Expected the complete fixed 16,000/4,000 development split")
    validation = [by_id[query_id] for query_id in valid_ids]
    if {by_id[q]["item_id"] for q in fit_ids} & {q["item_id"] for q in validation}:
        raise ValueError("Product leakage in the development split")
    if len(catalog) != 50_000:
        raise ValueError("All 50,000 catalog products must be indexed")
    catalog_ids = [product["item_id"] for product in catalog]
    catalog_set = set(catalog_ids)
    if any(query["item_id"] not in catalog_set for query in validation):
        raise ValueError("Unknown validation target")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Building separate-field index over all 50,000 products...", flush=True)
    build_start = time.perf_counter()
    model = FieldBM25(catalog)
    index_path = args.output_dir / "catalog_fields.sqlite3"
    model.save_index(index_path)
    build_seconds = time.perf_counter() - build_start
    runs, rankings = [], {}
    metadata = {"source_sha256": sources, "code_sha256": code, "split_sha256": sha256(args.split),
                "seed": 42, "catalog_products": len(catalog), "validation_queries": len(validation),
                "environment": {"python": platform.python_version(), "sqlite": sqlite3.sqlite_version,
                                "platform": platform.platform(), "processor": platform.processor(),
                                "workers": args.workers},
                "index_build_seconds": build_seconds, "pipeline_external_llm_api_calls": 0,
                "selection_criterion": "local Recall@10, then Recall@1, then earlier simpler configuration"}

    def signature(configuration, mode):
        payload = dict(configuration=configuration, mode=mode, sources=sources, code=code, split=metadata["split_sha256"])
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def progress(name):
        def report(done, total):
            if done == total or done % 640 == 0:
                print(f"{name}: {done:,}/{total:,}", flush=True)
        return report

    def save_run(name, predictions, configuration, mode, elapsed, diagnostics, provenance):
        metrics = score_predictions(validation, predictions)
        folder = args.output_dir / name
        check = write_submission(folder / "validation_predictions.csv", valid_ids, predictions, model.item_ids)
        record = dict(name=name, mode=mode, configuration=configuration, metrics=metrics,
                      retrieval_seconds=elapsed, zero_match_queries=diagnostics,
                      provenance=provenance, validation_csv=check,
                      signature=signature(configuration, mode), kaggle_recall_at_10=None)
        write_json(folder / "metrics.json", record)
        runs.append(record)
        rankings[name] = predictions
        write_json(args.output_dir / "comparison.json", dict(metadata, runs=runs))
        print(f"{name}: {metrics}", flush=True)
        return record

    def cached(name, configuration, mode):
        if not args.resume:
            return False
        folder = args.output_dir / name
        if not (folder / "metrics.json").exists() or not (folder / "validation_predictions.csv").exists():
            return False
        previous = read_json(folder / "metrics.json")
        if previous["signature"] != signature(configuration, mode):
            return False
        check = validate_submission(folder / "validation_predictions.csv", catalog_ids, valid_ids)
        if check["sha256"] != previous["validation_csv"]["sha256"]:
            return False
        save_run(name, load_predictions(folder / "validation_predictions.csv"), configuration, mode,
                 previous["retrieval_seconds"], previous["zero_match_queries"], previous["provenance"])
        return True

    configurations = [
        ("baseline", (1, 1, 1, 1)),
        ("field_weighted", (4, 2, 2, 1)),
        ("title_emphasis", (8, 3, 2, 1)),
    ]
    for name, weights in configurations:
        model.configure(field_weights=weights)
        configuration = dict(model.configuration)
        if cached(name, configuration, "concat"):
            continue
        original_csv = args.baseline_run_dir / "validation_predictions.csv"
        baseline_record = read_json(ROOT / "artifacts/experiments/bm25_baseline.json")
        baseline_matches = (name == "baseline" and original_csv.exists()
                            and all(baseline_record["source_sha256"][key] == value for key, value in sources.items())
                            and baseline_record["split_sha256"] == metadata["split_sha256"]
                            and sha256(original_csv) == baseline_record["validation_csv"]["sha256"])
        if baseline_matches:
            validate_submission(original_csv, catalog_ids, valid_ids)
            predictions = load_predictions(original_csv)
            # Full-catalog equivalence check: the uniform field schema must
            # preserve the original concatenated baseline on actual queries.
            checked, _ = model.search_parallel(validation[:20], index_path, workers=args.workers)
            if checked["concat"] != predictions[:20]:
                raise ValueError("Uniform fields do not reproduce the original baseline")
            metadata["uniform_field_first20_match_original_baseline"] = True
            save_run(name, predictions, configuration, "concat", baseline_record["timing_seconds"]["validation_retrieval"],
                     baseline_record["diagnostics"]["zero_match_queries"], "verified original baseline predictions")
        else:
            stage = time.perf_counter()
            predictions, diagnostics = model.search_parallel(validation, index_path, workers=args.workers, progress=progress(name))
            save_run(name, predictions["concat"], configuration, "concat", time.perf_counter() - stage,
                     diagnostics["concat"], "full validation retrieval")

    best_concat = max(runs, key=lambda record: (record["metrics"]["recall_at_10"], record["metrics"]["recall_at_1"]))
    model.configure(**best_concat["configuration"])
    configuration = dict(model.configuration)
    modes = ("review_rrf", "hybrid_rrf")
    reused = [cached(mode, configuration, mode) for mode in modes]
    if not all(reused):
        # Both fusions reuse the same five review lists and concatenated list.
        # Do not rerun a mode already restored from a complete checkpoint.
        stage = time.perf_counter()
        predictions, diagnostics = model.search_parallel(validation, index_path, modes=modes,
                                                          workers=args.workers, progress=progress("review_fusion"))
        elapsed = time.perf_counter() - stage
        for mode, was_reused in zip(modes, reused):
            if not was_reused:
                save_run(mode, predictions[mode], configuration, mode, elapsed, diagnostics[mode],
                         "shared six-view retrieval; wall time shared between the two fusion variants")
    baseline_hits = {q["query_id"] for q, row in zip(validation, rankings["baseline"]) if q["item_id"] in row}
    for record in runs:
        hits = {q["query_id"] for q, row in zip(validation, rankings[record["name"]]) if q["item_id"] in row}
        record["hits_at_10"] = len(hits)
        record["gained_vs_baseline"] = len(hits - baseline_hits)
        record["lost_vs_baseline"] = len(baseline_hits - hits)
    selected = max(runs, key=lambda record: (record["metrics"]["recall_at_10"], record["metrics"]["recall_at_1"]))
    summary = dict(metadata, runs=runs, selected=selected["name"],
                   selection_completed_before_test_retrieval=True, fusion_parent=best_concat["name"],
                   baseline_kaggle_recall_at_10=baseline_record["kaggle_recall_at_10"],
                   baseline_kaggle_provenance="user-reported; not used for selection")
    selected_predictions = rankings[selected["name"]]
    cases = [{"query_id": query["query_id"], "target_item_id": query["item_id"],
              "rank_at_10": row.index(query["item_id"]) + 1 if query["item_id"] in row else None,
              "predicted_item_ids": row} for query, row in zip(validation, selected_predictions)]
    write_json(args.output_dir / "selected_validation_cases.json", cases)
    write_json(args.output_dir / "comparison.json", summary)
    print(f"Selected {selected['name']} using local validation: {selected['metrics']}", flush=True)
    if args.predict_test:
        test = read_json(args.data_dir / "test_queries.json")
        if len(test) != 19_858 or any("item_id" in query or "category_path" in query for query in test):
            raise ValueError("Expected the complete unlabeled test set")
        model.configure(**selected["configuration"])
        stage = time.perf_counter()
        predictions, diagnostics = model.search_parallel(test, index_path, modes=(selected["mode"],),
                                                          workers=args.workers, progress=progress("test"))
        path = args.output_dir / f"{args.student_id}_final.csv"
        summary["test_csv"] = write_submission(path, [query["query_id"] for query in test],
                                                predictions[selected["mode"]], model.item_ids)
        summary["test_zero_match_queries"] = diagnostics[selected["mode"]]
        summary["test_retrieval_seconds"] = time.perf_counter() - stage
    summary["total_seconds"] = time.perf_counter() - started
    write_json(args.output_dir / "comparison.json", summary)
    model.close()
    print("Saved comparison and validated outputs:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
