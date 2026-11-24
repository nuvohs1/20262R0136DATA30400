"""Compute Precision@N, Recall@N, NDCG@N and MRR@N for saved validation rankings (pure Python).

Reads the validation CSVs written by the BM25, dense and learning-to-rank scripts
(4,000 held-out development queries, full-catalog rankings, ten items per query) and the
gold items from train_gold.json. Nothing is re-ranked or tuned here.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from review_retrieval.common import read_json, sha256
from review_retrieval.metrics import mean_metrics

RUNS = {
    "BM25 (original baseline)": "outputs/bm25_baseline/validation_predictions.csv",
    "BM25 v2 (metadata only)": "outputs/bm25_v2/validation_predictions.csv",
    "Dense: zero-shot BERT": "outputs/dense_eval/bert-base-uncased/validation_dense_mean.csv",
    "Dense: LTR fine-tuned BERT": "outputs/dense_eval/bert-ltr/validation_dense_mean.csv",
    "Hybrid: BM25 v2 + LTR BERT (alpha 3)": "outputs/dense_eval/bert-ltr/validation_hybrid_3.0.csv",
}


def load_rankings(path):
    with open(path, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    if rows[0][0] != "query_id":
        raise ValueError(f"{path}: unexpected header")
    return {row[0]: row[1:] for row in rows[1:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "project_release")
    parser.add_argument("--split", type=Path, default=ROOT / "artifacts/dev_split.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/experiments/ranking_metrics.json")
    args = parser.parse_args()
    split = read_json(args.split)
    if sha256(args.data_dir / "train_gold.json") != split["source_sha256"]:
        raise ValueError("Training data does not match the fixed split fingerprint")
    gold = {q["query_id"]: q["item_id"] for q in read_json(args.data_dir / "train_gold.json")}
    valid_ids = split["validation_query_ids"]

    results = {}
    for name, rel in RUNS.items():
        path = ROOT / rel
        if not path.exists():
            print(f"skip (missing): {rel}")
            continue
        ranking = load_rankings(path)
        if set(ranking) != set(valid_ids):
            raise ValueError(f"{rel} does not cover exactly the validation queries")
        results[name] = {"file": rel, **{k: round(v, 5) for k, v in
                         mean_metrics([gold[q] for q in valid_ids], [ranking[q] for q in valid_ids]).items()}}

    cols = ["recall@1", "recall@5", "recall@10", "precision@5", "precision@10", "ndcg@5", "ndcg@10", "mrr@10"]
    print("| Method | " + " | ".join(cols) + " |")
    print("| --- |" + " --- |" * len(cols))
    for name, m in results.items():
        print(f"| {name} | " + " | ".join(f"{m[c]:.4f}" for c in cols) + " |")
    args.output.write_text(json.dumps({
        "evaluation": "local development validation: 4,000 held-out queries, full 50,000-product catalog, one relevant item per query",
        "definitions": "see src/review_retrieval/metrics.py", "split_sha256": sha256(args.split), "results": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
