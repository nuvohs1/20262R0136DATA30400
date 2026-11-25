"""Evaluate the multi-stage ranker and write the Kaggle file (NumPy only).

Stage 1: BM25 v2 + alpha * LTR BERT, top-K candidates (scripts/make_candidates.py).
Stage 2: BERT cross-encoder scores (scripts/train_cross_encoder.py, scripts/score_cross_encoder.py).
Final:   (1 - w) * z(stage 1) + w * z(cross-encoder) within each query's K candidates.

w is selected on the tuning slice (model A scores); validation is evaluated once with that w;
test uses model B scores and the same w.

Self-training run (run_pseudo.sh): --dev-model outputs/pseudo/ce-dev --full-model outputs/pseudo/ce-full
--output-dir outputs/pseudo/submission --pseudo-dev outputs/pseudo/pseudo_dev_summary.json
"""

import argparse
import csv
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from review_retrieval.common import read_json, sha256, write_json, write_submission
from review_retrieval.metrics import mean_metrics
from review_retrieval.rerank import WEIGHTS, rerank_positions, select_weight


def ranked_lists(cands, scores, weight):
    order = rerank_positions(np.asarray(cands["first_stage_scores"]), scores, weight)
    return [[row[p] for p in pos] for row, pos in zip(cands["items"], order)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "project_release")
    parser.add_argument("--candidates-dir", type=Path, default=ROOT / "outputs/rerank")
    parser.add_argument("--dev-model", type=Path, default=ROOT / "outputs/rerank/ce-dev")
    parser.add_argument("--full-model", type=Path, default=ROOT / "outputs/rerank/ce-full")
    parser.add_argument("--student-id", default="2024320020")
    parser.add_argument("--output-dir", type=Path, help="Default: <candidates-dir>/submission")
    parser.add_argument("--method", default="multi-stage: BM25 v2 + LTR BERT candidates -> BERT cross-encoder reranking")
    parser.add_argument("--pseudo-dev", type=Path, help="Pseudo-label summary of the dev stage (self-training run)")
    args = parser.parse_args()
    if not args.student_id.isalnum():
        parser.error("--student-id must be alphanumeric")
    gold = {q["query_id"]: q["item_id"] for q in read_json(args.data_dir / "train_gold.json")}
    load = lambda name: read_json(args.candidates_dir / f"candidates_{name}.json")
    tune, valid = load("tune"), load("validation")
    tune_scores = np.load(args.dev_model / "scores_tune.npy")
    valid_scores = np.load(args.dev_model / "scores_validation.npy")

    # 1) Choose w on the tuning slice.
    tune_gold = [gold[q] for q in tune["query_ids"]]
    weight, curve = select_weight(tune["items"], tune_gold, np.asarray(tune["first_stage_scores"]), tune_scores)
    print(f"Tuning slice R@10 by weight: {curve}\nSelected w = {weight}", flush=True)

    # 2) Validation, once.
    valid_gold = [gold[q] for q in valid["query_ids"]]
    results = {}
    for name, w in (("first_stage", 0.0), ("cross_encoder_only", 1.0), (f"fused_w{weight}", weight)):
        results[name] = {k: round(v, 5) for k, v in mean_metrics(valid_gold, ranked_lists(valid, valid_scores, w)).items()}
        print(f"  validation {name}: R@1 {results[name]['recall@1']:.4f} R@10 {results[name]['recall@10']:.4f} "
              f"NDCG@10 {results[name]['ndcg@10']:.4f}", flush=True)
    catalog_ids = sorted(p["item_id"] for p in read_json(args.data_dir / "catalog.json"))
    out_dir = args.output_dir or args.candidates_dir / "submission"
    write_submission(out_dir / "validation_reranked.csv", valid["query_ids"], ranked_lists(valid, valid_scores, weight), catalog_ids)

    # 3) Test.
    test = load("test")
    test_scores = np.load(args.full_model / "scores_test.npy")
    path = out_dir / f"{args.student_id}_final.csv"
    check = write_submission(path, test["query_ids"], ranked_lists(test, test_scores, weight), catalog_ids)
    first_stage_csv = ROOT / f"outputs/dense_eval/bert-ltr/hybrid_bm25_expansion_dense/{args.student_id}_final.csv"
    consistent = None
    if first_stage_csv.exists():
        with open(first_stage_csv, encoding="utf-8", newline="") as stream:
            previous = {row[0]: row[1:] for row in list(csv.reader(stream))[1:]}
        consistent = all(previous[q] == row[:10] for q, row in zip(test["query_ids"], test["items"]))
        print(f"Test candidates' top 10 equal the hybrid submission: {consistent}", flush=True)
    print(f"Wrote {path}", flush=True)
    pseudo_report = None
    if args.pseudo_dev:
        # Reported after the final evaluation only; no choice depends on it.
        labels = read_json(args.pseudo_dev.with_name("pseudo_dev.json"))["queries"]
        valid_set = set(valid["query_ids"])
        hits = [p["item_id"] == gold[p["query_id"]] for p in labels if p["query_id"] in valid_set]
        pseudo_report = {**read_json(args.pseudo_dev), "validation_pseudo_label_precision":
                         round(float(np.mean(hits)), 5) if hits else None, "validation_pseudo_labels": len(hits)}
        print(f"Validation pseudo-label precision: {pseudo_report['validation_pseudo_label_precision']} "
              f"({len(hits)} queries)", flush=True)
    write_json(out_dir / "metrics.json", {
        "method": args.method, "pseudo_labels": pseudo_report,
        "evaluation": "local development validation (4,000 queries, full catalog); not a Kaggle score",
        "candidate_recall": {"tune": tune["candidate_recall"], "validation": valid["candidate_recall"]},
        "first_stage": {"tune": tune["first_stage"], "validation": valid["first_stage"], "test": test["first_stage"]},
        "weight_grid": list(WEIGHTS), "tuning_slice_recall_at_10_by_weight": curve, "selected_weight": weight,
        "validation": results, "test_csv": check, "test_top10_candidates_match_hybrid_csv": consistent,
        "dev_model_log": read_json(args.dev_model / "train_log.json"),
        "full_model_log": read_json(args.full_model / "train_log.json"),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "kaggle_recall_at_10": None,
    })


if __name__ == "__main__":
    main()
