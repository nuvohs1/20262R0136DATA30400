"""Reranking with category labels as auxiliary information (NumPy only).

    final = (1 - w) * z(stage 1) + w * z(cross-encoder) + beta * z(category)
Stage 1 and the cross-encoder are the step-4b ranker (outputs/rerank); the category score comes
from scripts/train_category.py. (w, beta) are chosen together on the tuning slice; beta = 0 is
the step-4b ranker. Validation is evaluated once; test uses the full-data models and the same
(w, beta).
"""

import argparse
import csv
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from review_retrieval.category import BETAS, fused_positions, select_weights
from review_retrieval.common import read_json, write_json, write_submission
from review_retrieval.metrics import mean_metrics
from review_retrieval.rerank import WEIGHTS


def ranked(cands, cross, cat, w, beta):
    order = fused_positions(np.asarray(cands["first_stage_scores"]), cross, cat, w, beta)
    return [[row[p] for p in pos] for row, pos in zip(cands["items"], order)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "project_release")
    parser.add_argument("--rerank-dir", type=Path, default=ROOT / "outputs/rerank")
    parser.add_argument("--category-dir", type=Path, default=ROOT / "outputs/category")
    parser.add_argument("--student-id", default="2024320020")
    args = parser.parse_args()
    if not args.student_id.isalnum():
        parser.error("--student-id must be alphanumeric")
    R, C = args.rerank_dir, args.category_dir
    gold = {q["query_id"]: q["item_id"] for q in read_json(args.data_dir / "train_gold.json")}
    load = lambda name: read_json(R / f"candidates_{name}.json")
    tune, valid, test = load("tune"), load("validation"), load("test")
    ce = {"tune": np.load(R / "ce-dev/scores_tune.npy"), "validation": np.load(R / "ce-dev/scores_validation.npy"),
          "test": np.load(R / "ce-full/scores_test.npy")}
    cat = {"tune": np.load(C / "dev/cat_scores_tune.npy"), "validation": np.load(C / "dev/cat_scores_validation.npy"),
           "test": np.load(C / "full/cat_scores_test.npy")}
    rerank_w = read_json(R / "submission/metrics.json")["selected_weight"]

    # 1) (w, beta) on the tuning slice.
    tune_gold = [gold[q] for q in tune["query_ids"]]
    (w, beta), curve = select_weights(tune["items"], tune_gold, np.asarray(tune["first_stage_scores"]), ce["tune"], cat["tune"])
    best_by_beta = {b: max(curve[(x, b)] for x in WEIGHTS) for b in BETAS}
    print(f"Tuning slice best R@10 per beta: {best_by_beta}\nSelected w = {w}, beta = {beta}", flush=True)

    # 2) Validation, once.
    valid_gold = [gold[q] for q in valid["query_ids"]]
    runs = {f"rerank_w{rerank_w}": (rerank_w, 0.0), f"stage1_plus_category_beta{beta}": (0.0, beta),
            f"rerank_plus_category_w{w}_beta{beta}": (w, beta)}
    results = {}
    for name, (rw, rb) in runs.items():
        results[name] = {k: round(v, 5) for k, v in mean_metrics(valid_gold, ranked(valid, ce["validation"], cat["validation"], rw, rb)).items()}
        print(f"  validation {name}: R@1 {results[name]['recall@1']:.4f} R@10 {results[name]['recall@10']:.4f} "
              f"NDCG@10 {results[name]['ndcg@10']:.4f}", flush=True)
    # Category accuracy of validation queries (reported only; nothing depends on it).
    leaf_paths = sorted({l.strip() for l in (args.data_dir / "category_taxonomy.txt").read_text(encoding="utf-8").splitlines() if l.strip()})
    class_of = {p: i for i, p in enumerate(leaf_paths)}
    path_of = {q["query_id"]: q["category_path"] for q in read_json(args.data_dir / "train_gold.json")}
    probs = np.load(C / "dev/query_probs_validation.npy").astype(np.float32)
    truth = np.array([class_of[path_of[q]] for q in valid["query_ids"]])
    top5 = np.argsort(-probs, axis=1)[:, :5]
    accuracy = {"query_acc@1": round(float(np.mean(top5[:, 0] == truth)), 5),
                "query_acc@5": round(float(np.mean([t in r for t, r in zip(truth, top5)])), 5)}
    print(f"  validation query category accuracy: {accuracy}", flush=True)

    catalog_ids = sorted(p["item_id"] for p in read_json(args.data_dir / "catalog.json"))
    out = C / "submission"
    write_submission(out / "validation_ranked.csv", valid["query_ids"], ranked(valid, ce["validation"], cat["validation"], w, beta), catalog_ids)

    # 3) Test.
    path = out / f"{args.student_id}_final.csv"
    check = write_submission(path, test["query_ids"], ranked(test, ce["test"], cat["test"], w, beta), catalog_ids)
    hybrid = ROOT / f"outputs/dense_eval/bert-ltr/hybrid_bm25_expansion_dense/{args.student_id}_final.csv"
    consistent = None
    if hybrid.exists():
        with open(hybrid, encoding="utf-8", newline="") as stream:
            previous = {r[0]: r[1:] for r in list(csv.reader(stream))[1:]}
        consistent = all(previous[q] == r[:10] for q, r in zip(test["query_ids"], test["items"]))
    print(f"Wrote {path}; test candidates match the hybrid submission: {consistent}", flush=True)
    write_json(out / "metrics.json", {
        "method": "multi-stage ranking + category auxiliary score (label-GCN multi-task classifier)",
        "evaluation": "local development validation (4,000 queries, full catalog); not a Kaggle score",
        "weight_grid": list(WEIGHTS), "beta_grid": list(BETAS),
        "tuning_slice_best_recall_at_10_by_beta": best_by_beta,
        "tuning_slice_recall_at_10": {f"w{a}_beta{b}": v for (a, b), v in curve.items()},
        "selected": {"weight": w, "beta": beta}, "validation": results,
        "validation_query_category_accuracy": accuracy, "test_csv": check,
        "test_top10_candidates_match_hybrid_csv": consistent,
        "dev_model_log": read_json(C / "dev/train_log.json"), "full_model_log": read_json(C / "full/train_log.json"),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "kaggle_recall_at_10": None,
    })


if __name__ == "__main__":
    main()
