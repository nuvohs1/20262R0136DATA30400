"""Evaluate rerankers on the test-like proxy validation and write Kaggle files (NumPy only).

Inputs (scripts/make_memory_candidates.py, scripts/score_cross_encoder.py):
  outputs/memory/candidates_{exp,mem}_{proxy_tune,proxy_validation,test}.json
  outputs/memory/scores/{model}_{variant}_{split}.npy  for model in {rerank, hardneg}
  (rerank = step-4b cross-encoder trained with BM25 negatives; hardneg = cross-encoder trained
  with hybrid negatives). Model A scores the proxy splits, model B scores test.
  ens = both cross-encoders averaged: (z(rerank) + z(hardneg)) / 2 within each query's
  candidates (no extra training; the two models differ in their training negatives).
  aug = augmented cross-encoder (run_augce.sh; scored on the exp variant only);
  ens3 = rerank, hardneg and aug averaged the same way.
Pairs whose score files are missing are skipped.

For every (first-stage variant, cross-encoder) pair, the fusion weight w is chosen on the proxy
tuning slice. The pair with the best proxy tuning-slice Recall@10 is selected (ties: exp before
mem, then the MODELS order, i.e. the simpler or earlier pipeline). Proxy validation is reported
once for every pair, overall and separately for queries whose product has / has not labelled
memory. The fusion weights used by the earlier submissions (rerank w 0.5, hardneg w 0.6) are
also reported on the proxy, to check whether the proxy agrees with what was observed before.
"""

import argparse
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from review_retrieval.common import read_json, write_json, write_submission
from review_retrieval.metrics import mean_metrics
from review_retrieval.rerank import WEIGHTS, rerank_positions, row_zscore, select_weight

VARIANTS = ("exp", "mem")
MODELS = ("rerank", "hardneg", "ens", "aug", "ens3")  # tie-break order: simpler / earlier first
PARTS = {"ens": ("rerank", "hardneg"), "ens3": ("rerank", "hardneg", "aug")}
SPLITS = ("proxy_tune", "proxy_validation", "test")
EARLIER_WEIGHTS = {"rerank": 0.5, "hardneg": 0.6}


def ranked_lists(cands, scores, weight):
    order = rerank_positions(np.asarray(cands["first_stage_scores"]), scores, weight)
    return [[row[p] for p in pos] for row, pos in zip(cands["items"], order)]


def report(cands, scores, weight, gold):
    lists = ranked_lists(cands, scores, weight)
    flags = cands["has_memory"]
    out = {}
    for name, keep in (("all", None), ("with_memory", True), ("without_memory", False)):
        idx = [i for i, f in enumerate(flags) if keep is None or f == keep]
        out[name] = {k: round(v, 5) for k, v in mean_metrics([gold[i] for i in idx], [lists[i] for i in idx]).items()}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "project_release")
    parser.add_argument("--memory-dir", type=Path, default=ROOT / "outputs/memory")
    parser.add_argument("--student-id", default="2024320020")
    args = parser.parse_args()
    if not args.student_id.isalnum():
        parser.error("--student-id must be alphanumeric")
    d = args.memory_dir
    gold_by_id = {q["query_id"]: q["item_id"] for q in read_json(args.data_dir / "train_gold.json")}
    catalog_ids = sorted(p["item_id"] for p in read_json(args.data_dir / "catalog.json"))
    cands = {(v, s): read_json(d / f"candidates_{v}_{s}.json") for v in VARIANTS for s in ("proxy_tune", "proxy_validation", "test")}
    def scores(m, v, s):
        if m in PARTS:
            return sum(row_zscore(scores(p, v, s)) for p in PARTS[m]) / len(PARTS[m])
        return np.load(d / "scores" / f"{m}_{v}_{s}.npy")

    def available(m, v):
        return all((d / "scores" / f"{p}_{v}_{s}.npy").exists() for p in PARTS.get(m, (m,)) for s in SPLITS)

    tuning, results = {}, {}
    for v in VARIANTS:
        tune, valid = cands[(v, "proxy_tune")], cands[(v, "proxy_validation")]
        tune_gold = [gold_by_id[q] for q in tune["query_ids"]]
        valid_gold = [gold_by_id[q] for q in valid["query_ids"]]
        for m in MODELS:
            if not available(m, v):
                continue
            t_scores = scores(m, v, "proxy_tune")
            w, curve = select_weight(tune["items"], tune_gold, np.asarray(tune["first_stage_scores"]), t_scores)
            tuning[f"{v}+{m}"] = {"selected_weight": w, "recall_at_10_by_weight": curve, "best": curve[w]}
            print(f"{v}+{m}: proxy tuning R@10 by w {curve} -> w {w}", flush=True)
    key = lambda name: (tuning[name]["best"], -VARIANTS.index(name.split("+")[0]), -MODELS.index(name.split("+")[1]))
    selected = max(tuning, key=key)
    print(f"Selected on the proxy tuning slice: {selected} (R@10 {tuning[selected]['best']:.4f})", flush=True)

    # Proxy validation, once, for every pair (choices above are already fixed).
    for v in VARIANTS:
        valid = cands[(v, "proxy_validation")]
        valid_gold = [gold_by_id[q] for q in valid["query_ids"]]
        results[f"{v} first stage"] = report(valid, np.zeros((len(valid["items"]), len(valid["items"][0]))), 0.0, valid_gold)
        for m in MODELS:
            if f"{v}+{m}" not in tuning:
                continue
            s = scores(m, v, "proxy_validation")
            w = tuning[f"{v}+{m}"]["selected_weight"]
            results[f"{v}+{m} (w {w}, proxy-tuned)"] = report(valid, s, w, valid_gold)
            if v == "exp" and m in EARLIER_WEIGHTS:
                results[f"{v}+{m} (w {EARLIER_WEIGHTS[m]}, as submitted before)"] = report(valid, s, EARLIER_WEIGHTS[m], valid_gold)
    for name, r in results.items():
        print(f"  proxy validation {name}: R@10 {r['all']['recall@10']:.4f} "
              f"(memory {r['with_memory']['recall@10']:.4f}, no memory {r['without_memory']['recall@10']:.4f})", flush=True)

    out = d / "submission"
    files = {}
    for name in tuning:
        v, m = name.split("+")
        test = cands[(v, "test")]
        path = out / f"{v}_{m}" / f"{args.student_id}_final.csv"
        files[name] = write_submission(path, test["query_ids"],
                                       ranked_lists(test, scores(m, v, "test"), tuning[name]["selected_weight"]), catalog_ids)
        print(f"Wrote {path}", flush=True)
    v, m = selected.split("+")
    final = out / f"{args.student_id}_final.csv"
    files["selected"] = write_submission(final, cands[(v, "test")]["query_ids"],
                                         ranked_lists(cands[(v, "test")], scores(m, v, "test"), tuning[selected]["selected_weight"]),
                                         catalog_ids)
    write_json(out / "metrics.json", {
        "method": "test-like proxy validation (labelled-review memory) for the multi-stage ranker",
        "evaluation": "local proxy: 3-review queries; half of the products have 2 labelled reviews (seed 42); "
                      "w and the pair chosen on the proxy tuning slice; proxy validation reported once; not a Kaggle score",
        "memory_selection": read_json(d / "memory_selection.json"),
        "candidate_recall": {f"{v}/{s}": cands[(v, s)]["candidate_recall"] for v in VARIANTS for s in ("proxy_tune", "proxy_validation")},
        "first_stage": {f"{v}/{s}": cands[(v, s)]["first_stage"] for v in VARIANTS for s in ("proxy_tune", "proxy_validation", "test")},
        "weight_grid": list(WEIGHTS), "proxy_tuning": tuning, "selected": selected,
        "proxy_validation": results, "test_csv": files,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    })
    print(f"Final file ({selected}): {final}", flush=True)


if __name__ == "__main__":
    main()
