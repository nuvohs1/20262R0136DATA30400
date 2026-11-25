"""Pseudo-labels for self-training the cross-encoder (NumPy only, CPU).

Inputs come from run_rerank.sh: first-stage candidates, cross-encoder scores, and the fusion
weight w selected on the tuning slice (outputs/rerank/submission/metrics.json).

  --stage dev   pool = tuning slice + validation queries (reviews only), scored by model A.
                The margin threshold is chosen here, on the tuning slice: the smallest margin
                whose pseudo-label precision reaches --min-precision. Validation labels are
                not read.
  --stage full  pool = the 19,858 test queries, scored by model B, with the threshold and w
                from the dev stage.

Writes outputs/pseudo/pseudo_{dev,full}.json: for every selected query, the pseudo positive and
up to --negative-pool hard negatives (stage-1 candidates without the pseudo positive).
"""

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from review_retrieval.common import read_json, sha256, write_json
from review_retrieval.pseudo import choose_threshold, negatives_for, precision_curve, top_and_margin


def load_pool(candidates_path, scores_path, weight):
    cands = read_json(candidates_path)
    scores = np.load(scores_path)
    first = np.asarray(cands["first_stage_scores"], dtype=np.float64)
    if scores.shape != first.shape:
        raise ValueError(f"{scores_path} does not match {candidates_path}")
    top, margin = top_and_margin(first, scores, weight)
    return cands, first, top, margin


def select(cands, first, top, margin, threshold, pool):
    out = []
    for n, q in enumerate(cands["query_ids"]):
        if margin[n] >= threshold:
            positive = cands["items"][n][top[n]]
            out.append({"query_id": q, "item_id": positive, "margin": round(float(margin[n]), 5),
                        "negatives": negatives_for(cands["items"][n], first[n], positive, pool)})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("dev", "full"), required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "project_release")
    parser.add_argument("--rerank-dir", type=Path, default=ROOT / "outputs/rerank")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/pseudo")
    parser.add_argument("--min-precision", type=float, default=0.9)
    parser.add_argument("--negative-pool", type=int, default=30)
    args = parser.parse_args()
    R = args.rerank_dir
    weight = read_json(R / "submission/metrics.json")["selected_weight"]
    train_sha = sha256(args.data_dir / "train_gold.json")

    if args.stage == "dev":
        tune, t_first, t_top, t_margin = load_pool(R / "candidates_tune.json", R / "ce-dev/scores_tune.npy", weight)
        gold = {q["query_id"]: q["item_id"] for q in read_json(args.data_dir / "train_gold.json")}
        correct = [row[p] == gold[q] for q, row, p in zip(tune["query_ids"], tune["items"], t_top)]
        curve = precision_curve(t_margin, correct)
        threshold = choose_threshold(curve, args.min_precision)
        del gold  # only tuning-slice labels were compared above
        valid, v_first, v_top, v_margin = load_pool(R / "candidates_validation.json", R / "ce-dev/scores_validation.npy", weight)
        queries = (select(tune, t_first, t_top, t_margin, threshold, args.negative_pool)
                   + select(valid, v_first, v_top, v_margin, threshold, args.negative_pool))
        info = {"pool": {"tune": len(tune["query_ids"]), "validation": len(valid["query_ids"])},
                "tuning_slice_curve": curve, "tuning_slice_precision": curve[threshold]["precision"],
                "selected": {"tune": int((t_margin >= threshold).sum()), "validation": int((v_margin >= threshold).sum())},
                "scored_by": "cross-encoder model A (outputs/rerank/ce-dev)", "query_source": "train_gold.json reviews only"}
    else:
        dev = read_json(args.output_dir / "pseudo_dev.json")
        threshold, weight = dev["threshold"], dev["weight"]
        test, s_first, s_top, s_margin = load_pool(R / "candidates_test.json", R / "ce-full/scores_test.npy", weight)
        queries = select(test, s_first, s_top, s_margin, threshold, args.negative_pool)
        info = {"pool": {"test": len(test["query_ids"])}, "selected": {"test": len(queries)},
                "scored_by": "cross-encoder model B (outputs/rerank/ce-full)", "query_source": "test_queries.json",
                "test_queries_sha256": sha256(args.data_dir / "test_queries.json")}

    record = {"set": f"pseudo-{args.stage}", "weight": weight, "threshold": threshold,
              "min_precision": args.min_precision, "negative_pool": args.negative_pool,
              "train_gold_sha256": train_sha, **info, "queries": queries}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / f"pseudo_{args.stage}.json", record)
    write_json(args.output_dir / f"pseudo_{args.stage}_summary.json", {k: v for k, v in record.items() if k != "queries"})
    print(f"{record['set']}: w {weight}, margin threshold {threshold}, selected {info['selected']} "
          f"of {info['pool']}", flush=True)


if __name__ == "__main__":
    main()
