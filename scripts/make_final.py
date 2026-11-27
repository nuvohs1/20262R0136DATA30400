"""Write the final Kaggle file (NumPy only).

Final pipeline (docs/final_selection.md):
  stage 1: BM25 v2 with labelled-review expansion + 3 * z(LTR BERT model B), top 50 per query
           (scripts/make_candidates.py -> outputs/rerank/candidates_test.json)
  stage 2: two BERT cross-encoders (model B), trained with BM25 negatives (rerank) and with
           hybrid negatives (hardneg), averaged as (z(rerank) + z(hardneg)) / 2 per query
  fusion : (1 - w) * z(stage 1) + w * z(stage 2)

w was chosen on the proxy tuning slice (training data only, scripts/run_memory_eval.py, pair
"exp+ens"). A full run reads it from outputs/memory/submission/metrics.json; the quick run reads
the committed record artifacts/experiments/memory_proxy.json. No setting is read from test data.
"""

import argparse
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from review_retrieval.common import read_json, sha256, write_json, write_submission
from review_retrieval.rerank import rerank_positions, row_zscore

PAIR = "exp+ens"


def ensemble_ranking(items, first_stage_scores, rerank_scores, hardneg_scores, weight):
    stage2 = (row_zscore(rerank_scores) + row_zscore(hardneg_scores)) / 2
    order = rerank_positions(np.asarray(first_stage_scores), stage2, weight)
    return [[row[p] for p in pos] for row, pos in zip(items, order)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "project_release")
    parser.add_argument("--candidates", type=Path, default=ROOT / "outputs/rerank/candidates_test.json")
    parser.add_argument("--rerank-scores", type=Path, default=ROOT / "outputs/rerank/ce-full/scores_test.npy")
    parser.add_argument("--hardneg-scores", type=Path, default=ROOT / "outputs/hardneg/ce-full/scores_test.npy")
    parser.add_argument("--selection", type=Path, default=ROOT / "outputs/memory/submission/metrics.json",
                        help="Proxy selection record with proxy_tuning['exp+ens']['selected_weight']")
    parser.add_argument("--reference", type=Path, default=ROOT / "artifacts/experiments/memory_proxy.json",
                        help="Committed record with the SHA-256 of the submitted file")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/final")
    parser.add_argument("--student-id", default="2024320020")
    args = parser.parse_args()
    if not args.student_id.isalnum():
        parser.error("--student-id must be alphanumeric")

    cands = read_json(args.candidates)
    if cands["split"] != "test" or cands.get("test_queries_sha256") != sha256(args.data_dir / "test_queries.json"):
        raise ValueError("Candidates must be the test candidates built from the current test_queries.json")
    rerank, hardneg = np.load(args.rerank_scores), np.load(args.hardneg_scores)
    shape = (len(cands["items"]), len(cands["items"][0]))
    if rerank.shape != shape or hardneg.shape != shape:
        raise ValueError(f"Score shapes {rerank.shape}, {hardneg.shape} do not match candidates {shape}")
    weight = float(read_json(args.selection)["proxy_tuning"][PAIR]["selected_weight"])

    catalog_ids = sorted(p["item_id"] for p in read_json(args.data_dir / "catalog.json"))
    lists = ensemble_ranking(cands["items"], cands["first_stage_scores"], rerank, hardneg, weight)
    path = args.output_dir / f"{args.student_id}_final.csv"
    check = write_submission(path, cands["query_ids"], lists, catalog_ids)
    reference = read_json(args.reference)["test_csv"][PAIR]["sha256"] if args.reference.exists() else None
    same = check["sha256"] == reference
    write_json(args.output_dir / "metrics.json", {
        "method": "BM25 v2 + review expansion + 3*z(LTR BERT) top 50 -> mean of two BERT cross-encoders, fused with w",
        "pair": PAIR, "weight": weight, "weight_source": str(args.selection.relative_to(ROOT) if args.selection.is_relative_to(ROOT) else args.selection),
        "first_stage": cands["first_stage"], "test_csv": check,
        "reference_sha256": reference, "matches_submitted_file": same,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    })
    print(f"Final file: {path} ({check})", flush=True)
    if reference:
        print("Matches the submitted file (Kaggle 0.77188)." if same else
              "SHA-256 differs from the submitted file. GPU kernels can reorder near-tied scores; "
              "the row differences are listed by scripts/compare_submissions.py.", flush=True)


if __name__ == "__main__":
    main()
