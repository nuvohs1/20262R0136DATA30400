"""Mine cross-encoder hard negatives from the hybrid first stage (CPU, NumPy only).

The step-4b cross-encoder was trained with negatives from BM25 v2 alone, but at inference it
reranks the top 50 of the hybrid BM25 v2 + alpha * z(LTR BERT). This script mines negatives from
that same hybrid, so training negatives look like the candidates the reranker has to separate.

For each training query, the hybrid ranks the full catalog; the top products other than the
correct one are kept (item IDs only). Same leakage protocol as scripts/mine_negatives.py:
  dev : the 12,000 development-training queries (development-training minus the tuning slice);
        BM25 RSJ weights from those 12,000 queries; dense embeddings from LTR model A, which was
        trained on exactly these queries and never on the tuning slice or validation.
  full: all 20,000 training queries; RSJ from all of them; LTR model B.
Review expansion is disabled: a training query's own reviews would otherwise be in the review
index and reveal the correct product. Only the correct product is excluded; nothing else about
the labels is used. alpha is the value selected on the tuning slice by scripts/run_dense.py.
"""

import argparse
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_candidates import candidates, load_embeddings
from review_retrieval.bm25_v2 import Retriever
from review_retrieval.common import read_json, sha256, write_json
from review_retrieval.sparse_bm25 import product_field_tokens, review_token_lists
from run_bm25_v2 import CONFIG as BM25_CONFIG, without_expansion


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "project_release")
    parser.add_argument("--split", type=Path, default=ROOT / "artifacts/dev_split.json")
    parser.add_argument("--dev-embeddings", type=Path, default=ROOT / "outputs/dense/bert-ltr-dev")
    parser.add_argument("--full-embeddings", type=Path, default=ROOT / "outputs/dense/bert-ltr-full")
    parser.add_argument("--alpha-from", type=Path, default=ROOT / "outputs/dense_eval/bert-ltr/metrics.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/hardneg")
    parser.add_argument("--depth", type=int, default=51, help="Retrieve this many; keep up to depth-1 negatives")
    parser.add_argument("--sets", default="dev,full")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    random.seed(42)
    np.seterr(all="ignore")
    alpha = float(read_json(args.alpha_from)["selected_alpha"])

    catalog = sorted(read_json(args.data_dir / "catalog.json"), key=lambda p: p["item_id"])
    train = read_json(args.data_dir / "train_gold.json")
    split = read_json(args.split)
    if sha256(args.data_dir / "train_gold.json") != split["source_sha256"]:
        raise ValueError("Training data does not match the fixed split fingerprint")
    item_ids = [p["item_id"] for p in catalog]
    position = {item: n for n, item in enumerate(item_ids)}
    by_id = {q["query_id"]: q for q in train}
    row = {q["query_id"]: n for n, q in enumerate(train)}
    fit_ids, valid_ids = split["train_query_ids"], split["validation_query_ids"]
    tune_ids = sorted(random.Random(42).sample(sorted(fit_ids), 4000))
    rest_ids = sorted(set(fit_ids) - set(tune_ids))

    stem = BM25_CONFIG["stem"]
    print("Tokenizing...", flush=True)
    fields = [product_field_tokens(p, stem) for p in catalog]
    tokens = {q["query_id"]: review_token_lists(q, stem) for q in train}
    retriever = Retriever(without_expansion(BM25_CONFIG), item_ids, fields)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sets = {"dev": (rest_ids, args.dev_embeddings), "full": ([q["query_id"] for q in train], args.full_embeddings)}
    for name in args.sets.split(","):
        start = time.perf_counter()
        ids, emb_dir = sets[name]
        eligibility, (catalog_emb, train_emb) = load_embeddings(
            emb_dir, ("catalog", "train_reviews"), ("catalog.json", "train_gold.json"), args.data_dir)
        if eligibility["kind"] != "fine-tuned checkpoint" or eligibility["provenance"].get("negatives_set") != name:
            raise ValueError(f"'{name}' negatives need the LTR model trained on the same queries ({name})")
        train_emb = train_emb.reshape(len(train), 5, -1)
        print(f"Mining '{name}' hybrid negatives for {len(ids):,} queries ({emb_dir.name}, alpha {alpha})...", flush=True)
        retriever.learn([(by_id[q]["item_id"], tokens[q]) for q in ids])
        ranked = candidates(retriever, catalog_emb, [tokens[q] for q in ids], [train_emb[row[q]] for q in ids],
                            alpha, args.depth, args.workers)
        negatives, hits = {}, 0
        for q, (r, _) in zip(ids, ranked):
            target = position[by_id[q]["item_id"]]
            hits += target in r[:10]
            negatives[q] = [item_ids[i] for i in r if i != target][:args.depth - 1]
        record = {
            "set": name, "train_query_ids": ids,
            "tune_query_ids": tune_ids if name == "dev" else [],
            "validation_query_ids_excluded": valid_ids if name == "dev" else [],
            "negatives": negatives,
            "first_stage": {"bm25": "v2 metadata only, no review expansion", "dense": emb_dir.name,
                            "dense_negatives_set": eligibility["provenance"]["negatives_set"],
                            "dense_checkpoint_checked": True, "alpha": alpha, "depth": args.depth,
                            "rsj_source": f"the {len(ids):,} queries of this set"},
            "first_stage_recall_at_10_on_these_training_queries": hits / len(ids),
            "train_gold_sha256": split["source_sha256"],
            "catalog_sha256": sha256(args.data_dir / "catalog.json"),
        }
        write_json(args.output_dir / f"negatives_{name}.json", record)
        summary = {k: v for k, v in record.items() if k not in ("negatives", "train_query_ids", "tune_query_ids",
                                                                 "validation_query_ids_excluded")}
        write_json(args.output_dir / f"negatives_{name}_summary.json",
                   {**summary, "train_queries": len(ids), "negatives_sha256": sha256(args.output_dir / f"negatives_{name}.json")})
        print(f"  wrote negatives_{name}.json (hybrid R@10 on these training queries {hits / len(ids):.4f}, "
              f"{time.perf_counter() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
