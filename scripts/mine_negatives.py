"""Mine hard negatives for learning-to-rank fine-tuning (local, NumPy only).

First stage = BM25 v2 (metadata only, no review expansion). No pretrained model is
involved, so the negatives cannot carry information from any external checkpoint.
For each training query we keep the top-ranked products other than the correct one.
Only item IDs are written.

Two sets are produced so that no evaluation label is used for training:
  dev : training queries = development-training minus the tuning slice (12,000);
        BM25 RSJ weights are learned from those 12,000 queries only. The tuning
        slice (4,000) selects epochs; the validation partition is untouched.
  full: all 20,000 training queries, RSJ learned from all of them (final test model).
Review expansion is disabled: a training query's own reviews would otherwise be in
the review index and trivially reveal the correct product.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from review_retrieval.bm25_v2 import Retriever
from review_retrieval.common import read_json, sha256, write_json
from review_retrieval.sparse_bm25 import product_field_tokens, review_token_lists, top_k
from run_bm25_v2 import CONFIG as BM25_CONFIG, without_expansion

_W = {}


def _init(retriever, depth):
    _W.update(retriever=retriever, depth=depth)


def _mine(batch):
    return [top_k(_W["retriever"].scores(reviews), _W["depth"]).tolist() for reviews in batch]


def mine(retriever, reviews, depth, workers):
    items = list(reviews)
    batches = [items[i:i + 64] for i in range(0, len(items), 64)]
    ranked = []
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(retriever, depth)) as ex:
        for part in ex.map(_mine, batches):
            ranked.extend(part)
            if len(ranked) % 2048 < 64 or len(ranked) == len(items):
                print(f"  mined {len(ranked):,}/{len(items):,}", flush=True)
    return ranked


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "project_release")
    parser.add_argument("--split", type=Path, default=ROOT / "artifacts/dev_split.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ltr")
    parser.add_argument("--depth", type=int, default=51, help="Retrieve this many; keep up to depth-1 negatives")
    parser.add_argument("--sets", default="dev,full")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    random.seed(42)
    np.seterr(all="ignore")

    catalog = sorted(read_json(args.data_dir / "catalog.json"), key=lambda p: p["item_id"])
    train = read_json(args.data_dir / "train_gold.json")
    split = read_json(args.split)
    if sha256(args.data_dir / "train_gold.json") != split["source_sha256"]:
        raise ValueError("Training data does not match the fixed split fingerprint")
    item_ids = [p["item_id"] for p in catalog]
    position = {item: n for n, item in enumerate(item_ids)}
    by_id = {q["query_id"]: q for q in train}
    fit_ids, valid_ids = split["train_query_ids"], split["validation_query_ids"]
    tune_ids = sorted(random.Random(42).sample(sorted(fit_ids), 4000))
    rest_ids = sorted(set(fit_ids) - set(tune_ids))

    stem = BM25_CONFIG["stem"]
    print("Tokenizing...", flush=True)
    fields = [product_field_tokens(p, stem) for p in catalog]
    tokens = {q["query_id"]: review_token_lists(q, stem) for q in train}
    retriever = Retriever(without_expansion(BM25_CONFIG), item_ids, fields)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sets = {"dev": rest_ids, "full": [q["query_id"] for q in train]}
    for name in args.sets.split(","):
        start = time.perf_counter()
        ids = sets[name]
        print(f"Mining '{name}' negatives for {len(ids):,} queries...", flush=True)
        retriever.learn([(by_id[q]["item_id"], tokens[q]) for q in ids])
        ranked = mine(retriever, [tokens[q] for q in ids], args.depth, args.workers)
        negatives, hits = {}, 0
        for q, r in zip(ids, ranked):
            target = position[by_id[q]["item_id"]]
            hits += target in r[:10]
            negatives[q] = [item_ids[i] for i in r if i != target][:args.depth - 1]
        record = {
            "set": name, "train_query_ids": ids,
            "tune_query_ids": tune_ids if name == "dev" else [],
            "validation_query_ids_excluded": valid_ids if name == "dev" else [],
            "negatives": negatives,
            "first_stage": {"bm25": "v2 metadata only, no review expansion", "dense": None,
                            "depth": args.depth,
                            "rsj_source": f"the {len(ids):,} queries of this set"},
            "first_stage_recall_at_10_on_these_training_queries": hits / len(ids),
            "train_gold_sha256": split["source_sha256"],
            "catalog_sha256": sha256(args.data_dir / "catalog.json"),
        }
        write_json(args.output_dir / f"negatives_{name}.json", record)
        print(f"  wrote negatives_{name}.json (first-stage R@10 on these queries {hits / len(ids):.4f}, "
              f"{time.perf_counter() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
