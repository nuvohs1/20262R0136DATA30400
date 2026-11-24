"""First stage of the multi-stage ranker: top-K candidates from BM25 v2 + alpha * LTR BERT (NumPy only).

Writes outputs/rerank/candidates_{tune,validation,test}.json with item IDs and first-stage
scores (no review text). Same leakage protocol as scripts/run_dense.py:
  tune       : 4,000-query tuning slice; BM25 RSJ from the other 12,000 development-training
               queries; dense embeddings from LTR model A (never trained on these queries).
  validation : 4,000 held-out queries; RSJ from the 16,000 development-training queries; model A.
  test       : RSJ and labelled-review expansion from all 20,000 training queries; model B.
alpha is the value selected on the tuning slice by scripts/run_dense.py (read from its metrics).
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

from review_retrieval.bm25_v2 import Retriever, zscore
from review_retrieval.common import read_json, sha256, write_json
from review_retrieval.model_policy import check_embedding_manifest
from review_retrieval.sparse_bm25 import product_field_tokens, review_token_lists, top_k
from run_bm25_v2 import CONFIG as BM25_CONFIG, with_expansion, without_expansion

_W = {}


def _init(retriever, catalog, alpha, depth):
    _W.update(retriever=retriever, catalog=catalog.astype(np.float32), alpha=alpha, depth=depth)


def _candidates(batch):
    out = []
    for reviews, embeddings in batch:
        dense = (_W["catalog"] @ embeddings.astype(np.float32).T).mean(axis=1)
        scores = _W["retriever"].scores(reviews) + _W["alpha"] * zscore(dense)
        idx = top_k(scores, _W["depth"])
        out.append((idx.tolist(), [round(float(s), 5) for s in scores[idx]]))
    return out


def candidates(retriever, catalog, reviews, embeddings, alpha, depth, workers):
    items = list(zip(reviews, embeddings))
    batches = [items[i:i + 64] for i in range(0, len(items), 64)]
    result = []
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(retriever, catalog, alpha, depth)) as ex:
        for part in ex.map(_candidates, batches):
            result.extend(part)
    return result


def load_embeddings(directory, parts, sources, data_dir):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name in sources:
        if manifest["source_sha256"][name] != sha256(data_dir / name):
            raise ValueError(f"{directory} was computed from a different {name}")
    eligibility = check_embedding_manifest(manifest)
    return eligibility, [np.load(directory / f"{p}.npy") for p in parts]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "project_release")
    parser.add_argument("--split", type=Path, default=ROOT / "artifacts/dev_split.json")
    parser.add_argument("--dev-embeddings", type=Path, default=ROOT / "outputs/dense/bert-ltr-dev")
    parser.add_argument("--test-embeddings", type=Path, default=ROOT / "outputs/dense/bert-ltr-full")
    parser.add_argument("--alpha-from", type=Path, default=ROOT / "outputs/dense_eval/bert-ltr/metrics.json")
    parser.add_argument("--depth", type=int, default=50, help="Candidates per query (K)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/rerank")
    parser.add_argument("--splits", default="tune,validation,test")
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
    by_id = {q["query_id"]: q for q in train}
    row = {q["query_id"]: n for n, q in enumerate(train)}
    fit_ids, valid_ids = split["train_query_ids"], split["validation_query_ids"]
    tune_ids = sorted(random.Random(42).sample(sorted(fit_ids), 4000))
    rest_ids = sorted(set(fit_ids) - set(tune_ids))

    stem = BM25_CONFIG["stem"]
    print("Tokenizing...", flush=True)
    fields = [product_field_tokens(p, stem) for p in catalog]
    tokens = {q["query_id"]: review_token_lists(q, stem) for q in train}
    retriever = Retriever(with_expansion(BM25_CONFIG), item_ids, fields)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wanted = args.splits.split(",")

    if {"tune", "validation"} & set(wanted):
        eligibility, (catalog_emb, train_emb) = load_embeddings(
            args.dev_embeddings, ("catalog", "train_reviews"), ("catalog.json", "train_gold.json"), args.data_dir)
        if eligibility["kind"] == "fine-tuned checkpoint" and eligibility["provenance"].get("negatives_set") != "dev":
            raise ValueError("Tuning/validation candidates must use the development model (model A)")
        train_emb = train_emb.reshape(len(train), 5, -1)
        retriever.config = without_expansion(BM25_CONFIG)
        for name, ids, rsj_ids in (("tune", tune_ids, rest_ids), ("validation", valid_ids, fit_ids)):
            if name not in wanted:
                continue
            start = time.perf_counter()
            retriever.learn([(by_id[q]["item_id"], tokens[q]) for q in rsj_ids])
            ranked = candidates(retriever, catalog_emb, [tokens[q] for q in ids], [train_emb[row[q]] for q in ids],
                                alpha, args.depth, args.workers)
            gold = [by_id[q]["item_id"] for q in ids]
            items = [[item_ids[i] for i in r] for r, _ in ranked]
            recall = {f"recall_at_{k}": float(np.mean([g in it[:k] for g, it in zip(gold, items)])) for k in (10, 20, 30, 50, args.depth)}
            write_json(args.output_dir / f"candidates_{name}.json", {
                "split": name, "query_ids": ids, "items": items, "first_stage_scores": [s for _, s in ranked],
                "first_stage": {"bm25": "v2 metadata only", "rsj_queries": len(rsj_ids), "dense": str(args.dev_embeddings.name),
                                "alpha": alpha, "depth": args.depth},
                "candidate_recall": recall, "train_gold_sha256": split["source_sha256"],
            })
            print(f"  {name}: {len(ids):,} queries, candidate recall {recall} ({time.perf_counter() - start:.0f}s)", flush=True)

    if "test" in wanted:
        start = time.perf_counter()
        test = read_json(args.data_dir / "test_queries.json")
        _, (catalog_emb, test_emb) = load_embeddings(
            args.test_embeddings, ("catalog", "test_reviews"), ("catalog.json", "test_queries.json"), args.data_dir)
        test_emb = test_emb.reshape(len(test), 5, -1)
        retriever.config = with_expansion(BM25_CONFIG)
        retriever.learn([(q["item_id"], tokens[q["query_id"]]) for q in train])
        ranked = candidates(retriever, catalog_emb, [review_token_lists(q, stem) for q in test], list(test_emb),
                            alpha, args.depth, args.workers)
        write_json(args.output_dir / "candidates_test.json", {
            "split": "test", "query_ids": [q["query_id"] for q in test],
            "items": [[item_ids[i] for i in r] for r, _ in ranked], "first_stage_scores": [s for _, s in ranked],
            "first_stage": {"bm25": "v2 + labelled-review expansion", "rsj_queries": len(train),
                            "dense": str(args.test_embeddings.name), "alpha": alpha, "depth": args.depth},
            "test_queries_sha256": sha256(args.data_dir / "test_queries.json"),
        })
        print(f"  test: {len(test):,} queries ({time.perf_counter() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
