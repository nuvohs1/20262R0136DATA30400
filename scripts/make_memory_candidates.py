"""Test-like proxy validation and labelled-review memory for the first stage (CPU, NumPy only).

Why: on test, many correct products already have labelled reviews in train_gold.json, and the
first stage uses them (BM25 review expansion). No validation product has labelled reviews, so
the earlier validation could not show that information, and a reranker tuned there can
override it on test. The proxy (review_retrieval.memory.build_proxy, seed 42) gives half of the
tuning-slice / validation queries 2 of their own reviews as labelled memory and keeps 3 reviews
as the query. Only course training data is used; nothing is estimated from test queries.

Two first-stage variants are written for each split (top-K candidates, item IDs and scores):
  exp : BM25 v2 with review expansion + alpha * z(LTR BERT) (the test first stage of step 4b);
  mem : exp + a dense memory bonus mu * max(0, z(memory) - tau), where the memory score is the
        inner product between the query's mean review embedding and the mean embedding of a
        product's labelled reviews. (mu, tau) is chosen on the proxy tuning slice by Recall@10.
Splits and what they may use:
  proxy_tune       : the 4,000 tuning-slice queries (3 reviews each). RSJ from the other 12,000
                     development-training queries; memory = those 12,000 queries' reviews + the
                     proxy memory; dense = LTR model A.
  proxy_validation : the 4,000 validation queries (3 reviews each). RSJ from the 16,000
                     development-training queries; memory = their reviews + the proxy memory;
                     model A. Nothing is chosen here.
  test             : the test queries (5 reviews). RSJ and memory from all 20,000 training
                     queries; LTR model B. Uses the (mu, tau) from the tuning slice.
RSJ query weights are always learned from the full labelled queries only, not from the proxy
memory reviews.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_candidates import load_embeddings
from review_retrieval.bm25_v2 import Retriever, zscore
from review_retrieval.common import read_json, sha256, write_json
from review_retrieval.memory import MEMORY_SHARE, MUS, TAUS, build_proxy, memory_bonus, select_memory_weights
from review_retrieval.sparse_bm25 import product_field_tokens, review_token_lists, top_k
from run_bm25_v2 import CONFIG as BM25_CONFIG, with_expansion

_W = {}


def _init(state):
    _W.update(state)


def _run(batch):
    out = []
    for reviews, embedding in batch:
        base = _W["retriever"].scores(reviews) + _W["alpha"] * zscore(_W["catalog"] @ embedding)
        per_combo = []
        for mu, tau in _W["combos"]:
            s = base.copy()
            if mu:
                s[_W["memory_positions"]] += memory_bonus(embedding, _W["memory_matrix"], mu, tau)
            idx = top_k(s, _W["depth"])
            per_combo.append((idx.tolist(), [round(float(x), 5) for x in s[idx]]))
        out.append(per_combo)
    return out


def first_stage(state, reviews, embeddings, workers):
    items = list(zip(reviews, embeddings))
    batches = [items[i:i + 32] for i in range(0, len(items), 32)]
    result = []
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(state,)) as ex:
        for part in ex.map(_run, batches):
            result.extend(part)
            if len(result) % 1024 < 32 or len(result) == len(items):
                print(f"    {len(result):,}/{len(items):,}", flush=True)
    return result


def build_retriever(item_ids, fields, rsj_pairs, memory_pairs):
    retriever = Retriever(with_expansion(BM25_CONFIG), item_ids, fields)
    retriever.learn(rsj_pairs)
    weights = retriever.term_weights
    retriever.learn(rsj_pairs + memory_pairs)  # review-expansion index over all labelled reviews
    retriever.term_weights = weights  # RSJ weights from the full labelled queries only
    return retriever


def memory_matrix(products, embeddings_by_product, position):
    """products: sorted item IDs with memory; returns (positions, (M, d) mean embeddings)."""
    return (np.array([position[p] for p in products], dtype=np.int64),
            np.stack([embeddings_by_product[p] for p in products]).astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "project_release")
    parser.add_argument("--split", type=Path, default=ROOT / "artifacts/dev_split.json")
    parser.add_argument("--dev-embeddings", type=Path, default=ROOT / "outputs/dense/bert-ltr-dev")
    parser.add_argument("--full-embeddings", type=Path, default=ROOT / "outputs/dense/bert-ltr-full")
    parser.add_argument("--alpha-from", type=Path, default=ROOT / "outputs/dense_eval/bert-ltr/metrics.json")
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/memory")
    parser.add_argument("--splits", default="proxy_tune,proxy_validation,test")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    random.seed(42)
    np.seterr(all="ignore")
    alpha = float(read_json(args.alpha_from)["selected_alpha"])
    wanted = args.splits.split(",")

    catalog = sorted(read_json(args.data_dir / "catalog.json"), key=lambda p: p["item_id"])
    train = read_json(args.data_dir / "train_gold.json")
    split = read_json(args.split)
    if sha256(args.data_dir / "train_gold.json") != split["source_sha256"]:
        raise ValueError("Training data does not match the fixed split fingerprint")
    item_ids = [p["item_id"] for p in catalog]
    position = {item: n for n, item in enumerate(item_ids)}
    by_id = {q["query_id"]: q for q in train}
    row = {q["query_id"]: n for n, q in enumerate(train)}
    if len({q["item_id"] for q in train}) != len(train):
        raise ValueError("Expected one training query per product")
    fit_ids, valid_ids = split["train_query_ids"], split["validation_query_ids"]
    tune_ids = sorted(random.Random(42).sample(sorted(fit_ids), 4000))
    rest_ids = sorted(set(fit_ids) - set(tune_ids))

    stem = BM25_CONFIG["stem"]
    print("Tokenizing...", flush=True)
    fields = [product_field_tokens(p, stem) for p in catalog]
    tokens = {q["query_id"]: review_token_lists(q, stem) for q in train}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.output_dir / "memory_selection.json"

    def write(variant, name, ids, ranked, extra):
        items = [[item_ids[i] for i in r] for r, _ in ranked]
        record = {"split": name, "variant": variant, "query_ids": ids, "items": items,
                  "first_stage_scores": [s for _, s in ranked], "train_gold_sha256": split["source_sha256"], **extra}
        if name != "test":
            gold = [by_id[q]["item_id"] for q in ids]
            record["candidate_recall"] = {f"recall_at_{k}": float(np.mean([g in it[:k] for g, it in zip(gold, items)]))
                                          for k in (10, 20, 30, 50)}
            print(f"  {variant}/{name}: candidate recall {record['candidate_recall']}", flush=True)
        write_json(args.output_dir / f"candidates_{variant}_{name}.json", record)

    if {"proxy_tune", "proxy_validation"} & set(wanted):
        eligibility, (catalog_emb, train_emb) = load_embeddings(
            args.dev_embeddings, ("catalog", "train_reviews"), ("catalog.json", "train_gold.json"), args.data_dir)
        if eligibility["kind"] == "fine-tuned checkpoint" and eligibility["provenance"].get("negatives_set") != "dev":
            raise ValueError("Proxy splits must use the development model (model A)")
        train_emb = train_emb.reshape(len(train), 5, -1).astype(np.float32)
        for name, ids, labelled_ids in (("proxy_tune", tune_ids, rest_ids), ("proxy_validation", valid_ids, fit_ids)):
            if name not in wanted:
                continue
            start = time.perf_counter()
            proxy = build_proxy(ids, seed=42)
            rsj_pairs = [(by_id[q]["item_id"], tokens[q]) for q in labelled_ids]
            memory_pairs = [(by_id[q]["item_id"], [tokens[q][i] for i in proxy[q]["memory"]]) for q in ids if proxy[q]["memory"]]
            retriever = build_retriever(item_ids, fields, rsj_pairs, memory_pairs)
            memory = {by_id[q]["item_id"]: train_emb[row[q]].mean(axis=0) for q in labelled_ids}
            memory.update({by_id[q]["item_id"]: train_emb[row[q], proxy[q]["memory"]].mean(axis=0) for q in ids if proxy[q]["memory"]})
            positions, matrix = memory_matrix(sorted(memory), memory, position)
            if name == "proxy_tune":
                combos = [(0.0, 0.0)] + [(m, t) for m in MUS if m for t in TAUS]
            else:
                chosen = read_json(selection_path)["selected"]
                combos = [(0.0, 0.0), (chosen["mu"], chosen["tau"])]
            state = dict(retriever=retriever, catalog=catalog_emb.astype(np.float32), alpha=alpha, depth=args.depth,
                         memory_positions=positions, memory_matrix=matrix, combos=combos)
            print(f"{name}: {len(ids):,} queries (3 reviews each), {len(memory_pairs):,} with proxy memory, "
                  f"{len(memory):,} products with memory, {len(combos)} first-stage settings", flush=True)
            ranked = first_stage(state, [[tokens[q][i] for i in proxy[q]["query"]] for q in ids],
                                 [train_emb[row[q], proxy[q]["query"]].mean(axis=0) for q in ids], args.workers)
            gold = [position[by_id[q]["item_id"]] for q in ids]
            if name == "proxy_tune":
                grid = {}
                for c, combo in enumerate(combos):
                    grid[combo] = float(np.mean([g in r[c][0][:10] for g, r in zip(gold, ranked)]))
                full = {(m, t): grid[(0.0, 0.0)] if m == 0 else grid[(m, t)] for m in MUS for t in TAUS}
                mu, tau = select_memory_weights(full)
                write_json(selection_path, {
                    "rule": "proxy tuning-slice first-stage Recall@10; ties -> smaller mu, then larger tau",
                    "memory_share": MEMORY_SHARE, "alpha": alpha,
                    "tuning_recall_at_10": {f"mu{m}_tau{t}": v for (m, t), v in sorted(full.items())},
                    "selected": {"mu": mu, "tau": tau}})
                print(f"  selected mu {mu}, tau {tau} (R@10 {full[(mu, tau)]:.4f}; no memory {full[(0.0, 0.0)]:.4f})", flush=True)
                chosen_index = combos.index((mu, tau)) if mu else 0
            else:
                chosen_index = 1 if combos[1][0] else 0
                mu, tau = combos[1]
            extra = {"review_indices": [proxy[q]["query"] for q in ids],
                     "has_memory": [bool(proxy[q]["memory"]) for q in ids],
                     "first_stage": {"bm25": "v2 + review expansion (labelled + proxy memory)", "rsj_queries": len(labelled_ids),
                                     "dense": args.dev_embeddings.name, "alpha": alpha, "depth": args.depth}}
            write("exp", name, ids, [r[0] for r in ranked], extra)
            write("mem", name, ids, [r[chosen_index] for r in ranked],
                  {**extra, "first_stage": {**extra["first_stage"], "memory_mu": mu, "memory_tau": tau}})
            print(f"  {name} done ({time.perf_counter() - start:.0f}s)", flush=True)

    if "test" in wanted:
        start = time.perf_counter()
        chosen = read_json(selection_path)["selected"]
        test = read_json(args.data_dir / "test_queries.json")
        _, (catalog_emb, train_emb, test_emb) = load_embeddings(
            args.full_embeddings, ("catalog", "train_reviews", "test_reviews"),
            ("catalog.json", "train_gold.json", "test_queries.json"), args.data_dir)
        train_emb = train_emb.reshape(len(train), 5, -1).astype(np.float32)
        test_emb = test_emb.reshape(len(test), 5, -1).astype(np.float32)
        pairs = [(q["item_id"], tokens[q["query_id"]]) for q in train]
        retriever = build_retriever(item_ids, fields, pairs, [])
        memory = {q["item_id"]: train_emb[row[q["query_id"]]].mean(axis=0) for q in train}
        positions, matrix = memory_matrix(sorted(memory), memory, position)
        combos = [(0.0, 0.0), (chosen["mu"], chosen["tau"])]
        state = dict(retriever=retriever, catalog=catalog_emb.astype(np.float32), alpha=alpha, depth=args.depth,
                     memory_positions=positions, memory_matrix=matrix, combos=combos)
        print(f"test: {len(test):,} queries, memory of {len(memory):,} products, mu {chosen['mu']} tau {chosen['tau']}", flush=True)
        ranked = first_stage(state, [review_token_lists(q, stem) for q in test], list(test_emb.mean(axis=1)), args.workers)
        extra = {"first_stage": {"bm25": "v2 + review expansion (all training queries)", "rsj_queries": len(train),
                                 "dense": args.full_embeddings.name, "alpha": alpha, "depth": args.depth},
                 "test_queries_sha256": sha256(args.data_dir / "test_queries.json")}
        ids = [q["query_id"] for q in test]
        write("exp", "test", ids, [r[0] for r in ranked], extra)
        write("mem", "test", ids, [r[1 if chosen["mu"] else 0] for r in ranked],
              {**extra, "first_stage": {**extra["first_stage"], "memory_mu": chosen["mu"], "memory_tau": chosen["tau"]}})
        previous = ROOT / "outputs/rerank/candidates_test.json"
        if previous.exists():
            same = read_json(previous)["items"] == [[item_ids[i] for i in r[0][0]] for r in ranked]
            print(f"  exp/test candidates equal the step-4b test candidates: {same}", flush=True)
        print(f"  test done ({time.perf_counter() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
