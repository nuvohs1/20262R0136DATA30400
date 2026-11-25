"""ANN with an IVF index for the LTR BERT bi-encoder: speed vs. accuracy (NumPy only, CPU).

The bi-encoder's dense score (mean of the five review-product inner products) equals one inner
product with the mean review embedding, so each query is a single vector. An IVF index
(spherical k-means, --nlist cells) over the 50,000 product embeddings is compared with exact
search for several nprobe values:
  * ANN recall@10: share of the exact top 10 that IVF also returns (approximation quality);
  * Recall@10 / Recall@50 of the correct product (task quality);
  * products scored per query and milliseconds per query (cost).
nprobe is chosen on the tuning slice as the smallest value whose ANN recall@10 reaches --target;
validation is reported once. Embeddings: LTR model A (outputs/dense/bert-ltr-dev).
"""

import argparse
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from review_retrieval.common import read_json, sha256, write_json
from review_retrieval.ivf import IVFIndex, exact_search, spherical_kmeans
from review_retrieval.model_policy import check_embedding_manifest


def evaluate(queries, gold_rows, search):
    """search(query) -> (top-50 rows, products scored). Returns metrics and the top-10 lists."""
    tops, scored = [], 0
    start = time.perf_counter()
    for q in queries:
        rows, n = search(q)
        tops.append(rows)
        scored += n
    seconds = time.perf_counter() - start
    return {"recall@10": float(np.mean([g in t[:10] for g, t in zip(gold_rows, tops)])),
            "recall@50": float(np.mean([g in t for g, t in zip(gold_rows, tops)])),
            "products_scored_per_query": scored / len(queries),
            "ms_per_query": 1000 * seconds / len(queries)}, [t[:10] for t in tops]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "project_release")
    parser.add_argument("--split", type=Path, default=ROOT / "artifacts/dev_split.json")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "outputs/dense/bert-ltr-dev")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ann")
    parser.add_argument("--nlist", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--nprobes", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--target", type=float, default=0.95, help="ANN recall@10 required on the tuning slice")
    args = parser.parse_args()
    nprobes = [int(n) for n in args.nprobes.split(",")]

    manifest = read_json(args.embeddings / "manifest.json")
    for name in ("catalog.json", "train_gold.json"):
        if manifest["source_sha256"][name] != sha256(args.data_dir / name):
            raise ValueError(f"{args.embeddings} was computed from a different {name}")
    eligibility = check_embedding_manifest(manifest)
    if eligibility["kind"] == "fine-tuned checkpoint" and eligibility["provenance"]["negatives_set"] != "dev":
        raise ValueError("Tuning/validation queries must use the development model (model A)")
    train = read_json(args.data_dir / "train_gold.json")
    split = read_json(args.split)
    if sha256(args.data_dir / "train_gold.json") != split["source_sha256"]:
        raise ValueError("Training data does not match the fixed split fingerprint")
    catalog = np.load(args.embeddings / "catalog.npy").astype(np.float32)
    reviews = np.load(args.embeddings / "train_reviews.npy").astype(np.float32).reshape(len(train), 5, -1).mean(axis=1)
    item_ids = sorted(p["item_id"] for p in read_json(args.data_dir / "catalog.json"))
    position = {item: n for n, item in enumerate(item_ids)}
    row = {q["query_id"]: n for n, q in enumerate(train)}
    gold = {q["query_id"]: position[q["item_id"]] for q in train}
    tune_ids = sorted(random.Random(42).sample(sorted(split["train_query_ids"]), 4000))
    splits = {"tune": tune_ids, "validation": split["validation_query_ids"]}

    print(f"Building IVF index: {len(catalog):,} products, nlist {args.nlist}...", flush=True)
    start = time.perf_counter()
    centroids, assignment, history = spherical_kmeans(catalog, args.nlist, args.iterations, seed=42)
    index = IVFIndex(catalog, centroids, assignment)
    build_seconds = time.perf_counter() - start
    sizes = index.list_sizes()
    print(f"  built in {build_seconds:.1f}s; list sizes min {sizes.min()} median {int(np.median(sizes))} max {sizes.max()}", flush=True)

    results = {}
    for name, ids in splits.items():
        queries = reviews[[row[q] for q in ids]]
        gold_rows = [gold[q] for q in ids]
        exact, exact_top = evaluate(queries, gold_rows, lambda q: (exact_search(catalog, q, 50), len(catalog)))
        results[name] = {"exact": exact}
        print(f"{name} exact: {exact}", flush=True)
        for nprobe in nprobes:
            m, top = evaluate(queries, gold_rows, lambda q: index.search(q, nprobe, 50))
            m["ann_recall@10"] = float(np.mean([len(set(a.tolist()) & set(e.tolist())) / 10 for a, e in zip(top, exact_top)]))
            m["speedup_vs_exact"] = exact["ms_per_query"] / m["ms_per_query"]
            results[name][f"nprobe_{nprobe}"] = m
            print(f"{name} nprobe {nprobe}: {m}", flush=True)
        if name == "tune":
            ok = [n for n in nprobes if results["tune"][f"nprobe_{n}"]["ann_recall@10"] >= args.target]
            selected = min(ok) if ok else max(nprobes)
            print(f"Selected nprobe = {selected} (target ANN recall@10 {args.target})", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "metrics.json", {
        "method": "IVF (spherical k-means) approximate inner-product search over LTR BERT product embeddings",
        "evaluation": "local; tuning slice selects nprobe, validation reported once; dense retrieval only",
        "embeddings": str(args.embeddings.relative_to(ROOT)) if args.embeddings.is_relative_to(ROOT) else str(args.embeddings),
        "nlist": args.nlist, "iterations": args.iterations, "kmeans_mean_cosine_by_iteration": history,
        "list_sizes": {"min": int(sizes.min()), "median": float(np.median(sizes)), "max": int(sizes.max())},
        "build_seconds": build_seconds, "target_ann_recall_at_10": args.target, "selected_nprobe": selected,
        "results": results,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "cpus": os.cpu_count(),
                        "note": "timings are single-process wall-clock and vary between machines"},
    })
    print(f"Saved {args.output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
