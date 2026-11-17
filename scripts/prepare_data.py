"""Audit course data locally and save a reproducible product-disjoint split."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re


ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def review_hashes(rows):
    # No raw review content is printed or persisted in audit outputs.
    return [
        hashlib.sha256(re.sub(r"\s+", " ", review["text"]).strip().encode()).hexdigest()
        for row in rows for review in row["reviews"]
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "project_release")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    names = ("catalog.json", "train_gold.json", "test_queries.json", "category_taxonomy.txt")
    fingerprints = {name: fingerprint(args.data_dir / name) for name in names}
    catalog, train, test = [
        json.loads((args.data_dir / name).read_text(encoding="utf-8")) for name in names[:3]
    ]
    taxonomy = (args.data_dir / names[3]).read_text(encoding="utf-8").splitlines()
    taxonomy = [line.strip() for line in taxonomy if line.strip()]
    require(len(catalog) == 50_000, "Expected 50,000 catalog products")
    require(len(train) == 20_000, "Expected 20,000 training queries")
    require(len(test) == 19_858, "Expected 19,858 test queries")
    require(len(taxonomy) == len(set(taxonomy)) == 500, "Invalid taxonomy")
    catalog_ids = [row["item_id"] for row in catalog]
    catalog_set = set(catalog_ids)
    require(len(catalog_set) == len(catalog_ids), "Duplicate catalog IDs")
    require(all(isinstance(item_id, str) and item_id for item_id in catalog_ids), "Invalid catalog IDs")
    for rows, label in ((train, "train"), (test, "test")):
        ids = [row["query_id"] for row in rows]
        require(len(ids) == len(set(ids)), f"Duplicate {label} query IDs")
        require(all(isinstance(query_id, str) and query_id for query_id in ids), f"Invalid {label} query IDs")
        require(all(len(row["reviews"]) == 5 for row in rows), f"Invalid {label} review count")
        require(all(isinstance(r.get("text"), str) for row in rows for r in row["reviews"]), f"Invalid {label} review text")
    require(not ({r["query_id"] for r in train} & {r["query_id"] for r in test}), "Overlapping train/test query IDs")
    require(all(row["item_id"] in catalog_set for row in train), "Unknown training product ID")
    require(all(row["category_path"] in set(taxonomy) for row in train), "Unknown training category")
    require(all("item_id" not in row and "category_path" not in row for row in test), "Unexpected test label fields")

    groups = defaultdict(list)
    for row in train:
        groups[row["item_id"]].append(row)
    by_category = defaultdict(list)
    for item_id, rows in groups.items():
        categories = {row["category_path"] for row in rows}
        require(len(categories) == 1, "Inconsistent category labels for one product")
        by_category[next(iter(categories))].append(item_id)
    rng = random.Random(SEED)
    valid_items = set()
    singleton_categories = []
    for category in sorted(by_category):
        items = sorted(by_category[category])
        rng.shuffle(items)
        if len(items) == 1:
            singleton_categories.append(category)
            continue
        n_valid = max(1, min(len(items) - 1, round(len(items) * 0.2)))
        valid_items.update(items[:n_valid])
    dev_train = [row for row in train if row["item_id"] not in valid_items]
    dev_valid = [row for row in train if row["item_id"] in valid_items]
    require(dev_train and dev_valid, "Empty development partition")
    require(not ({r["item_id"] for r in dev_train} & valid_items), "Development product leakage")

    train_hashes, test_hashes = review_hashes(train), review_hashes(test)
    fit_hashes, valid_hashes = set(review_hashes(dev_train)), set(review_hashes(dev_valid))
    split = {
        "seed": SEED,
        "strategy": "category-stratified product groups; approximately 20% validation per category",
        "source_sha256": fingerprints["train_gold.json"],
        "train_query_ids": [row["query_id"] for row in dev_train],
        "validation_query_ids": [row["query_id"] for row in dev_valid],
    }
    encoded_split = (json.dumps(split, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    audit = {
        "source_sha256": fingerprints,
        "counts": {
            "catalog_products": len(catalog), "training_queries": len(train),
            "test_queries": len(test), "taxonomy_categories": len(taxonomy),
            "training_unique_products": len(groups), "training_categories": len(by_category),
            "training_reviews": len(train_hashes), "test_reviews": len(test_hashes),
        },
        "catalog_empty_or_missing_fields": {
            field: sum(not str(row.get(field) or "").strip() for row in catalog)
            for field in ("title", "store", "features", "description")
        },
        "training_queries_per_category": dict(sorted(Counter(r["category_path"] for r in train).items())),
        "duplicate_review_texts": {
            "normalization": "collapse whitespace and strip; case preserved",
            "training_repeated_occurrences": len(train_hashes) - len(set(train_hashes)),
            "test_repeated_occurrences": len(test_hashes) - len(set(test_hashes)),
            "distinct_shared_train_test": len(set(train_hashes) & set(test_hashes)),
            "distinct_shared_development_partitions": len(fit_hashes & valid_hashes),
        },
        "development_split": {
            "seed": SEED, "training_queries": len(dev_train), "validation_queries": len(dev_valid),
            "training_products": len(groups) - len(valid_items), "validation_products": len(valid_items),
            "overlapping_products": 0, "singleton_categories_retained_in_training": singleton_categories,
            "manifest_sha256": hashlib.sha256(encoded_split).hexdigest(),
        },
        "checks": {
            "expected_counts": True, "unique_ids_within_files": True,
            "disjoint_train_test_query_ids": True, "five_reviews_per_query": True,
            "training_products_in_catalog": True, "training_categories_in_taxonomy": True,
            "test_labels_absent": True, "development_products_disjoint": True,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dev_split.json").write_bytes(encoded_split)
    (args.output_dir / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"counts": audit["counts"], "split": audit["development_split"],
                      "duplicates": audit["duplicate_review_texts"], "checks": audit["checks"]}, indent=2))


if __name__ == "__main__":
    main()
