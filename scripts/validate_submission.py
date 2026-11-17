"""Validate the exact course test submission format without transmitting data."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from review_retrieval.common import read_json, validate_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "project_release")
    args = parser.parse_args()
    catalog = read_json(args.data_dir / "catalog.json")
    test = read_json(args.data_dir / "test_queries.json")
    if len(catalog) != 50_000 or len(test) != 19_858:
        raise ValueError("Expected the complete original catalog and test queries")
    print(validate_submission(args.submission, [p["item_id"] for p in catalog], [q["query_id"] for q in test]))


if __name__ == "__main__":
    main()
