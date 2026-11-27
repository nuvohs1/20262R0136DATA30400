"""Compare two submission files row by row (no data leaves the machine).

Reports how many rows have the same top 10 in the same order, the same top-10 set, and the same
top 1. Used to check a reproduced file against the submitted one.
"""

import argparse
import csv
from pathlib import Path


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        return {row[0]: row[1:] for row in reader}


def compare(a, b):
    if a.keys() != b.keys():
        raise ValueError("The files contain different query IDs")
    n = len(a)
    return {
        "rows": n,
        "same_order": sum(a[q] == b[q] for q in a),
        "same_set": sum(set(a[q]) == set(b[q]) for q in a),
        "same_top1": sum(a[q][0] == b[q][0] for q in a),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    print(compare(read_rows(args.first), read_rows(args.second)))


if __name__ == "__main__":
    main()
