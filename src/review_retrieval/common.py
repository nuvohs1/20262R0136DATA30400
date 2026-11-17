"""Shared local text preparation, input fingerprints, and prediction checks."""

import csv
import hashlib
import html
import json
from pathlib import Path
import re
import unicodedata


TEXT_FIELDS = ("title", "store", "features", "description")
PREPROCESSING_VERSION = "nfkc_html_media_lower_alnum_v1"
MEDIA = re.compile(r"\[\[(?:VIDEOID|VIDEO|IMAGEID|IMAGE):[^\]]*\]\]", re.I)
TAGS = re.compile(r"<[^>]*>")
TOKENS = re.compile(r"[a-z0-9]+")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenize(text):
    """Keep model numbers; remove markup/media IDs without external dictionaries."""
    text = unicodedata.normalize("NFKC", html.unescape(text or ""))
    text = MEDIA.sub(" ", text)
    text = TAGS.sub(" ", text)
    return TOKENS.findall(text.lower())


def product_tokens(product):
    return [token for field in TEXT_FIELDS for token in tokenize(product.get(field) or "")]


def query_tokens(query):
    if len(query["reviews"]) != 5:
        raise ValueError("Each query must contain exactly five reviews")
    return [token for review in query["reviews"] for token in tokenize(review["text"])]


def validate_submission(path, catalog_ids, expected_query_ids):
    """Reject wrong headers/order/count, duplicate candidates, and unknown IDs."""
    expected_query_ids = list(expected_query_ids)
    catalog_ids = set(catalog_ids)
    if len(expected_query_ids) != len(set(expected_query_ids)):
        raise ValueError("Expected query IDs must be unique")
    fields = ["query_id"] + [f"item_id_{rank}" for rank in range(1, 11)]
    count = 0
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise ValueError("Submission header must be query_id,item_id_1,...,item_id_10")
        for count, row in enumerate(reader, start=1):
            if set(row) != set(fields) or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row {count}")
            if count > len(expected_query_ids) or row["query_id"] != expected_query_ids[count - 1]:
                raise ValueError(f"Unexpected query ID or order at row {count}")
            predictions = [row[field] for field in fields[1:]]
            if len(set(predictions)) != 10:
                raise ValueError(f"Candidates must be distinct at row {count}")
            if not set(predictions) <= catalog_ids:
                raise ValueError(f"Unknown catalog ID at row {count}")
    if count != len(expected_query_ids):
        raise ValueError(f"Expected {len(expected_query_ids)} rows; found {count}")
    return {"rows": count, "candidates_per_row": 10, "valid": True, "sha256": sha256(path)}


def write_submission(path, query_ids, predictions, catalog_ids):
    path = Path(path)
    query_ids, predictions = list(query_ids), list(predictions)
    if len(query_ids) != len(predictions) or any(len(row) != 10 for row in predictions):
        raise ValueError("Predictions must contain exactly ten candidates for every query")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["query_id"] + [f"item_id_{rank}" for rank in range(1, 11)])
        writer.writerows([query_id, *items] for query_id, items in zip(query_ids, predictions))
    return validate_submission(path, catalog_ids, query_ids)
