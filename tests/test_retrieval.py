"""Independent numerical and submission-format checks on synthetic data only."""

from collections import Counter
import csv
from contextlib import contextmanager
import math
from pathlib import Path
import sys
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from review_retrieval.bm25 import BM25
from review_retrieval.common import product_tokens, query_tokens, tokenize, validate_submission, write_submission


def query(text):
    return {"reviews": [{"text": text}] + [{"text": ""} for _ in range(4)]}


@contextmanager
def temporary_directory():
    directory = Path(__file__).resolve().parents[1] / "tmp" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        for child in directory.iterdir():
            child.unlink()
        directory.rmdir()


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.catalog = [{"item_id": f"I{i:05d}", "title": text} for i, text in enumerate([
            "blue blue plate", "blue cup", "cup cup cup red", "green plate", "red plate plate bowl",
            "blue bowl", "green bowl bowl", "cup plate", "other", "", "blue plate",
        ])]

    def test_sqlite_bm25_matches_independent_scalar_formula(self):
        documents = [Counter(product_tokens(p)) for p in self.catalog]
        lengths = [sum(doc.values()) for doc in documents]
        average = sum(lengths) / len(lengths)
        model = BM25(list(reversed(self.catalog)), max_query_terms=0, max_document_frequency=1)
        queries = [query("blue blue plate"), query("cup bowl"), query("unknownword"), query("")]
        for q in queries:
            expected = {}
            for product, doc, length in zip(self.catalog, documents, lengths):
                score = 0
                for term in sorted(set(query_tokens(q))):
                    tf = doc[term]
                    df = sum(term in other for other in documents)
                    idf = math.log((len(documents) - df + 0.5) / (df + 0.5))
                    if idf <= 0:
                        idf = 1e-6
                    if tf:
                        score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / average))
                expected[product["item_id"]] = score
            scored, _ = model.search_one(q)
            for item_id, score in scored:
                self.assertAlmostEqual(score, expected[item_id], places=10)
            ordering = sorted(expected, key=lambda item_id: (-expected[item_id], item_id))[:10]
            self.assertEqual([item_id for item_id, _ in scored], ordering)
        self.assertEqual(model.search(queries)[0], model.search(queries)[0])
        model.close()

    def test_common_terms_have_positive_floor(self):
        catalog = [dict(p, title=p["title"] + " common") for p in self.catalog]
        model = BM25(catalog, max_query_terms=0, max_document_frequency=1)
        rows, _ = model.search_one(query("common"))
        self.assertTrue(all(0 < score < 1e-5 for _, score in rows))
        model.close()

    def test_preprocessing_removes_markup_and_media_but_keeps_numbers(self):
        self.assertEqual(tokenize("ＮＥＳＣＯ<br>RG-1400 &amp; Blue [[VIDEOID:secret42]]"),
                         ["nesco", "rg", "1400", "blue"])

    def test_submission_rejects_corruption(self):
        ids = [p["item_id"] for p in self.catalog]
        with temporary_directory() as directory:
            path = Path(directory) / "submission.csv"
            write_submission(path, ["Q0", "Q1"], [ids[:10], ids[1:]], ids)
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.reader(stream))
            for corrupted in (rows[:2], [rows[0], rows[2], rows[1]],
                              [rows[0], ["Q0", *([ids[0]] * 10)], rows[2]],
                              [rows[0], ["Q0", "UNKNOWN", *ids[1:10]], rows[2]],
                              [["wrong", *rows[0][1:]], *rows[1:]],
                              [rows[0], rows[1][:-1], rows[2]]):
                with path.open("w", newline="", encoding="utf-8") as stream:
                    csv.writer(stream).writerows(corrupted)
                with self.assertRaises(ValueError):
                    validate_submission(path, ids, ["Q0", "Q1"])

    def test_zero_match_ties_use_item_id_order(self):
        model = BM25(self.catalog)
        predictions, diagnostics = model.search([query("unseenword")])
        self.assertEqual(predictions[0], sorted(model.item_ids)[:10])
        self.assertEqual(diagnostics["zero_match_queries"], 1)
        model.close()

    def test_equal_scores_tie_by_item_id(self):
        catalog = [{"item_id": p["item_id"], "title": "lamp"} for p in reversed(self.catalog)]
        model = BM25(catalog, max_document_frequency=1)
        rows, _ = model.search_one(query("lamp"), k=3)
        self.assertEqual([item for item, _ in rows], sorted(p["item_id"] for p in catalog)[:3])
        self.assertEqual(len({score for _, score in rows}), 1)
        model.close()

    def test_query_filtering_uses_catalog_frequency(self):
        model = BM25(self.catalog, max_query_terms=1, max_document_frequency=0.2)
        rows, _ = model.search_one(query("blue blue other red"))
        self.assertEqual(rows[0][0], "I00008")
        predictions, diagnostics = model.search([query("blue")])
        self.assertEqual(diagnostics["zero_match_queries"], 1)
        self.assertEqual(predictions[0], sorted(model.item_ids)[:10])
        model.close()

    def test_parallel_and_serial_predictions_match(self):
        model = BM25(self.catalog, max_query_terms=0, max_document_frequency=1)
        queries = [query("blue plate"), query("cup bowl"), query("unknownword")] * 23
        expected = model.search(queries)
        with temporary_directory() as directory:
            path = directory / "index.sqlite3"
            model.save_index(path)
            self.assertEqual(model.search_parallel(queries, path, workers=2), expected)
        model.close()


if __name__ == "__main__":
    unittest.main()
