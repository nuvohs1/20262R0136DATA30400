"""Field weighting and fusion checks, using exclusively synthetic examples."""

from collections import Counter
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from review_retrieval.bm25 import BM25
from review_retrieval.bm25_variants import FieldBM25
from review_retrieval.common import TEXT_FIELDS, query_tokens, tokenize
from test_retrieval import query, temporary_directory


class VariantTests(unittest.TestCase):
    def setUp(self):
        self.catalog = [{"item_id": f"I{i:05d}", "title": title, "store": store,
                         "features": features, "description": description}
                        for i, (title, store, features, description) in enumerate([
                            ("blue plate", "acme", "blue", "large plate"),
                            ("red cup", "blue", "small", "blue cup cup"),
                            ("bowl", "acme", "round", "blue blue bowl"),
                            ("blue cup", "acme", "red", "small red"),
                            ("wooden tray", "brand", "large", "blue plate"),
                            ("green dish", "brand", "small", ""),
                            ("cloth", "", "", "white"),
                            ("lamp", "", "", "green"),
                            ("rug", "", "", ""), ("", "", "", ""),
                            ("pot", "", "", "small"),
                        ])]

    def test_uniform_fields_match_original_concatenated_baseline(self):
        original = BM25(self.catalog, max_query_terms=0, max_document_frequency=1)
        fields = FieldBM25(self.catalog, field_weights=(1, 1, 1, 1), max_query_terms=0, max_document_frequency=1)
        queries = [query("blue plate"), query("blue cup"), query("unknownword")]
        self.assertEqual(original.search(queries)[0], fields.search(queries)[0]["concat"])
        for q in queries[:2]:
            expected = original.search_one(q)[0]
            actual = fields.score_tokens(query_tokens(q))
            for (item_a, score_a), (item_b, score_b) in zip(expected, actual):
                self.assertEqual(item_a, item_b)
                self.assertAlmostEqual(score_a, score_b, places=10)
        original.close()
        fields.close()

    def test_weighted_scores_match_independent_scalar_formula(self):
        weights = (4, 2, 2, 0.5)
        model = FieldBM25(self.catalog, field_weights=weights, max_query_terms=0, max_document_frequency=1)
        documents = [[Counter(tokenize(p[field])) for field in TEXT_FIELDS] for p in self.catalog]
        lengths = [sum(sum(field.values()) for field in document) for document in documents]
        average = sum(lengths) / len(lengths)
        expected = {}
        for product, document, length in zip(self.catalog, documents, lengths):
            score = 0
            for term in ("blue", "plate"):
                tf = sum(weight * field[term] for weight, field in zip(weights, document))
                df = sum(any(term in field for field in other) for other in documents)
                idf = math.log((len(documents) - df + 0.5) / (df + 0.5))
                idf = idf if idf > 0 else 1e-6
                score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / average))
            expected[product["item_id"]] = score
        actual = model.score_tokens(["blue", "plate"])
        for item_id, score in actual:
            self.assertAlmostEqual(score, expected[item_id], places=10)
        ordered = sorted(expected, key=lambda item_id: (-expected[item_id], item_id))[:10]
        self.assertEqual(model.finish(actual), ordered)
        model.close()

    def test_rank_cursor_collects_all_boundary_ties(self):
        catalog = [{"item_id": f"I{i:05d}", "title": "lamp"} for i in range(40, -1, -1)]
        model = FieldBM25(catalog, max_document_frequency=1)
        self.assertEqual(model.finish(model.score_tokens(["lamp"])), sorted(p["item_id"] for p in catalog)[:10])
        model.close()

    def test_rrf_uses_rank_weights_and_excludes_empty_views(self):
        model = FieldBM25(self.catalog, rrf_constant=60)
        lists = [[("A", 100), ("B", 1)], [], [("B", 900), ("C", 0.1)]]
        fused = dict(model.fuse(lists, [2, 1, 1]))
        self.assertAlmostEqual(fused["A"], 2 / 61)
        self.assertAlmostEqual(fused["B"], 2 / 62 + 1 / 61)
        self.assertAlmostEqual(fused["C"], 1 / 62)
        self.assertEqual(model.fuse([[], []]), [])
        model.close()

    def test_all_modes_match_between_serial_and_parallel(self):
        model = FieldBM25(self.catalog, max_document_frequency=1)
        queries = [query("blue plate"), query("unseenword"), query("cup")] * 7
        modes = ("concat", "review_rrf", "hybrid_rrf")
        expected = model.search(queries, modes)
        with temporary_directory() as directory:
            path = directory / "fields.sqlite3"
            model.save_index(path)
            self.assertEqual(model.search_parallel(queries, path, modes, workers=2), expected)
        for predictions in expected[0].values():
            self.assertTrue(all(len(row) == len(set(row)) == 10 for row in predictions))
        model.close()


if __name__ == "__main__":
    unittest.main()
