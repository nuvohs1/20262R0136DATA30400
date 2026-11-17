"""Optional array backend must match SQLite scores, rankings, and ties."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from review_retrieval.bm25_numpy import NumpyBM25
except ModuleNotFoundError:
    NumpyBM25 = None
from review_retrieval.bm25_query_variants import QueryBM25
import test_bm25_variants as fixtures
from test_retrieval import query


@unittest.skipIf(NumpyBM25 is None, "Optional NumPy backend not installed")
class NumpyTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.VariantTests()
        fixture.setUp()
        self.catalog = fixture.catalog

    def test_scores_and_rankings_match_sqlite(self):
        for weights in ((1, 1, 1, 1), (4, 2, 2, 0.5)):
            for frequency in ("binary", "log_count"):
                config = dict(field_weights=weights, query_tf=frequency, max_document_frequency=1)
                scalar = QueryBM25(self.catalog, **config)
                fast = NumpyBM25(self.catalog, **config)
                for tokens in (["blue", "plate"], ["blue"] * 8 + ["plate"] * 3, ["unknownword"]):
                    a, b = scalar.score_tokens(tokens), fast.score_tokens(tokens)
                    self.assertEqual([item for item, _ in a], [item for item, _ in b])
                    for (_, score_a), (_, score_b) in zip(a, b):
                        self.assertAlmostEqual(score_a, score_b, places=10)
                queries = [query("blue blue plate"), query("unknownword")]
                self.assertEqual(scalar.search(queries)[0], fast.search(queries)[0])
                scalar.close()
                fast.close()

    def test_uniform_score_ties_and_zero_match(self):
        catalog = [{"item_id": f"I{i:05d}", "title": "lamp"} for i in range(40, -1, -1)]
        model = NumpyBM25(catalog, max_document_frequency=1)
        predictions, _ = model.search([query("lamp"), query("unseenword")])
        self.assertEqual(predictions["concat"][0], sorted(p["item_id"] for p in catalog)[:10])
        self.assertEqual(predictions["concat"][0], predictions["concat"][1])
        model.close()
