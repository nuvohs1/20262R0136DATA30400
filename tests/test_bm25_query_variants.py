"""Check repeated-phrase semantics and deterministic parallel query weighting."""

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from review_retrieval.bm25_query_variants import QueryBM25
from review_retrieval.bm25_variants import FieldBM25
import test_bm25_variants as fixtures
from test_retrieval import query, temporary_directory


class QueryVariantTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.VariantTests()
        fixture.setUp()
        self.catalog = fixture.catalog

    def test_query_frequency_multiplies_each_term_score(self):
        reference = FieldBM25(self.catalog, field_weights=(1, 1, 1, 1), max_document_frequency=1)
        weighted = QueryBM25(self.catalog, field_weights=(1, 1, 1, 1), max_document_frequency=1, query_tf="log_count")
        expected = {}
        tokens = ["blue"] * 8 + ["plate"] * 3
        for term, count in (("blue", 8), ("plate", 3)):
            factor = min(4, 1 + int(math.log2(count)))
            for item_id, score in reference.score_tokens([term], k=len(self.catalog)):
                expected[item_id] = expected.get(item_id, 0) + factor * score
        rows = weighted.score_tokens(tokens, k=len(self.catalog))
        for item_id, score in rows:
            self.assertAlmostEqual(score, expected[item_id], places=10)
        self.assertEqual([item for item, _ in rows], sorted(expected, key=lambda item: (-expected[item], item)))
        reference.close()
        weighted.close()

    def test_binary_mode_preserves_field_model(self):
        original = FieldBM25(self.catalog, max_document_frequency=1)
        weighted = QueryBM25(self.catalog, max_document_frequency=1, query_tf="binary")
        queries = [query("blue blue plate"), query("unknownword")]
        self.assertEqual(original.search(queries), weighted.search(queries))
        original.close()
        weighted.close()

    def test_weighted_parallel_matches_serial(self):
        model = QueryBM25(self.catalog, max_document_frequency=1, query_tf="log_count")
        queries = [query("blue blue blue cup plate"), query("unknownword")] * 9
        expected = model.search(queries)
        with temporary_directory() as directory:
            path = directory / "query.sqlite3"
            model.save_index(path)
            self.assertEqual(model.search_parallel(queries, path, workers=2), expected)
        model.close()


if __name__ == "__main__":
    unittest.main()
