"""Synthetic checks for the NumPy BM25F scorer, RSJ weights, fusion, and tie-breaking."""

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import numpy as np
    from review_retrieval.bm25_v2 import fuse, zscore
    from review_retrieval.sparse_bm25 import SparseBM25, light_stem, top_k
except ModuleNotFoundError:
    np = None


DOCS = [
    ["blue blue plate", "acme", "", ""],
    ["blue cup", "", "ceramic cup", ""],
    ["cup cup cup red", "zeta", "", "a red cup"],
    ["green plate", "", "", ""],
    ["red plate plate bowl", "acme", "", ""],
    ["blue bowl", "", "", ""],
    ["green bowl bowl", "", "", ""],
    ["cup plate", "", "", ""],
    ["other", "", "", ""],
    ["", "", "", ""],
    ["blue plate", "", "", ""],
]


def fields():
    return [[text.split() for text in doc] for doc in DOCS]


def reviews(text):
    return [text.split(), [], [], [], []]


@unittest.skipIf(np is None, "NumPy is required for BM25 v2")
class SparseBM25Tests(unittest.TestCase):
    def setUp(self):
        self.ids = [f"I{i:02d}" for i in range(len(DOCS))]
        self.model = SparseBM25(self.ids, fields())

    def scalar(self, query_terms, k1, b, weights):
        n = len(DOCS)
        docs = fields()
        lengths = np.array([[len(f) for f in doc] for doc in docs], dtype=float)
        averages = lengths.mean(axis=0)
        scores = np.zeros(n)
        for term in query_terms:
            df = sum(any(term in f for f in doc) for doc in docs)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for d, doc in enumerate(docs):
                tf = sum(weights[f] * doc[f].count(term) / (1 - b + b * lengths[d, f] / (averages[f] or 1))
                         for f in range(4) if doc[f].count(term))
                scores[d] += idf * tf * (k1 + 1) / (tf + k1) if tf else 0.0
        return scores

    def test_bm25f_matches_scalar_formula(self):
        for k1, b, weights in ((1.2, 0.75, (1, 1, 1, 1)), (4.0, 1.0, (5, 1, 1, 0.5)), (0.9, 0.3, (2, 0, 1, 1))):
            self.model.fit(k1, b, weights)
            got = self.model.score_query(reviews("blue cup acme"), mode="binary")
            np.testing.assert_allclose(got, self.scalar(["blue", "cup", "acme"], k1, b, weights), rtol=1e-5)

    def test_query_modes_and_selection(self):
        self.model.fit()
        terms, weights = self.model.query_terms(reviews("cup cup cup blue unknownword"), mode="log")
        vocab = {v: k for k, v in self.model.vocabulary.items()}
        self.assertEqual([vocab[t] for t in terms], ["blue", "cup"])
        self.assertAlmostEqual(weights[1], 1 + math.log(3))
        terms, _ = self.model.query_terms(reviews("cup blue zeta"), mode="binary", max_terms=1)
        self.assertEqual([vocab[t] for t in terms], ["zeta"])  # rarest term has the highest IDF

    def test_relevance_weights_reward_terms_that_match_targets(self):
        self.model.fit()
        # "cup" in reviews always co-occurs with a target containing "cup"; "blue" never does.
        pairs = [(reviews("cup blue"), 2), (reviews("cup blue"), 7), (reviews("cup blue"), 1)]
        weights = self.model.relevance_weights([p[0] for p in pairs], [p[1] for p in pairs], prior_strength=1.0)
        vocab = self.model.vocabulary
        self.assertGreater(weights[vocab["cup"]], 0)
        self.assertGreater(weights[vocab["cup"]], weights[vocab["blue"]] - 1e-12)
        self.assertTrue(np.all(weights >= 0))

    def test_top_k_breaks_ties_by_item_order_and_respects_mask(self):
        scores = np.array([1.0, 3.0, 3.0, 0.5, 3.0, 2.0])
        self.assertEqual(top_k(scores, 3).tolist(), [1, 2, 4])
        self.assertEqual(top_k(scores, 4).tolist(), [1, 2, 4, 5])
        mask = np.array([True, False, True, True, True, True])
        self.assertEqual(top_k(scores, 3, mask).tolist(), [2, 4, 5])

    def test_fusion_only_rewards_outstanding_review_matches(self):
        meta = np.array([0.0, 1.0, 2.0, 3.0])
        review = np.array([0.0, 0.0, 10.0])
        positions = np.array([0, 1, 2])
        fused = fuse(meta, review, positions, weight=1.0, threshold=0.5)
        expected = zscore(meta)
        expected[2] += max(zscore(review)[2] - 0.5, 0)
        np.testing.assert_allclose(fused, expected)
        np.testing.assert_allclose(fuse(meta, review, positions, weight=0.0, threshold=0.0), zscore(meta))

    def test_light_stem_is_conservative(self):
        self.assertEqual(light_stem("curtains"), "curtain")
        self.assertEqual(light_stem("batteries"), "battery")
        self.assertEqual(light_stem("boxes"), "box")
        self.assertEqual(light_stem("glass"), "glass")
        self.assertEqual(light_stem("x100s"), "x100s")


if __name__ == "__main__":
    unittest.main()
