from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
    from review_retrieval.category import (build_taxonomy, fused_positions, match_scores, normalized_adjacency,
                                           select_weights)
    from review_retrieval.rerank import rerank_positions
except ImportError:  # NumPy unavailable
    np = None


@unittest.skipIf(np is None, "NumPy is required")
class CategoryTest(unittest.TestCase):
    def test_taxonomy(self):
        nodes, leaves, edges = build_taxonomy(["A > B > C", "A > B > D", "E > F > G"])
        self.assertEqual(len(nodes), 7)  # A, A>B, A>B>C, A>B>D, E, E>F, E>F>G
        self.assertEqual([nodes[i] for i in leaves], ["A > B > C", "A > B > D", "E > F > G"])
        self.assertEqual(len(edges), 5)
        adj = normalized_adjacency(len(nodes), edges)
        self.assertTrue(np.allclose(adj, adj.T))
        self.assertTrue(np.all(np.diag(adj) > 0))

    def test_match_scores(self):
        query = np.array([[0.9, 0.1]])
        products = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        scores = match_scores(query, products, [[0, 1, 2]])
        self.assertTrue(scores[0, 0] > scores[0, 2] > scores[0, 1])

    def test_beta_zero_equals_rerank(self):
        rng = np.random.default_rng(0)
        first, cross, cat = rng.normal(size=(5, 12)), rng.normal(size=(5, 12)), rng.normal(size=(5, 12))
        self.assertEqual(fused_positions(first, cross, cat, 0.5, 0.0).tolist(), rerank_positions(first, cross, 0.5).tolist())

    def test_select_weights_prefers_category_when_it_helps(self):
        items = [[f"i{j}" for j in range(12)]]
        first = np.arange(12, 0, -1, dtype=float)[None, :]
        cross = first.copy()
        cat = np.zeros((1, 12)); cat[0, 11] = 10.0  # only the category score finds i11
        (w, beta), curve = select_weights(items, ["i11"], first, cross, cat)
        self.assertGreater(beta, 0.0)
        self.assertEqual(curve[(w, beta)], 1.0)
        self.assertEqual(curve[(0.5, 0.0)], 0.0)


if __name__ == "__main__":
    unittest.main()
