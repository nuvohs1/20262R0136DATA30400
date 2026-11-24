from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
    from review_retrieval.rerank import recall_at_10, rerank_positions, row_zscore, select_weight
except ImportError:  # NumPy unavailable
    np = None


@unittest.skipIf(np is None, "NumPy is required")
class RerankTest(unittest.TestCase):
    def test_zscore_constant_row(self):
        z = row_zscore([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]])
        self.assertTrue(np.all(z[0] == 0))
        self.assertAlmostEqual(z[1].mean(), 0.0)

    def test_weights_zero_and_one(self):
        first = np.array([[3.0, 2.0, 1.0]])
        cross = np.array([[0.0, 1.0, 5.0]])
        self.assertEqual(rerank_positions(first, cross, 0.0, k=3).tolist(), [[0, 1, 2]])
        self.assertEqual(rerank_positions(first, cross, 1.0, k=3).tolist(), [[2, 1, 0]])

    def test_ties_keep_first_stage_order(self):
        first = np.array([[2.0, 2.0, 2.0]])
        cross = np.array([[1.0, 1.0, 1.0]])
        self.assertEqual(rerank_positions(first, cross, 0.5, k=3).tolist(), [[0, 1, 2]])

    def test_select_weight(self):
        items = [[f"i{j}" for j in range(12)]]
        first = np.arange(12, 0, -1, dtype=float)[None, :]   # first stage puts i11 last
        cross = np.zeros((1, 12)); cross[0, 11] = 10.0        # cross-encoder promotes i11
        self.assertEqual(recall_at_10(items, ["i11"], first, cross, 0.0), 0.0)
        best, curve = select_weight(items, ["i11"], first, cross)
        self.assertEqual(curve[1.0], 1.0)
        self.assertLess(best, 1.0 + 1e-9)
        self.assertEqual(curve[best], 1.0)


if __name__ == "__main__":
    unittest.main()
