from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import numpy as np
    from learn_fusion import features, fit_listwise
except ImportError:  # NumPy unavailable
    np = None


@unittest.skipIf(np is None, "NumPy is required")
class FusionTest(unittest.TestCase):
    def test_listwise_fit_finds_the_informative_feature(self):
        rng = random.Random(42)
        n, k = 300, 10
        x = np.array([[[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(k)] for _ in range(n)])
        gold = np.argmax(x[:, :, 0] + 0.3 * np.array([[rng.gauss(0, 1) for _ in range(k)] for _ in range(n)]), axis=1)
        w = fit_listwise(x, gold)
        self.assertGreater(w[0], 1.0)
        self.assertLess(abs(w[1]), 0.5 * w[0])
        self.assertTrue(np.array_equal(w, fit_listwise(x, gold)))  # deterministic

    def test_unmatched_queries_are_skipped_and_features_shape(self):
        first = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
        x = features(first, [first, -first])
        self.assertEqual(x.shape, (2, 3, 4))
        self.assertAlmostEqual(x[0, 0, 3], 1.0)  # rank 0 -> 1 / log2(2)
        self.assertAlmostEqual(x[1, 2, 3], 1.0)
        w = fit_listwise(x, np.array([0, -1]))
        self.assertTrue(np.isfinite(w).all())


if __name__ == "__main__":
    unittest.main()
