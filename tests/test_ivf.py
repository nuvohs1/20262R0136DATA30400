from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
    from review_retrieval.ivf import IVFIndex, exact_search, spherical_kmeans
except ImportError:  # NumPy unavailable
    np = None


def unit_vectors(n, d, seed):
    r = random.Random(seed)
    x = np.array([[r.gauss(0, 1) for _ in range(d)] for _ in range(n)], dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@unittest.skipIf(np is None, "NumPy is required")
class IVFTest(unittest.TestCase):
    def setUp(self):
        self.x = unit_vectors(300, 16, 0)
        self.q = unit_vectors(20, 16, 1)

    def test_kmeans_is_deterministic_and_improves(self):
        c1, a1, h1 = spherical_kmeans(self.x, 8, iterations=5, seed=42)
        c2, a2, _ = spherical_kmeans(self.x, 8, iterations=5, seed=42)
        self.assertTrue(np.array_equal(a1, a2) and np.array_equal(c1, c2))
        self.assertGreaterEqual(h1[-1], h1[0])
        self.assertTrue(np.allclose(np.linalg.norm(c1, axis=1), 1.0, atol=1e-5))

    def test_full_probe_equals_exact(self):
        c, a, _ = spherical_kmeans(self.x, 8, iterations=5, seed=42)
        index = IVFIndex(self.x, c, a)
        self.assertEqual(index.list_sizes().sum(), len(self.x))
        for q in self.q:
            rows, scored = index.search(q, 8, 10)
            self.assertEqual(scored, len(self.x))
            self.assertEqual(set(rows.tolist()), set(exact_search(self.x, q, 10).tolist()))

    def test_fewer_probes_score_fewer_products(self):
        c, a, _ = spherical_kmeans(self.x, 8, iterations=5, seed=42)
        index = IVFIndex(self.x, c, a)
        self.assertLess(index.search(self.q[0], 1, 10)[1], len(self.x))


if __name__ == "__main__":
    unittest.main()
