from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
    from review_retrieval.memory import build_proxy, memory_bonus, select_memory_weights, MUS, TAUS
except ImportError:  # NumPy unavailable
    np = None


@unittest.skipIf(np is None, "NumPy is required")
class MemoryTest(unittest.TestCase):
    def test_proxy_is_deterministic_and_disjoint(self):
        ids = [f"Q{i:04d}" for i in range(400)]
        a, b = build_proxy(ids, seed=42), build_proxy(list(reversed(ids)), seed=42)
        self.assertEqual(a, b)
        self.assertEqual(sum(bool(v["memory"]) for v in a.values()), 200)
        for v in a.values():
            self.assertEqual(len(v["query"]), 3)
            self.assertIn(len(v["memory"]), (0, 2))
            self.assertFalse(set(v["query"]) & set(v["memory"]))
            self.assertTrue(set(v["query"]) <= set(range(5)))
        self.assertNotEqual(a, build_proxy(ids, seed=43))

    def test_bonus_is_thresholded(self):
        memory = np.eye(4, dtype=np.float32)
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.assertTrue(np.array_equal(memory_bonus(q, memory, 0.0, 0.0), np.zeros(4)))
        bonus = memory_bonus(q, memory, 2.0, 1.0)
        self.assertTrue((bonus >= 0).all())
        self.assertEqual(int(np.argmax(bonus)), 0)
        self.assertEqual(int((bonus > 0).sum()), 1)

    def test_selection_ties_prefer_less_memory(self):
        results = {(m, t): 0.5 for m in MUS for t in TAUS}
        self.assertEqual(select_memory_weights(results), (0.0, max(TAUS)))
        results[(2.0, 1.0)] = 0.6
        self.assertEqual(select_memory_weights(results), (2.0, 1.0))


if __name__ == "__main__":
    unittest.main()
