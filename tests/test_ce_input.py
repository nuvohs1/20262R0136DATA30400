from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
    from review_retrieval.ce_input import allocate, assemble, sample_reviews
except ImportError:  # NumPy unavailable
    np = None


@unittest.skipIf(np is None, "NumPy is required")
class CrossEncoderInputTest(unittest.TestCase):
    def test_allocate_water_fills(self):
        self.assertEqual(allocate([10, 20, 30], 100), [10, 20, 30])
        caps = allocate([10, 200, 300, 15, 400], 320)
        self.assertEqual(sum(caps), 320)
        self.assertEqual(caps[0], 10)
        self.assertEqual(caps[3], 15)
        self.assertLessEqual(max(caps) - min(c for c, n in zip(caps, [10, 200, 300, 15, 400]) if c < n), 1)
        self.assertEqual(allocate([100] * 5, 320), [64] * 5)

    def test_assemble_layout(self):
        reviews = [np.arange(1, 6), np.arange(10, 100)]
        out = assemble(reviews, 0, budget=20)
        self.assertEqual(len(out), 21)
        self.assertEqual(list(out[:6]), [1, 2, 3, 4, 5, 0])
        old = assemble(reviews, 0, cap=3)
        self.assertEqual(list(old), [1, 2, 3, 0, 10, 11, 12])

    def test_sampling_is_seeded(self):
        a = [sample_reviews(5, random.Random(42)) for _ in range(3)]
        rng1, rng2 = random.Random(7), random.Random(7)
        draws1 = [sample_reviews(5, rng1) for _ in range(200)]
        draws2 = [sample_reviews(5, rng2) for _ in range(200)]
        self.assertEqual(draws1, draws2)
        self.assertEqual(a[0], a[1])
        self.assertTrue(all(len(d) in (3, 4, 5) and len(set(d)) == len(d) for d in draws1))
        self.assertTrue(any(len(d) == 5 for d in draws1) and any(len(d) < 5 for d in draws1))


if __name__ == "__main__":
    unittest.main()
