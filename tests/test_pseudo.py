from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
    from review_retrieval.pseudo import choose_threshold, negatives_for, precision_curve, top_and_margin
except ImportError:  # NumPy unavailable
    np = None


@unittest.skipIf(np is None, "NumPy is required")
class PseudoLabelTest(unittest.TestCase):
    def test_top_and_margin(self):
        first = np.array([[3.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
        cross = np.array([[3.0, 2.0, 1.0], [0.0, 5.0, 0.0]])
        top, margin = top_and_margin(first, cross, 0.5)
        self.assertEqual(top.tolist(), [0, 1])
        self.assertTrue(np.all(margin > 0))
        _, tie = top_and_margin(np.ones((1, 3)), np.ones((1, 3)), 0.5)
        self.assertEqual(tie.tolist(), [0.0])

    def test_threshold_reaches_precision(self):
        margins = [0.1, 0.2, 1.0, 1.5, 2.0]
        correct = [False, False, True, True, False]
        curve = precision_curve(margins, correct, grid=(0.0, 0.5, 1.6, 2.5))
        self.assertEqual(curve[0.0]["coverage"], 1.0)
        self.assertAlmostEqual(curve[0.5]["precision"], 2 / 3, places=4)
        self.assertIsNone(curve[2.5]["precision"])
        with self.assertRaises(ValueError):
            choose_threshold(curve, 0.9)
        curve = precision_curve(margins, [False, False, True, True, True], grid=(0.0, 0.5, 1.6))
        self.assertEqual(choose_threshold(curve, 0.9), 0.5)

    def test_negatives_exclude_positive(self):
        self.assertEqual(negatives_for(["a", "b", "c", "d"], [4, 3, 2, 1], "b", 2), ["a", "c"])


if __name__ == "__main__":
    unittest.main()
