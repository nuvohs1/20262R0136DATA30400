from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import numpy as np
    from make_final import ensemble_ranking
except ImportError:  # NumPy unavailable
    np = None
from compare_submissions import compare


@unittest.skipIf(np is None, "NumPy is required")
class FinalTest(unittest.TestCase):
    def test_weight_zero_keeps_first_stage_order(self):
        items = [[f"P{i}" for i in range(12)]]
        first = [list(range(12, 0, -1))]
        a = np.arange(12, dtype=float)[None]
        self.assertEqual(ensemble_ranking(items, first, a, a, 0.0)[0], items[0][:10])

    def test_ensemble_is_mean_of_zscores(self):
        items = [[f"P{i}" for i in range(12)]]
        first = [[0.0] * 12]
        rerank = np.arange(12, dtype=float)[None]        # prefers P11
        hardneg = 100 * np.arange(12, dtype=float)[None]  # same order, other scale
        self.assertEqual(ensemble_ranking(items, first, rerank, hardneg, 1.0)[0][0], "P11")
        hardneg = -1000 * np.arange(12, dtype=float)[None]  # exact opposite -> z-scores cancel
        self.assertEqual(ensemble_ranking(items, first, rerank, hardneg, 1.0)[0], items[0][:10])


class CompareTest(unittest.TestCase):
    def test_compare_counts(self):
        a = {"Q1": ["A", "B"], "Q2": ["C", "D"]}
        b = {"Q1": ["A", "B"], "Q2": ["D", "C"]}
        self.assertEqual(compare(a, b), {"rows": 2, "same_order": 1, "same_set": 2, "same_top1": 1})


if __name__ == "__main__":
    unittest.main()
