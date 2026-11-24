import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from review_retrieval.metrics import mean_metrics, query_metrics, rank_of


class MetricsTest(unittest.TestCase):
    def test_rank(self):
        self.assertEqual(rank_of("b", ["a", "b", "c"]), 2)
        self.assertIsNone(rank_of("z", ["a", "b"]))

    def test_single_query(self):
        m = query_metrics("c", ["a", "b", "c"] + [str(i) for i in range(7)])
        self.assertEqual(m["recall@1"], 0.0)
        self.assertEqual(m["recall@5"], 1.0)
        self.assertAlmostEqual(m["precision@5"], 0.2)
        self.assertAlmostEqual(m["precision@10"], 0.1)
        self.assertAlmostEqual(m["ndcg@10"], 1 / math.log2(4))
        self.assertAlmostEqual(m["mrr@10"], 1 / 3)
        self.assertEqual(query_metrics("a", ["a"])["ndcg@1"], 1.0)

    def test_miss_and_mean(self):
        self.assertEqual(sum(query_metrics("x", ["a", "b"]).values()), 0.0)
        m = mean_metrics(["a", "x"], [["a", "b"], ["a", "b"]])
        self.assertAlmostEqual(m["recall@10"], 0.5)
        self.assertAlmostEqual(m["mrr@10"], 0.5)


if __name__ == "__main__":
    unittest.main()
