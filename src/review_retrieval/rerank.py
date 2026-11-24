"""Second-stage fusion for reranking (NumPy only).

For each query the first stage supplies K candidates with scores; the cross-encoder adds a
score per candidate. Both are z-normalized within the query's K candidates and mixed:
    final = (1 - w) * z(first stage) + w * z(cross-encoder)
w = 0 keeps the first-stage order, w = 1 is the pure cross-encoder order. Ties keep the
first-stage order (stable sort over candidates already ranked by the first stage).
"""

import numpy as np

WEIGHTS = tuple(round(0.1 * i, 1) for i in range(11))


def row_zscore(x):
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return np.divide(x - mean, std, out=np.zeros_like(x), where=std > 0)


def rerank_positions(first, cross, weight, k=10):
    """first, cross: (queries, K). Returns (queries, k) candidate positions in final order."""
    final = (1.0 - weight) * row_zscore(first) + weight * row_zscore(cross)
    return np.argsort(-final, axis=1, kind="stable")[:, :k]


def recall_at_10(items, gold, first, cross, weight):
    """items: list of K-item lists; gold: list of correct item IDs."""
    order = rerank_positions(first, cross, weight)
    return float(np.mean([g in {row[p] for p in pos} for row, pos, g in zip(items, order, gold)]))


def select_weight(items, gold, first, cross, weights=WEIGHTS):
    """Best weight by Recall@10 (ties: the smaller weight, i.e. closer to the first stage)."""
    scores = {w: recall_at_10(items, gold, first, cross, w) for w in weights}
    best = max(weights, key=lambda w: (scores[w], -w))
    return best, scores
