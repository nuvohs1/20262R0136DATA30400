"""Self-training with pseudo-labels for the reranker (NumPy only).

An unlabeled query (five reviews, no product) receives a pseudo-label when the current
multi-stage ranker is confident about its top product:
    final       = (1 - w) * z(stage 1) + w * z(cross-encoder)      (review_retrieval.rerank)
    pseudo item = argmax final
    confidence  = final[top 1] - final[top 2]                      (margin, in z units)
Queries whose margin reaches a threshold are added to the cross-encoder's training data with
the pseudo item as the positive. The threshold is the smallest margin at which pseudo-label
precision on the labelled tuning slice reaches a fixed target (default 0.9, the confidence
level used in the course's self-training example). Validation and test labels are never used.
"""

import numpy as np

from .rerank import row_zscore

MARGIN_GRID = tuple(round(0.05 * i, 2) for i in range(81))  # 0.00 ... 4.00


def fused_scores(first, cross, weight):
    return (1.0 - weight) * row_zscore(first) + weight * row_zscore(cross)


def top_and_margin(first, cross, weight):
    """Returns (top-1 candidate position, margin between the best two fused scores) per query."""
    final = fused_scores(first, cross, weight)
    order = np.argsort(-final, axis=1, kind="stable")
    rows = np.arange(final.shape[0])
    return order[:, 0], final[rows, order[:, 0]] - final[rows, order[:, 1]]


def precision_curve(margins, correct, grid=MARGIN_GRID):
    """Pseudo-label precision and coverage for each margin threshold (labelled queries only)."""
    margins, correct = np.asarray(margins), np.asarray(correct, dtype=bool)
    curve = {}
    for t in grid:
        chosen = margins >= t
        n = int(chosen.sum())
        curve[t] = {"coverage": round(n / len(margins), 5),
                    "precision": round(float(correct[chosen].mean()), 5) if n else None}
    return curve


def choose_threshold(curve, min_precision):
    """Smallest threshold whose precision reaches min_precision (most pseudo-labels at that quality)."""
    for t in sorted(curve):
        if curve[t]["precision"] is not None and curve[t]["precision"] >= min_precision:
            return t
    raise ValueError(f"No threshold reaches precision {min_precision}")


def negatives_for(items, first_scores, positive, pool):
    """Hard negatives: the stage-1 candidates in stage-1 order, without the pseudo positive."""
    order = np.argsort(-np.asarray(first_scores), kind="stable")
    return [items[i] for i in order if items[i] != positive][:pool]
