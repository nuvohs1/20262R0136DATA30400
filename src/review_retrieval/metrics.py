"""Ranking metrics for queries with exactly one relevant product (pure Python).

With a single relevant item r at 1-based rank k (k = None if it is outside the list):
  Recall@N    = 1 if k <= N else 0              (the Kaggle metric at N = 10)
  Precision@N = Recall@N / N                    (at most one of the N results can be relevant)
  NDCG@N      = 1 / log2(k + 1) if k <= N        (ideal DCG = 1: the item at rank 1)
  MRR@N       = 1 / k if k <= N
These follow the standard graded definitions (DCG = sum g_i / log2(i + 1), nDCG = DCG / IDCG)
specialised to one relevant document with gain 1.
"""

import math


def rank_of(target, ranked):
    """1-based rank of target in ranked, or None."""
    for position, item in enumerate(ranked, start=1):
        if item == target:
            return position
    return None


def query_metrics(target, ranked, cutoffs=(1, 5, 10)):
    k = rank_of(target, ranked)
    out = {}
    for n in cutoffs:
        hit = k is not None and k <= n
        out[f"recall@{n}"] = 1.0 if hit else 0.0
        out[f"precision@{n}"] = (1.0 / n) if hit else 0.0
        out[f"ndcg@{n}"] = (1.0 / math.log2(k + 1)) if hit else 0.0
        out[f"mrr@{n}"] = (1.0 / k) if hit else 0.0
    return out


def mean_metrics(targets, rankings, cutoffs=(1, 5, 10)):
    """Average the per-query metrics; targets and rankings are aligned lists."""
    targets, rankings = list(targets), list(rankings)
    if len(targets) != len(rankings) or not targets:
        raise ValueError("targets and rankings must be non-empty and aligned")
    total = {}
    for target, ranked in zip(targets, rankings):
        for name, value in query_metrics(target, ranked, cutoffs).items():
            total[name] = total.get(name, 0.0) + value
    return {name: value / len(targets) for name, value in total.items()}
