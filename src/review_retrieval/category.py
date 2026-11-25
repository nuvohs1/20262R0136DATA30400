"""Category labels as auxiliary information (NumPy only).

The training file gives each labelled query (and thus its product) a 4-level category path
from category_taxonomy.txt (500 leaves). Catalog products have no category field, so a
classifier predicts category distributions for queries (from their reviews) and for products
(from their metadata); see scripts/train_category.py. A candidate's category score is
    cat(q, p) = log( sum_c P(c | q) * P(c | p) + eps )
the log-probability that query and product fall into the same leaf category. It joins the
reranking fusion as a third, z-normalized term:
    final = (1 - w) * z(stage 1) + w * z(cross-encoder) + beta * z(category)
"""

import numpy as np

from .rerank import WEIGHTS, row_zscore

BETAS = tuple(round(0.1 * i, 1) for i in range(11))
SEPARATOR = " > "


def build_taxonomy(paths):
    """paths: leaf category paths. Returns (node names, leaf node indices, edges (child, parent)).

    Nodes are all path prefixes (levels 1-4), sorted for a fixed order; leaves keep the order of
    sorted(paths), which is also the class order of the classifier.
    """
    leaves = sorted(set(paths))
    nodes = sorted({SEPARATOR.join(p.split(SEPARATOR)[:d]) for p in leaves for d in range(1, len(p.split(SEPARATOR)) + 1)})
    index = {n: i for i, n in enumerate(nodes)}
    edges = []
    for n in nodes:
        parts = n.split(SEPARATOR)
        if len(parts) > 1:
            edges.append((index[n], index[SEPARATOR.join(parts[:-1])]))
    return nodes, [index[p] for p in leaves], edges


def normalized_adjacency(num_nodes, edges):
    """D^-1/2 (A + I) D^-1/2 of the undirected taxonomy tree (GCN propagation matrix)."""
    a = np.eye(num_nodes)
    for child, parent in edges:
        a[child, parent] = a[parent, child] = 1.0
    d = 1.0 / np.sqrt(a.sum(axis=1))
    return a * d[:, None] * d[None, :]


def match_scores(query_probs, product_probs, candidate_positions, eps=1e-9):
    """query_probs (Q, C); product_probs (N, C); candidate_positions (Q, K) catalog rows -> (Q, K)."""
    q = np.asarray(query_probs, dtype=np.float64)
    p = np.asarray(product_probs)
    out = np.empty(np.asarray(candidate_positions).shape)
    for n, row in enumerate(candidate_positions):
        out[n] = np.log(p[row].astype(np.float64) @ q[n] + eps)
    return out


def fused_positions(first, cross, cat, weight, beta, k=10):
    final = (1.0 - weight) * row_zscore(first) + weight * row_zscore(cross) + beta * row_zscore(cat)
    return np.argsort(-final, axis=1, kind="stable")[:, :k]


def select_weights(items, gold, first, cross, cat, weights=WEIGHTS, betas=BETAS):
    """Best (w, beta) by Recall@10; ties prefer the smaller beta, then the smaller w."""
    curve = {}
    for beta in betas:
        for w in weights:
            order = fused_positions(first, cross, cat, w, beta)
            curve[(w, beta)] = float(np.mean([g in {row[p] for p in pos} for row, pos, g in zip(items, order, gold)]))
    best = max(curve, key=lambda wb: (curve[wb], -wb[1], -wb[0]))
    return best, curve
