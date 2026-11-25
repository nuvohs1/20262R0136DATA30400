"""Approximate nearest-neighbour search with an inverted-file (IVF) index (NumPy only).

Build: spherical k-means splits the L2-normalized product embeddings into `nlist` clusters
(Voronoi cells); each product is stored in the list of its nearest centroid.
Search: a query is compared with the centroids, only the products in the `nprobe` closest
lists are scored exactly (inner product), and the best k of those are returned.
nprobe = nlist scores every product and equals exact search.

All randomness (k-means initialisation) comes from random.Random(seed).
"""

import random

import numpy as np


def _normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms > 0, norms, 1.0)


def spherical_kmeans(x, k, iterations=20, seed=42, batch=8192):
    """x: (N, d) float32, rows L2-normalized. Returns (centroids (k, d), assignment (N,), history)."""
    x = np.asarray(x, dtype=np.float32)
    centroids = x[sorted(random.Random(seed).sample(range(len(x)), k))].copy()
    history = []
    for _ in range(iterations + 1):
        assign = np.empty(len(x), dtype=np.int64)
        objective = 0.0
        for i in range(0, len(x), batch):
            sims = x[i:i + batch] @ centroids.T
            assign[i:i + batch] = sims.argmax(axis=1)
            objective += float(sims.max(axis=1).sum())
        history.append(objective / len(x))  # mean cosine similarity to the assigned centroid
        if len(history) == iterations + 1:
            break
        sums = np.zeros_like(centroids)
        np.add.at(sums, assign, x)
        empty = np.bincount(assign, minlength=k) == 0
        sums[empty] = centroids[empty]  # keep the previous centroid for an empty cluster
        centroids = _normalize(sums).astype(np.float32)
    return centroids, assign, history


class IVFIndex:
    def __init__(self, vectors, centroids, assignment):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.centroids = np.asarray(centroids, dtype=np.float32)
        order = np.argsort(assignment, kind="stable")
        bounds = np.searchsorted(assignment[order], np.arange(len(centroids) + 1))
        self.lists = [order[bounds[c]:bounds[c + 1]] for c in range(len(centroids))]

    def list_sizes(self):
        return np.array([len(l) for l in self.lists])

    def search(self, query, nprobe, k=10):
        """query: (d,). Returns (top-k product rows, number of products scored)."""
        query = np.asarray(query, dtype=np.float32)
        probe = np.argsort(-(self.centroids @ query), kind="stable")[:nprobe]
        rows = np.concatenate([self.lists[c] for c in probe])
        scores = self.vectors[rows] @ query
        k = min(k, len(rows))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.lexsort((rows[top], -scores[top]))]  # score descending, then row ascending
        return rows[top], len(rows)


def exact_search(vectors, query, k=10):
    scores = np.asarray(vectors, dtype=np.float32) @ np.asarray(query, dtype=np.float32)
    top = np.argpartition(-scores, k - 1)[:k]
    return top[np.lexsort((top, -scores[top]))]
