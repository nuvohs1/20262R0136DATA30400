"""Labelled-review memory and the test-like proxy validation (NumPy only).

On test, many correct products already have five labelled reviews in train_gold.json (another
query of the same product). In the development split every product has exactly one query, so
no validation product ever has labelled reviews, and local validation cannot show the value
of that information. The proxy recreates the situation inside the tuning slice / validation:

  * a fixed share (MEMORY_SHARE = 0.5, a neutral prior: "unknown, either case equally likely";
    it is not estimated from test data) of the queries is chosen with random.Random(seed);
  * for each chosen query, 2 of its 5 reviews are moved into the labelled memory of its own
    product, and the query keeps the other 3;
  * every other query also keeps 3 reviews (the other 2 are dropped), so both halves have
    queries of the same length.
The memory reviews and the query reviews of a product never overlap.

Memory score: a product with labelled reviews gets the inner product between the query's mean
review embedding and the mean embedding of its labelled reviews. The score is z-normalized
over the products that have memory, and only the part above a threshold tau counts:
    bonus = mu * max(0, z_memory - tau)
which mirrors the thresholded BM25 review-expansion bonus (review_retrieval.bm25_v2.fuse).
"""

import random

import numpy as np

MEMORY_SHARE = 0.5
MEMORY_REVIEWS = 2
MUS = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0)
TAUS = (0.0, 1.0, 2.0, 3.0)


def build_proxy(query_ids, seed=42, share=MEMORY_SHARE, memory_reviews=MEMORY_REVIEWS, reviews=5):
    """Returns {query_id: {"query": [review indices], "memory": [review indices]}} (deterministic)."""
    ids = sorted(query_ids)
    rng = random.Random(seed)
    chosen = set(rng.sample(ids, round(share * len(ids))))
    proxy = {}
    for q in ids:
        order = list(range(reviews))
        rng.shuffle(order)
        held, kept = sorted(order[:memory_reviews]), sorted(order[memory_reviews:])
        proxy[q] = {"query": kept, "memory": held if q in chosen else []}
    return proxy


def memory_bonus(query_embedding, memory_matrix, mu, tau):
    """memory_matrix: (M, d) mean labelled-review embeddings of the M products with memory."""
    if not mu or len(memory_matrix) == 0:
        return np.zeros(len(memory_matrix))
    s = memory_matrix @ query_embedding
    std = s.std()
    z = (s - s.mean()) / (std if std > 1e-12 else 1.0)
    return mu * np.maximum(z - tau, 0.0)


def select_memory_weights(results, mus=MUS, taus=TAUS):
    """results[(mu, tau)] = tuning-slice Recall@10. Ties: smaller mu, then larger tau (less memory)."""
    return max(((m, t) for m in mus for t in taus), key=lambda k: (results[k], -k[0], k[1]))
