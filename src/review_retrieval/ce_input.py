"""Review-side input construction for the cross-encoder (NumPy only, no tokenizer).

Two changes over the fixed per-review cut used by the first cross-encoders (64 tokens each):

- Shared review budget. The reviews of a query share one token budget. Short reviews leave their
  unused tokens to long ones: each review gets min(length, cap), with the largest cap that keeps
  the total within the budget (water filling). With 5 reviews and a budget of 320, the total
  query length never exceeds the old 5 x 64 layout, so the product passage keeps the same room.
- Review augmentation (training only). Each time a query is drawn, its reviews are shuffled and,
  with probability 1 - keep_all, only 3 or 4 of them are kept. The model cannot rely on review
  order and also learns from shorter queries. A seeded random.Random drives it, so it is
  reproducible.
"""

import numpy as np


def allocate(lengths, budget):
    """Per-review token caps: min(length, c) with the largest integer c whose total fits the budget."""
    lengths = [int(n) for n in lengths]
    if sum(lengths) <= budget:
        return lengths
    lo, hi = 0, max(lengths)
    while lo < hi:  # largest c with sum(min(n, c)) <= budget
        mid = (lo + hi + 1) // 2
        if sum(min(n, mid) for n in lengths) <= budget:
            lo = mid
        else:
            hi = mid - 1
    caps = [min(n, lo) for n in lengths]
    spare = budget - sum(caps)  # hand out leftover tokens one by one, in order, to reviews still cut
    for i, n in enumerate(lengths):
        if spare and caps[i] < n:
            caps[i] += 1
            spare -= 1
    return caps


def assemble(reviews, sep_id, budget=None, cap=None):
    """reviews: list of int arrays (token IDs) -> one array 'r1 [SEP] r2 [SEP] ... rk'.
    budget: shared budget for the review tokens (separators not counted); cap: fixed cut per review."""
    if budget is not None:
        caps = allocate([len(r) for r in reviews], budget)
    else:
        caps = [cap if cap is not None else len(r) for r in reviews]
    sep = np.asarray([sep_id], dtype=np.int32)
    parts = []
    for j, (r, c) in enumerate(zip(reviews, caps)):
        if j:
            parts.append(sep)
        parts.append(np.asarray(r[:c], dtype=np.int32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)


def sample_reviews(n, rng, keep_all=0.5, smaller=(3, 4)):
    """Indices of the reviews to use for one training draw: shuffled; all n with probability
    keep_all, else a random size from `smaller`."""
    order = list(range(n))
    rng.shuffle(order)
    if rng.random() >= keep_all:
        order = order[:min(n, rng.choice(smaller))]
    return order
