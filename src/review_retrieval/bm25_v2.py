"""Improved BM25 pipeline: tuned BM25F + supervised RSJ query weights + review expansion.

All statistics that use labels (RSJ weights, the review-expansion index) are built
only from the queries passed in as `labelled`. For validation this is the
development-training partition; for the test submission it is all of train_gold.json.
"""

from concurrent.futures import ProcessPoolExecutor
import numpy as np

from .sparse_bm25 import SparseBM25, top_k


def zscore(scores):
    std = scores.std()
    return (scores - scores.mean()) / (std if std > 1e-12 else 1.0)


def fuse(meta_scores, review_scores, review_positions, weight, threshold):
    """Metadata z-score plus a thresholded bonus for products with similar labelled reviews.

    Only review matches that stand out from the review index (z above `threshold`)
    contribute, so ordinary review-to-review overlap does not displace products that
    have no labelled reviews.
    """
    fused = zscore(meta_scores)
    if review_scores is not None and weight:
        bonus = np.maximum(zscore(review_scores) - threshold, 0.0) * weight
        fused[review_positions] += bonus
    return fused


class Retriever:
    def __init__(self, config, item_ids, catalog_fields):
        self.config = config
        self.item_ids = list(item_ids)
        self.position = {item: n for n, item in enumerate(self.item_ids)}
        meta = config["metadata"]
        self.meta = SparseBM25(self.item_ids, catalog_fields).fit(meta["k1"], meta["b"], meta["field_weights"])
        self.meta.post_tf = None  # free per-field counts once contributions are computed
        self.term_weights = None
        self.reviews = None
        self.review_positions = None

    def learn(self, labelled):
        """labelled: list of (item_id, five review token lists) from the allowed partition only."""
        targets = [self.position[item] for item, _ in labelled]
        self.term_weights = self.meta.relevance_weights(
            [reviews for _, reviews in labelled], targets, prior_strength=self.config["rsj_prior_strength"])
        expansion = self.config["review_expansion"]
        if expansion["enabled"]:
            documents = {}
            for item, reviews in labelled:
                documents.setdefault(item, []).extend(token for tokens in reviews for token in tokens)
            items = sorted(documents)
            self.reviews = SparseBM25(items, [[documents[item]] for item in items]).fit(expansion["k1"], expansion["b"])
            self.reviews.post_tf = None
            self.review_positions = np.array([self.position[item] for item in items])
        return self

    def scores(self, reviews):
        query = dict(self.config["query"])
        meta_scores = self.meta.score_query(reviews, term_weights=self.term_weights, **query)
        expansion = self.config["review_expansion"]
        review_scores = None
        if self.reviews is not None and expansion["enabled"]:
            review_scores = self.reviews.score_query(reviews, **expansion["query"])
        return fuse(meta_scores, review_scores, self.review_positions, expansion["weight"], expansion["threshold"])

    def rank(self, reviews, k=10):
        return top_k(self.scores(reviews), k)


_WORKER = None


def _init_worker(retriever):
    global _WORKER
    _WORKER = retriever


def _rank_batch(batch):
    return [_WORKER.rank(reviews).tolist() for reviews in batch]


def rank_all(retriever, queries, workers=1, batch_size=64, progress=None):
    """Rank many queries; returns catalog positions (deterministic, order preserved)."""
    batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
    results = []
    if workers <= 1:
        for batch in batches:
            results.extend(_rank_batch_local(retriever, batch))
            if progress:
                progress(len(results), len(queries))
        return results
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(retriever,)) as executor:
        for ranked in executor.map(_rank_batch, batches):
            results.extend(ranked)
            if progress:
                progress(len(results), len(queries))
    return results


def _rank_batch_local(retriever, batch):
    return [retriever.rank(reviews).tolist() for reviews in batch]
