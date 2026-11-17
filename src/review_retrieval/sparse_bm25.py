"""Inverted-index BM25F with tunable k1, b, field weights, and query-term weighting.

Only NumPy core arrays are used (no SciPy and no NumPy random generators, which the
recorded Windows environment blocks). The index is built from catalog metadata only;
reviews are never added to product documents.
"""

from collections import Counter
import math

import numpy as np

from .common import TEXT_FIELDS, tokenize


def light_stem(token):
    """Conservative English plural folding with no external dictionary.

    Tokens containing digits (model numbers, sizes) are left untouched. Only common
    plural suffixes are folded: "curtains" -> "curtain", "boxes" -> "box",
    "batteries" -> "battery".
    """
    if len(token) <= 3 or not token.isalpha():
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("sses", "shes", "ches", "xes", "zes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def normalize_tokens(tokens, stem):
    return [light_stem(token) for token in tokens] if stem else list(tokens)


def product_field_tokens(product, stem=True):
    return [normalize_tokens(tokenize(product.get(field) or ""), stem) for field in TEXT_FIELDS]


def review_token_lists(query, stem=True):
    if len(query["reviews"]) != 5:
        raise ValueError("Each query must contain exactly five reviews")
    return [normalize_tokens(tokenize(review["text"]), stem) for review in query["reviews"]]


QUERY_MODES = ("binary", "log", "sqrt", "reviews")


class SparseBM25:
    """BM25F over any fixed set of text fields (the four metadata fields by default).

    Each field's term frequency is length-normalized with its own b, multiplied by the
    field weight, summed, and saturated once with k1 (Robertson-Zaragoza BM25F).
    IDF uses the positive form log(1 + (N - df + 0.5) / (df + 0.5)).
    """

    def __init__(self, item_ids, field_token_lists):
        self.item_ids = list(item_ids)
        n = len(self.item_ids)
        if n < 10 or len(set(self.item_ids)) != n or len(field_token_lists) != n:
            raise ValueError("Catalog must have at least ten distinct products")
        self.vocabulary = {}
        self.n_fields = len(field_token_lists[0])
        if any(len(fields) != self.n_fields for fields in field_token_lists):
            raise ValueError("Every document needs the same number of fields")
        self.field_lengths = np.zeros((self.n_fields, n), dtype=np.float64)
        terms, docs, fields, counts = [], [], [], []
        for doc, product_fields in enumerate(field_token_lists):
            for f, tokens in enumerate(product_fields):
                self.field_lengths[f, doc] = len(tokens)
                for term, count in Counter(tokens).items():
                    terms.append(self.vocabulary.setdefault(term, len(self.vocabulary)))
                    docs.append(doc)
                    fields.append(f)
                    counts.append(count)
        terms = np.asarray(terms, np.int64)
        docs = np.asarray(docs, np.int64)
        # One posting per (term, doc); field frequencies kept side by side.
        key = terms * n + docs
        unique_keys, inverse = np.unique(key, return_inverse=True)
        self.post_terms = unique_keys // n
        self.post_docs = (unique_keys % n).astype(np.int32)
        self.post_tf = np.zeros((self.n_fields, len(unique_keys)), dtype=np.float64)
        np.add.at(self.post_tf, (np.asarray(fields), inverse), np.asarray(counts, np.float64))
        vocabulary_size = len(self.vocabulary)
        self.term_start = np.searchsorted(self.post_terms, np.arange(vocabulary_size + 1))
        self.document_frequency = np.diff(self.term_start)
        df = self.document_frequency
        self.idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        self.post_weight = None

    def fit(self, k1=1.2, b=0.75, field_weights=None):
        """Precompute each posting's BM25F contribution (without the query weight)."""
        field_weights = tuple(field_weights) if field_weights is not None else (1.0,) * self.n_fields
        b = tuple(b) if isinstance(b, (tuple, list)) else (b,) * self.n_fields
        if k1 <= 0 or len(b) != self.n_fields or len(field_weights) != self.n_fields:
            raise ValueError("Invalid BM25F configuration")
        pseudo_tf = np.zeros(len(self.post_docs), dtype=np.float64)
        for f in range(self.n_fields):
            if field_weights[f] == 0:
                continue
            lengths = self.field_lengths[f]
            average = lengths.mean() or 1.0
            norm = 1.0 - b[f] + b[f] * lengths / average
            pseudo_tf += field_weights[f] * self.post_tf[f] / np.maximum(norm[self.post_docs], 1e-9)
        weight = pseudo_tf * (k1 + 1.0) / (pseudo_tf + k1)
        self.post_weight = (weight * self.idf[self.post_terms]).astype(np.float32)
        self.config = {"k1": k1, "b": list(b), "field_weights": list(field_weights)}
        return self

    def relevance_weights(self, review_lists_by_query, target_indices, prior_strength=10.0):
        """Supervised Robertson-Sparck Jones term weights learned from labelled training pairs.

        For each catalog term t, among training queries whose reviews mention t, estimate
        p_t = P(t occurs in the correct product's metadata) with a Beta prior centred on
        the pooled rate, and q_t = df_t / N for a random product. The RSJ weight is
        log[p (1 - q) / (q (1 - p))]. Only the supplied (training-partition) pairs are used.
        """
        n, vocabulary_size = len(self.item_ids), len(self.vocabulary)
        mentioned = np.zeros(vocabulary_size, dtype=np.float64)
        matched = np.zeros(vocabulary_size, dtype=np.float64)
        for review_lists, target in zip(review_lists_by_query, target_indices):
            terms = {self.vocabulary[t] for tokens in review_lists for t in tokens if t in self.vocabulary}
            if not terms:
                continue
            terms = np.fromiter(terms, dtype=np.int64)
            mentioned[terms] += 1
            start, end = self.term_start[terms], self.term_start[terms + 1]
            for term, s, e in zip(terms, start, end):
                # Posting lists are sorted by document, so membership is a binary search.
                k = s + np.searchsorted(self.post_docs[s:e], target)
                if k < e and self.post_docs[k] == target:
                    matched[term] += 1
        pooled = matched.sum() / max(mentioned.sum(), 1.0)
        p = (matched + prior_strength * pooled) / (mentioned + prior_strength)
        q = np.clip(self.document_frequency / n, 1e-6, 1 - 1e-6)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        weights = np.log(p * (1 - q) / (q * (1 - p)))
        return np.maximum(weights, 0.0)

    def query_terms(self, review_lists, mode="log", max_document_frequency=1.0, max_terms=0, idf_power=0.0,
                    term_weights=None, term_weight_power=1.0, select_by="idf"):
        """Return (term_ids, query_weights) for one five-review query.

        mode: 'binary' (each distinct term once), 'log' (1 + log tf), 'sqrt' (sqrt tf),
        or 'reviews' (number of the five reviews that mention the term).
        idf_power: multiply each query weight by idf ** idf_power (extra emphasis on
        rare, specific terms on top of the IDF already inside BM25).
        term_weights: optional per-term multipliers (e.g. relevance_weights()), raised to
        term_weight_power.
        max_terms: keep only the highest-priority terms; 0 keeps all. Priority is
        IDF * (1 + log tf) when select_by='idf', or the final query weight when
        select_by='weight'. Ties are broken alphabetically.
        """
        n = len(self.item_ids)
        if mode == "reviews":
            counts = Counter(term for tokens in review_lists for term in set(tokens))
        elif mode in QUERY_MODES:
            counts = Counter(term for tokens in review_lists for term in tokens)
        else:
            raise ValueError(f"Unknown query mode {mode}")
        if select_by not in ("idf", "weight"):
            raise ValueError(f"Unknown selection rule {select_by}")
        kept = []
        for term, count in counts.items():
            column = self.vocabulary.get(term)
            if column is None or self.document_frequency[column] > max_document_frequency * n:
                continue
            if mode == "binary":
                weight = 1.0
            elif mode == "log":
                weight = 1.0 + math.log(count)
            elif mode == "sqrt":
                weight = math.sqrt(count)
            else:
                weight = float(count)
            if idf_power:
                weight *= self.idf[column] ** idf_power
            if term_weights is not None:
                weight *= term_weights[column] ** term_weight_power
            if weight <= 0:
                continue
            priority = weight if select_by == "weight" else (1.0 + math.log(count)) * self.idf[column]
            kept.append((-priority, term, column, weight))
        if max_terms:
            kept.sort()
            kept = kept[:max_terms]
        kept.sort(key=lambda row: row[1])
        return [row[2] for row in kept], [row[3] for row in kept]

    def score_terms(self, term_ids, weights):
        """Dense score vector over all products for one weighted bag of terms."""
        if self.post_weight is None:
            raise ValueError("Call fit() first")
        if not term_ids:
            return np.zeros(len(self.item_ids), dtype=np.float64)
        starts, ends = self.term_start[term_ids], self.term_start[np.asarray(term_ids) + 1]
        lengths = ends - starts
        index = np.repeat(starts - np.concatenate(([0], np.cumsum(lengths)[:-1])), lengths) + np.arange(lengths.sum())
        values = self.post_weight[index].astype(np.float64) * np.repeat(np.asarray(weights, np.float64), lengths)
        return np.bincount(self.post_docs[index], weights=values, minlength=len(self.item_ids))

    def score_query(self, review_lists, **options):
        return self.score_terms(*self.query_terms(review_lists, **options))


def top_k(scores, k=10, allowed=None):
    """Top-k product indices for one score vector.

    Ties are broken by product index, i.e. ascending item ID when the catalog is sorted.
    allowed: optional boolean mask; disallowed products are never returned.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if allowed is not None:
        scores = np.where(allowed, scores, -np.inf)
    pool = min(len(scores), k)
    candidates = np.argpartition(-scores, pool - 1)[:pool]
    cutoff = scores[candidates].min()
    above = np.flatnonzero(scores > cutoff)
    above = above[np.lexsort((above, -scores[above]))]
    tied = np.flatnonzero(scores == cutoff)
    return np.concatenate([above, tied])[:k]


def recall_at(ranked_indices, target_indices, ks=(1, 5, 10)):
    ranked = np.asarray(ranked_indices)
    target = np.asarray(target_indices)[:, None]
    return {f"recall@{k}": float((ranked[:, :k] == target).any(axis=1).mean()) for k in ks}
