"""Optional deterministic NumPy scoring with the same SQLite BM25 formula.

Only NumPy core arrays are used. No NumPy random generators, SciPy, or model
downloads are used; the original SQLite implementation remains available.
"""

from array import array
from collections import Counter
import math

import numpy as np

from .bm25_query_variants import QueryBM25
from .common import TEXT_FIELDS, tokenize


class NumpyBM25(QueryBM25):
    def __init__(self, catalog, **configuration):
        super().__init__(catalog, **configuration)
        postings = {}
        lengths = []
        weights = self.configuration["field_weights"]
        for doc_id, product in enumerate(sorted(catalog, key=lambda p: p["item_id"])):
            counts = Counter()
            length = 0
            for field, weight in zip(TEXT_FIELDS, weights):
                field_counts = Counter(tokenize(product.get(field) or ""))
                length += sum(field_counts.values())
                for term, frequency in field_counts.items():
                    counts[term] += weight * frequency
            lengths.append(length)
            for term, frequency in counts.items():
                if term not in postings:
                    postings[term] = (array("I"), array("d"))
                postings[term][0].append(doc_id)
                postings[term][1].append(frequency)
        lengths = np.asarray(lengths, dtype=np.float64)
        norm = 1.2 * (0.25 + 0.75 * lengths / self.average_document_length)
        self.term_scores = {}
        for term, (raw_ids, raw_frequencies) in postings.items():
            ids = np.frombuffer(raw_ids, dtype=np.uint32)
            tf = np.frombuffer(raw_frequencies, dtype=np.float64)
            df = self.document_frequency[term]
            if len(ids) != df:
                raise ValueError("NumPy and SQLite disagree on document frequency")
            idf = math.log((len(self.item_ids) - df + 0.5) / (df + 0.5))
            idf = idf if idf > 0 else 1e-6
            values = idf * ((tf * 2.2) / (tf + norm[ids]))
            self.term_scores[term] = (ids, values)
        self.frozen_field_weights = tuple(weights)
        self.connection.close()

    def configure(self, **configuration):
        if hasattr(self, "frozen_field_weights"):
            requested = tuple(configuration.get("field_weights", (4, 2, 2, 1)))
            if requested != self.frozen_field_weights:
                raise ValueError("Build a new NumPy index when changing field weights")
        super().configure(**configuration)

    def score_tokens(self, tokens, k=10, term_limit=None):
        self.last_near_tie = False
        limit = self.configuration["max_query_terms"] if term_limit is None else term_limit
        terms = self.select_terms(tokens, limit)
        if not terms:
            return []
        scores = np.zeros(len(self.item_ids), dtype=np.float64)
        # Each term's posting list has distinct document IDs. Repeated query
        # terms are accumulated separately, in the same order as SQLite.
        for term in terms:
            ids, values = self.term_scores[term]
            scores[ids] += values
        k = min(k, len(scores))
        inspect = min(k + 1, len(scores))
        threshold = np.partition(scores, len(scores) - inspect)[len(scores) - inspect]
        above = np.flatnonzero(scores > threshold)
        tied = np.flatnonzero(scores == threshold)[:inspect - len(above)]
        selected = np.concatenate((above, tied))
        selected = selected[np.lexsort((selected, -scores[selected]))]
        positive = [float(scores[i]) for i in selected if scores[i] > 0]
        self.last_near_tie = any(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                                 for a, b in zip(positive, positive[1:]))
        return [(self.item_ids[int(i)], float(scores[i])) for i in selected[:k] if scores[i] > 0]
