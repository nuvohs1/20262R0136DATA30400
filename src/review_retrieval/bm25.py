"""SQLite FTS5 BM25: a local full-catalog baseline with no external packages."""

from collections import Counter
import math
import sqlite3

from .common import product_tokens, query_tokens


class BM25:
    def __init__(self, catalog, max_query_terms=64, max_document_frequency=0.2):
        # FTS5 fixes k1=1.2 and b=0.75. Query terms are binary OR phrases.
        catalog = sorted(catalog, key=lambda item: item["item_id"])
        self.item_ids = [item["item_id"] for item in catalog]
        if len(catalog) < 10 or len(set(self.item_ids)) != len(catalog):
            raise ValueError("Catalog must have at least ten distinct products")
        if max_query_terms < 0 or not 0 < max_document_frequency <= 1:
            raise ValueError("Invalid query filter configuration")
        self.max_query_terms = max_query_terms
        self.max_document_frequency = max_document_frequency
        self.connection = sqlite3.connect(":memory:")
        self.connection.execute("CREATE VIRTUAL TABLE products USING fts5(item_id UNINDEXED, text, tokenize='ascii')")
        self.connection.execute("CREATE VIRTUAL TABLE vocabulary USING fts5vocab(products, 'row')")
        total_tokens, self.empty_documents = 0, 0
        with self.connection:
            for product in catalog:
                tokens = product_tokens(product)
                total_tokens += len(tokens)
                self.empty_documents += int(not tokens)
                self.connection.execute("INSERT INTO products(item_id, text) VALUES (?, ?)",
                                        (product["item_id"], " ".join(tokens)))
            self.connection.execute("INSERT INTO products(products) VALUES ('optimize')")
        self.document_frequency = dict(self.connection.execute("SELECT term, doc FROM vocabulary"))
        self.average_document_length = total_tokens / len(catalog)
        if not total_tokens:
            raise ValueError("Catalog has no indexable terms")

    def search_one(self, query, k=10):
        counts = Counter(query_tokens(query))
        n = len(self.item_ids)
        terms = [term for term in counts if term in self.document_frequency
                 and self.document_frequency[term] / n <= self.max_document_frequency]
        # Catalog-derived filtering requires no outside stopword dictionary.
        # Select informative terms by IDF * (1 + log(query frequency)).
        terms.sort(key=lambda term: (
            -(1 + math.log(counts[term])) * math.log(1 + (
                n - self.document_frequency[term] + 0.5) / (self.document_frequency[term] + 0.5)), term))
        if self.max_query_terms:
            terms = terms[:self.max_query_terms]
        terms.sort()
        if not terms:
            return [(item, 0.0) for item in self.item_ids[:k]], True
        expression = " OR ".join(f'"{term}"' for term in terms)
        # A negative FTS5 BM25 rank is better. Ties use lexicographic item ID.
        rows = list(self.connection.execute(
            "SELECT item_id, bm25(products) AS score FROM products "
            "WHERE products MATCH ? ORDER BY score, item_id LIMIT ?", (expression, k)))
        zero_match = not rows
        if len(rows) < k:
            found = {item for item, _ in rows}
            for item in self.item_ids:
                if item not in found:
                    rows.append((item, 0.0))
                    if len(rows) == k:
                        break
        return [(item, -score) for item, score in rows], zero_match

    def search(self, queries, k=10, progress=None):
        if not 1 <= k <= len(self.item_ids):
            raise ValueError("Invalid number of candidates")
        predictions, zero_match_queries = [], 0
        for count, query in enumerate(queries, start=1):
            rows, zero_match = self.search_one(query, k)
            predictions.append([item for item, _ in rows])
            zero_match_queries += int(zero_match)
            if progress:
                progress(count, len(queries))
        return predictions, {"zero_match_queries": zero_match_queries}

    def close(self):
        self.connection.close()
