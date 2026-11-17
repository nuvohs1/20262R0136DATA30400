"""Field-weighted BM25 and review-level reciprocal rank fusion, locally."""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import math
from pathlib import Path
import sqlite3

from .common import TEXT_FIELDS, query_tokens, tokenize


_MODEL = None
_MODES = None


def _initialize(index_path, configuration, modes):
    global _MODEL, _MODES
    _MODEL = FieldBM25.open_index(index_path, **configuration)
    _MODES = modes


def _batch(queries):
    return _MODEL.search(queries, _MODES)


class FieldBM25:
    """FTS5 column weights modify term frequency before BM25 saturation.

    Document length remains the unweighted sum of the four fields; this is
    SQLite's weighted BM25, not independent per-field length normalization.
    """

    def __init__(self, catalog, **configuration):
        self.configure(**configuration)
        self.item_ids = sorted(product["item_id"] for product in catalog)
        if len(self.item_ids) < 10 or len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("Expected at least ten distinct catalog products")
        self.connection = sqlite3.connect(":memory:")
        self.connection.execute("CREATE VIRTUAL TABLE products USING fts5("
                                "item_id UNINDEXED, title, store, features, description, tokenize='ascii')")
        self.connection.execute("CREATE VIRTUAL TABLE vocabulary USING fts5vocab(products, 'row')")
        total_tokens = 0
        with self.connection:
            for product in sorted(catalog, key=lambda p: p["item_id"]):
                fields = [tokenize(product.get(field) or "") for field in TEXT_FIELDS]
                total_tokens += sum(map(len, fields))
                self.connection.execute("INSERT INTO products VALUES (?, ?, ?, ?, ?)",
                                        [product["item_id"], *(" ".join(tokens) for tokens in fields)])
            self.connection.execute("INSERT INTO products(products) VALUES ('optimize')")
        self.document_frequency = dict(self.connection.execute("SELECT term, doc FROM vocabulary"))
        self.average_document_length = total_tokens / len(catalog)
        if not total_tokens:
            raise ValueError("Catalog contains no indexable terms")

    def configure(self, field_weights=(4, 2, 2, 1), max_query_terms=64,
                  max_document_frequency=0.2, review_terms=24, fusion_depth=100,
                  rrf_constant=60, concat_weight=2):
        weights = tuple(float(value) for value in field_weights)
        if len(weights) != 4 or any(not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError("Require four finite positive field weights")
        if max_query_terms < 0 or review_terms < 1 or not 0 < max_document_frequency <= 1:
            raise ValueError("Invalid query filtering configuration")
        if fusion_depth < 10 or not math.isfinite(rrf_constant) or rrf_constant <= 0:
            raise ValueError("Invalid fusion configuration")
        if not math.isfinite(concat_weight) or concat_weight <= 0:
            raise ValueError("Concatenated-view weight must be positive")
        self.configuration = dict(field_weights=list(weights), max_query_terms=max_query_terms,
                                  max_document_frequency=max_document_frequency, review_terms=review_terms,
                                  fusion_depth=fusion_depth, rrf_constant=rrf_constant,
                                  concat_weight=concat_weight)
        self.rank_function = "bm25(0," + ",".join(format(weight, ".17g") for weight in weights) + ")"

    @classmethod
    def open_index(cls, path, **configuration):
        model = cls.__new__(cls)
        model.configure(**configuration)
        model.connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
        model.connection.execute("PRAGMA mmap_size=268435456")
        model.item_ids = [row[0] for row in model.connection.execute("SELECT item_id FROM products ORDER BY item_id")]
        model.document_frequency = dict(model.connection.execute("SELECT term, doc FROM vocabulary"))
        return model

    def save_index(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(str(path))
        try:
            self.connection.backup(target)
        finally:
            target.close()

    def select_terms(self, tokens, limit):
        counts = Counter(tokens)
        n = len(self.item_ids)
        terms = [term for term in counts if term in self.document_frequency
                 and self.document_frequency[term] / n <= self.configuration["max_document_frequency"]]
        terms.sort(key=lambda term: (
            -(1 + math.log(counts[term])) * math.log(1 + (
                n - self.document_frequency[term] + 0.5) / (self.document_frequency[term] + 0.5)), term))
        return sorted(terms[:limit] if limit else terms)

    def score_tokens(self, tokens, k=10, term_limit=None):
        limit = self.configuration["max_query_terms"] if term_limit is None else term_limit
        terms = self.select_terms(tokens, limit)
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        # Native ORDER BY rank enables FTS5's rank-sorted cursor. Expand at
        # the cutoff until every boundary tie is included, then sort by ID.
        fetch = min(k + 1, len(self.item_ids))
        while True:
            rows = list(self.connection.execute(
                "SELECT item_id, rank FROM products WHERE products MATCH ? "
                "AND rank MATCH ? ORDER BY rank LIMIT ?", (expression, self.rank_function, fetch)))
            if len(rows) < fetch or fetch == len(self.item_ids) or len(rows) <= k or rows[k - 1][1] != rows[-1][1]:
                break
            fetch = min(len(self.item_ids), fetch * 2)
        rows.sort(key=lambda row: (row[1], row[0]))
        return [(item_id, -score) for item_id, score in rows[:k]]

    def finish(self, scored, k=10):
        ranked = [item_id for item_id, _ in scored[:k]]
        if len(ranked) < k:
            found = set(ranked)
            for item_id in self.item_ids:
                if item_id not in found:
                    ranked.append(item_id)
                    if len(ranked) == k:
                        break
        return ranked

    def fuse(self, lists, weights=None):
        scores = defaultdict(float)
        weights = [1] * len(lists) if weights is None else weights
        for ranked, weight in zip(lists, weights):
            for rank, (item_id, _) in enumerate(ranked, start=1):
                scores[item_id] += weight / (self.configuration["rrf_constant"] + rank)
        return sorted(scores.items(), key=lambda row: (-row[1], row[0]))

    def search_one(self, query, modes=("concat",)):
        unknown = set(modes) - {"concat", "review_rrf", "hybrid_rrf"}
        if unknown:
            raise ValueError(f"Unknown retrieval modes: {sorted(unknown)}")
        concatenated_tokens = query_tokens(query)
        fusion = bool(set(modes) & {"review_rrf", "hybrid_rrf"})
        depth = self.configuration["fusion_depth"] if fusion else 10
        concat = self.score_tokens(concatenated_tokens, k=depth) if set(modes) & {"concat", "hybrid_rrf"} else []
        views = [self.score_tokens(tokenize(review["text"]), k=depth,
                                   term_limit=self.configuration["review_terms"])
                 for review in query["reviews"]] if fusion else []
        scores = {}
        if "concat" in modes:
            scores["concat"] = concat
        if "review_rrf" in modes:
            scores["review_rrf"] = self.fuse(views)
        if "hybrid_rrf" in modes:
            scores["hybrid_rrf"] = self.fuse([concat, *views], [self.configuration["concat_weight"], *([1] * 5)])
        return {mode: self.finish(scored) for mode, scored in scores.items()}, {
            mode: int(not scored) for mode, scored in scores.items()}

    def search(self, queries, modes=("concat",)):
        predictions = {mode: [] for mode in modes}
        zero_matches = {mode: 0 for mode in modes}
        for query in queries:
            result, diagnostics = self.search_one(query, modes)
            for mode in modes:
                predictions[mode].append(result[mode])
                zero_matches[mode] += diagnostics[mode]
        return predictions, zero_matches

    def search_parallel(self, queries, index_path, modes=("concat",), workers=8, progress=None):
        if workers < 1:
            raise ValueError("Worker count must be positive")
        if workers == 1:
            return self.search(queries, modes)
        predictions, zero_matches = {mode: [] for mode in modes}, {mode: 0 for mode in modes}
        batches = (queries[start:start + 16] for start in range(0, len(queries), 16))
        with ProcessPoolExecutor(max_workers=workers, initializer=_initialize,
                                 initargs=(str(Path(index_path).resolve()), self.configuration, tuple(modes))) as executor:
            for result, diagnostics in executor.map(_batch, batches):
                for mode in modes:
                    predictions[mode].extend(result[mode])
                    zero_matches[mode] += diagnostics[mode]
                if progress:
                    progress(len(predictions[modes[0]]), len(queries))
        return predictions, zero_matches

    def close(self):
        self.connection.close()
