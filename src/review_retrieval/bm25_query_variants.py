"""Isolate BM25 query length and quantized logarithmic query frequency."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import math
from pathlib import Path

from .bm25_variants import FieldBM25


_MODEL = None
_MODES = None


def _initialize(path, configuration, modes):
    global _MODEL, _MODES
    _MODEL = QueryBM25.open_index(path, **configuration)
    _MODES = modes


def _batch(queries):
    return _MODEL.search(queries, _MODES)


class QueryBM25(FieldBM25):
    def configure(self, query_tf="binary", **configuration):
        if query_tf not in ("binary", "log_count"):
            raise ValueError("Unknown query frequency weighting")
        super().configure(**configuration)
        self.configuration["query_tf"] = query_tf

    def select_terms(self, tokens, limit):
        tokens = list(tokens)
        selected = super().select_terms(tokens, limit)
        if self.configuration["query_tf"] == "binary":
            return selected
        counts = Counter(tokens)
        # FTS5 treats repeated OR phrases as separate BM25 contributions.
        # First choose up to 'limit' distinct terms, then repeat each phrase
        # by min(4, 1 + floor(log2(query_count))). This is a quantized weight.
        return [term for term in selected for _ in range(min(4, 1 + int(math.log2(counts[term]))))]

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
