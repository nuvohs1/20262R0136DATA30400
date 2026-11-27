"""Shared input construction and scoring for the BERT cross-encoder (GPU stage).

Input for one (query, product) pair, at most --max-length tokens:
  [CLS] review_1 [SEP] review_2 [SEP] ... review_5 [SEP] product passage [SEP]
Each review is truncated to --review-tokens WordPiece tokens (segment A); the passage
(Title | Store | Features | Description, as in dense retrieval) fills the rest (segment B).
With --review-budget (the augmented cross-encoder), the reviews instead share one token budget
(review_retrieval.ce_input.allocate), and training draws shuffled review subsets.
Token IDs are kept as NumPy int32 arrays and pairs are assembled per batch, so millions of
candidate pairs never have to be materialized at once.
"""

import time

import numpy as np
import torch

from review_retrieval.ce_input import assemble
from review_retrieval.dense_text import review_texts

REVIEW_CAP = 256  # tokens kept per review before the shared budget is applied


class PairEncoder:
    def __init__(self, tokenizer, passage_texts, max_length=512, review_tokens=64, review_budget=None):
        self.max_length, self.review_tokens, self.review_budget = max_length, review_tokens, review_budget
        self.cls, self.sep, self.pad = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.passages = self._ids(passage_texts, max_length)

    def _ids(self, texts, limit):
        out = []
        for start in range(0, len(texts), 10_000):
            enc = self.tokenizer(texts[start:start + 10_000], add_special_tokens=False, truncation=True,
                                 max_length=limit)["input_ids"]
            out.extend(np.asarray(ids, dtype=np.int32) for ids in enc)
        return out

    def review_ids(self, queries, review_indices=None):
        """Per query, the token IDs of each review (cut at the per-review limit of this layout).
        review_indices (proxy validation) keeps only the listed reviews, in their original order."""
        texts = [review_texts(q) for q in queries]
        if review_indices is not None:
            texts = [[q[i] for i in idx] for q, idx in zip(texts, review_indices)]
        limit = REVIEW_CAP if self.review_budget else self.review_tokens
        flat = self._ids([t for q in texts for t in q], limit)
        out, start = [], 0
        for q in texts:
            out.append(flat[start:start + len(q)])
            start += len(q)
        return out

    def join(self, reviews):
        """List of per-review token arrays -> query token IDs (segment A)."""
        if self.review_budget:
            return assemble(reviews, self.sep, budget=self.review_budget)
        return assemble(reviews, self.sep, cap=self.review_tokens)

    def queries(self, queries, review_indices=None):
        """Token IDs of the concatenated reviews for each query record."""
        return [self.join(r) for r in self.review_ids(queries, review_indices)]

    def length(self, query, item):
        return min(self.max_length, len(query) + len(self.passages[item]) + 3)

    def batch(self, pairs, query_ids, device):
        """pairs: list of (query index, item index) -> model inputs on device."""
        rows = []
        for q, item in pairs:
            query = query_ids[q]
            passage = self.passages[item][:self.max_length - 3 - len(query)]
            ids = np.concatenate(([self.cls], query, [self.sep], passage, [self.sep])).astype(np.int64)
            types = np.concatenate((np.zeros(len(query) + 2, np.int64), np.ones(len(passage) + 1, np.int64)))
            rows.append((ids, types))
        width = max(len(ids) for ids, _ in rows)
        input_ids = np.full((len(rows), width), self.pad, dtype=np.int64)
        token_types = np.zeros((len(rows), width), dtype=np.int64)
        mask = np.zeros((len(rows), width), dtype=np.int64)
        for r, (ids, types) in enumerate(rows):
            input_ids[r, :len(ids)], token_types[r, :len(ids)], mask[r, :len(ids)] = ids, types, 1
        return {"input_ids": torch.from_numpy(input_ids).to(device),
                "token_type_ids": torch.from_numpy(token_types).to(device),
                "attention_mask": torch.from_numpy(mask).to(device)}


@torch.inference_mode()
def score_candidates(model, encoder, query_ids, candidate_items, device, batch_size=256, label="scoring"):
    """candidate_items: (queries, K) item indices -> (queries, K) float32 cross-encoder scores."""
    model.eval()
    n_queries, depth = len(candidate_items), len(candidate_items[0])
    pairs = [(q, int(item)) for q in range(n_queries) for item in candidate_items[q]]
    order = sorted(range(len(pairs)), key=lambda i: encoder.length(query_ids[pairs[i][0]], pairs[i][1]))
    scores = np.zeros(len(pairs), dtype=np.float32)
    start = time.perf_counter()
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        batch = encoder.batch([pairs[i] for i in idx], query_ids, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(**batch).logits[:, 0]
        scores[idx] = logits.float().cpu().numpy()
        done = min(b + batch_size, len(order))
        if (b // batch_size) % 500 == 0 or done == len(order):
            print(f"  {label}: {done:,}/{len(order):,} pairs ({time.perf_counter() - start:.0f}s)", flush=True)
    return scores.reshape(n_queries, depth)
