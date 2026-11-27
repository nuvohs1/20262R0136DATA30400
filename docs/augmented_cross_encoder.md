# Augmented cross-encoder

## Why

On validation, the hybrid first stage already has the correct product among its 50 candidates for 87.9% of queries, but the final Recall@10 is 0.71. The reranker is the bottleneck: on the tuning slice, either cross-encoder alone (w = 1) scores *below* the first stage alone (0.640 / 0.666 vs 0.656). Two input limits are visible:

- **Reviews are cut at 64 tokens each.** About half of the review text is dropped (51% kept, estimated from word counts), even when other reviews of the same query are short and leave unused room.
- **The reviews are always seen in the same order, and always as five.** The proxy queries (and noisy test queries) look different.

The two earlier cross-encoders also differ only in their negatives, and averaging them helped (proxy validation 0.5995 → 0.6043, Kaggle 0.76916 → 0.77188). That suggests a third, more different model can add to the ensemble.

## What changes (`run_augce.sh`)

Same base model (`google-bert/bert-base-uncased`, verified revision), same listwise loss, learning rate, batch, group size and 2-epoch schedule as before. Three things change:

| Change | Detail |
| --- | --- |
| Shared review budget | The five reviews share 320 tokens (the old total of 5 × 64). Each review gets min(length, c), with the largest c that fits (`review_retrieval.ce_input.allocate`). Reviews are first cut at 256 tokens. Review text kept rises from about 51% to 57%. |
| Review augmentation (training only) | Every time a query is drawn, its reviews are shuffled; in half of the draws only 3 or 4 of them are kept. Seeded by `random.Random(42 + epoch)`. |
| Mixed negatives | Each group of 8 has the correct product, 4 negatives from the hybrid first stage and 3 from BM25 (both mined from the same query set). |

Scoring always uses all reviews of a query (3 on the proxy, 5 on test) in their original order with the shared budget.

**Leakage controls are unchanged.** Model A is trained on the 12,000 development-training queries, and its epoch is chosen on the tuning slice. Model B is trained on all 20,000 training queries and stops at A's epoch. Validation, the proxy splits and test never enter training. Both negative pools come from the query set being trained on, and the script refuses pools from different sets.

## Evaluation

- **Old protocol.** Fusion weight on the tuning slice, validation once (`outputs/augce/submission/`).
- **Proxy.** `aug` and `ens3` = (z(rerank) + z(hardneg) + z(aug)) / 3 are added to `scripts/run_memory_eval.py` on the `exp` first stage. w is chosen on the proxy tuning slice, and proxy validation is read once.

## Results

Pending the AWS run (step 4h).
