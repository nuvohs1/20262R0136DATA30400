# Augmented cross-encoder

## Why

On validation, the hybrid first stage already has the correct product among its 50 candidates for 87.9% of queries, but the final Recall@10 is 0.71. The reranker is the bottleneck: the cross-encoders alone (w = 1) score at most level with the first stage alone (validation 0.638 / 0.668 vs 0.667). Two input limits are visible:

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

Run on the course AWS notebook (NVIDIA L4; model A about 33 minutes per epoch, model B 55). Records: `artifacts/experiments/augce_bert.json` and `artifacts/experiments/memory_proxy.json`.

**Training.** Epoch 2 was selected on the tuning slice (tuning-slice R@10 at the best weight: 0.696 after epoch 1, 0.708 after epoch 2). Earlier cross-encoders reached 0.695 and 0.702.

**Old protocol (validation, 4,000 queries, read once):**

| Reranker | Cross-encoder alone R@10 | Fused R@10 | R@1 | NDCG@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- |
| Step 4b (BM25 negatives, w 0.5) | 0.6380 | 0.7073 | 0.3767 | 0.5344 | 0.4801 |
| Hybrid negatives (w 0.6) | 0.6675 | 0.7145 | 0.3852 | 0.5409 | 0.4864 |
| **Augmented (w 0.6)** | **0.6765** | **0.7147** | **0.3945** | **0.5463** | **0.4934** |

The first stage alone is 0.6673 on validation.

The augmented model is the first cross-encoder that clearly beats the first stage on its own (0.6765 vs 0.6673; the hybrid-negative model only ties it, and the step-4b model is 3 points below). Its gain is mostly at the top of the list (R@1 +0.9 points over the hybrid-negative model, at the same R@10).

**Proxy (exp first stage; w chosen on the proxy tuning slice; proxy validation read once):**

| Ranker | Proxy tuning R@10 | Proxy validation R@10 | Memory half | No-memory half | MRR@10 |
| --- | --- | --- | --- | --- | --- |
| exp + rerank (step 4b, Kaggle 0.76916) | 0.5908 | 0.5995 | 0.6125 | 0.5865 | 0.3821 |
| exp + ens (Kaggle 0.77188) | 0.5923 | 0.6043 | 0.6125 | 0.5960 | 0.3889 |
| exp + aug (w 0.6) | 0.6003 | 0.6115 | 0.6155 | 0.6075 | 0.3960 |
| **exp + ens3 (w 0.6)** | **0.6040** | **0.6133** | **0.6220** | 0.6045 | **0.3977** |

- The augmented model alone already beats the two-model ensemble on both halves.
- Averaging all three is best overall (+0.9 points over `exp + ens` on proxy validation), and on both halves of the proxy.
- `ens3` is selected by the learned-fusion rule as well (`docs/learned_fusion.md`).

Test files: `outputs/memory/submission/exp_ens3/` (SHA-256 `e1c6aef8…`) and `exp_aug/` (`329aa3b2…`).

**Kaggle (public, recorded after the choice).** `exp + ens3` scored 0.77127 and `exp + aug` 0.76906, against 0.77188 for `exp + ens`. This is the first time the proxy and Kaggle disagree on the order.

A likely reason is a bias of the proxy toward this model. Proxy queries have 3 reviews, and the augmented model was trained on 3- or 4-review subsets in half of its draws. Test queries always have 5 reviews. On the old validation, which also has 5-review queries, the augmented reranker is level with the hybrid-negative one after fusion (0.7147 vs 0.7145). Its proxy lead (0.6115 vs 0.5965) therefore mostly reflects the shorter queries.

The training augmentation and the proxy share the same 3-review shape, so the proxy is not a fair judge of augmentation. This is recorded as a limitation. The final file stays `exp + ens` (`docs/final_selection.md`).

**Note on the proxy rule.** On the proxy tuning slice, `mem + ens` (0.6068) is still above `exp + ens3` (0.6040), so `run_memory_eval.py` still names `mem + ens` as its pick. The memory bonus has already been rejected: every `mem` file scored below its `exp` counterpart on Kaggle (`docs/final_selection.md`). It was therefore not scored with the new model, and `exp + ens3` is the candidate.
