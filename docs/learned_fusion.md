# Learned fusion of stage 1 and the cross-encoders

`scripts/learn_fusion.py` (NumPy, CPU, under a minute) tests whether a learned combination beats the plain three-model average (`ens3`).

## Method

**Features.** Five per candidate, all computed within the query's 50 candidates:

- z(stage 1);
- z(rerank), z(hardneg) and z(aug);
- 1 / log2(stage-1 rank + 2).

**Model.** One linear score per candidate. It is trained with a listwise softmax cross-entropy (the correct product against its 49 competitors) plus L2 regularisation (λ = 1, fixed in advance). Training uses Newton's method with a backtracking line search, starting from zero. The problem is convex and has no randomness.

**Training data.** The proxy tuning slice only, i.e. training queries. Test inputs are never used to fit anything.

**Selection rule (fixed before any result).**

1. Run 5-fold cross-validation on the proxy tuning slice. The folds come from `random.Random(42)` over the sorted query IDs.
2. Compare `ens3` (w chosen on the training folds) with the learned fusion (fitted on the training folds).
3. The learned fusion is used only if its cross-validated Recall@10 is higher. Ties go to `ens3`.

The stage-1 components (BM25 and dense scores separately) were not available locally. Stage 1 enters as its fused score and its rank.

## Results

Full record: `artifacts/experiments/learned_fusion.json`.

| | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean |
| --- | --- | --- | --- | --- | --- | --- |
| ens3 | 0.5875 | 0.6088 | 0.6050 | 0.6150 | 0.6038 | **0.6040** |
| learned | 0.5887 | 0.6100 | 0.6012 | 0.6138 | 0.6000 | 0.6028 |

**`ens3` is kept by the rule.** On proxy validation the two are level (0.6132 vs 0.6140, a difference of 3 queries out of 4,000).

**Fitted weights.** Refitted on the whole slice:

| Feature | Weight |
| --- | --- |
| z(stage 1) | 0.561 |
| z(rerank) | 0.193 |
| z(hardneg) | 0.160 |
| z(aug) | 0.555 |
| rank feature | −0.015 |

The learned model gives the augmented cross-encoder about three times the weight of each earlier one. This agrees with its better single-model results. However, the equal average is already close to the best linear combination on this data, so the extra parameters do not pay off.

**First attempt.** It used plain Newton steps without a line search, diverged, and scored near zero. The cross-validation rule rejected it automatically. The line search fixed it, and the numbers above come from the fixed version.
