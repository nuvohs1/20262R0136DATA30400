# Category labels as auxiliary information (label GCN, multi-task)

## Idea

Every labelled training query carries the category path of its product (4 levels, 500 leaf categories in the official `category_taxonomy.txt`; 40 training queries per leaf). Catalog products and test queries have no category. If both a query and a candidate product can be placed in the taxonomy, a product from a different category than the query is unlikely to be correct. That gives a signal that is complementary to text matching. This step follows the course's label-embedding (P2c) and label-GNN (P5b) classifiers and adds the predicted category agreement to the reranker as a third score.

Only course files are used (`train_gold.json` categories, `category_taxonomy.txt`, catalog metadata, reviews). No external model or data is involved. All inputs come from the LTR fine-tuned `bert-base-uncased` bi-encoder, which is already verified by `model_policy.py`.

## Model (`scripts/train_category.py`)

- **Label embeddings with a GCN.**
  - The taxonomy tree has 687 nodes (13 + 38 + 136 + 500), and each is linked to its parent.
  - Node features are the bi-encoder embeddings of the node names, for example "Kitchen & Dining > Dining & Entertaining".
  - A 2-layer GCN, `H' = Â · ReLU(Â H W₁) W₂` with `Â = D^−½ (A + I) D^−½`, mixes each category with its parent and children.
  - The 500 leaf rows are the class embeddings. Sibling categories ("Salt Shakers", "Pepper Mills") share information through their parent.
- **Multi-task inputs in one label space.**
  - A query MLP reads the mean of the five review embeddings.
  - A product MLP reads the product embedding.
  - Both predict the leaf by inner product with the label embeddings: `logits = MLP(x) · Eᵀ`.
  - The loss is `CE(query → leaf) + CE(product → leaf)` on each labelled pair, because a product has the same category as its query.
- **Training settings.** Hidden size 256, AdamW (lr 1e-3, weight decay 1e-4), batch 256, up to 30 epochs, seed 42.
- **Category score.** The score of a candidate is `cat(q, p) = log Σ_c P(c | q) · P(c | p)`, the probability that the query and the product share a leaf. Products whose category is known from the labelled queries use that label. All other products, and all queries, use the classifier's prediction.
- **Fusion.** `final = (1 − w) · z(stage 1) + w · z(cross-encoder) + β · z(category)` over the 50 step-4b candidates. w and β are chosen together on the tuning slice from {0, 0.1, …, 1}². β = 0 is exactly the step-4b ranker.

## Leakage protocol

| Model | Labelled queries (categories used) | Embeddings | Used for |
| --- | --- | --- | --- |
| Category model A (`outputs/category/dev`) | 12,000 development-training | LTR model A | epoch (tuning-slice pair log-probability), (w, β), validation once |
| Category model B (`outputs/category/full`) | all 20,000 | LTR model B | test, stopped at A's epoch |

Validation categories are read only after the final evaluation, to report the classifier's accuracy. Tuning-slice categories select the epoch, as tuning-slice labels did in every earlier step. Candidates and cross-encoder scores are the step-4b ones, so the test top 10 before reranking still equal the hybrid submission.

## How to run

On the course AWS notebook (step 4d), after `run_rerank.sh`:

```bash
bash run_category.sh 2024320020   # encode -> dev -> full -> eval
```

The final file is `outputs/category/submission/2024320020_final.csv`, and metrics are in `outputs/category/submission/metrics.json`.

## Results

Run on the course AWS notebook. Full record: `artifacts/experiments/category_gcn.json`.

**Classifier (model A).** Tuning-slice metrics improve steadily. The pair log-probability peaks at epoch 17 (−1.96), so epoch 17 is selected; later epochs start to overfit (−2.16 at epoch 30). At epoch 17 the scores are:

| Tuning slice, epoch 17 | Value |
| --- | --- |
| Query → leaf, accuracy@1 | 0.494 |
| Query → leaf, accuracy@5 | 0.789 |
| Product → leaf, accuracy@1 | 0.590 |
| P(query and correct product share a leaf), geometric mean | 0.14 |

On validation (read once, after the evaluation), query accuracy is 0.509 @1 and 0.795 @5 over 500 classes, in line with the tuning slice. Products are easier to classify than queries, because their titles often name the product type.

**Fusion.** The best tuning-slice Recall@10 for each β (best w for that β) is:

| β | 0.0 | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R@10 | **0.6950** | 0.6930 | 0.6890 | 0.6815 | 0.6693 | 0.6590 | 0.6415 |

Every positive β lowers Recall@10. The selection is therefore **β = 0**, which is exactly the step-4b ranker. Validation (0.7073 Recall@10) and the test file are identical to step 4b; the test CSV has the same SHA-256.

**Likely reasons the category score does not help** (interpretation; the per-candidate category scores were not analysed further):

- **The candidates already share the query's category.** Stage 1 (BM25 + bi-encoder) and the cross-encoder both read product titles and features, so the 50 candidates of a query are mostly products of the same kind. A category score cannot separate them; it only re-ranks between near-siblings.
- **The classifier is too uncertain at the leaf level.** With about 50% query accuracy over 500 leaves, the chance that a query and its correct product receive the same leaf is only about 0.14. Among similar candidates, this noise outweighs the signal.
- **Categories are already implicit.** The cross-encoder is trained with hard negatives from the same BM25 neighbourhood, so it has learned to tell apart products of related categories.

The classifier itself works (79.5% top-5 accuracy over 500 classes, using 40 examples per class). Category information is simply redundant with the text signals the ranker already uses. The step is kept as a documented negative result. No Kaggle submission is needed, because the file is identical to step 4b.
