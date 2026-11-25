"""Category classifier with a label GCN, trained multi-task on queries and products (PyTorch).

Category labels (category_path) exist only for labelled training queries; catalog products have
none. Following the course's label-embedding / label-GNN classifiers:
  * label embeddings: every node of the 4-level taxonomy (category_taxonomy.txt) starts from the
    embedding of its name (the LTR bi-encoder, passage side) and a 2-layer GCN over the taxonomy
    tree mixes parents and children; the 500 leaf rows are the class embeddings;
  * two inputs, one shared label space (multi-task): a query MLP reads the mean of the five
    review embeddings, a product MLP reads the product embedding; logits = projection . labels;
  * loss = cross-entropy(query -> leaf) + cross-entropy(product of that query -> leaf).

  --set dev   labelled = the 12,000 development-training queries outside the tuning slice
              (embeddings and label names from LTR model A); the epoch is chosen on the tuning
              slice by the mean log-probability that a query and its correct product share a
              category. Writes category scores for the tuning-slice and validation candidates.
  --set full  labelled = all 20,000 training queries (LTR model B), stops at the dev epoch.
              Writes category scores for the test candidates.
Products whose category is known from the labelled queries use that label (one-hot); all other
products and every query use the classifier's prediction. Validation labels are never read.
"""

import argparse
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from encode_dense import encode, set_seeds
from review_retrieval.category import build_taxonomy, match_scores, normalized_adjacency
from review_retrieval.common import read_json, sha256, write_json
from review_retrieval.model_policy import check_embedding_manifest, check_model


class CategoryModel(nn.Module):
    def __init__(self, dim, hidden, node_features, adjacency, leaves):
        super().__init__()
        self.register_buffer("h0", node_features)
        self.register_buffer("adj", adjacency)
        self.register_buffer("leaves", leaves)
        self.gcn1, self.gcn2 = nn.Linear(dim, hidden), nn.Linear(hidden, hidden)
        mlp = lambda: nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, hidden))
        self.query, self.product = mlp(), mlp()

    def labels(self):
        h = torch.relu(self.adj @ self.gcn1(self.h0))
        return (self.adj @ self.gcn2(h))[self.leaves]

    def forward(self, queries=None, products=None):
        e = self.labels()
        return (None if queries is None else self.query(queries) @ e.T,
                None if products is None else self.product(products) @ e.T)


def load_embeddings(directory, data_dir, parts, model_set):
    manifest = read_json(directory / "manifest.json")
    for name in ("catalog.json", "train_gold.json", "test_queries.json"):
        if manifest["source_sha256"][name] != sha256(data_dir / name):
            raise ValueError(f"{directory} was computed from a different {name}")
    eligibility = check_embedding_manifest(manifest)
    if eligibility["kind"] != "fine-tuned checkpoint" or eligibility["provenance"]["negatives_set"] != model_set:
        raise ValueError(f"{directory} must come from the LTR model trained on the '{model_set}' set")
    return eligibility, {p: np.load(directory / f"{p}.npy").astype(np.float32) for p in parts}


@torch.no_grad()
def probabilities(model, x, side, device, batch=4096):
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        t = torch.from_numpy(x[i:i + batch]).to(device)
        logits = model(queries=t)[0] if side == "query" else model(products=t)[1]
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=("dev", "full"), required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "project_release")
    parser.add_argument("--split", type=Path, default=ROOT / "artifacts/dev_split.json")
    parser.add_argument("--rerank-dir", type=Path, default=ROOT / "outputs/rerank")
    parser.add_argument("--output-dir", type=Path, help="Default: outputs/category/<set>")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    set_seeds(42)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.output_dir or ROOT / f"outputs/category/{args.set}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = read_json(args.data_dir / "train_gold.json")
    split = read_json(args.split)
    train_sha = sha256(args.data_dir / "train_gold.json")
    if train_sha != split["source_sha256"]:
        raise ValueError("Training data does not match the fixed split fingerprint")
    row = {q["query_id"]: n for n, q in enumerate(train)}
    by_id = {q["query_id"]: q for q in train}
    tune_ids = sorted(random.Random(42).sample(sorted(split["train_query_ids"]), 4000))
    if args.set == "dev":
        labelled = sorted(set(split["train_query_ids"]) - set(tune_ids))
        held_out = set(tune_ids) | set(split["validation_query_ids"])
        if set(labelled) & held_out:
            raise ValueError("Labelled queries overlap the tuning or validation queries")
    else:
        labelled = [q["query_id"] for q in train]

    # Taxonomy and leaf classes (official file); labels of the labelled queries only.
    paths = [line.strip() for line in (args.data_dir / "category_taxonomy.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    nodes, leaves, edges = build_taxonomy(paths)
    leaf_paths = sorted(set(paths))
    class_of = {p: i for i, p in enumerate(leaf_paths)}
    y = {q: class_of[by_id[q]["category_path"]] for q in labelled}
    if args.set == "dev":
        y_tune = {q: class_of[by_id[q]["category_path"]] for q in tune_ids}  # tuning slice: epoch selection only

    # Embeddings from the LTR bi-encoder of the same set; label names encoded by its checkpoint.
    emb_dir = ROOT / f"outputs/dense/bert-ltr-{args.set}"
    parts = ("catalog", "train_reviews") + (("test_reviews",) if args.set == "full" else ())
    eligibility, emb = load_embeddings(emb_dir, args.data_dir, parts, args.set)
    queries = {"train": emb["train_reviews"].reshape(len(train), 5, -1).mean(axis=1)}
    if args.set == "full":
        test = read_json(args.data_dir / "test_queries.json")
        queries["test"] = emb["test_reviews"].reshape(len(test), 5, -1).mean(axis=1)
    catalog = emb["catalog"]
    item_ids = sorted(p["item_id"] for p in read_json(args.data_dir / "catalog.json"))
    position = {item: n for n, item in enumerate(item_ids)}

    checkpoint = ROOT / f"outputs/ltr/bert-ltr-{args.set}/checkpoint"
    check_model(checkpoint, train_gold_sha256=train_sha)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    encoder = AutoModel.from_pretrained(checkpoint).to(device).eval()
    names = encode(encoder, tokenizer, nodes, "", 64, 256, device, "category names").astype(np.float32)
    del encoder
    torch.cuda.empty_cache() if device.type == "cuda" else None

    model = CategoryModel(catalog.shape[1], args.hidden, torch.from_numpy(names),
                          torch.from_numpy(normalized_adjacency(len(nodes), edges).astype(np.float32)),
                          torch.tensor(leaves)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    xq = torch.from_numpy(queries["train"][[row[q] for q in labelled]]).to(device)
    xp = torch.from_numpy(catalog[[position[by_id[q]["item_id"]] for q in labelled]]).to(device)
    yt = torch.tensor([y[q] for q in labelled], device=device)

    def tune_metrics():
        pq = probabilities(model, queries["train"][[row[q] for q in tune_ids]], "query", device)
        pp = probabilities(model, catalog[[position[by_id[q]["item_id"]] for q in tune_ids]], "product", device)
        gold = np.array([y_tune[q] for q in tune_ids])
        top5 = np.argsort(-pq, axis=1)[:, :5]
        return {"pair_log_prob": float(np.mean(np.log((pq * pp).sum(axis=1) + 1e-9))),
                "query_acc@1": float(np.mean(top5[:, 0] == gold)),
                "query_acc@5": float(np.mean([g in t for g, t in zip(gold, top5)])),
                "product_acc@1": float(np.mean(pp.argmax(axis=1) == gold))}

    log = {"args": {k: str(v) for k, v in vars(args).items()}, "labelled_queries": len(labelled), "classes": len(leaf_paths),
           "taxonomy_nodes": len(nodes), "embeddings": str(emb_dir.relative_to(ROOT)),
           "embedding_eligibility": {k: eligibility[k] for k in ("kind", "base_model", "base_revision")},
           "category_labels": "category_path of the labelled training queries only", "epochs": [],
           "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": str(device)}}
    best, best_state = -np.inf, None
    last = args.stop_after_epoch or args.epochs
    for epoch in range(1, last + 1):
        model.train()
        gen = torch.Generator().manual_seed(42 + epoch)
        order = torch.randperm(len(labelled), generator=gen).to(device)
        start, total = time.perf_counter(), 0.0
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            lq, lp = model(xq[idx], xp[idx])
            loss = nn.functional.cross_entropy(lq, yt[idx]) + nn.functional.cross_entropy(lp, yt[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        entry = {"epoch": epoch, "mean_loss": total / len(labelled), "seconds": time.perf_counter() - start}
        if args.set == "dev":
            entry["tuning_slice"] = tune_metrics()
            if entry["tuning_slice"]["pair_log_prob"] > best:
                best, log["selected_epoch"] = entry["tuning_slice"]["pair_log_prob"], epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch}: {entry}", flush=True)
        log["epochs"].append(entry)
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        log["selected_epoch"] = last
    torch.save(model.state_dict(), out_dir / "category_model.pt")

    # Product distributions: known label for labelled products, prediction otherwise.
    product_probs = probabilities(model, catalog, "product", device)
    for q in labelled:
        p = position[by_id[q]["item_id"]]
        product_probs[p] = 0.0
        product_probs[p, y[q]] = 1.0
    splits = ("tune", "validation") if args.set == "dev" else ("test",)
    for name in splits:
        cands = read_json(args.rerank_dir / f"candidates_{name}.json")
        source = queries["test"] if name == "test" else queries["train"]
        index = {q["query_id"]: n for n, q in enumerate(test)} if name == "test" else row
        pq = probabilities(model, source[[index[q] for q in cands["query_ids"]]], "query", device)
        scores = match_scores(pq, product_probs, [[position[i] for i in r] for r in cands["items"]])
        np.save(out_dir / f"cat_scores_{name}.npy", scores.astype(np.float32))
        if name == "validation":
            np.save(out_dir / "query_probs_validation.npy", pq.astype(np.float16))  # accuracy reported by the evaluation only
        print(f"  {name}: category scores {scores.shape}", flush=True)
    log["completed"] = True
    write_json(out_dir / "train_log.json", log)
    print(f"Saved {out_dir} (epoch {log['selected_epoch']})", flush=True)


if __name__ == "__main__":
    main()
