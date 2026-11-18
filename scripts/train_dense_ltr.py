"""Fine-tune BERT as a bi-encoder with a listwise learning-to-rank loss on review-product pairs (GPU, e.g. Colab).

Base model: google-bert/bert-base-uncased at the verified revision (BookCorpus + English
Wikipedia, MLM/NSP only; docs/model_eligibility.md). review_retrieval.model_policy refuses
any other base. The saved checkpoint carries provenance.json (base model, revision, and
the SHA-256 of the course training file it was fine-tuned on).

Objective (listwise softmax cross-entropy, i.e. ListNet top-1 / InfoNCE):
  for a batch of B queries, each query's candidate list is the B correct products in
  the batch plus h hard negatives per query (shared across the batch). With query
  vector q (mean of the five review embeddings, re-normalized) and product vector p,
  loss = -log softmax_j(q . p_j / tau)[correct]. Duplicate copies of a query's correct
  product elsewhere in the batch are masked out.

Hard negatives come from scripts/mine_negatives.py (BM25 only, item IDs only), drawn without
replacement from the top `--negative-pool` first-stage results with a seeded RNG.

Protocol:
  --negatives outputs/ltr/negatives_dev.json  : trains on 12,000 development-training
      queries; after each epoch, Recall@10 of dense retrieval on the 4,000-query tuning
      slice (full catalog) selects the checkpoint. Validation queries are never used.
  --negatives outputs/ltr/negatives_full.json --stop-after-epoch K : trains on all
      20,000 training queries with the same schedule, stopping at the selected epoch K.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from encode_dense import encode, set_seeds
from review_retrieval.common import read_json, sha256
from review_retrieval.dense_text import product_passage, review_texts
from review_retrieval.model_policy import ALLOWED_CHECKPOINTS, DEFAULT_CHECKPOINT, PROVENANCE_FILE, check_model


def embed(model, tokenizer, texts, max_length, device):
    batch = tokenizer(texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to(device)
    hidden = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
    return torch.nn.functional.normalize(pooled.float(), dim=-1)


def recall_on_slice(model, tokenizer, passages, item_ids, queries, device, args):
    """Dense-only Recall@1/10 over the full catalog for the given labelled queries."""
    model.eval()
    common = dict(model=model, tokenizer=tokenizer, batch_size=256, device=device)
    catalog = encode(texts=passages, prefix=args.passage_prefix, max_length=args.passage_max_length, label="eval catalog", **common)
    texts = [t for q in queries for t in review_texts(q)]
    reviews = encode(texts=texts, prefix=args.query_prefix, max_length=256, label="eval reviews", **common)
    model.train()
    catalog_t = torch.from_numpy(catalog).to(device).float()
    query_t = torch.from_numpy(reviews.reshape(len(queries), 5, -1)).to(device).float().mean(1)
    position = {item: n for n, item in enumerate(item_ids)}
    targets = torch.tensor([position[q["item_id"]] for q in queries], device=device)
    hits1 = hits10 = 0
    for s in range(0, len(queries), 512):
        top = (query_t[s:s + 512] @ catalog_t.T).topk(10, dim=1).indices
        hits10 += (top == targets[s:s + 512, None]).any(1).sum().item()
        hits1 += (top[:, 0] == targets[s:s + 512]).sum().item()
    return {"recall_at_1": hits1 / len(queries), "recall_at_10": hits10 / len(queries)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "project_release")
    parser.add_argument("--negatives", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--revision", default=ALLOWED_CHECKPOINTS[DEFAULT_CHECKPOINT]["revision"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4, help="Length of the learning-rate schedule")
    parser.add_argument("--stop-after-epoch", type=int, help="Stop and save after this epoch (full-data run)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hard-negatives", type=int, default=3)
    parser.add_argument("--negative-pool", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--review-max-length", type=int, default=192)
    parser.add_argument("--passage-max-length", type=int, default=256)
    parser.add_argument("--query-prefix", default="", help="Must match encode_dense.py (default: none)")
    parser.add_argument("--passage-prefix", default="", help="Must match encode_dense.py (default: none)")
    args = parser.parse_args()
    set_seeds(42)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("A GPU is required for fine-tuning (use the Colab notebook).")

    record = read_json(args.negatives)
    train = {q["query_id"]: q for q in read_json(args.data_dir / "train_gold.json")}
    if sha256(args.data_dir / "train_gold.json") != record["train_gold_sha256"]:
        raise ValueError("Negatives were mined from a different train_gold.json")
    catalog = sorted(read_json(args.data_dir / "catalog.json"), key=lambda p: p["item_id"])
    item_ids = [p["item_id"] for p in catalog]
    passage = {p["item_id"]: product_passage(p) for p in catalog}
    train_ids = record["train_query_ids"]
    tune = [train[q] for q in record["tune_query_ids"]]
    overlap = set(train_ids) & (set(record["tune_query_ids"]) | set(record["validation_query_ids_excluded"]))
    if overlap:
        raise ValueError("Training queries overlap the tuning or validation queries")
    if record["first_stage"].get("dense"):
        raise ValueError("Negatives must come from the BM25-only first stage (re-run scripts/mine_negatives.py)")
    negatives = record["negatives"]

    eligibility = check_model(args.model, args.revision)
    train_sha = sha256(args.data_dir / "train_gold.json")
    provenance = {
        "base_model": eligibility["base_model"], "base_revision": eligibility["base_revision"],
        "fine_tuning_data": "course train_gold.json only", "train_gold_sha256": train_sha,
        "catalog_sha256": sha256(args.data_dir / "catalog.json"), "negatives_sha256": sha256(args.negatives),
        "negatives_set": record["set"], "train_queries": len(train_ids),
        "objective": "listwise softmax cross-entropy (in-batch + BM25 hard negatives)",
        "query_prefix": args.query_prefix, "passage_prefix": args.passage_prefix,
    }

    def save(epoch):
        target = args.output_dir / "checkpoint"
        model.save_pretrained(target)
        tokenizer.save_pretrained(target)
        (target / PROVENANCE_FILE).write_text(json.dumps({**provenance, "epoch": epoch}, indent=2) + "\n", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModel.from_pretrained(args.model, revision=args.revision).to(device)
    model.gradient_checkpointing_enable()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = len(train_ids) // args.batch_size  # drop the incomplete last batch
    total = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total * args.warmup_ratio), total)
    scaler = torch.amp.GradScaler("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = {"args": {k: str(v) for k, v in vars(args).items()}, "provenance": provenance, "train_queries": len(train_ids),
           "steps_per_epoch": steps_per_epoch, "epochs": [],
           "environment": {"python": platform.python_version(), "torch": torch.__version__,
                           "transformers": __import__("transformers").__version__,
                           "device": torch.cuda.get_device_name(0)}}
    best = -1.0
    last_epoch = args.stop_after_epoch or args.epochs
    for epoch in range(1, last_epoch + 1):
        rng = random.Random(42 + epoch)
        order = list(train_ids)
        rng.shuffle(order)
        start, running = time.perf_counter(), 0.0
        for step in range(steps_per_epoch):
            batch = order[step * args.batch_size:(step + 1) * args.batch_size]
            positives = [train[q]["item_id"] for q in batch]
            hard = [item for q in batch for item in rng.sample(negatives[q][:args.negative_pool], args.hard_negatives)]
            candidates = positives + hard
            reviews = [args.query_prefix + t for q in batch for t in review_texts(train[q])]
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                review_vectors = embed(model, tokenizer, reviews, args.review_max_length, device)
                query = torch.nn.functional.normalize(review_vectors.view(len(batch), 5, -1).mean(1), dim=-1)
                products = embed(model, tokenizer, [args.passage_prefix + passage[i] for i in candidates],
                                 args.passage_max_length, device)
            logits = (query @ products.T) / args.temperature
            duplicate = torch.tensor([[c == positives[i] and j != i for j, c in enumerate(candidates)]
                                      for i in range(len(batch))], device=device)
            logits = logits.masked_fill(duplicate, float("-inf"))
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(batch), device=device))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if (step + 1) % 50 == 0 or step + 1 == steps_per_epoch:
                print(f"epoch {epoch} step {step + 1}/{steps_per_epoch} loss {running / (step + 1):.4f} "
                      f"({time.perf_counter() - start:.0f}s)", flush=True)
        entry = {"epoch": epoch, "mean_loss": running / steps_per_epoch, "seconds": time.perf_counter() - start}
        if tune:
            entry["tuning_slice_dense"] = recall_on_slice(model, tokenizer, [passage[i] for i in item_ids],
                                                          item_ids, tune, device, args)
            print(f"epoch {epoch} tuning slice: {entry['tuning_slice_dense']}", flush=True)
            if entry["tuning_slice_dense"]["recall_at_10"] > best:
                best = entry["tuning_slice_dense"]["recall_at_10"]
                log["selected_epoch"] = epoch
                save(epoch)
        log["epochs"].append(entry)
        (args.output_dir / "train_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    if not tune:
        log["selected_epoch"] = last_epoch
        save(last_epoch)
    log["completed"] = True
    (args.output_dir / "train_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    print(f"Saved checkpoint (epoch {log['selected_epoch']}) to {args.output_dir / 'checkpoint'}", flush=True)


if __name__ == "__main__":
    main()
