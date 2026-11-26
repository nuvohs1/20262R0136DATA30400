"""Train a BERT cross-encoder reranker with a listwise softmax loss (GPU, e.g. AWS SageMaker L4).

Base model: google-bert/bert-base-uncased at the verified revision (review_retrieval.model_policy),
with a single-logit classification head. Training data: course train_gold.json only.

Objective: for every training query, a group of G candidates = its correct product + G-1
negatives drawn without replacement (random.Random(42 + epoch)) from the top
--negative-pool first-stage results of scripts/mine_negatives.py (BM25) or
scripts/mine_hybrid_negatives.py (BM25 + LTR BERT hybrid, the reranker's own first stage). Each pair is scored
independently; the loss is softmax cross-entropy over the G scores (listwise, ListNet top-1).

Protocol (same as the bi-encoder):
  --negatives outputs/ltr/negatives_dev.json --tune-candidates outputs/rerank/candidates_tune.json
      model A: 12,000 development-training queries; after each epoch the tuning-slice
      candidates are reranked, and the epoch with the best Recall@10 (best fusion weight)
      is kept. Validation queries are never used.
  --negatives outputs/ltr/negatives_full.json --stop-after-epoch K
      model B: all 20,000 training queries, same schedule, stops at model A's epoch.

Self-training (scripts/make_pseudo_labels.py): --pseudo outputs/pseudo/pseudo_dev.json adds
pseudo-labelled tuning-slice and validation queries to model A's data (reviews only, labels
never read); --pseudo outputs/pseudo/pseudo_full.json adds pseudo-labelled test queries to
model B's data. Training restarts from the base checkpoint; pseudo queries are mixed with the
labelled ones and use their own hard negatives (stage-1 candidates).
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cross_encoder_common import PairEncoder, score_candidates
from encode_dense import set_seeds
from review_retrieval.common import read_json, sha256
from review_retrieval.dense_text import product_passage
from review_retrieval.model_policy import (ALLOWED_CHECKPOINTS, DEFAULT_CHECKPOINT, PROVENANCE_FILE,
                                           PSEUDO_FINE_TUNING_DATA, check_model)
from review_retrieval.rerank import select_weight


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "project_release")
    parser.add_argument("--negatives", type=Path, required=True)
    parser.add_argument("--tune-candidates", type=Path, help="Tuning-slice candidates (model A only)")
    parser.add_argument("--pseudo", type=Path, help="Pseudo-labels from scripts/make_pseudo_labels.py (self-training)")
    parser.add_argument("--model", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--revision", default=ALLOWED_CHECKPOINTS[DEFAULT_CHECKPOINT]["revision"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2, help="Length of the learning-rate schedule")
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--batch-queries", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--negative-pool", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--review-tokens", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    args = parser.parse_args()
    set_seeds(42)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if not torch.cuda.is_available():
        raise SystemExit("A GPU is required.")
    device = torch.device("cuda")

    record = read_json(args.negatives)
    train_sha = sha256(args.data_dir / "train_gold.json")
    if train_sha != record["train_gold_sha256"]:
        raise ValueError("Negatives were mined from a different train_gold.json")
    stage = record["first_stage"]
    if stage.get("dense"):
        # Hybrid negatives (scripts/mine_hybrid_negatives.py): the dense part must be our own LTR
        # bert-base-uncased checkpoint (checked by model_policy at mining time), trained on the same
        # query set, so no other model and no held-out query influences the negatives.
        if not stage.get("dense_checkpoint_checked") or stage.get("dense_negatives_set") != record["set"]:
            raise ValueError("Hybrid negatives must come from the verified LTR model of the same query set")
    negative_source = "hybrid BM25 + LTR BERT" if stage.get("dense") else "BM25"
    train = {q["query_id"]: q for q in read_json(args.data_dir / "train_gold.json")}
    train_ids = record["train_query_ids"]
    held_out = set(record["tune_query_ids"]) | set(record["validation_query_ids_excluded"])
    if set(train_ids) & held_out:
        raise ValueError("Training queries overlap the tuning or validation queries")
    tune = read_json(args.tune_candidates) if args.tune_candidates else None
    if tune is not None and (set(tune["query_ids"]) & set(train_ids) or tune["split"] != "tune"):
        raise ValueError("Tuning candidates must be the tuning slice, disjoint from the training queries")

    pseudo = read_json(args.pseudo) if args.pseudo else None
    pseudo_records = {}
    if pseudo is not None:
        if pseudo["set"] != f"pseudo-{record['set']}" or pseudo["train_gold_sha256"] != train_sha:
            raise ValueError("Pseudo-labels must match the negatives set (pseudo-dev with dev, pseudo-full with full)")
        if record["set"] == "dev":
            # Reviews of tuning-slice / validation queries only; their item_id and category_path are dropped.
            source = {q: {"query_id": q, "reviews": train[q]["reviews"]} for q in held_out}
        else:
            test = read_json(args.data_dir / "test_queries.json")
            if sha256(args.data_dir / "test_queries.json") != pseudo["test_queries_sha256"]:
                raise ValueError("Pseudo-labels were made for a different test_queries.json")
            source = {q["query_id"]: q for q in test}
        pseudo_records = {p["query_id"]: {"query_id": p["query_id"], "reviews": source[p["query_id"]]["reviews"],
                                          "item_id": p["item_id"]} for p in pseudo["queries"]}
        if set(pseudo_records) & set(train_ids):
            raise ValueError("Pseudo-labelled queries overlap the labelled training queries")
        for p in pseudo["queries"]:
            record["negatives"][p["query_id"]] = p["negatives"]

    eligibility = check_model(args.model, args.revision)
    provenance = {
        "base_model": eligibility["base_model"], "base_revision": eligibility["base_revision"],
        "fine_tuning_data": PSEUDO_FINE_TUNING_DATA if pseudo else "course train_gold.json only", "train_gold_sha256": train_sha,
        "catalog_sha256": sha256(args.data_dir / "catalog.json"), "negatives_sha256": sha256(args.negatives),
        "negatives_set": record["set"], "train_queries": len(train_ids),
        "objective": f"cross-encoder, listwise softmax cross-entropy over (1 positive + {negative_source} negatives) groups",
        "negative_source": negative_source,
        "max_length": args.max_length, "review_tokens": args.review_tokens,
    }
    if pseudo is not None:
        provenance.update({"pseudo_labels_sha256": sha256(args.pseudo), "pseudo_set": pseudo["set"],
                           "pseudo_queries": len(pseudo_records), "pseudo_threshold": pseudo["threshold"]})

    catalog = sorted(read_json(args.data_dir / "catalog.json"), key=lambda p: p["item_id"])
    item_ids = [p["item_id"] for p in catalog]
    position = {item: n for n, item in enumerate(item_ids)}
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, revision=args.revision, num_labels=1).to(device)
    model.gradient_checkpointing_enable()
    print("Tokenizing catalog and reviews...", flush=True)
    encoder = PairEncoder(tokenizer, [product_passage(p) for p in catalog], args.max_length, args.review_tokens)
    labelled_ids = list(train_ids)
    train_ids = labelled_ids + sorted(pseudo_records)
    examples = {**{q: train[q] for q in labelled_ids}, **pseudo_records}
    query_index = {q: n for n, q in enumerate(train_ids)}
    query_ids = encoder.queries([examples[q] for q in train_ids])
    if tune is not None:
        tune_queries = encoder.queries([train[q] for q in tune["query_ids"]])
        tune_items = [[position[i] for i in row] for row in tune["items"]]
        tune_first = np.asarray(tune["first_stage_scores"], dtype=np.float64)
        tune_gold = [train[q]["item_id"] for q in tune["query_ids"]]

    def save(epoch, extra=None):
        target = args.output_dir / "checkpoint"
        model.save_pretrained(target)
        tokenizer.save_pretrained(target)
        (target / PROVENANCE_FILE).write_text(json.dumps({**provenance, "epoch": epoch, **(extra or {})}, indent=2) + "\n",
                                              encoding="utf-8")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = len(train_ids) // args.batch_queries
    total = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total * args.warmup_ratio), total)
    scaler = torch.amp.GradScaler("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = {"args": {k: str(v) for k, v in vars(args).items()}, "provenance": provenance,
           "steps_per_epoch": steps_per_epoch, "epochs": [],
           "environment": {"python": platform.python_version(), "torch": torch.__version__,
                           "transformers": __import__("transformers").__version__,
                           "device": torch.cuda.get_device_name(0)}}
    best = -1.0
    last_epoch = args.stop_after_epoch or args.epochs
    targets = torch.zeros(args.batch_queries, dtype=torch.long, device=device)  # positive is first in each group
    for epoch in range(1, last_epoch + 1):
        model.train()
        rng = random.Random(42 + epoch)
        order = list(train_ids)
        rng.shuffle(order)
        start, running = time.perf_counter(), 0.0
        for step in range(steps_per_epoch):
            batch_q = order[step * args.batch_queries:(step + 1) * args.batch_queries]
            pairs = []
            for q in batch_q:
                group = [examples[q]["item_id"]] + rng.sample(record["negatives"][q][:args.negative_pool], args.group_size - 1)
                pairs.extend((query_index[q], position[item]) for item in group)
            inputs = encoder.batch(pairs, query_ids, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**inputs).logits[:, 0]
            loss = torch.nn.functional.cross_entropy(logits.float().view(len(batch_q), args.group_size), targets)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if (step + 1) % 100 == 0 or step + 1 == steps_per_epoch:
                print(f"epoch {epoch} step {step + 1}/{steps_per_epoch} loss {running / (step + 1):.4f} "
                      f"({time.perf_counter() - start:.0f}s)", flush=True)
        entry = {"epoch": epoch, "mean_loss": running / steps_per_epoch, "seconds": time.perf_counter() - start}
        if tune is not None:
            scores = score_candidates(model, encoder, tune_queries, tune_items, device, args.eval_batch_size, "tune")
            weight, curve = select_weight(tune["items"], tune_gold, tune_first, scores)
            entry["tuning_slice"] = {"recall_at_10_by_weight": curve, "best_weight": weight}
            print(f"epoch {epoch} tuning slice R@10 by weight: {curve}", flush=True)
            if curve[weight] > best:
                best = curve[weight]
                log["selected_epoch"] = epoch
                save(epoch)
                np.save(args.output_dir / "scores_tune.npy", scores)
        log["epochs"].append(entry)
        (args.output_dir / "train_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    if tune is None:
        log["selected_epoch"] = last_epoch
        save(last_epoch)
    log["completed"] = True
    (args.output_dir / "train_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    print(f"Saved checkpoint (epoch {log['selected_epoch']}) to {args.output_dir / 'checkpoint'}", flush=True)


if __name__ == "__main__":
    main()
