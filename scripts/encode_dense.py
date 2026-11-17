"""Encode catalog products and every review with a pretrained E5 encoder (run on a GPU, e.g. Colab).

No training happens here: the encoder is used as-is (zero-shot), so no labels are
read. Outputs are L2-normalized float16 embeddings, one row per product or review:

  outputs/dense/<tag>/catalog.npy        50,000 x d, rows in ascending item_id order
  outputs/dense/<tag>/train_reviews.npy  20,000*5 x d, train_gold.json order, 5 rows per query
  outputs/dense/<tag>/test_reviews.npy   19,858*5 x d, test_queries.json order, 5 rows per query
  outputs/dense/<tag>/manifest.json      model revision, settings, input fingerprints, row order

Model eligibility (intfloat/e5-base-v2, MIT): contrastive pre-training on CCPairs
(Wikipedia, Reddit, Common Crawl, StackExchange, S2ORC, News, SimpleWiki, GooAQ,
WikiHow, Yahoo Answers) and fine-tuning on MS-MARCO, NQ, and NLI (Wang et al., 2022,
arXiv:2212.03533, Sec. 4-5 and Table 9). No Amazon data is used for training; Amazon
datasets appear only in the MTEB evaluation tables.
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
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from review_retrieval.common import read_json, sha256
from review_retrieval.dense_text import product_passage, review_texts


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.inference_mode()
def encode(model, tokenizer, texts, prefix, max_length, batch_size, device, label):
    """Mean-pooled, L2-normalized embeddings; batches are length-sorted, output keeps input order."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float16)
    start = time.perf_counter()
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        batch = tokenizer([prefix + texts[i] for i in idx], max_length=max_length, padding=True,
                          truncation=True, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
        out[idx] = pooled.cpu().numpy().astype(np.float16)
        done = min(b + batch_size, len(order))
        if done == len(order) or (b // batch_size) % 200 == 0:
            print(f"  {label}: {done:,}/{len(order):,} ({time.perf_counter() - start:.0f}s)", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "project_release")
    parser.add_argument("--model", default="intfloat/e5-base-v2")
    parser.add_argument("--tag", default="e5-base-v2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/dense")
    parser.add_argument("--passage-max-length", type=int, default=256)
    parser.add_argument("--review-max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--parts", default="catalog,train,test", help="Comma-separated subset to (re)encode")
    args = parser.parse_args()
    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no GPU found; encoding ~250k texts on CPU will be very slow.", flush=True)
    out_dir = args.output_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    revision = getattr(model.config, "_commit_hash", None)

    catalog = sorted(read_json(args.data_dir / "catalog.json"), key=lambda p: p["item_id"])
    train = read_json(args.data_dir / "train_gold.json")
    test = read_json(args.data_dir / "test_queries.json")
    if len(catalog) != 50_000 or len(train) != 20_000 or len(test) != 19_858:
        raise ValueError("Expected the complete original course files")
    parts = set(args.parts.split(","))
    timing = {}
    common = dict(model=model, tokenizer=tokenizer, batch_size=args.batch_size, device=device)
    if "catalog" in parts:
        t = time.perf_counter()
        emb = encode(texts=[product_passage(p) for p in catalog], prefix="passage: ",
                     max_length=args.passage_max_length, label="catalog", **common)
        np.save(out_dir / "catalog.npy", emb)
        timing["catalog"] = time.perf_counter() - t
    for name, queries in (("train", train), ("test", test)):
        if name in parts:
            t = time.perf_counter()
            texts = [text for q in queries for text in review_texts(q)]
            emb = encode(texts=texts, prefix="query: ", max_length=args.review_max_length, label=name, **common)
            np.save(out_dir / f"{name}_reviews.npy", emb)
            timing[name] = time.perf_counter() - t

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.update({
        "model": args.model, "model_revision": revision, "pooling": "mean over attention mask, L2-normalized",
        "prefixes": {"catalog": "passage: ", "reviews": "query: "},
        "passage_max_length": args.passage_max_length, "review_max_length": args.review_max_length,
        "dtype": "float16", "hidden_size": model.config.hidden_size,
        "row_order": {"catalog": "ascending item_id", "train_reviews": "train_gold.json order x 5 reviews",
                      "test_reviews": "test_queries.json order x 5 reviews"},
        "catalog_item_ids_sha256": __import__("hashlib").sha256(
            "\n".join(p["item_id"] for p in catalog).encode()).hexdigest(),
        "source_sha256": {n: sha256(args.data_dir / n) for n in ("catalog.json", "train_gold.json", "test_queries.json")},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": __import__("transformers").__version__,
                        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
    })
    manifest.setdefault("timing_seconds", {}).update(timing)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved embeddings to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
