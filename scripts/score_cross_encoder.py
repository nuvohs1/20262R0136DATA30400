"""Score first-stage candidates with a trained cross-encoder checkpoint (GPU).

  --checkpoint outputs/rerank/ce-dev/checkpoint  --candidates outputs/rerank/candidates_validation.json
  --checkpoint outputs/rerank/ce-full/checkpoint --candidates outputs/rerank/candidates_test.json
The checkpoint must carry provenance.json from scripts/train_cross_encoder.py (allowed base
model, course training data only). Validation must be scored with model A (dev).
Writes a (queries, K) float32 array next to the checkpoint: scores_<split>.npy (or --output).
Proxy candidates (scripts/make_memory_candidates.py) carry review_indices: only those reviews of
each query are given to the cross-encoder.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cross_encoder_common import PairEncoder, score_candidates
from encode_dense import set_seeds
from review_retrieval.common import read_json, sha256
from review_retrieval.dense_text import product_passage
from review_retrieval.model_policy import check_model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "project_release")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, help="Default: <checkpoint>/../scores_<split>.npy")
    args = parser.parse_args()
    set_seeds(42)
    if not torch.cuda.is_available():
        raise SystemExit("A GPU is required.")
    device = torch.device("cuda")
    eligibility = check_model(args.checkpoint, train_gold_sha256=sha256(args.data_dir / "train_gold.json"))
    provenance = eligibility["provenance"]
    candidates = read_json(args.candidates)
    is_test = candidates["split"] == "test"
    if not is_test and provenance["negatives_set"] != "dev":
        raise ValueError("Tuning/validation candidates must be scored with the development model (model A)")

    catalog = sorted(read_json(args.data_dir / "catalog.json"), key=lambda p: p["item_id"])
    position = {p["item_id"]: n for n, p in enumerate(catalog)}
    source = "test_queries.json" if is_test else "train_gold.json"
    records = {q["query_id"]: q for q in read_json(args.data_dir / source)}
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(device)
    encoder = PairEncoder(tokenizer, [product_passage(p) for p in catalog], provenance["max_length"], provenance["review_tokens"])
    query_ids = encoder.queries([records[q] for q in candidates["query_ids"]], candidates.get("review_indices"))
    items = [[position[i] for i in row] for row in candidates["items"]]
    scores = score_candidates(model, encoder, query_ids, items, device, args.batch_size, candidates["split"])
    out = args.output or args.checkpoint.parent / f"scores_{candidates['split']}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, scores)
    print(f"Saved {out} {scores.shape}", flush=True)


if __name__ == "__main__":
    main()
