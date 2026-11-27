"""Pack the three model-B checkpoints the final file needs, with a SHA-256 manifest.

  outputs/ltr/bert-ltr-full/checkpoint   LTR BERT bi-encoder (first stage, test embeddings)
  outputs/rerank/ce-full/checkpoint      cross-encoder trained with BM25 negatives
  outputs/hardneg/ce-full/checkpoint     cross-encoder trained with hybrid negatives

All three are fine-tuned from google-bert/bert-base-uncased on train_gold.json only (each carries
provenance.json). Writes outputs/model_pack/models_<id>.zip (stored, not compressed) and
outputs/model_pack/model_manifest.json. The manifest is committed as artifacts/model_manifest.json;
scripts/fetch_models.py refuses any file that does not match it.
"""

import argparse
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from review_retrieval.common import sha256, write_json

CHECKPOINTS = ("outputs/ltr/bert-ltr-full/checkpoint", "outputs/rerank/ce-full/checkpoint",
               "outputs/hardneg/ce-full/checkpoint")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student-id", default="2024320020")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/model_pack")
    args = parser.parse_args()
    files = sorted(p for c in CHECKPOINTS for p in (ROOT / c).rglob("*") if p.is_file())
    for c in CHECKPOINTS:
        if not (ROOT / c / "provenance.json").is_file():
            raise SystemExit(f"Missing {c}/provenance.json")
    manifest = {p.relative_to(ROOT).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in files}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / f"models_{args.student_id}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for name in manifest:
            z.write(ROOT / name, name)
    record = {"archive": archive.name, "archive_sha256": sha256(archive), "checkpoints": list(CHECKPOINTS), "files": manifest}
    write_json(args.output_dir / "model_manifest.json", record)
    print(f"Wrote {archive} ({archive.stat().st_size / 1e9:.2f} GB, {len(manifest)} files)", flush=True)


if __name__ == "__main__":
    main()
