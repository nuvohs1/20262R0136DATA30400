"""Pretrained-model eligibility policy (pure Python; used by the GPU and the local scripts).

Course rule (TA ruling): use only checkpoints whose training data is publicly documented
well enough to verify that they were not trained on Amazon data for downstream tasks
(QA, retrieval, ...). Amazon text inside generic language-model pretraining (MLM) would be
acceptable, but every checkpoint must be verified individually from its official docs.

Every script that loads model weights or consumes embeddings calls this module:
  * a Hugging Face checkpoint must be listed in ALLOWED_CHECKPOINTS and pinned to the
    exact revision recorded there (other names or revisions raise an error);
  * a local checkpoint directory must contain the provenance.json written by
    scripts/train_dense_ltr.py, naming an allowed base checkpoint and the course
    training file it was fine-tuned on (verified by SHA-256).
"""

from pathlib import Path
import json

ALLOWED_CHECKPOINTS = {
    "google-bert/bert-base-uncased": {
        "revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
        "license": "apache-2.0",
        "training_data": ["BookCorpus (11,038 unpublished books)",
                          "English Wikipedia (excluding lists, tables and headers)"],
        "objectives": ["masked language modeling (MLM)", "next sentence prediction (NSP)"],
        "amazon_data": "none",
        "downstream_training": "none (self-supervised pretraining only)",
        "documentation": [
            "https://huggingface.co/google-bert/bert-base-uncased/blob/86b5e0934494bd15c9632b12f734a8a67f723594/README.md#training-data",
            "Devlin et al., 2019, BERT, arXiv:1810.04805, Sec. 3.1 (pre-training data: BooksCorpus and English Wikipedia)",
        ],
        "notes": "Explicitly named as usable in the course tutorial slides (honor code, p.13).",
    },
}

ALIASES = {"bert-base-uncased": "google-bert/bert-base-uncased"}
DEFAULT_CHECKPOINT = "google-bert/bert-base-uncased"
PROVENANCE_FILE = "provenance.json"


class IneligibleModelError(ValueError):
    pass


def canonical(name):
    return ALIASES.get(name, name)


def check_hub_checkpoint(name, revision):
    """Return (canonical name, record) or raise IneligibleModelError."""
    name = canonical(name)
    record = ALLOWED_CHECKPOINTS.get(name)
    if record is None:
        raise IneligibleModelError(
            f"{name!r} is not in ALLOWED_CHECKPOINTS. Only checkpoints whose training data is publicly "
            "documented and verified (see docs/model_eligibility.md) may be used.")
    if revision != record["revision"]:
        raise IneligibleModelError(f"{name!r} must be loaded at the verified revision {record['revision']}, got {revision!r}")
    return name, record


def check_local_checkpoint(path, train_gold_sha256=None):
    """A fine-tuned checkpoint is eligible if its base is allowed and it was trained on course data only."""
    path = Path(path)
    file = path / PROVENANCE_FILE
    if not file.exists():
        raise IneligibleModelError(f"{path} has no {PROVENANCE_FILE}; its origin cannot be verified")
    provenance = json.loads(file.read_text(encoding="utf-8"))
    check_hub_checkpoint(provenance["base_model"], provenance["base_revision"])
    if provenance.get("fine_tuning_data") != "course train_gold.json only":
        raise IneligibleModelError(f"{path}: fine-tuning data is not restricted to the course training file")
    if train_gold_sha256 and provenance.get("train_gold_sha256") != train_gold_sha256:
        raise IneligibleModelError(f"{path}: fine-tuned on a different train_gold.json")
    return provenance


def check_model(name_or_path, revision=None, train_gold_sha256=None):
    """Validate a model before loading it. Returns a JSON-serializable eligibility record."""
    if Path(name_or_path).is_dir():
        provenance = check_local_checkpoint(name_or_path, train_gold_sha256)
        base = canonical(provenance["base_model"])
        return {"kind": "fine-tuned checkpoint", "base_model": base, "base_revision": provenance["base_revision"],
                "base_record": ALLOWED_CHECKPOINTS[base], "provenance": provenance}
    name, record = check_hub_checkpoint(name_or_path, revision)
    return {"kind": "pretrained checkpoint", "base_model": name, "base_revision": revision, "base_record": record}


def check_embedding_manifest(manifest):
    """Refuse embeddings that were not produced by an eligible model (e.g. older, banned runs)."""
    eligibility = manifest.get("eligibility")
    if not eligibility:
        raise IneligibleModelError("Embedding manifest has no eligibility record; re-encode with scripts/encode_dense.py")
    check_hub_checkpoint(eligibility["base_model"], eligibility["base_revision"])
    if eligibility["kind"] == "fine-tuned checkpoint":
        if eligibility["provenance"].get("fine_tuning_data") != "course train_gold.json only":
            raise IneligibleModelError("Embeddings come from a checkpoint fine-tuned on non-course data")
    return eligibility
