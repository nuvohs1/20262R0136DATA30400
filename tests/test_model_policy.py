import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from review_retrieval.model_policy import (ALLOWED_CHECKPOINTS, DEFAULT_CHECKPOINT, PROVENANCE_FILE,
                                           PSEUDO_FINE_TUNING_DATA, IneligibleModelError, check_embedding_manifest, check_model)

REVISION = ALLOWED_CHECKPOINTS[DEFAULT_CHECKPOINT]["revision"]


class ModelPolicyTest(unittest.TestCase):
    def test_allowed_checkpoint_and_alias(self):
        self.assertEqual(check_model(DEFAULT_CHECKPOINT, REVISION)["base_model"], DEFAULT_CHECKPOINT)
        self.assertEqual(check_model("bert-base-uncased", REVISION)["base_model"], DEFAULT_CHECKPOINT)

    def test_unlisted_checkpoints_are_refused(self):
        for name in ("intfloat/e5-base-v2", "sentence-transformers/all-MiniLM-L6-v2", "roberta-base"):
            with self.assertRaises(IneligibleModelError):
                check_model(name, "main")

    def test_wrong_revision_is_refused(self):
        with self.assertRaises(IneligibleModelError):
            check_model(DEFAULT_CHECKPOINT, "main")
        with self.assertRaises(IneligibleModelError):
            check_model(DEFAULT_CHECKPOINT, None)

    def test_local_checkpoint_requires_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(IneligibleModelError):
                check_model(tmp)
            provenance = {"base_model": DEFAULT_CHECKPOINT, "base_revision": REVISION,
                          "fine_tuning_data": "course train_gold.json only", "train_gold_sha256": "abc"}
            (Path(tmp) / PROVENANCE_FILE).write_text(json.dumps(provenance), encoding="utf-8")
            self.assertEqual(check_model(tmp, train_gold_sha256="abc")["kind"], "fine-tuned checkpoint")
            with self.assertRaises(IneligibleModelError):
                check_model(tmp, train_gold_sha256="different")
            (Path(tmp) / PROVENANCE_FILE).write_text(json.dumps({**provenance, "base_model": "intfloat/e5-base-v2"}),
                                                     encoding="utf-8")
            with self.assertRaises(IneligibleModelError):
                check_model(tmp)

    def test_self_trained_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            provenance = {"base_model": DEFAULT_CHECKPOINT, "base_revision": REVISION,
                          "fine_tuning_data": PSEUDO_FINE_TUNING_DATA, "train_gold_sha256": "abc"}
            (Path(tmp) / PROVENANCE_FILE).write_text(json.dumps(provenance), encoding="utf-8")
            with self.assertRaises(IneligibleModelError):  # no pseudo-label fingerprint
                check_model(tmp)
            provenance.update(pseudo_labels_sha256="def", pseudo_set="pseudo-full")
            (Path(tmp) / PROVENANCE_FILE).write_text(json.dumps(provenance), encoding="utf-8")
            self.assertEqual(check_model(tmp, train_gold_sha256="abc")["kind"], "fine-tuned checkpoint")
            (Path(tmp) / PROVENANCE_FILE).write_text(json.dumps({**provenance, "fine_tuning_data": "external"}),
                                                     encoding="utf-8")
            with self.assertRaises(IneligibleModelError):
                check_model(tmp)

    def test_embedding_manifest(self):
        with self.assertRaises(IneligibleModelError):
            check_embedding_manifest({"model": "intfloat/e5-base-v2"})
        manifest = {"eligibility": check_model(DEFAULT_CHECKPOINT, REVISION)}
        self.assertEqual(check_embedding_manifest(manifest)["base_model"], DEFAULT_CHECKPOINT)


if __name__ == "__main__":
    unittest.main()
