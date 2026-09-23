"""Input alignment checks for script 04; no model download required."""

import csv
import runpy
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / "SCRIPTS/04_create_user_embeddings.py"))


class UserEmbeddingTests(unittest.TestCase):
    def load_rows(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["user_id", "profile_text"])
                writer.writeheader()
                writer.writerows(rows)
            return MODULE["load_profiles"](path)

    def test_order_is_not_sorted(self):
        texts, metadata = self.load_rows([
            {"user_id": "z", "profile_text": "First person"},
            {"user_id": "a", "profile_text": "Second person"},
        ])
        self.assertEqual(texts, ["First person", "Second person"])
        self.assertEqual(metadata, [{"user_id": "z"}, {"user_id": "a"}])

    def test_duplicate_id_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate user ID"):
            self.load_rows([{"user_id": "a", "profile_text": "text"}] * 2)

    def test_empty_text_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.load_rows([{"user_id": "a", "profile_text": " "}])

    def test_model_revision_matches_career_script(self):
        career = runpy.run_path(str(ROOT / "SCRIPTS/02_create_career_embeddings.py"))
        for key in ("MODEL_NAME", "MODEL_REVISION"):
            self.assertEqual(MODULE[key], career[key])


if __name__ == "__main__":
    unittest.main()
