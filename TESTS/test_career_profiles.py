"""Run with: .venv/Scripts/python.exe -m unittest discover -s TESTS"""

import csv
import runpy
import tempfile
import unittest
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
prepare = runpy.run_path(str(ROOT / "SCRIPTS/01_prepare_career_profiles.py"))
embed = runpy.run_path(str(ROOT / "SCRIPTS/02_create_career_embeddings.py"))


class ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            prepare["MODEL_NAME"], revision=prepare["MODEL_REVISION"])

    def test_whole_tasks_reserved_despite_many_skills(self):
        skills = [f"Specialized skill {i}" for i in range(80)]
        tasks = [
            {"text": "Analyze laboratory samples and record the results.", "core": True, "support": 95},
            {"text": "Communicate findings to customers and explain next steps.", "core": True, "support": 90},
            {"text": "Maintain equipment and document routine maintenance.", "core": False, "support": 60},
        ]
        args = ("Lab Analyst", "Study samples to identify their properties.", skills, tasks, self.tokenizer)
        result = prepare["shorten_profile"](*args)
        text, count, skill_ids, task_ids = result
        self.assertEqual(result, prepare["shorten_profile"](*args))
        self.assertGreaterEqual(len(task_ids), 2)
        self.assertTrue(245 <= count <= 250)
        self.assertEqual(count, len(self.tokenizer.encode(text, add_special_tokens=True)))
        self.assertIn(args[1], text)
        self.assertIn(0, skill_ids)
        for i in task_ids:
            self.assertIn(tasks[i]["text"], text)

    def test_sparse_source_is_not_padded(self):
        text, count, skills, tasks = prepare["shorten_profile"](
            "Analyst", "Analyze information.", [], [], self.tokenizer)
        self.assertLess(count, 245)
        self.assertEqual(text, "Occupation: Analyst Description: Analyze information.")
        self.assertEqual((skills, tasks), ([], []))

    def test_oversized_description_fails_without_slicing(self):
        with self.assertRaisesRegex(ValueError, "description exceed"):
            prepare["shorten_profile"]("Analyst", "A complete sentence. " * 100,
                                        [], [], self.tokenizer)

    def test_embedding_loader_prefers_short_profile_preserves_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["O*NET-SOC Code", "Title", "text", "profile_text"])
                writer.writeheader()
                for code in ["22", "11"]:
                    writer.writerow({"O*NET-SOC Code": code, "Title": code,
                                     "text": "legacy long text", "profile_text": "short " + code})
            texts, metadata = embed["load_profiles"](path)
            self.assertEqual(texts, ["short 22", "short 11"])
            self.assertEqual([r["O*NET-SOC Code"] for r in metadata], ["22", "11"])


if __name__ == "__main__":
    unittest.main()
