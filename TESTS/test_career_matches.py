"""Known-vector checks for ranking, tie handling, and input validation."""

import runpy
import unittest
from pathlib import Path

import numpy as np

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "SCRIPTS/05_match_careers.py"))


class CareerMatchTests(unittest.TestCase):
    def test_descending_similarity_and_code_ties(self):
        users = np.array([[1., 0.], [0., 1.]])
        careers = np.array([[0., 1.], [1., 0.], [1., 0.], [-1., 0.]])
        user_metadata = [{"user_id": "z"}, {"user_id": "a"}]
        career_metadata = [{"O*NET-SOC Code": code, "Title": code} for code in ["C", "B", "A", "D"]]
        rows = MODULE["rank_matches"](users, careers, user_metadata, career_metadata, top_k=3)
        self.assertEqual([r["career_code"] for r in rows[:3]], ["A", "B", "C"])
        self.assertEqual([r["cosine_similarity"] for r in rows[:3]], [1., 1., 0.])
        self.assertEqual([r["user_id"] for r in rows], ["z"] * 3 + ["a"] * 3)
        self.assertEqual(rows[3]["career_code"], "C")

    def test_non_normalized_vectors_rejected(self):
        with self.assertRaisesRegex(ValueError, "unit normalized"):
            MODULE["validate_embeddings"](np.zeros((1, 384)), [{}], 1, "Test")

    def test_misaligned_count_rejected(self):
        with self.assertRaisesRegex(ValueError, "aligned rows"):
            MODULE["validate_embeddings"](np.zeros((2, 384)), [{}], 2, "Test")


if __name__ == "__main__":
    unittest.main()
