"""Checks for source-grounded user profile extraction and token budgeting."""

import runpy
import unittest
from pathlib import Path

from transformers import AutoTokenizer

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "SCRIPTS/03_prepare_user_profiles.py"))


class UserProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(MODULE["MODEL_NAME"], revision=MODULE["MODEL_REVISION"])

    def test_balanced_source_grounded_profile(self):
        resume = (
            "ANALYST Summary Experienced analyst. "
            "Experience Designed reports to support budget decisions. "
            "Analyzed records to identify errors. Analyzed records to identify errors. "
            "Education Bachelor of Science : Accounting Example University "
            "Skills Python, SQL, Excel, budget analysis"
        )
        interests = "I enjoy investigating discrepancies and improving financial reporting."
        result = MODULE["build_profile"](resume, interests, self.tokenizer)
        text, count, _, selected = result
        self.assertEqual(result, MODULE["build_profile"](resume, interests, self.tokenizer))
        self.assertIn(interests, text)
        self.assertEqual(count, len(self.tokenizer.encode(text, add_special_tokens=True)))
        self.assertLessEqual(count, 250)
        for category in ("Education", "Experience", "Skills"):
            self.assertTrue(selected[category])
        for units in selected.values():
            for unit in units:
                self.assertIn(unit, MODULE["normalize"](resume))
        self.assertLessEqual(text.count("Analyzed records to identify errors."), 1)

    def test_degree_context_and_abbreviations(self):
        title, sections = MODULE["extract_resume"](
            "TEACHER Jane Doe Education M.Ed : Secondary Education 2007 St. Martin's University "
            "B.S : Biology 1998 Example University Skills Teaching, biology"
        )
        self.assertEqual(title, "TEACHER")
        self.assertEqual(len(sections["Education"]), 1)
        self.assertIn("M.Ed : Secondary Education", sections["Education"][0])
        self.assertIn("St. Martin's University", sections["Education"][0])
        self.assertIn("B.S : Biology", sections["Education"][0])

    def test_postposed_degree_keeps_institution(self):
        _, sections = MODULE["extract_resume"](
            "ANALYST Education 5/1998 Emory University Health education and Behavioral Research "
            "Masters of Public Health 5/1995 Harvard University Special Student Program "
            "Skills Research"
        )
        self.assertEqual(len(sections["Education"]), 1)
        self.assertIn("Emory University Health education and Behavioral Research Masters of Public Health",
                      sections["Education"][0])

    def test_missing_interests_fails(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            MODULE["build_profile"]("ANALYST Experience Analyzed data.", "", self.tokenizer)

    def test_parenthesized_skill_list_stays_intact(self):
        self.assertEqual(MODULE["split_list"]("Office (Excel, Word), SQL; Python"),
                         ["Office (Excel, Word)", " SQL", " Python"])


if __name__ == "__main__":
    unittest.main()
