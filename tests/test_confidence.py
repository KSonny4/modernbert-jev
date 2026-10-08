"""Tests for confidence formulas against TypeSafe's documented examples."""

import unittest

from modernbert_jev.confidence import choice_confidence, expected_score, score_confidence


class ConfidenceTest(unittest.TestCase):
    def test_choice_example(self):
        # (0.6, 0.3, 0.1) and (0.6, 0.2, 0.2) both give 0.4.
        self.assertAlmostEqual(choice_confidence([0.6, 0.3, 0.1]), 0.4)
        self.assertAlmostEqual(choice_confidence([0.6, 0.2, 0.2]), 0.4)

    def test_choice_bounds(self):
        self.assertAlmostEqual(choice_confidence([0.5, 0.5]), 0.0)
        self.assertAlmostEqual(choice_confidence([1.0, 0.0]), 1.0)
        self.assertAlmostEqual(choice_confidence([1.0]), 1.0)
        self.assertAlmostEqual(choice_confidence([]), 0.0)

    def test_score_adjacent_vs_opposite(self):
        # Torn between adjacent levels: 0.25. Between opposite ends: 0.
        self.assertAlmostEqual(score_confidence([0.0, 0.5, 0.5]), 0.25)
        self.assertAlmostEqual(score_confidence([0.5, 0.0, 0.5]), 0.0)

    def test_score_docs_example(self):
        # bug_severity (0, 0.57, 0.43): 1 - 0.43 / (2/3) ~= 0.355.
        self.assertAlmostEqual(score_confidence([0.0, 0.57, 0.43]), 0.355, places=3)

    def test_score_certain(self):
        self.assertAlmostEqual(score_confidence([0.0, 1.0, 0.0]), 1.0)
        self.assertAlmostEqual(score_confidence([1.0]), 1.0)

    def test_expected_score(self):
        self.assertAlmostEqual(expected_score([0.0, 0.5, 0.5]), 1.5)
        self.assertAlmostEqual(expected_score([0.25, 0.5, 0.25]), 1.0)


if __name__ == "__main__":
    unittest.main()
