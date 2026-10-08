"""Tests for config-driven NLI label mapping."""

import unittest

from modernbert_jev.runtime import label_indices


class LabelTest(unittest.TestCase):
    def test_cross_encoder_order(self):
        c, e = label_indices({"0": "contradiction", "1": "entailment", "2": "neutral"})
        self.assertEqual((c, e), (0, 1))

    def test_glue_order(self):
        c, e = label_indices({"0": "entailment", "1": "neutral", "2": "contradiction"})
        self.assertEqual((c, e), (2, 0))

    def test_case_insensitive(self):
        c, e = label_indices({"0": "ENTAILMENT", "1": "NEUTRAL", "2": "CONTRADICTION"})
        self.assertEqual((c, e), (2, 0))

    def test_fallback(self):
        c, e = label_indices(None)
        self.assertEqual((c, e), (0, 1))
        c, e = label_indices({"0": "LABEL_0", "1": "LABEL_1"})
        self.assertEqual((c, e), (0, 1))


if __name__ == "__main__":
    unittest.main()
