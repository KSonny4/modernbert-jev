"""Tests for chunked session runs in DecisionModel._logits."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from modernbert_jev import runtime
from modernbert_jev.runtime import DecisionModel


class CharTokenizer:
    """Right-padded character tokenizer with the encode_batch shape used by runtime."""

    def encode_batch(self, pairs):
        seqs = [[ord(c) for c in f"{a}|{b}"] for a, b in pairs]
        width = max(map(len, seqs))
        return [
            SimpleNamespace(
                ids=s + [0] * (width - len(s)),
                attention_mask=[1] * len(s) + [0] * (width - len(s)),
                type_ids=[0] * width,
            )
            for s in seqs
        ]


class FakeSession:
    """Records feed shapes; each row's logits depend on that row's real content."""

    def __init__(self):
        self.shapes = []

    def run(self, _outputs, feed):
        ids, mask = feed["input_ids"], feed["attention_mask"]
        self.shapes.append(ids.shape)
        return [np.stack([ids.sum(axis=1), mask.sum(axis=1), ids[:, 0]], axis=1)]


def make_model():
    model = object.__new__(DecisionModel)
    model.tokenizer = CharTokenizer()
    model.session = FakeSession()
    model.input_names = ["input_ids", "attention_mask"]
    return model


PAIRS = [("x" * n, "hyp") for n in (5, 300, 2, 80, 1, 600, 40, 7)]


class LogitsChunkTest(unittest.TestCase):
    def test_chunked_matches_unchunked_and_respects_budget(self):
        model = make_model()
        with mock.patch.object(runtime, "BATCH_ATTN_BUDGET", 2**40):
            full, full_tokens = model._logits(PAIRS)
        self.assertEqual(len(model.session.shapes), 1)

        model = make_model()
        budget = 700 * 700
        with mock.patch.object(runtime, "BATCH_ATTN_BUDGET", budget):
            chunked, tokens = model._logits(PAIRS)

        np.testing.assert_array_equal(chunked, full)
        self.assertGreater(len(model.session.shapes), 1)
        for rows, width in model.session.shapes:
            self.assertTrue(rows == 1 or rows * width * width <= budget)
        self.assertEqual(tokens, full_tokens)

    def test_token_count_is_sum_of_mask(self):
        model = make_model()
        expected = sum(len(f"{a}|{b}") for a, b in PAIRS)
        with mock.patch.object(runtime, "BATCH_ATTN_BUDGET", 1):
            _, tokens = model._logits(PAIRS)
        self.assertEqual(tokens, expected)

    def test_short_request_is_one_call(self):
        model = make_model()
        logits, _ = model._logits([("short", "hyp"), ("a", "b"), ("c", "d")])
        self.assertEqual(len(model.session.shapes), 1)
        self.assertEqual(logits.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
