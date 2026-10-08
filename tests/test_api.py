"""Tests for TypeSafe-compatible request parsing."""

import unittest

from modernbert_jev.api import ApiError, parse_request


QUICKSTART = {
    "state": "The integration keeps failing. Please help ASAP.",
    "model": "modernbert-jev",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this",
            "criteria": {
                "billing": "Payment or subscription issues",
                "technical": "Bugs or integration problems",
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated the customer appears",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "The message conveys urgency",
        },
    },
}


class ParseTest(unittest.TestCase):
    def test_quickstart_shape(self):
        req = parse_request(QUICKSTART)
        self.assertEqual(req.model, "modernbert-jev")
        self.assertEqual(set(req.questions), {"department", "frustration", "is_urgent"})
        self.assertEqual(
            req.questions["department"].criteria["technical"], "Bugs or integration problems"
        )
        self.assertEqual(len(req.questions["frustration"].criteria), 3)

    def test_rejects_unknown_type(self):
        bad = {"state": "x", "model": "m", "questions": {"q": {"type": "essay"}}}
        with self.assertRaises(ApiError):
            parse_request(bad)

    def test_rejects_empty_criteria(self):
        bad = {
            "state": "x",
            "model": "m",
            "questions": {"q": {"type": "choice", "instructions": "pick", "criteria": {}}},
        }
        with self.assertRaises(ApiError):
            parse_request(bad)

    def test_rejects_missing_state(self):
        with self.assertRaises(ApiError):
            parse_request({"model": "m", "questions": {}})

    def test_json_state_allowed(self):
        raw = dict(QUICKSTART, state={"ticket": "help", "priority": 1})
        self.assertEqual(parse_request(raw).state["priority"], 1)


if __name__ == "__main__":
    unittest.main()
