"""Per-question context reaches the premise; string instructions do not change it."""

import unittest

from modernbert_jev.api import parse_request, state_text
from modernbert_jev.runtime import DecisionModel


class ContextPremiseTest(unittest.TestCase):
    def test_dict_candidates_get_own_premise(self):
        state = {"query": "Can you delete my alarm at 7:00 a.m.?", "task": "Rank."}
        raw = {
            "state": state,
            "model": "m",
            "questions": {
                "doc_0": {
                    "type": "choice",
                    "instructions": {"candidate": "ALARM_TOOL", "task": "Assess."},
                    "criteria": {"no": "Not relevant.", "yes": "Relevant."},
                },
                "doc_1": {
                    "type": "choice",
                    "instructions": {"candidate": "WEATHER_TOOL", "task": "Assess."},
                    "criteria": {"no": "Not relevant.", "yes": "Relevant."},
                },
                "plain": {
                    "type": "choice",
                    "instructions": "Which team",
                    "criteria": {"no": "Not relevant.", "yes": "Relevant."},
                },
            },
        }
        request = parse_request(raw)

        seen = {}

        def fake_ask_choice(self, premise, question):
            seen[len(seen)] = premise
            return None, 0

        model = object.__new__(DecisionModel)
        model.name = "m"
        model._ask_choice = fake_ask_choice.__get__(model)
        model.ask(request)

        self.assertEqual(len(seen), 3)
        # Order follows dict insertion: doc_0, doc_1, plain.
        self.assertIn("ALARM_TOOL", seen[0])
        self.assertNotIn("WEATHER_TOOL", seen[0])
        self.assertIn("WEATHER_TOOL", seen[1])
        self.assertNotIn("ALARM_TOOL", seen[1])
        self.assertEqual(seen[2], state_text(state))


if __name__ == "__main__":
    unittest.main()
