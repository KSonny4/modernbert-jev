"""Train inputs for --data jevman must match the runtime text, windows and label index."""

import json
import os
import tempfile
import types
import unittest

try:
    import torch
    from modernbert_jev.api import state_text
    from modernbert_jev.runtime import MAX_STATE_CHARS, NULL_PREMISE, PMI_ALPHA, DecisionModel
    from modernbert_jev.train import decision_scores, jevman_example, load_jsonl, split_held_out
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

LONG_LOG = " ".join(f"step{i}" for i in range(700))  # 700 words: 3 windows

ROWS = [
    {
        "submission": "semif", "game": 1, "frame": 118, "game_score": 3210,
        "state": {"pacman": [3, 4], "ghosts": [[5, 4]]},
        "question": {
            "id": "move",
            "instructions": "Which way should Pac-Man move?",
            "criteria": {
                "left": "Eat dots, away from the ghost",
                "right": "DANGER: moves toward a ghost",
            },
        },
        "choice": "left",
    },
    {
        "submission": "semif", "game": 2, "frame": 7, "game_score": 40,
        "state": "Pac-Man at the corner. Power pellet ahead.",
        "question": {
            "id": "move",
            "instructions": "Which way should Pac-Man move?",
            "criteria": {
                "up": "Walk into the TRAP corridor",
                "down": "Take the power pellet",
            },
        },
        "choice": "down",
    },
    {
        "submission": "semif", "game": 3, "frame": 9, "game_score": 900,
        "state": {"pacman": [1, 1], "maze": "#" * 2000, "log": LONG_LOG},
        "question": {
            "id": "move",
            "instructions": "Which way should Pac-Man move?",
            "criteria": {
                "left": "Eat dots, away from the ghost",
                "right": "DANGER: moves toward a ghost",
            },
        },
        "choice": "right",
    },
    {
        "submission": "semi", "game": 4, "frame": 2, "game_score": 10,
        "state": "Pac-Man at the corner. Power pellet ahead.",
        "question": {
            "id": "move",
            "instructions": "Which way should Pac-Man move?",
            "criteria": {
                "up": "Walk into the TRAP corridor",
                "down": "Take the power pellet",
            },
        },
        "choice": "up",
    },
]


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class JevmanExampleTest(unittest.TestCase):
    def test_pairs_match_runtime_text_and_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "decisions.jsonl")
            with open(path, "w") as fh:
                for row in ROWS:
                    fh.write(json.dumps(row) + "\n")
            rows = load_jsonl(path)
        self.assertEqual(len(rows), len(ROWS))
        for row in rows:
            ex = jevman_example(row)
            labels = list(row["question"]["criteria"])
            self.assertEqual(
                ex["hyps"], [_runtime_hyp(row["question"], label) for label in labels]
            )
            self.assertEqual(ex["target"], labels.index(row["choice"]))
            self.assertNotIn("maze", ex["premises"][0])

    def test_instructions_premise_is_question_text(self):
        ex = jevman_example(ROWS[0], "instructions")
        self.assertEqual(ex["premises"], [state_text(ROWS[0]["question"]["instructions"])[:MAX_STATE_CHARS]])

    def test_long_state_keeps_all_windows(self):
        ex = jevman_example(ROWS[2])
        expected = json.dumps(
            {"log": LONG_LOG, "pacman": [1, 1]}, sort_keys=True, ensure_ascii=False
        )
        self.assertGreater(len(ex["premises"]), 1)
        self.assertEqual(ex["premises"], DecisionModel._windows(expected))

    def test_decision_scores_max_over_windows_minus_null(self):
        ex = {"premises": ["w1", "w2"], "hyps": ["h1", "h2"], "target": 0, "desc": ["a", "b"]}
        entail = {("w1", "h1"): 1.0, ("w2", "h1"): 5.0, ("w1", "h2"): 3.0, ("w2", "h2"): 2.0,
                  (NULL_PREMISE, "h1"): 2.0, (NULL_PREMISE, "h2"): 1.0}
        seen: list[tuple[str, str]] = []

        def tokenizer(a, b, **_):
            seen[:] = list(zip(a, b))
            return {"idx": torch.arange(len(a))}

        def model(idx):
            vals = [[0.0, entail[pair]] for pair in seen]
            return types.SimpleNamespace(logits=torch.tensor(vals))

        (scores,) = decision_scores(model, tokenizer, [ex], "cpu", 16, 1)
        expected = [5.0 - PMI_ALPHA * 2.0, 3.0 - PMI_ALPHA * 1.0]
        for got, want in zip(scores.tolist(), expected):
            self.assertAlmostEqual(got, want, places=5)

    def test_split_holds_out_whole_games(self):
        rows = [{"submission": "a", "game": g} for g in range(1, 11)]
        rows = [{**r, "group": (r["submission"], r["game"])} for r in rows]
        train, held = split_held_out(rows, seed=7)
        self.assertEqual(len(held), 1)
        self.assertEqual(len(train) + len(held), len(rows))
        self.assertFalse({r["group"] for r in train} & {r["group"] for r in held})


def _runtime_hyp(question_dict, label):
    from modernbert_jev.api import ChoiceQuestion

    question = ChoiceQuestion(
        instructions=question_dict["instructions"],
        criteria=dict(question_dict["criteria"]),
    )
    return DecisionModel._choice_hypothesis(question, label)


if __name__ == "__main__":
    unittest.main()
