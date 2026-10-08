"""Decision runtime: NLI cross-encoder plus calibrated heads, ONNX CPU only.

A model directory holds ``model.onnx``, ``tokenizer.json``, ``config.json``
and an optional ``calibration.json`` with one temperature per head. Every
question becomes one batched premise/hypothesis inference: choice and score
softmax over per-option entailment logits, noul renormalizes entailment
against contradiction.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from modernbert_jev.api import (
    AnyAnswer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
    state_text,
)
from modernbert_jev.confidence import choice_confidence, expected_score, score_confidence

ort.set_default_logger_severity(3)

# Fallback NLI label order (cross-encoder style) when config lacks names.
DEFAULT_CONTRACTION = 0
DEFAULT_ENTAILMENT = 1


def label_indices(id2label: object) -> tuple[int, int]:
    """(contradiction_idx, entailment_idx) from a config id2label map."""
    if isinstance(id2label, dict):
        names = {int(k): str(v).lower() for k, v in id2label.items()}
        contra = next((i for i, n in names.items() if "contra" in n), None)
        entail = next((i for i, n in names.items() if "entail" in n), None)
        if contra is not None and entail is not None:
            return contra, entail
    return DEFAULT_CONTRACTION, DEFAULT_ENTAILMENT

# Characters of state kept before tokenization; keeps long inputs bounded.
MAX_STATE_CHARS = 20000
# Word-window chunking for states that overflow the encoder window.
CHUNK_WORDS = 300
CHUNK_OVERLAP = 75
CHUNK_CAP = int(os.environ.get("MJEV_CHUNK_CAP", "6"))
# Null-premise subtraction weight for choice (contextual calibration).
# 0.5 halves many-option bias while leaving few-option sets unchanged.
PMI_ALPHA = float(os.environ.get("MJEV_PMI", "0.5"))
NULL_PREMISE = "N/A"
# States below this length take a single inference pass (fast path).
SINGLE_PASS_CHARS = 1500


@dataclass(frozen=True)
class Calibration:
    choice: float = 1.0
    score: float = 1.0
    noul: float = 1.0

    @classmethod
    def from_dict(cls, raw: object) -> "Calibration":
        if not isinstance(raw, dict):
            return cls()
        def _temp(key: str) -> float:
            try:
                value = float(raw.get(key, 1.0))
            except (TypeError, ValueError):
                return 1.0
            return value if value > 0 else 1.0

        return cls(choice=_temp("choice"), score=_temp("score"), noul=_temp("noul"))


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / max(temperature, 1e-6)
    scaled -= scaled.max()
    exp = np.exp(scaled)
    return exp / exp.sum()


class DecisionModel:
    """A loaded decision model directory, ready to answer questions."""

    def __init__(self, model_dir: str, threads: int = 1) -> None:
        self.model_dir = model_dir
        self.name = os.path.basename(os.path.abspath(model_dir))
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.max_length = 512
        self.contra_idx, self.entail_idx = DEFAULT_CONTRACTION, DEFAULT_ENTAILMENT
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as fh:
                raw = json.load(fh)
            self.max_length = int(raw.get("max_position_embeddings", 512) or 512)
            self.contra_idx, self.entail_idx = label_indices(raw.get("id2label"))
        pad_id = self.tokenizer.token_to_id("[PAD]")
        self.tokenizer.enable_padding(pad_id=pad_id if pad_id is not None else 0)
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.calibration = Calibration()
        cal_path = os.path.join(model_dir, "calibration.json")
        if os.path.exists(cal_path):
            with open(cal_path) as fh:
                self.calibration = Calibration.from_dict(json.load(fh))
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, threads)
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            os.path.join(model_dir, "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [i.name for i in self.session.get_inputs()]

    def _logits(self, pairs: list[tuple[str, str]]) -> tuple[np.ndarray, int]:
        enc = self.tokenizer.encode_batch(pairs)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed: dict[str, np.ndarray] = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = ids
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = mask
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in enc], dtype=np.int64
            )
        logits = self.session.run(None, feed)[0]
        return np.asarray(logits), int(mask.sum())

    @staticmethod
    def _windows(premise: str) -> list[str]:
        """Split a long premise into overlapping word windows."""
        words = premise.split()
        if len(words) <= CHUNK_WORDS:
            return [premise]
        step = CHUNK_WORDS - CHUNK_OVERLAP
        if (len(words) - CHUNK_OVERLAP + step - 1) // step > CHUNK_CAP:
            step = max(1, (len(words) - CHUNK_WORDS) // (CHUNK_CAP - 1))
        out = [" ".join(words[i : i + CHUNK_WORDS]) for i in range(0, len(words), step)]
        if len(out) > CHUNK_CAP:
            out = out[: CHUNK_CAP - 1] + [" ".join(words[-CHUNK_WORDS:])]
        return out

    def _window_logits(
        self, premise: str, hypotheses: list[str]
    ) -> tuple[np.ndarray, int]:
        """Max-pooled (windows, hypotheses, 3) NLI logits in one batch."""
        windows = [premise] if len(premise) <= SINGLE_PASS_CHARS else self._windows(premise)
        pairs = [(window, hyp) for window in windows for hyp in hypotheses]
        logits, tokens = self._logits(pairs)
        return logits.reshape(len(windows), len(hypotheses), 3), tokens

    # Hypothesis template variant: v1 prefixes instructions, v2 uses bare
    # statements, v3 additionally strips noul instructions to the core
    # question. Selected by MJEV_TEMPLATE, default v3.
    template: str = os.environ.get("MJEV_TEMPLATE", "v3")
    # Choice hypothesis content: description only, label only, or both.
    hyp: str = os.environ.get("MJEV_HYP", "desc")

    @classmethod
    def _choice_hypothesis(cls, question: ChoiceQuestion, label: str) -> str:
        desc = question.criteria[label].strip()
        words = label.replace("_", " ")
        statement = desc if desc else words
        if cls.template == "v1":
            text = f"{question.instructions.strip()}: {label}"
            if desc:
                text += f" - {desc}"
            return text
        if cls.hyp == "label":
            statement = words
        elif cls.hyp == "both" and desc:
            statement = f"{words}: {desc}"
        return statement[0].upper() + statement[1:] if statement else label

    @staticmethod
    def _core_question(instructions: str) -> str:
        """Reduce instructions to the bare question: up to the first "?".

        Falls back to the first sentence for declarative instructions.
        Deliberately no aux-inversion: heuristic rewriting corrupts
        multi-word subjects, and the NLI head matches keywords anyway.
        """
        text = instructions.strip()
        head, mark, _ = text.partition("?")
        if mark:
            return (head + mark).strip()
        return text.split(". ")[0].strip() or text

    @classmethod
    def _noul_hypothesis(cls, question: NoulQuestion) -> str:
        if cls.template == "v1":
            hypothesis = question.instructions.strip()
        elif cls.template == "v3":
            hypothesis = cls._core_question(question.instructions)
        else:
            hypothesis = question.instructions.strip().split(". ")[0]
        criteria = question.criteria
        if isinstance(criteria, dict) and "true" in criteria and "false" in criteria:
            hypothesis += f" (Yes means: {criteria['true']} No means: {criteria['false']})"
        elif isinstance(criteria, str) and criteria.strip():
            hypothesis += f" ({criteria.strip()})"
        elif isinstance(criteria, list) and criteria:
            hypothesis += " (" + "; ".join(str(c) for c in criteria) + ")"
        elif criteria and cls.template == "v1":
            hypothesis += f" ({json.dumps(criteria, sort_keys=True)})"
        return hypothesis

    def _pmi_adjust(self, hyps: list[str], pooled: np.ndarray) -> tuple[np.ndarray, int]:
        """Subtract per-hypothesis null-premise bias (contextual calibration)."""
        if PMI_ALPHA == 0:
            return pooled, 0
        null_pairs = [(NULL_PREMISE, hyp) for hyp in hyps]
        null_logits, null_tokens = self._logits(null_pairs)
        return pooled - PMI_ALPHA * null_logits[:, self.entail_idx], null_tokens

    def _ask_choice(
        self, premise: str, question: ChoiceQuestion
    ) -> tuple[AnyAnswer, int]:
        labels = list(question.criteria)
        hyps = [self._choice_hypothesis(question, label) for label in labels]
        stacked, tokens = self._window_logits(premise, hyps)
        pooled = stacked[:, :, self.entail_idx].max(axis=0)
        pooled, null_tokens = self._pmi_adjust(hyps, pooled)
        tokens += null_tokens
        probs = _softmax(pooled, self.calibration.choice)
        dist = {label: float(p) for label, p in zip(labels, probs)}
        best = labels[int(np.argmax(probs))]
        ordered = [dist[label] for label in labels]
        return (
            ChoiceAnswer(choice=best, confidence=choice_confidence(ordered), probabilities=dist),
            tokens,
        )

    def _ask_score(
        self, premise: str, question: ScoreQuestion
    ) -> tuple[AnyAnswer, int]:
        if self.template == "v1":
            hyps = [f"{question.instructions.strip()}: {level}" for level in question.criteria]
        else:
            hyps = [level.strip() for level in question.criteria]
        stacked, tokens = self._window_logits(premise, hyps)
        best = stacked[:, :, self.entail_idx].max(axis=0)
        probs = _softmax(best, self.calibration.score)
        dist = {str(i): float(p) for i, p in enumerate(probs)}
        ordered = [dist[str(i)] for i in range(len(question.criteria))]
        return (
            ScoreAnswer(
                score=float(expected_score(ordered)),
                confidence=score_confidence(ordered),
                legend={str(i): level for i, level in enumerate(question.criteria)},
                probabilities=dist,
            ),
            tokens,
        )

    def _ask_noul(
        self, premise: str, question: NoulQuestion
    ) -> tuple[AnyAnswer, int]:
        hypothesis = self._noul_hypothesis(question)
        stacked, tokens = self._window_logits(premise, [hypothesis])
        best_window = int(np.argmax(stacked[:, 0, self.entail_idx]))
        dist = _softmax(stacked[best_window, 0], self.calibration.noul)
        mass = dist[self.entail_idx] + dist[self.contra_idx]
        value = float(dist[self.entail_idx] / mass) if mass > 0 else 0.5
        return NoulAnswer(noul=value), tokens

    def ask(self, request: SystemOneRequest) -> SystemOneResponse:
        """Answer every question in one request against the same state."""
        premise = state_text(request.state)[:MAX_STATE_CHARS]
        answers: dict[str, AnyAnswer] = {}
        total_tokens = 0
        for name, question in request.questions.items():
            if isinstance(question, ChoiceQuestion):
                answer, tokens = self._ask_choice(premise, question)
            elif isinstance(question, ScoreQuestion):
                answer, tokens = self._ask_score(premise, question)
            elif isinstance(question, NoulQuestion):
                answer, tokens = self._ask_noul(premise, question)
            else:  # pragma: no cover - parse_request only builds these three
                raise TypeError(f"unknown question type: {type(question)!r}")
            answers[name] = answer
            total_tokens += tokens
        return SystemOneResponse(
            model=self.name, answers=answers, input_tokens=total_tokens
        )
