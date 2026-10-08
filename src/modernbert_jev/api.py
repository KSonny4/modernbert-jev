"""TypeSafe-compatible request/response models for ``POST /v1/systemone``.

Covers the three question primitives (choice, score, noul) with the same
JSON shapes as the TypeSafe API, so existing clients can point at this
server by changing only the base URL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union

MAX_QUESTIONS = 256
MAX_OPTIONS = 256

QuestionType = Literal["choice", "score", "noul"]


class ApiError(ValueError):
    """A request the server must reject with HTTP 400."""


@dataclass(frozen=True)
class ChoiceQuestion:
    instructions: str
    criteria: dict[str, str]


@dataclass(frozen=True)
class ScoreQuestion:
    instructions: str
    criteria: list[str]


@dataclass(frozen=True)
class NoulQuestion:
    instructions: str
    criteria: Any = None


AnyQuestion = Union[ChoiceQuestion, ScoreQuestion, NoulQuestion]


@dataclass(frozen=True)
class SystemOneRequest:
    state: Any
    model: str
    questions: dict[str, AnyQuestion]


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    confidence: float
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "choice": self.choice,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
        }


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    confidence: float
    legend: dict[str, str]
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "score",
            "score": self.score,
            "confidence": self.confidence,
            "legend": dict(self.legend),
            "probabilities": dict(self.probabilities),
        }


@dataclass(frozen=True)
class NoulAnswer:
    noul: float

    def to_dict(self) -> dict[str, Any]:
        return {"type": "noul", "noul": self.noul}


AnyAnswer = Union[ChoiceAnswer, ScoreAnswer, NoulAnswer]


@dataclass(frozen=True)
class SystemOneResponse:
    model: str
    answers: dict[str, AnyAnswer]
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "answers": {k: v.to_dict() for k, v in self.answers.items()},
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            },
        }


def _need_instructions(raw: Any, name: str) -> str:
    if not isinstance(raw, dict):
        raise ApiError(f"question {name!r} must be an object")
    instructions = raw.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ApiError(f"question {name!r} needs non-empty 'instructions'")
    return instructions


def _parse_choice(raw: dict[str, Any], name: str) -> ChoiceQuestion:
    instructions = _need_instructions(raw, name)
    criteria = raw.get("criteria")
    if not isinstance(criteria, dict) or not criteria:
        raise ApiError(f"choice question {name!r} needs a non-empty 'criteria' map")
    if len(criteria) > MAX_OPTIONS:
        raise ApiError(f"choice question {name!r} has more than {MAX_OPTIONS} options")
    clean: dict[str, str] = {}
    for label, desc in criteria.items():
        if not isinstance(label, str) or not label:
            raise ApiError(f"choice question {name!r} has a non-string option label")
        if not isinstance(desc, str):
            raise ApiError(f"choice question {name!r} option {label!r} needs a string description")
        clean[label] = desc
    return ChoiceQuestion(instructions=instructions, criteria=clean)


def _parse_score(raw: dict[str, Any], name: str) -> ScoreQuestion:
    instructions = _need_instructions(raw, name)
    criteria = raw.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        raise ApiError(f"score question {name!r} needs a non-empty 'criteria' list")
    if len(criteria) > MAX_OPTIONS:
        raise ApiError(f"score question {name!r} has more than {MAX_OPTIONS} levels")
    if not all(isinstance(level, str) for level in criteria):
        raise ApiError(f"score question {name!r} levels must all be strings")
    return ScoreQuestion(instructions=instructions, criteria=list(criteria))


def _parse_noul(raw: dict[str, Any], name: str) -> NoulQuestion:
    instructions = _need_instructions(raw, name)
    return NoulQuestion(instructions=instructions, criteria=raw.get("criteria"))


def parse_request(raw: Any) -> SystemOneRequest:
    """Parse and validate a ``POST /v1/systemone`` body."""
    if not isinstance(raw, dict):
        raise ApiError("request body must be a JSON object")
    if "state" not in raw:
        raise ApiError("request needs 'state'")
    model = raw.get("model")
    if not isinstance(model, str) or not model:
        raise ApiError("request needs a 'model' name")
    questions = raw.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ApiError("request needs a non-empty 'questions' map")
    if len(questions) > MAX_QUESTIONS:
        raise ApiError(f"request has more than {MAX_QUESTIONS} questions")
    parsed: dict[str, AnyQuestion] = {}
    for name, item in questions.items():
        if not isinstance(item, dict):
            raise ApiError(f"question {name!r} must be an object")
        qtype = item.get("type")
        if qtype == "choice":
            parsed[name] = _parse_choice(item, name)
        elif qtype == "score":
            parsed[name] = _parse_score(item, name)
        elif qtype == "noul":
            parsed[name] = _parse_noul(item, name)
        else:
            raise ApiError(f"question {name!r} has unknown type {qtype!r}")
    return SystemOneRequest(state=raw["state"], model=model, questions=parsed)


def state_text(state: Any) -> str:
    """Render request state (text or JSON) as model input text."""
    if isinstance(state, str):
        return state
    import json

    return json.dumps(state, sort_keys=True, ensure_ascii=False)
