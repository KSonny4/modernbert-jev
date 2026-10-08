"""Confidence formulas, matching the TypeSafe API definitions.

Choice confidence measures how far the top probability sits above an even
split: ``(p_max - 1/n) / (1 - 1/n)``. Score confidence additionally counts how
far probability sits from the most likely level. Noul answers carry no
separate confidence; the noul value itself is the probability of yes.
"""

from __future__ import annotations


def choice_confidence(probabilities: list[float]) -> float:
    """Confidence of a choice answer from its probability distribution."""
    n = len(probabilities)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    peak = max(probabilities)
    return min(1.0, max(0.0, (n * peak - 1.0) / (n - 1.0)))


def score_confidence(probabilities: list[float]) -> float:
    """Confidence of a score answer from its per-level distribution."""
    n = len(probabilities)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    m = probabilities.index(max(probabilities))
    spread = sum(p * abs(i - m) for i, p in enumerate(probabilities))
    even_spread = sum(abs(i - (n - 1) / 2.0) for i in range(n)) / n
    return max(0.0, 1.0 - spread / even_spread)


def expected_score(probabilities: list[float]) -> float:
    """Expected level of a score answer; may fall between two levels."""
    return sum(i * p for i, p in enumerate(probabilities))
