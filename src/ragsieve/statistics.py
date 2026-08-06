from __future__ import annotations

from collections.abc import Sequence


def empirical_upper_tail_probability(value: float, reference: Sequence[float]) -> float:
    """Finite mid-rank upper-tail probability against a query-local reference tail."""
    if not reference:
        raise ValueError("the reference tail must not be empty")
    greater = sum(float(other) > float(value) for other in reference)
    equal = sum(float(other) == float(value) for other in reference)
    return (0.5 + greater + 0.5 * equal) / (len(reference) + 1.0)


def simes_probability(probabilities: Sequence[float]) -> float:
    """Simes combination for a sparse, variable-size family of hypotheses."""
    if not probabilities:
        raise ValueError("at least one probability is required")
    ordered = sorted(float(value) for value in probabilities)
    count = len(ordered)
    return min(1.0, min(count * value / rank for rank, value in enumerate(ordered, 1)))
