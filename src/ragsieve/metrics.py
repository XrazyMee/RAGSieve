from __future__ import annotations

from collections.abc import Sequence


def auroc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """Probability that a random positive outranks a random negative."""
    if not positive or not negative:
        return float("nan")
    ordered = sorted(
        [(float(value), 1) for value in positive] + [(float(value), 0) for value in negative],
        key=lambda item: item[0],
    )
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        stop = index
        while stop + 1 < len(ordered) and ordered[stop + 1][0] == ordered[index][0]:
            stop += 1
        average_rank = (index + stop) / 2.0 + 1.0
        rank_sum += average_rank * sum(ordered[position][1] for position in range(index, stop + 1))
        index = stop + 1
    count = len(positive)
    return (rank_sum - count * (count + 1) / 2.0) / (count * len(negative))


def tpr_at_fpr(
    positive: Sequence[float], negative: Sequence[float], *, target: float = 0.05
) -> float:
    """True-positive rate at a strict threshold with at most the target FPR."""
    if not positive or not negative:
        return float("nan")
    if not 0.0 <= target < 1.0:
        raise ValueError("target must be in [0, 1)")
    allowed = int(len(negative) * target)
    threshold = sorted((float(value) for value in negative), reverse=True)[allowed]
    return sum(float(value) > threshold for value in positive) / len(positive)


def detection_metrics(
    positive: Sequence[float],
    negative: Sequence[float],
    *,
    threshold: float = 1.0,
    target_fpr: float = 0.05,
) -> dict[str, float | int]:
    """Metrics used by the paper's document-level detection tables."""
    if not positive or not negative:
        raise ValueError("both positive and negative scores are required")
    return {
        "positive_documents": len(positive),
        "clean_documents": len(negative),
        "auroc": auroc(positive, negative),
        "tpr_at_5pct_fpr": tpr_at_fpr(positive, negative, target=target_fpr),
        "fixed_threshold_tpr": sum(value >= threshold for value in positive) / len(positive),
        "fixed_threshold_clean_fpr": sum(value >= threshold for value in negative) / len(negative),
    }
