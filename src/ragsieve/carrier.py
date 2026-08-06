from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch


def language_model_profile(
    losses: torch.Tensor, *, windows: Sequence[int] = (8, 16)
) -> dict[str, float]:
    """Summarize local bursts and change points in a causal-LM loss trace."""
    if not windows or any(window < 1 for window in windows):
        raise ValueError("windows must contain positive integers")
    values = losses.float()
    output: dict[str, float] = {}
    for window in windows:
        if values.numel() == 0:
            burst = change_point = 0.0
        else:
            rolling = (
                values.unfold(0, window, 1).mean(dim=1)
                if values.numel() >= window
                else values.mean().reshape(1)
            )
            burst = float(rolling.max() - rolling.median())
            if values.numel() >= 2 * window:
                boundaries = range(window, values.numel() - window + 1, max(1, window // 2))
                change_point = max(
                    (
                        abs(
                            float(
                                values[index - window : index].mean()
                                - values[index : index + window].mean()
                            )
                        )
                        for index in boundaries
                    ),
                    default=0.0,
                )
            else:
                change_point = 0.0
        output[f"lm_window_burst_{window}"] = burst
        output[f"lm_change_point_{window}"] = change_point
    return output


def variance_adjusted_transition_evidence(
    signals: Mapping[str, float],
    *,
    windows: Sequence[int] = (8, 16),
    reference_window: int = 16,
    reference_bits: float = 5.0,
) -> float:
    """Information-gated multiscale carrier evidence without corpus calibration."""
    if not windows or any(window < 1 for window in windows):
        raise ValueError("windows must contain positive integers")
    if reference_window < 1 or reference_bits <= 0.0:
        raise ValueError("reference_window and reference_bits must be positive")
    maximum_ratio = 0.0
    for window in windows:
        threshold_nats = reference_bits * math.log(2.0) * math.sqrt(reference_window / window)
        burst = float(signals[f"lm_window_burst_{window}"])
        change = float(signals[f"lm_change_point_{window}"])
        maximum_ratio = max(maximum_ratio, burst / threshold_nats, change / threshold_nats)
    return max(0.0, maximum_ratio - 1.0)


def semantic_tail_excess(probability: float, *, alpha: float = 0.05) -> float:
    """Normalized semantic-transition evidence beyond a fixed tail level."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    bounded = max(math.ulp(1.0), min(1.0, float(probability)))
    return max(0.0, math.log(alpha / bounded) / math.log(1.0 / alpha))
