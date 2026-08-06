from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, Self

import torch

from .anchor import anchor_integrity_features
from .carrier import (
    language_model_profile,
    semantic_tail_excess,
    variance_adjusted_transition_evidence,
)
from .models import BERTScoreTraceScorer, CausalSurprisalScorer
from .statistics import empirical_upper_tail_probability


class CausalScorer(Protocol):
    def loss_profiles_batch(
        self, texts: Sequence[str], *, batch_size: int
    ) -> list[torch.Tensor]: ...


class SemanticScorer(Protocol):
    def encode_queries(self, texts: Sequence[str], *, batch_size: int) -> list[torch.Tensor]: ...

    def alignment_jumps_batch(
        self,
        texts: Sequence[str],
        query_embeddings: Sequence[torch.Tensor],
        *,
        batch_size: int,
    ) -> list[float]: ...


@dataclass(frozen=True)
class DetectorConfig:
    """The fixed paper configuration."""

    candidate_count: int = 5
    context_size: int = 20
    causal_model: str = "Qwen/Qwen3-0.6B-Base"
    semantic_model: str = "bert-base-uncased"
    semantic_layer: int = 9
    ppl_windows: tuple[int, ...] = (8, 16)
    reference_window: int = 16
    reference_bits: float = 5.0
    semantic_tail_alpha: float = 0.05
    decision_threshold: float = 1.0
    causal_batch_size: int = 5
    semantic_batch_size: int = 20

    def __post_init__(self) -> None:
        if self.candidate_count < 2:
            raise ValueError("candidate_count must be at least two")
        if self.context_size <= self.candidate_count:
            raise ValueError("context_size must include a nonempty reference tail")


class RAGSieveQueryDetector:
    """Apply the same online document detector to every retrieval event."""

    def __init__(
        self,
        config: DetectorConfig | None = None,
        *,
        device: str | int = "cuda:0",
        causal_scorer: CausalScorer | None = None,
        semantic_scorer: SemanticScorer | None = None,
    ) -> None:
        self.config = config or DetectorConfig()
        self.causal = causal_scorer or CausalSurprisalScorer(
            self.config.causal_model, device=device
        )
        self.semantic = semantic_scorer or BERTScoreTraceScorer(
            self.config.semantic_model,
            layer=self.config.semantic_layer,
            device=device,
        )

    def detect(self, query: str, documents: Sequence[str]) -> list[dict[str, float | bool]]:
        """Score top-ranked candidates using only the query and its retrieved documents."""
        config = self.config
        if len(documents) < config.context_size:
            raise ValueError(
                f"expected at least {config.context_size} documents, got {len(documents)}"
            )
        texts = list(documents[: config.context_size])
        base = anchor_integrity_features(texts, query, candidate_count=config.candidate_count)

        loss_profiles = self.causal.loss_profiles_batch(
            texts[: config.candidate_count], batch_size=config.causal_batch_size
        )
        language_profiles = [
            language_model_profile(profile, windows=config.ppl_windows) for profile in loss_profiles
        ]

        query_embedding = self.semantic.encode_queries([query], batch_size=1)[0]
        jumps = self.semantic.alignment_jumps_batch(
            texts,
            [query_embedding] * len(texts),
            batch_size=config.semantic_batch_size,
        )
        reference_jumps = jumps[config.candidate_count :]

        output: list[dict[str, float | bool]] = []
        for index in range(config.candidate_count):
            semantic_probability = empirical_upper_tail_probability(jumps[index], reference_jumps)
            ppl_evidence = variance_adjusted_transition_evidence(
                language_profiles[index],
                windows=config.ppl_windows,
                reference_window=config.reference_window,
                reference_bits=config.reference_bits,
            )
            semantic_evidence = semantic_tail_excess(
                semantic_probability, alpha=config.semantic_tail_alpha
            )
            score = float(base[index]["base_score"]) + ppl_evidence + semantic_evidence
            output.append(
                {
                    **base[index],
                    "ppl_transition_evidence": ppl_evidence,
                    "semantic_alignment_jump": jumps[index],
                    "semantic_tail_probability": semantic_probability,
                    "semantic_tail_evidence": semantic_evidence,
                    "score": score,
                    "flagged": score >= config.decision_threshold,
                }
            )
        return output

    def close(self) -> None:
        for scorer in (self.causal, self.semantic):
            close = getattr(scorer, "close", None)
            if close is not None:
                close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
