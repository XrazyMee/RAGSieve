from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import torch
import torch.nn.functional as F

from .integrity import character_script, script_anomaly

_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class GraphDetectorConfig:
    """Fixed RSG configuration used in the paper."""

    neighbor_count: int = 16
    dense_core_count: int = 4
    support_scale: int = 2
    semantic_threshold: float = 0.85
    lexical_ceiling: float = 0.60
    multiscript_gate: int = 3
    target_alert_fraction: float = 0.05
    decision_threshold: float = 0.5
    query_block_size: int = 512

    def __post_init__(self) -> None:
        if self.neighbor_count < 1:
            raise ValueError("neighbor_count must be positive")
        if not 1 <= self.dense_core_count <= self.neighbor_count:
            raise ValueError("dense_core_count must be in [1, neighbor_count]")
        if self.support_scale < 1:
            raise ValueError("support_scale must be positive")
        if not 0.0 <= self.semantic_threshold <= 1.0:
            raise ValueError("semantic_threshold must be in [0, 1]")
        if not 0.0 <= self.lexical_ceiling <= 1.0:
            raise ValueError("lexical_ceiling must be in [0, 1]")
        if not 0.0 < self.target_alert_fraction < 1.0:
            raise ValueError("target_alert_fraction must be in (0, 1)")
        if self.query_block_size < 1:
            raise ValueError("query_block_size must be positive")


def lexical_jaccard(left: str, right: str) -> float:
    """Case-folded alphanumeric word-set Jaccard similarity."""
    left_words = set(_WORD.findall(left.casefold()))
    right_words = set(_WORD.findall(right.casefold()))
    union = left_words | right_words
    return len(left_words & right_words) / len(union) if union else 0.0


def graph_script_features(text: str, *, multiscript_gate: int = 3) -> dict[str, float]:
    """Measure token-internal script fragmentation without penalizing bilingual blocks."""
    base = script_anomaly(text)
    tagged = [(character, character_script(character)) for character in text]
    scripts = [script for _, script in tagged if script is not None]
    adjacent_transitions = sum(
        left_script is not None and right_script is not None and left_script != right_script
        for (_, left_script), (_, right_script) in pairwise(tagged)
    )
    intratoken_rate = adjacent_transitions / max(1, len(scripts) - 1)
    script_count = len(set(scripts))
    integrity_score = float(adjacent_transitions > 0 or script_count >= multiscript_gate)
    return {
        **base,
        "alphabetic_script_count": float(script_count),
        "intratoken_script_transition_count": float(adjacent_transitions),
        "intratoken_script_transition_rate": intratoken_rate,
        "script_integrity_score": integrity_score,
    }


def exact_cosine_knn(
    embeddings: torch.Tensor,
    *,
    k: int = 16,
    query_block_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an exact cosine top-k graph without materializing its full similarity matrix."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional matrix")
    if len(embeddings) < 2:
        raise ValueError("at least two documents are required")
    if not embeddings.is_floating_point():
        raise TypeError("embeddings must use a floating-point dtype")
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("embeddings contain non-finite values")
    if bool((embeddings.norm(dim=1) == 0).any()):
        raise ValueError("embeddings contain a zero vector")

    vectors = F.normalize(embeddings.float(), p=2, dim=1)
    neighbor_count = min(k, len(vectors) - 1)
    all_scores: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []
    for start in range(0, len(vectors), query_block_size):
        stop = min(start + query_block_size, len(vectors))
        similarities = vectors[start:stop] @ vectors.T
        rows = torch.arange(stop - start, device=vectors.device)
        columns = torch.arange(start, stop, device=vectors.device)
        similarities[rows, columns] = -math.inf
        scores, indices = torch.topk(
            similarities,
            k=neighbor_count,
            dim=1,
            largest=True,
            sorted=True,
        )
        all_scores.append(scores.cpu())
        all_indices.append(indices.cpu())
    return torch.cat(all_scores), torch.cat(all_indices)


def _eligible_graph(
    texts: Sequence[str],
    neighbor_scores: torch.Tensor,
    neighbor_indices: torch.Tensor,
    config: GraphDetectorConfig,
) -> list[dict[int, float]]:
    graph: list[dict[int, float]] = []
    for row, text in enumerate(texts):
        neighbors: dict[int, float] = {}
        for similarity, neighbor in zip(
            neighbor_scores[row].tolist(),
            neighbor_indices[row].tolist(),
            strict=True,
        ):
            if similarity < config.semantic_threshold:
                break
            if lexical_jaccard(text, texts[neighbor]) <= config.lexical_ceiling:
                neighbors[neighbor] = similarity
        graph.append(neighbors)
    return graph


def _empirical_upper_tail(values: torch.Tensor) -> torch.Tensor:
    _, inverse, counts = torch.unique(
        values.float(), sorted=True, return_inverse=True, return_counts=True
    )
    upper_counts = torch.flip(torch.cumsum(torch.flip(counts, dims=(0,)), dim=0), dims=(0,))
    return upper_counts[inverse].float() / (len(values) + 1)


def graph_scores_from_neighbors(
    texts: Sequence[str],
    neighbor_scores: torch.Tensor,
    neighbor_indices: torch.Tensor,
    config: GraphDetectorConfig | None = None,
) -> list[dict[str, float | int | bool]]:
    """Score a corpus from a precomputed top-k graph and document text alone."""
    settings = config or GraphDetectorConfig()
    if neighbor_scores.ndim != 2 or neighbor_indices.ndim != 2:
        raise ValueError("neighbor scores and indices must be matrices")
    if neighbor_scores.shape != neighbor_indices.shape:
        raise ValueError("neighbor scores and indices must have equal shapes")
    if neighbor_scores.shape[0] != len(texts):
        raise ValueError("the top-k graph must contain one row per document")
    if neighbor_scores.shape[1] < 1:
        raise ValueError("the top-k graph must contain at least one neighbor")
    if bool((neighbor_indices < 0).any()) or bool((neighbor_indices >= len(texts)).any()):
        raise ValueError("neighbor index is outside the corpus")

    graph = _eligible_graph(texts, neighbor_scores, neighbor_indices, settings)
    raw: list[dict[str, float | int]] = []
    for row, text in enumerate(texts):
        support = list(graph[row])
        support_factor = min(1.0, len(support) / settings.support_scale)
        background = float(neighbor_scores[row, -1])
        dense_core = sorted(graph[row].values(), reverse=True)[: settings.dense_core_count]
        mean_support = sum(dense_core) / len(dense_core) if dense_core else background
        local_excess = (mean_support - background) / max(1e-6, 1.0 - background)
        density = support_factor * max(0.0, min(1.0, local_excess))
        integrity = graph_script_features(
            text,
            multiscript_gate=settings.multiscript_gate,
        )
        raw.append(
            {
                **integrity,
                "eligible_neighbor_count": len(support),
                "local_density_contrast": density,
            }
        )

    strengths = torch.tensor([float(record["local_density_contrast"]) for record in raw])
    integrity_scores = torch.tensor([float(record["script_integrity_score"]) for record in raw])
    tail_probability = _empirical_upper_tail(strengths)
    integrity_fraction = float((integrity_scores >= settings.decision_threshold).float().mean())
    topology_fraction = max(0.0, settings.target_alert_fraction - integrity_fraction)
    topology_scores = (
        settings.decision_threshold * topology_fraction / tail_probability
        if topology_fraction > 0.0
        else torch.zeros_like(tail_probability)
    ).clamp(max=1.0)

    output: list[dict[str, float | int | bool]] = []
    for index, record in enumerate(raw):
        topology_score = float(topology_scores[index])
        score = max(topology_score, float(record["script_integrity_score"]))
        output.append(
            {
                **record,
                "topology_tail_probability": float(tail_probability[index]),
                "snapshot_integrity_alert_fraction": integrity_fraction,
                "snapshot_remaining_topology_tail_fraction": topology_fraction,
                "topological_coordination_score": topology_score,
                "score": score,
                "flagged": score >= settings.decision_threshold,
            }
        )
    return output


class RAGSieveGraphDetector:
    """Query-free corpus scanner operating on document text and document embeddings."""

    def __init__(
        self,
        config: GraphDetectorConfig | None = None,
        *,
        device: str | int = "cuda:0",
    ) -> None:
        self.config = config or GraphDetectorConfig()
        self.device = torch.device(f"cuda:{device}" if isinstance(device, int) else device)

    def detect(
        self, texts: Sequence[str], embeddings: torch.Tensor
    ) -> list[dict[str, float | int | bool]]:
        """Build the exact graph and score every document with one fixed rule."""
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have equal lengths")
        scores, indices = exact_cosine_knn(
            embeddings.to(self.device),
            k=self.config.neighbor_count,
            query_block_size=self.config.query_block_size,
        )
        return graph_scores_from_neighbors(texts, scores, indices, self.config)
