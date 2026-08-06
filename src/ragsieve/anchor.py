from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from scipy.stats import hypergeom

from .integrity import script_anomaly
from .statistics import empirical_upper_tail_probability, simes_probability

_WORD = re.compile(r"[a-z0-9]+")


def content_tokens(text: str, query: str, *, minimum_length: int = 3) -> set[str]:
    """Word types outside the query that can act as answer or claim anchors."""
    query_tokens = set(_WORD.findall(query.casefold()))
    return {
        token
        for token in _WORD.findall(text.casefold())
        if token not in query_tokens and (len(token) >= minimum_length or token.isdigit())
    }


def answer_anchor_probabilities(
    texts: Sequence[str], query: str, *, candidate_count: int = 5
) -> list[float]:
    """Test query-external token concentration among the candidate documents."""
    if candidate_count < 2 or len(texts) <= candidate_count:
        raise ValueError("answer-anchor testing requires candidates and a reference tail")
    token_sets = [content_tokens(text, query) for text in texts]
    candidate_frequency = Counter(
        token for tokens in token_sets[:candidate_count] for token in tokens
    )
    context_frequency = Counter(token for tokens in token_sets for token in tokens)
    token_probabilities = {
        token: float(
            hypergeom.sf(
                candidate_frequency[token] - 1,
                len(texts),
                context_frequency[token],
                candidate_count,
            )
        )
        for token in candidate_frequency
        if candidate_frequency[token] >= 2
    }
    output = []
    for tokens in token_sets[:candidate_count]:
        probabilities = [
            token_probabilities[token] for token in tokens if token in token_probabilities
        ]
        output.append(simes_probability(probabilities) if probabilities else 1.0)
    return output


def integrity_probabilities(texts: Sequence[str], *, candidate_count: int = 5) -> list[float]:
    """Rank candidate mixed-script mass against the same query's retrieval tail."""
    if candidate_count < 1 or len(texts) <= candidate_count:
        raise ValueError("integrity testing requires candidates and a reference tail")
    values = [script_anomaly(text)["mixed_script_fraction"] for text in texts]
    reference = values[candidate_count:]
    return [
        empirical_upper_tail_probability(value, reference) for value in values[:candidate_count]
    ]


def anchor_integrity_features(
    texts: Sequence[str], query: str, *, candidate_count: int = 5
) -> list[dict[str, float]]:
    """Compute the two normalized, query-local non-carrier evidence components."""
    if candidate_count < 2 or len(texts) <= candidate_count:
        raise ValueError("features require at least two candidates and a reference tail")
    anchor = answer_anchor_probabilities(texts, query, candidate_count=candidate_count)
    integrity = integrity_probabilities(texts, candidate_count=candidate_count)
    anchor_limit = math.log10(math.comb(len(texts), candidate_count))
    integrity_limit = math.log10(2.0 * (len(texts) - candidate_count + 1))

    output = []
    for anchor_probability, integrity_probability in zip(anchor, integrity, strict=True):
        answer_evidence = -math.log10(max(math.ulp(1.0), anchor_probability))
        integrity_evidence = -math.log10(max(math.ulp(1.0), integrity_probability))
        normalized_answer = answer_evidence / anchor_limit
        normalized_integrity = integrity_evidence / integrity_limit
        output.append(
            {
                "answer_anchor_probability": anchor_probability,
                "answer_anchor_evidence": answer_evidence,
                "normalized_answer_anchor_score": normalized_answer,
                "integrity_probability": integrity_probability,
                "integrity_evidence": integrity_evidence,
                "normalized_integrity_score": normalized_integrity,
                "base_score": normalized_answer + normalized_integrity,
            }
        )
    return output
