from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


@dataclass(frozen=True)
class RetrieverSpec:
    model_name: str
    pooling: str
    query_prefix: str = ""
    document_prefix: str = ""
    max_length: int = 512


RETRIEVERS: dict[str, RetrieverSpec] = {
    "bge-m3": RetrieverSpec("BAAI/bge-m3", "cls"),
    "e5-large-v2": RetrieverSpec(
        "intfloat/e5-large-v2",
        "mean",
        query_prefix="query: ",
        document_prefix="passage: ",
    ),
    "minilm-l6-v2": RetrieverSpec(
        "sentence-transformers/all-MiniLM-L6-v2",
        "mean",
    ),
}


def _device(value: str | int) -> torch.device:
    return torch.device(f"cuda:{value}" if isinstance(value, int) else value)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            yield value


def document_text(row: dict[str, Any]) -> str:
    text = row.get("text")
    if not isinstance(text, str):
        raise TypeError("every document must contain a string text field")
    title = row.get("title")
    if isinstance(title, str) and title.strip() and not text.startswith(title):
        return f"{title.strip()}\n\n{text}"
    return text


class DenseRetrieverEncoder:
    """Dense encoder used to build the paper's victim-retriever indices locally."""

    def __init__(self, encoder: str = "bge-m3", *, device: str | int = "cuda:0") -> None:
        if encoder not in RETRIEVERS:
            raise ValueError(f"unknown encoder {encoder!r}; choose from {sorted(RETRIEVERS)}")
        self.name = encoder
        self.spec = RETRIEVERS[encoder]
        self.device = _device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.model_name, use_fast=True)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(self.spec.model_name, dtype=dtype).to(self.device)
        self.model.eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(
        self,
        texts: Sequence[str],
        *,
        role: str,
        batch_size: int = 32,
    ) -> torch.Tensor:
        if role not in {"query", "document"}:
            raise ValueError("role must be 'query' or 'document'")
        prefix = self.spec.query_prefix if role == "query" else self.spec.document_prefix
        output: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = [prefix + text for text in texts[start : start + batch_size]]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.spec.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**encoded).last_hidden_state.float()
            if self.spec.pooling == "cls":
                pooled = hidden[:, 0]
            else:
                mask = encoded.attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            output.append(F.normalize(pooled, p=2, dim=-1).cpu())
        if not output:
            raise ValueError("cannot encode an empty sequence")
        return torch.cat(output)

    def close(self) -> None:
        del self.model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def _fingerprint(rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row.get("id", row.get("document_id"))).encode())
        digest.update(b"\0")
        digest.update(document_text(row).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def build_corpus_index(
    corpus_path: Path,
    output_dir: Path,
    *,
    encoder_name: str,
    device: str,
    batch_size: int,
) -> Path:
    rows = list(_read_jsonl(corpus_path))
    ids = [str(row.get("id", row.get("document_id"))) for row in rows]
    if any(value == "None" for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("corpus document ids must be present and unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "embeddings.npy"
    encoder = DenseRetrieverEncoder(encoder_name, device=device)
    matrix: np.memmap | None = None
    try:
        for start in range(0, len(rows), batch_size):
            vectors = encoder.encode(
                [document_text(row) for row in rows[start : start + batch_size]],
                role="document",
                batch_size=batch_size,
            ).numpy()
            if matrix is None:
                matrix = np.lib.format.open_memmap(
                    matrix_path,
                    mode="w+",
                    dtype=np.float16,
                    shape=(len(rows), vectors.shape[1]),
                )
            matrix[start : start + len(vectors)] = vectors.astype(np.float16)
        if matrix is None:
            raise ValueError("the corpus is empty")
        matrix.flush()
        dimension = int(matrix.shape[1])
    finally:
        encoder.close()
    (output_dir / "document_ids.json").write_text(
        json.dumps(ids, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "encoder": encoder_name,
        "model": RETRIEVERS[encoder_name].model_name,
        "documents": len(rows),
        "dimension": dimension,
        "dtype": "float16",
        "normalized": True,
        "corpus_sha256": _fingerprint(rows),
        "corpus": corpus_path.as_posix(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return matrix_path


def topk_search(
    query: torch.Tensor,
    corpus: np.ndarray | torch.Tensor,
    *,
    k: int,
    device: str | int = "cpu",
    block_size: int = 8192,
    exclude_indices: set[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact blockwise cosine search over a locally materialized corpus index."""
    target = _device(device)
    query_vector = F.normalize(query.float().reshape(1, -1), p=2, dim=-1).to(target)
    best_scores = torch.empty(0, dtype=torch.float32)
    best_indices = torch.empty(0, dtype=torch.int64)
    for start in range(0, len(corpus), block_size):
        stop = min(start + block_size, len(corpus))
        if isinstance(corpus, np.ndarray):
            block = torch.from_numpy(np.array(corpus[start:stop], copy=True))
        else:
            block = corpus[start:stop].detach().cpu()
        block = F.normalize(block.float(), p=2, dim=-1).to(target)
        scores = (block @ query_vector.T).squeeze(1).cpu()
        indices = torch.arange(start, stop)
        if exclude_indices:
            excluded = [index - start for index in exclude_indices if start <= index < stop]
            if excluded:
                scores[torch.tensor(excluded, dtype=torch.int64)] = -torch.inf
        best_scores = torch.cat((best_scores, scores))
        best_indices = torch.cat((best_indices, indices))
        keep = min(k, len(best_scores))
        best_scores, positions = torch.topk(best_scores, keep, sorted=True)
        best_indices = best_indices[positions]
    finite = torch.isfinite(best_scores)
    return best_scores[finite], best_indices[finite]


def flagged_document_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    flagged: set[str] = set()
    for row in _read_jsonl(path):
        document_id = row.get("document_id", row.get("id"))
        if document_id is None or not isinstance(row.get("flagged"), bool):
            raise ValueError(f"{path}: every row needs document_id and boolean flagged")
        if row["flagged"]:
            flagged.add(str(document_id))
    return flagged


def prepare_contexts(
    dataset_dir: Path,
    index_dir: Path,
    output_path: Path,
    *,
    attacks_path: Path | None,
    subset_path: Path | None,
    device: str,
    top_k: int = 100,
    batch_size: int = 32,
    exclude_predictions_path: Path | None = None,
) -> int:
    corpus_rows = list(_read_jsonl(dataset_dir / "corpus.jsonl"))
    corpus_ids = json.loads((index_dir / "document_ids.json").read_text(encoding="utf-8"))
    if corpus_ids != [str(row.get("id", row.get("document_id"))) for row in corpus_rows]:
        raise ValueError("the index document order does not match the corpus")
    matrix = np.load(index_dir / "embeddings.npy", mmap_mode="r")
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    encoder_name = str(manifest["encoder"])
    query_rows = {str(row["id"]): row for row in _read_jsonl(dataset_dir / "queries.jsonl")}
    if subset_path is not None:
        subset = json.loads(subset_path.read_text(encoding="utf-8"))
        query_ids = [str(value) for value in subset["query_ids"]]
    else:
        query_ids = list(query_rows)
    attack_rows = (
        {str(row["query_id"]): row for row in _read_jsonl(attacks_path)}
        if attacks_path is not None
        else {}
    )
    if attacks_path is not None:
        query_ids = [query_id for query_id in query_ids if query_id in attack_rows]
    excluded_ids = flagged_document_ids(exclude_predictions_path)
    excluded_indices = {
        index for index, document_id in enumerate(corpus_ids) if document_id in excluded_ids
    }
    encoder = DenseRetrieverEncoder(encoder_name, device=device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with output_path.open("w", encoding="utf-8", newline="\n") as output:
            for query_id in query_ids:
                query_row = query_rows[query_id]
                query_text = str(query_row["text"])
                query_vector = encoder.encode([query_text], role="query", batch_size=1)[0]
                clean_scores, clean_indices = topk_search(
                    query_vector,
                    matrix,
                    k=top_k,
                    device=device,
                    exclude_indices=excluded_indices,
                )
                if len(clean_indices) < top_k:
                    raise ValueError("fewer than top_k corpus documents remain after exclusion")
                conditions: list[tuple[str, list[dict[str, Any]], torch.Tensor | None]] = [
                    ("clean", [], None)
                ]
                if query_id in attack_rows:
                    attack = attack_rows[query_id]
                    documents = [
                        row
                        for row in attack.get("documents", [])
                        if str(row.get("id", row.get("document_id"))) not in excluded_ids
                    ]
                    attack_vectors = (
                        encoder.encode(
                            [document_text(row) for row in documents],
                            role="document",
                            batch_size=batch_size,
                        )
                        if documents
                        else None
                    )
                    conditions.append(
                        (str(attack.get("attack", "attack")), documents, attack_vectors)
                    )
                for condition, injected, injected_vectors in conditions:
                    ranked = [
                        (
                            float(score),
                            corpus_rows[int(index)],
                            False,
                        )
                        for score, index in zip(clean_scores, clean_indices, strict=True)
                    ]
                    if injected_vectors is not None:
                        injected_scores = injected_vectors @ query_vector
                        ranked.extend(
                            (float(score), row, True)
                            for score, row in zip(injected_scores, injected, strict=True)
                        )
                    ranked = sorted(ranked, key=lambda value: value[0], reverse=True)[:top_k]
                    attack = attack_rows.get(query_id, {})
                    record = {
                        "query_id": query_id,
                        "query": query_text,
                        "dataset": dataset_dir.name,
                        "encoder": encoder_name,
                        "condition": condition,
                        "correct_answers": attack.get(
                            "correct_answers", query_row.get("answers", [])
                        ),
                        "target_answer": attack.get("target_answer"),
                        "documents": [
                            {
                                "id": str(row.get("id", row.get("document_id"))),
                                "rank": rank,
                                "text": document_text(row),
                                "poison": poison,
                                "similarity": score,
                            }
                            for rank, (score, row, poison) in enumerate(ranked, 1)
                        ],
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
    finally:
        encoder.close()
    return written
