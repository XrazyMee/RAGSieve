from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            yield row


def _document_id(row: dict[str, Any]) -> str:
    value = row.get("document_id", row.get("id"))
    if value is None:
        raise ValueError("every document and prediction needs an id")
    return str(value)


def filter_contexts(
    contexts_path: Path,
    predictions_path: Path | None,
    output_path: Path,
    *,
    context_size: int = 5,
    graph_predictions_path: Path | None = None,
    condition: str | None = None,
) -> int:
    """Union RSQ and RSG flags, then refill from the unchanged retrieval ranking."""
    if graph_predictions_path is not None and condition is None:
        raise ValueError("specify condition to associate RSG predictions with a corpus snapshot")
    flagged: set[tuple[str, str, str]] = set()
    for row in _read_jsonl(predictions_path) if predictions_path is not None else ():
        if not isinstance(row.get("flagged"), bool):
            raise TypeError("every prediction needs a boolean flagged field")
        if row.get("query_id") is None:
            raise ValueError("RSQ predictions need query_id; use graph_predictions_path for RSG")
        if row["flagged"]:
            flagged.add(
                (
                    str(row.get("query_id")),
                    str(row.get("condition", "unspecified")),
                    _document_id(row),
                )
            )

    corpus_flags: set[str] = set()
    if graph_predictions_path is not None:
        for row in _read_jsonl(graph_predictions_path):
            if not isinstance(row.get("flagged"), bool):
                raise TypeError("every RSG prediction needs a boolean flagged field")
            if row["flagged"]:
                corpus_flags.add(_document_id(row))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in _read_jsonl(contexts_path):
            query_id = str(record.get("query_id"))
            record_condition = str(record.get("condition", "unspecified"))
            if condition is not None and record_condition != condition:
                continue
            documents = record.get("documents")
            if not isinstance(documents, list):
                raise TypeError("every context needs a documents list")
            removed = [
                _document_id(document)
                for document in documents
                if (query_id, record_condition, _document_id(document)) in flagged
                or _document_id(document) in corpus_flags
            ]
            retained = [
                document
                for document in documents
                if (query_id, record_condition, _document_id(document)) not in flagged
                and _document_id(document) not in corpus_flags
            ][:context_size]
            if len(retained) < context_size:
                raise ValueError(
                    f"{query_id}/{record_condition}: fewer than {context_size} documents remain"
                )
            output.write(
                json.dumps(
                    {
                        **record,
                        "documents": retained,
                        "removed_document_ids": removed,
                        "removed_count": len(removed),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    return written
