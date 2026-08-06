from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .detector import DetectorConfig, RAGSieveQueryDetector
from .filtering import filter_contexts
from .graph import GraphDetectorConfig, RAGSieveGraphDetector
from .metrics import detection_metrics
from .qa import load_openai_config, run_qa
from .retrieval import build_corpus_index, prepare_contexts


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                yield line_number, value


def _documents(record: dict[str, Any], *, line_number: int) -> list[dict[str, Any]]:
    documents = record.get("documents")
    if not isinstance(documents, list):
        raise TypeError(f"line {line_number}: documents must be a list")
    if any(not isinstance(document, dict) for document in documents):
        raise TypeError(f"line {line_number}: every document must be an object")
    ranks = [document.get("rank") for document in documents]
    if any(not isinstance(rank, int) for rank in ranks):
        raise TypeError(f"line {line_number}: every document rank must be an integer")
    if ranks != sorted(ranks) or len(ranks) != len(set(ranks)):
        raise ValueError(f"line {line_number}: document ranks must be unique and ascending")
    if any(not isinstance(document.get("text"), str) for document in documents):
        raise ValueError(f"line {line_number}: every document must contain text")
    return documents


def run_detection(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = DetectorConfig(
        causal_batch_size=args.causal_batch_size,
        semantic_batch_size=args.semantic_batch_size,
    )
    contexts = 0
    candidates = 0
    with (
        RAGSieveQueryDetector(config, device=args.device) as detector,
        output_path.open("w", encoding="utf-8", newline="\n") as output,
    ):
        for line_number, record in _read_jsonl(input_path):
            query = record.get("query")
            query_id = record.get("query_id")
            if not isinstance(query, str) or query_id is None:
                raise ValueError(f"line {line_number}: query_id and query are required")
            documents = _documents(record, line_number=line_number)
            scores = detector.detect(query, [document["text"] for document in documents])
            metadata = {
                key: record[key] for key in ("dataset", "encoder", "condition") if key in record
            }
            for index, score in enumerate(scores):
                document = documents[index]
                row: dict[str, Any] = {
                    **metadata,
                    "query_id": query_id,
                    "document_id": document.get("id"),
                    "rank": document["rank"],
                    **score,
                }
                if "poison" in document:
                    row["poison"] = bool(document["poison"])
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                candidates += 1
            contexts += 1
    print(f"wrote {candidates} candidate scores from {contexts} contexts to {output_path}")


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("dataset", "all")), str(row.get("encoder", "all"))


def run_evaluation(args: argparse.Namespace) -> None:
    rows = [row for _, row in _read_jsonl(Path(args.predictions))]
    if any("poison" not in row for row in rows):
        raise ValueError("every prediction row must contain a poison label")

    clean_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    attack_scores: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        dataset, encoder = _key(row)
        condition = str(row.get("condition", "unspecified"))
        if condition == args.clean_condition:
            if row["poison"]:
                raise ValueError("the clean condition contains a poison-labelled document")
            clean_scores[(dataset, encoder)].append(float(row["score"]))
        elif row["poison"]:
            attack_scores[(dataset, encoder, condition)].append(float(row["score"]))

    cells = []
    for (dataset, encoder, condition), positives in sorted(attack_scores.items()):
        negatives = clean_scores.get((dataset, encoder), [])
        if not negatives:
            raise ValueError(f"missing {args.clean_condition!r} negatives for {dataset}/{encoder}")
        cells.append(
            {
                "dataset": dataset,
                "encoder": encoder,
                "condition": condition,
                **detection_metrics(
                    positives,
                    negatives,
                    threshold=args.threshold,
                    target_fpr=args.target_fpr,
                ),
            }
        )
    if not cells:
        raise ValueError("no poison-labelled attack candidates were found")

    metric_names = (
        "auroc",
        "tpr_at_5pct_fpr",
        "fixed_threshold_tpr",
        "fixed_threshold_clean_fpr",
    )
    result = {
        "configuration": {
            "negative_condition": args.clean_condition,
            "target_fpr": args.target_fpr,
            "decision_threshold": args.threshold,
        },
        "macro": {
            name: sum(float(cell[name]) for cell in cells) / len(cells) for name in metric_names
        },
        "cells": cells,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(cells)} detection cells to {output_path}")


def _graph_documents(path: Path) -> tuple[list[str], list[str]]:
    document_ids: list[str] = []
    texts: list[str] = []
    for line_number, record in _read_jsonl(path):
        document_id = record.get("document_id", record.get("id"))
        text = record.get("text")
        if document_id is None or not isinstance(text, str):
            raise ValueError(f"{path}:{line_number}: document_id (or id) and text are required")
        document_ids.append(str(document_id))
        texts.append(text)
    if len(document_ids) != len(set(document_ids)):
        raise ValueError(f"{path}: document ids must be unique")
    return document_ids, texts


def run_graph_detection(args: argparse.Namespace) -> None:
    document_ids, texts = _graph_documents(Path(args.documents))
    matrix = np.load(args.embeddings, mmap_mode="r")
    if matrix.ndim != 2 or len(matrix) != len(texts):
        raise ValueError(
            f"embeddings have shape {matrix.shape}, expected ({len(texts)}, dimension)"
        )
    embeddings = torch.from_numpy(np.array(matrix, copy=True, order="C"))
    config = GraphDetectorConfig(query_block_size=args.query_block_size)
    detector = RAGSieveGraphDetector(config, device=args.device)
    results = detector.detect(texts, embeddings)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for document_id, result in zip(document_ids, results, strict=True):
            output.write(
                json.dumps({"document_id": document_id, **result}, ensure_ascii=False) + "\n"
            )
    print(f"wrote {len(results)} query-free document scores to {output_path}")


def run_graph_evaluation(args: argparse.Namespace) -> None:
    scores: dict[str, float] = {}
    for line_number, row in _read_jsonl(Path(args.predictions)):
        document_id = row.get("document_id")
        if document_id is None or "score" not in row:
            raise ValueError(
                f"{args.predictions}:{line_number}: document_id and score are required"
            )
        key = str(document_id)
        if key in scores:
            raise ValueError(f"duplicate prediction for document {key!r}")
        scores[key] = float(row["score"])

    positive: list[float] = []
    negative: list[float] = []
    labels: set[str] = set()
    metadata: dict[str, Any] = {}
    for line_number, row in _read_jsonl(Path(args.labels)):
        document_id = row.get("document_id", row.get("id"))
        if document_id is None or not isinstance(row.get("poison"), bool):
            raise ValueError(
                f"{args.labels}:{line_number}: document_id (or id) and boolean poison are required"
            )
        key = str(document_id)
        if key in labels:
            raise ValueError(f"duplicate label for document {key!r}")
        if key not in scores:
            raise ValueError(f"missing prediction for labelled document {key!r}")
        labels.add(key)
        (positive if row["poison"] else negative).append(scores[key])
        for field in ("dataset", "encoder", "condition"):
            if field in row:
                value = row[field]
                if field in metadata and metadata[field] != value:
                    raise ValueError(f"labels contain multiple {field} values")
                metadata[field] = value
    unlabelled = set(scores) - labels
    if unlabelled:
        raise ValueError(f"labels are missing for {len(unlabelled)} predictions")

    result = {
        **metadata,
        "target_fpr": args.target_fpr,
        "decision_threshold": args.threshold,
        **detection_metrics(
            positive,
            negative,
            threshold=args.threshold,
            target_fpr=args.target_fpr,
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote graph detection metrics to {output_path}")


def run_build_index(args: argparse.Namespace) -> None:
    matrix = build_corpus_index(
        Path(args.corpus),
        Path(args.output_dir),
        encoder_name=args.encoder,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(f"wrote local document embeddings to {matrix}")


def run_prepare_contexts(args: argparse.Namespace) -> None:
    count = prepare_contexts(
        Path(args.dataset_dir),
        Path(args.index_dir),
        Path(args.output),
        attacks_path=Path(args.attacks) if args.attacks else None,
        subset_path=Path(args.subset) if args.subset else None,
        device=args.device,
        top_k=args.top_k,
        batch_size=args.batch_size,
        exclude_predictions_path=(
            Path(args.exclude_predictions) if args.exclude_predictions else None
        ),
    )
    print(f"wrote {count} ranked retrieval contexts to {args.output}")


def run_filter_contexts(args: argparse.Namespace) -> None:
    count = filter_contexts(
        Path(args.contexts),
        Path(args.predictions) if args.predictions else None,
        Path(args.output),
        context_size=args.context_size,
        graph_predictions_path=Path(args.graph_predictions) if args.graph_predictions else None,
        condition=args.condition,
    )
    print(f"wrote {count} refilled generation contexts to {args.output}")


def run_qa_evaluation(args: argparse.Namespace) -> None:
    summary = run_qa(
        Path(args.input),
        Path(args.output),
        Path(args.summary),
        config=load_openai_config(Path(args.env_file)),
        workers=args.workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


RSQ_ABLATIONS = {
    "without-answer-anchor": "normalized_answer_anchor_score",
    "without-script-integrity": "normalized_integrity_score",
    "without-surprisal": "ppl_transition_evidence",
    "without-query-alignment": "semantic_tail_evidence",
}


def run_ablation(args: argparse.Namespace) -> None:
    """Recompute a published component ablation from saved detector evidence."""
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for _, row in _read_jsonl(Path(args.predictions)):
            if args.variant in RSQ_ABLATIONS:
                omitted = RSQ_ABLATIONS[args.variant]
                score = sum(
                    float(row[field]) for field in RSQ_ABLATIONS.values() if field != omitted
                )
                threshold = DetectorConfig().decision_threshold
            else:
                config = GraphDetectorConfig()
                threshold = config.decision_threshold
                if args.variant == "corpus-local-only":
                    # Without integrity alerts, the graph branch receives the full alert budget.
                    probability = float(row["topology_tail_probability"])
                    score = min(1.0, threshold * config.target_alert_fraction / probability)
                else:
                    score = float(row["script_integrity_score"])
            output.write(
                json.dumps(
                    {**row, "variant": args.variant, "score": score, "flagged": score >= threshold},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {args.variant} predictions to {output_path}")


def run_benchmark(args: argparse.Namespace) -> None:
    """Measure resident detector execution, excluding loading, retrieval, and QA."""
    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats must be positive and warmup must be nonnegative")
    device = torch.device(args.device)
    if args.mode == "rsq":
        items = [
            (record["query"], [row["text"] for row in _documents(record, line_number=line)])
            for line, record in _read_jsonl(Path(args.input))
        ]
        if not items:
            raise ValueError("the benchmark needs at least one retrieval context")
        detector = RAGSieveQueryDetector(device=device)
        unit = "query"
    else:
        if args.embeddings is None:
            raise ValueError("RSG benchmarking requires --embeddings")
        _, texts = _graph_documents(Path(args.input))
        matrix = np.load(args.embeddings, mmap_mode="r")
        embeddings = torch.from_numpy(np.array(matrix, copy=True)).to(device)
        items = [(texts, embeddings)]
        detector = RAGSieveGraphDetector(device=device)
        unit = "corpus_scan"

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    elapsed = []
    try:
        for index in range(args.warmup):
            detector.detect(*items[index % len(items)])
        synchronize()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.repeats):
            for item in items:
                synchronize()
                start = time.perf_counter()
                detector.detect(*item)
                synchronize()
                elapsed.append(time.perf_counter() - start)
        mean_seconds = float(np.mean(elapsed))
        result = {
            "mode": args.mode,
            "device": str(device),
            "unit": unit,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "measurements": len(elapsed),
            "mean_ms": 1000.0 * mean_seconds,
            "p50_ms": 1000.0 * float(np.percentile(elapsed, 50)),
            "p95_ms": 1000.0 * float(np.percentile(elapsed, 95)),
        }
        if args.mode == "rsq":
            result["queries_per_second"] = 1.0 / mean_seconds
        else:
            result["documents"] = len(texts)
            result["documents_per_second"] = len(texts) / mean_seconds
        if device.type == "cuda":
            result["peak_gpu_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
    finally:
        if args.mode == "rsq":
            detector.close()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote resident {args.mode.upper()} execution cost to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragsieve")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser(
        "build-index", help="encode a released text corpus into a local dense index"
    )
    index.add_argument("--corpus", required=True)
    index.add_argument("--output-dir", required=True)
    index.add_argument(
        "--encoder", choices=("bge-m3", "e5-large-v2", "minilm-l6-v2"), default="bge-m3"
    )
    index.add_argument("--device", default="cuda:0")
    index.add_argument("--batch-size", type=int, default=32)
    index.set_defaults(handler=run_build_index)

    prepare = subparsers.add_parser(
        "prepare-contexts",
        help="run exact full-corpus retrieval and optionally inject demo documents",
    )
    prepare.add_argument("--dataset-dir", required=True)
    prepare.add_argument("--index-dir", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--attacks")
    prepare.add_argument("--subset")
    prepare.add_argument(
        "--exclude-predictions", help="RSG predictions to quarantine before retrieval"
    )
    prepare.add_argument("--device", default="cuda:0")
    prepare.add_argument("--top-k", type=int, default=100)
    prepare.add_argument("--batch-size", type=int, default=32)
    prepare.set_defaults(handler=run_prepare_contexts)

    detect = subparsers.add_parser("detect", help="score top-five candidate documents")
    detect.add_argument("--input", required=True, help="ranked retrieval contexts (JSONL)")
    detect.add_argument("--output", required=True, help="candidate predictions (JSONL)")
    detect.add_argument("--device", default="cuda:0")
    detect.add_argument("--causal-batch-size", type=int, default=5)
    detect.add_argument("--semantic-batch-size", type=int, default=20)
    detect.set_defaults(handler=run_detection)

    evaluate = subparsers.add_parser("evaluate", help="compute document-level metrics")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--clean-condition", default="clean")
    evaluate.add_argument("--target-fpr", type=float, default=0.05)
    evaluate.add_argument("--threshold", type=float, default=1.0)
    evaluate.set_defaults(handler=run_evaluation)

    filter_parser = subparsers.add_parser(
        "filter-contexts", help="union RSQ/RSG flags and refill the generation top five"
    )
    filter_parser.add_argument("--contexts", required=True)
    filter_parser.add_argument("--predictions", help="RSQ query-level predictions")
    filter_parser.add_argument(
        "--graph-predictions", help="RSG predictions for one corpus snapshot"
    )
    filter_parser.add_argument(
        "--condition", help="select a condition; required with RSG predictions"
    )
    filter_parser.add_argument("--output", required=True)
    filter_parser.add_argument("--context-size", type=int, default=5)
    filter_parser.set_defaults(handler=run_filter_contexts)

    qa = subparsers.add_parser(
        "qa", help="run answer generation and semantic ASR judging after filtering"
    )
    qa.add_argument("--input", required=True, help="filtered Top-5 contexts (JSONL)")
    qa.add_argument("--output", required=True, help="per-query QA records (JSONL)")
    qa.add_argument("--summary", required=True, help="aggregate QA metrics (JSON)")
    qa.add_argument("--env-file", default=".env")
    qa.add_argument("--workers", type=int, default=8)
    qa.set_defaults(handler=run_qa_evaluation)

    graph = subparsers.add_parser("detect-graph", help="query-free RSG scan of a complete corpus")
    graph.add_argument("--documents", required=True, help="document text (JSONL)")
    graph.add_argument("--embeddings", required=True, help="aligned document vectors (.npy)")
    graph.add_argument("--output", required=True, help="document predictions (JSONL)")
    graph.add_argument("--device", default="cuda:0")
    graph.add_argument("--query-block-size", type=int, default=512)
    graph.set_defaults(handler=run_graph_detection)

    graph_evaluate = subparsers.add_parser(
        "evaluate-graph", help="evaluate a saved RSG corpus scan"
    )
    graph_evaluate.add_argument("--predictions", required=True)
    graph_evaluate.add_argument("--labels", required=True)
    graph_evaluate.add_argument("--output", required=True)
    graph_evaluate.add_argument("--target-fpr", type=float, default=0.05)
    graph_evaluate.add_argument("--threshold", type=float, default=0.5)
    graph_evaluate.set_defaults(handler=run_graph_evaluation)

    ablate = subparsers.add_parser(
        "ablate", help="recompute component ablations from saved evidence"
    )
    ablate.add_argument("--predictions", required=True)
    ablate.add_argument("--output", required=True)
    ablate.add_argument(
        "--variant",
        choices=(*RSQ_ABLATIONS, "corpus-local-only", "script-integrity-only"),
        required=True,
    )
    ablate.set_defaults(handler=run_ablation)

    benchmark = subparsers.add_parser(
        "benchmark", help="measure resident RSQ or RSG execution cost"
    )
    benchmark.add_argument("--mode", choices=("rsq", "rsg"), required=True)
    benchmark.add_argument("--input", required=True, help="RSQ contexts or RSG corpus documents")
    benchmark.add_argument("--embeddings", help="aligned document vectors for RSG")
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--device", default="cuda:0")
    benchmark.add_argument("--warmup", type=int, default=1)
    benchmark.add_argument("--repeats", type=int, default=3)
    benchmark.set_defaults(handler=run_benchmark)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)
