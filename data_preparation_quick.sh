#!/usr/bin/env bash
set -euo pipefail

device=${DEVICE:-"cuda:0"}
batch_size=${BATCH_SIZE:-32}

uv run ragsieve build-index \
  --corpus data/demo/corpus.jsonl \
  --output-dir data/indices/demo/bge-m3 \
  --encoder bge-m3 \
  --device "${device}" \
  --batch-size "${batch_size}"
