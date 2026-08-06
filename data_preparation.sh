#!/usr/bin/env bash
set -euo pipefail

datasets=${DATASETS:-"nq hotpotqa msmarco"}
encoders=${ENCODERS:-"bge-m3 e5-large-v2 minilm-l6-v2"}
device=${DEVICE:-"cuda:0"}
batch_size=${BATCH_SIZE:-32}

for dataset in ${datasets}; do
  for encoder in ${encoders}; do
    uv run ragsieve build-index \
      --corpus "data/datasets/${dataset}/corpus.jsonl" \
      --output-dir "data/indices/${dataset}/${encoder}" \
      --encoder "${encoder}" \
      --device "${device}" \
      --batch-size "${batch_size}"
  done
done
