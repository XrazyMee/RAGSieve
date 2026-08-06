#!/usr/bin/env bash
set -euo pipefail

device=${DEVICE:-"cuda:0"}
mkdir -p outputs/demo

if [[ ! -f data/indices/demo/bge-m3/embeddings.npy ]]; then
  DEVICE="${device}" bash data_preparation_quick.sh
fi

uv run ragsieve detect \
  --input data/demo/contexts.jsonl \
  --output outputs/demo/rsq-predictions.jsonl \
  --device "${device}"
uv run ragsieve evaluate \
  --predictions outputs/demo/rsq-predictions.jsonl \
  --output outputs/demo/rsq-metrics.json
uv run ragsieve filter-contexts \
  --contexts data/demo/contexts.jsonl \
  --predictions outputs/demo/rsq-predictions.jsonl \
  --output outputs/demo/filtered-top5.jsonl

uv run ragsieve detect-graph \
  --documents data/demo/corpus.jsonl \
  --embeddings data/indices/demo/bge-m3/embeddings.npy \
  --output outputs/demo/rsg-predictions.jsonl \
  --device "${device}"
uv run ragsieve evaluate-graph \
  --predictions outputs/demo/rsg-predictions.jsonl \
  --labels data/demo/labels.jsonl \
  --output outputs/demo/rsg-metrics.json

for condition in pr_w ipi_cem_c; do
  uv run ragsieve filter-contexts \
    --contexts data/demo/contexts.jsonl \
    --graph-predictions outputs/demo/rsg-predictions.jsonl \
    --condition "${condition}" \
    --output "outputs/demo/rsg-${condition}-top5.jsonl"
  uv run ragsieve filter-contexts \
    --contexts data/demo/contexts.jsonl \
    --predictions outputs/demo/rsq-predictions.jsonl \
    --graph-predictions outputs/demo/rsg-predictions.jsonl \
    --condition "${condition}" \
    --output "outputs/demo/joint-${condition}-top5.jsonl"
done

printf 'RSQ: outputs/demo/rsq-metrics.json\n'
printf 'RSG: outputs/demo/rsg-metrics.json\n'
printf 'Filtered top-5: outputs/demo/filtered-top5.jsonl\n'
printf 'RSG and joint top-5: outputs/demo/{rsg,joint}-{pr_w,ipi_cem_c}-top5.jsonl\n'
