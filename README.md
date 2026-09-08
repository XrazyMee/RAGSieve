# RAGSieve

Code for **RAGSieve: Detecting Knowledge-Poisoned Documents in RAG Without a Trusted Reference**.

RAGSieve detects knowledge-poisoning documents at two points in a retrieval-augmented
generation (RAG) system. **RSQ** is an online filter: it scores the five documents selected
for generation against ranks 6--20 for the current query. **RSG** is an offline scanner: it
scores every corpus document against its own semantic--lexical neighborhood. The package
contains the RSQ and RSG implementations used in the paper, exact full-corpus retrieval, Top-5
filtering and refill, document-level evaluation, and a compact runnable example.

## Repository contents

```text
data/datasets/       NQ, HotpotQA, and MS MARCO text knowledge bases
data/demo/           four target queries and 20 example poison documents
src/ragsieve/     RSQ, RSG, retrieval, filtering, and metrics
install.sh           environment installation
data_preparation.sh  local construction of the nine dense indices
run_demo.sh          end-to-end RSQ and RSG functionality check
```

The three released knowledge bases contain 128,044 NQ documents, 9,961 HotpotQA
documents, and 8,239 MS MARCO passages. They include raw text, 1,000 sampled queries,
qrels, target answers, and the fixed 100-query attack subset. Dense vectors are generated
locally and are never part of the repository.

The demo contains two PR-W and two CEM-C realizations from NQ, with five poison documents
per query. These attacks give a short, inspectable path through both detectors without
shipping the complete attack collection or attack-generation implementations.

## Installation

Python 3.11 or 3.12, `uv`, and a CUDA-capable GPU are recommended.

```bash
bash install.sh
```

The detector downloads the Hugging Face models named in the paper on first use.

For end-to-end QA, copy the shared OpenAI-compatible configuration and fill in the three
values:

```bash
cp .env.example .env
```

```dotenv
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=your-model-name
OPENAI_API_KEY=your-api-key
```

The same endpoint and model generate answers and perform semantic ASR judging.

## Quick evaluation

```bash
bash run_demo.sh
```

This command builds a local BGE-M3 index for the 520-document RSG snapshot, runs RSQ on
the prepared full-corpus retrieval contexts, filters and refills the generation Top-5,
and evaluates both detectors. Results are written to `outputs/demo/`:

- `rsq-metrics.json`: RSQ AUROC and poison detection at 5% clean removal;
- `rsg-metrics.json`: RSG metrics on the query-free corpus snapshot;
- `filtered-top5.jsonl`: the five documents retained for generation after RSQ.
- `rsg-*-top5.jsonl` and `joint-*-top5.jsonl`: retained contexts for each demo attack
  after RSG alone and after joint filtering. QA is a separate, opt-in command.

With the released examples, RSQ reaches 100% AUROC and 100% poison detection at 5% clean
removal for both PR-W and CEM-C. RSG reaches 100% AUROC and 100% poison detection at the
same clean-removal budget; its fixed rule removes 85% of the poison documents and 0% of
the clean documents. These values are a functionality check on the curated demo, not the
paper's aggregate result.

Ground-truth fields are read only by the evaluation commands. They are not passed to RSQ
or RSG.

## Build the complete local indices

```bash
bash data_preparation.sh
```

By default, this builds the 3 datasets x 3 retrievers evaluated in the paper: BGE-M3,
E5-large-v2, and all-MiniLM-L6-v2. The output is placed under the ignored
`data/indices/` directory. A subset can be selected without editing the script:

```bash
DATASETS="nq" ENCODERS="bge-m3" DEVICE="cuda:0" bash data_preparation.sh
```

Each index is produced from the complete released corpus with the paper's pooling,
prefix, truncation, and cosine-normalization settings.

## RSQ: online query-time filtering

Prepare a clean retrieval and a retrieval with the bundled attacks:

```bash
uv run ragsieve prepare-contexts \
  --dataset-dir data/datasets/nq \
  --index-dir data/indices/nq/bge-m3 \
  --subset data/demo/subset.json \
  --attacks data/demo/attacks.jsonl \
  --output outputs/demo/contexts.jsonl \
  --device cuda:0
```

`prepare-contexts` performs exact search over the full corpus once for each query, scores
the injected documents separately, and saves the top 100 in their original order. RSQ
scores candidates 1--5 against ranks 6--20; the remaining ranks are available for refill.
Run RSQ, compute document-level metrics, and refill the generation context:

```bash
uv run ragsieve detect \
  --input outputs/demo/contexts.jsonl \
  --output outputs/demo/rsq-predictions.jsonl \
  --device cuda:0

uv run ragsieve evaluate \
  --predictions outputs/demo/rsq-predictions.jsonl \
  --output outputs/demo/rsq-metrics.json

uv run ragsieve filter-contexts \
  --contexts outputs/demo/contexts.jsonl \
  --predictions outputs/demo/rsq-predictions.jsonl \
  --output outputs/demo/filtered-top5.jsonl

uv run ragsieve qa \
  --input outputs/demo/filtered-top5.jsonl \
  --output outputs/demo/qa-records.jsonl \
  --summary outputs/demo/qa-summary.json \
  --env-file .env
```

RSQ combines four evidence sources: answer-anchor concentration, writing-system
integrity, local language-model surprisal, and a query-alignment transition. All settings
in the CLI defaults are the paper settings.

## RSG: offline corpus inspection

RSG consumes document text and the vectors already maintained by the RAG index:

```bash
uv run ragsieve detect-graph \
  --documents data/demo/corpus.jsonl \
  --embeddings data/indices/demo/bge-m3/embeddings.npy \
  --output outputs/demo/rsg-predictions.jsonl \
  --device cuda:0

uv run ragsieve evaluate-graph \
  --predictions outputs/demo/rsg-predictions.jsonl \
  --labels data/demo/labels.jsonl \
  --output outputs/demo/rsg-metrics.json
```

RSG builds the exact cosine top-16 graph, retains semantic-near and lexical-far
neighbors, measures each document's density rise over its own neighborhood floor, and
combines the empirical corpus tail with writing-system integrity.

## Joint deployment

Joint deployment is serial: RSG excludes documents first, then RSQ scores the new
top five against surviving ranks 6--20. Flagged candidates are removed and the context
is refilled once from the surviving ranking; replacement documents are not rescored.
After `bash run_demo.sh`, the PR-W example is:

```bash
uv run ragsieve quarantine-contexts \
  --contexts data/demo/contexts.jsonl \
  --graph-predictions outputs/demo/rsg-predictions.jsonl \
  --condition pr_w \
  --output outputs/demo/serial-pr_w-contexts.jsonl
uv run ragsieve detect \
  --input outputs/demo/serial-pr_w-contexts.jsonl \
  --output outputs/demo/serial-pr_w-rsq.jsonl --device cuda:0
uv run ragsieve filter-contexts \
  --contexts outputs/demo/serial-pr_w-contexts.jsonl \
  --predictions outputs/demo/serial-pr_w-rsq.jsonl \
  --output outputs/demo/joint-pr_w-top5.jsonl

uv run ragsieve qa \
  --input outputs/demo/joint-pr_w-top5.jsonl \
  --output outputs/demo/joint-pr_w-qa.jsonl \
  --summary outputs/demo/joint-pr_w-qa-summary.json \
  --env-file .env
```

For RSG alone, run `filter-contexts` on the original contexts with only
`--graph-predictions` and `--condition`. For CEM-C, use `--condition ipi_cem_c` and separate
output paths. `--condition` binds a RSG scan to the retrieval condition being evaluated;
unpoisoned QA uses a separate RSG scan of the clean corpus. The bundled RSG snapshot
contains both demo attacks. Full-corpus and clean-QA steps are in
[the artifact guide](docs/ARTIFACT.md#full-corpus-rsg-and-joint-qa).

## Component ablations

Saved evidence is sufficient to run the published ablations without loading models again:

```bash
uv run ragsieve ablate \
  --predictions outputs/demo/rsq-predictions.jsonl \
  --variant without-answer-anchor \
  --output outputs/demo/rsq-without-answer-anchor.jsonl
uv run ragsieve evaluate \
  --predictions outputs/demo/rsq-without-answer-anchor.jsonl \
  --output outputs/demo/rsq-without-answer-anchor-metrics.json

uv run ragsieve ablate \
  --predictions outputs/demo/rsg-predictions.jsonl \
  --variant corpus-local-only \
  --output outputs/demo/rsg-corpus-local-only.jsonl
uv run ragsieve evaluate-graph \
  --predictions outputs/demo/rsg-corpus-local-only.jsonl \
  --labels data/demo/labels.jsonl \
  --output outputs/demo/rsg-corpus-local-only-metrics.json
```

RSQ also supports `without-script-integrity`, `without-surprisal`, and
`without-query-alignment`; RSG also supports `script-integrity-only`. The corpus-local
ablation assigns the full 5% alert budget to the graph branch. Ablated predictions can
be passed to the same filtering and QA commands.

## Detection cost

```bash
uv run ragsieve benchmark --mode rsq \
  --input data/demo/contexts.jsonl \
  --output outputs/demo/rsq-cost.json --device cuda:0
uv run ragsieve benchmark --mode rsg \
  --input data/demo/corpus.jsonl \
  --embeddings data/indices/demo/bge-m3/embeddings.npy \
  --output outputs/demo/rsg-cost.json --device cuda:0
```

Measurements exclude model/data loading, retrieval, and QA. Models and embeddings remain
resident; CUDA work is synchronized around each measurement. The command reports mean,
median, and P95 execution time, throughput, and peak GPU allocation. Use the full NQ index
for the paper's corpus-scan workload; the small demo is a functionality check. Workload
and environment details are in [the artifact guide](docs/ARTIFACT.md#detection-cost).

## Mapping to the paper

The release reproduces the RAGSieve entries and ablations; baseline entries are
obtained from the implementations cited in the paper. The mapping below follows the current preprint.

| Paper result | Artifact path |
|---|---|
| Table 1 and Figure C1: RSQ document detection | `detect` then `evaluate` over the nine prepared systems. |
| Table C1: online QA after RSQ | `filter-contexts` followed by `qa` on the retained Top-5. |
| Table C2 and Figure C2(a): RSQ ablation | `ablate` then `evaluate`, using the four leave-one-out variants above. |
| Table 2 and Figure C1: RSG detection | `detect-graph` then `evaluate-graph` for each corpus snapshot. |
| Table C4 and Figure C2(b): RSG ablation | `ablate` with `corpus-local-only` or `script-integrity-only`, then `evaluate-graph`. |
| Table C3: offline QA after RSG | `filter-contexts --graph-predictions ... --condition ...`, then `qa`. |
| Figure C3: injection volume | [Snapshot preparation](docs/ARTIFACT.md#full-corpus-rsg-and-joint-qa) with 1, 3, 5, or 10 documents per query, followed by the same detection commands. |
| Table 3 and Figure C4: serial deployment | `quarantine-contexts`, `detect` on survivors, `filter-contexts`, then `qa`. |
| Tables B2/B3 and Figure B1: per-system results | Run the commands once per dataset/retriever; retain the per-cell JSON metrics. |
| Table B4 and Figure C5: detection cost | `benchmark --mode rsq` or `benchmark --mode rsg` on the stated workload. |

The same detection and serial QA commands accept independently prepared adaptive attack
files for Tables D2, D4, and D5. Attack optimization code is not included.

The paper's system diagrams are explanatory rather than generated data figures.
Detailed schemas and the expected evaluation protocol are in
[`docs/ARTIFACT.md`](docs/ARTIFACT.md).

## License

The code is released under the MIT License; see [LICENSE](LICENSE). Dataset and model licenses remain with their respective owners.
