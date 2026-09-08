# Artifact guide

## Evaluation levels

The artifact supports three levels of use:

1. `run_demo.sh` verifies the complete detector path with four target queries and twenty
   poison documents.
2. The released text knowledge bases support clean retrieval and new attack evaluations
   without downloading dataset preprocessors.
3. Paper-scale evaluation applies the same commands to prepared attack files for all 100
   targets, six attacks, three datasets, and three retrievers.

The repository contains RAGSieve. Attack construction and competing defenses follow
the cited projects and are not duplicated here.

## Data schemas

### Corpus and query files

`corpus.jsonl` contains `id`, `text`, and optional source metadata. `queries.jsonl`
contains `id`, `text`, and `answers`. `qrels.jsonl` joins each query to relevant corpus
documents. `subset_seed2026_n100.json` records the fixed target IDs.

### Attack file

Each attack row has this form:

```json
{
  "query_id": "...",
  "query": "...",
  "correct_answers": ["..."],
  "target_answer": "...",
  "attack": "pr_w",
  "documents": [{"id": "...", "text": "..."}]
}
```

Optimization traces are unnecessary for evaluation and are omitted from the demo.

### Prepared RSQ context

Each line represents one retrieval event and contains a query plus its ranked documents.
`prepare-contexts` and the precomputed demo both retain the top 100 for serial filtering.

| Field | Description |
|---|---|
| `query_id`, `query` | Stable ID and the current query text. |
| `dataset`, `encoder`, `condition` | Evaluation grouping keys. |
| `documents[].id` | Stable document identifier. |
| `documents[].rank` | One-based exact-cosine rank. |
| `documents[].text` | Text scored by RSQ. |
| `documents[].poison` | Optional evaluation label. |

RSQ reads ranks 1--5 as generation candidates and ranks 6--20 as the query-local
reference. The evaluator consumes the optional labels only after predictions have been
written.

### RSG snapshot

The RSG document JSONL and `.npy` matrix must have identical row order. Predictions
contain a stable `document_id`, the corpus-local density score, the writing-system score,
the combined score, and `flagged`. Labels are stored separately and joined only by
`evaluate-graph`.

## Detector outputs

`ragsieve detect` writes one row for each of the five RSQ candidates. Besides IDs and
the combined `score`, it records:

- answer-anchor and script-integrity evidence;
- local surprisal transition evidence;
- query-alignment jump and empirical tail probability;
- the fixed-threshold `flagged` decision.

`ragsieve detect-graph` writes one row for every corpus document with:

- eligible-neighbor count and corpus-local density contrast;
- empirical upper-tail probability;
- script-integrity evidence;
- the combined score and `flagged` decision.

These component fields are the inputs to the component tables and figures.

## Paper configuration

The CLI defaults are the paper configuration. RSQ uses five candidates, a top-20
retrieval window, Qwen3-0.6B-Base for surprisal, and BERT-base-uncased layer 9 for query
alignment. RSG uses 16 exact neighbors, four dense-core neighbors, semantic similarity
0.85, lexical Jaccard 0.60, and a 5% corpus alert fraction.

For each reported cell, preserve the dataset, retriever, attack, injection volume, and
original retrieval order. The main document-level metric reports poison detection when
clean-document removal is at most 5%. End-to-end evaluation removes flagged documents,
traverses the unchanged ranking to refill five documents, and then runs the QA prompt.
Joint evaluation applies RSG before RSQ. `quarantine-contexts` removes corpus flags
from an exact saved ranking, preserves all survivors for refill, and renumbers them.
RSQ then scores survivor ranks 1--5 against survivor ranks 6--20, followed by one refill
without rescoring replacements. At least 20 survivors are required; otherwise retrieve
a deeper pool or use `prepare-contexts --exclude-predictions` to search the filtered index.
Original-context RSQ predictions must not be reused for changed candidates or references.

## End-to-end QA configuration

Answer generation and semantic ASR judging use one OpenAI-compatible endpoint. The
`.env` file contains `OPENAI_BASE_URL`, `OPENAI_MODEL`, and `OPENAI_API_KEY`.
`ragsieve qa` applies the answer and judge prompts in `PROMPTS.md` with temperature 0
and reports ASR, token F1, and exact match by condition.

## Full-corpus RSG and joint QA

Run `bash data_preparation.sh` first. The following NQ/BGE-M3 example selects PR-W from
the bundled attacks and constructs its corpus snapshot. For paper-scale evaluation,
replace the attack file with the prepared 100-target file for the desired attack and
retriever. Set `count` to 1, 3, 5, or 10 for the injection-volume experiment; ten documents
repeat the five passages with distinct IDs. Use a separate output directory for each run.

```bash
uv run python - <<'PY'
import json
from pathlib import Path

dataset, attack, count = "nq", "pr_w", 5
attack_file = Path("data/demo/attacks.jsonl")
out = Path("outputs/nq-bge-m3-pr_w")
out.mkdir(parents=True, exist_ok=True)
clean = [json.loads(line) for line in Path(f"data/datasets/{dataset}/corpus.jsonl").read_text(encoding="utf-8").splitlines()]
selected = []
for line in attack_file.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    if row["attack"] != attack:
        continue
    source = row["documents"][:5]
    documents = (source * 2)[:count] if count == 10 else source[:count]
    documents = [
        {**doc, "id": doc["id"] if i < len(source) else f"{doc['id']}_copy1"}
        for i, doc in enumerate(documents)
    ]
    selected.append({**row, "documents": documents})
poison = [doc for row in selected for doc in row["documents"]]
labels = [{"document_id": doc["id"], "poison": False} for doc in clean]
labels += [{"document_id": doc["id"], "poison": True} for doc in poison]
for name, rows in (("attacks", selected), ("corpus", clean + poison), ("labels", labels)):
    (out / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
PY

snapshot=outputs/nq-bge-m3-pr_w
uv run ragsieve build-index --corpus "$snapshot/corpus.jsonl" \
  --output-dir "$snapshot/index" --encoder bge-m3 --device cuda:0
uv run ragsieve detect-graph --documents "$snapshot/corpus.jsonl" \
  --embeddings "$snapshot/index/embeddings.npy" \
  --output "$snapshot/rsg-pr_w.jsonl" --device cuda:0
uv run ragsieve evaluate-graph --predictions "$snapshot/rsg-pr_w.jsonl" \
  --labels "$snapshot/labels.jsonl" --output "$snapshot/rsg-metrics.json"

# The clean-QA condition uses a scan without injected documents.
uv run ragsieve detect-graph --documents data/datasets/nq/corpus.jsonl \
  --embeddings data/indices/nq/bge-m3/embeddings.npy \
  --output "$snapshot/rsg-clean.jsonl" --device cuda:0

# Save the exact ranking; standalone RSQ metrics use the original contexts.
uv run ragsieve prepare-contexts --dataset-dir data/datasets/nq \
  --index-dir data/indices/nq/bge-m3 --attacks "$snapshot/attacks.jsonl" \
  --subset data/datasets/nq/subset_seed2026_n100.json \
  --top-k 100 --output "$snapshot/contexts.jsonl" --device cuda:0
uv run ragsieve detect --input "$snapshot/contexts.jsonl" \
  --output "$snapshot/rsq.jsonl" --device cuda:0
uv run ragsieve evaluate --predictions "$snapshot/rsq.jsonl" \
  --output "$snapshot/rsq-metrics.json"

for condition in clean pr_w; do
  uv run ragsieve quarantine-contexts --contexts "$snapshot/contexts.jsonl" \
    --graph-predictions "$snapshot/rsg-${condition}.jsonl" \
    --condition "$condition" --output "$snapshot/serial-${condition}-contexts.jsonl"
  uv run ragsieve detect --input "$snapshot/serial-${condition}-contexts.jsonl" \
    --output "$snapshot/serial-${condition}-rsq.jsonl" --device cuda:0
  uv run ragsieve filter-contexts --contexts "$snapshot/serial-${condition}-contexts.jsonl" \
    --predictions "$snapshot/serial-${condition}-rsq.jsonl" \
    --condition "$condition" --output "$snapshot/joint-${condition}-top5.jsonl"
  uv run ragsieve qa --input "$snapshot/joint-${condition}-top5.jsonl" \
    --output "$snapshot/joint-${condition}-qa.jsonl" \
    --summary "$snapshot/joint-${condition}-qa-summary.json" --env-file .env
done
```

For RSG-only QA, apply `filter-contexts` to the original contexts with the corresponding
`--graph-predictions` and `--condition`, then run `qa`. For an unfiltered reference,
omit both prediction arguments;
`filter-contexts` then selects the original Top-5. Each invocation operates on one
dataset/retriever. Keep the resulting per-cell metrics and macro-average cells with
equal weight, rather than pooling all document or QA rows.

## Detection cost

`benchmark` uses one warm-up call and three measured repetitions by default; these can
be set with `--warmup` and `--repeats`. RSQ is timed per retrieval event after its models
are loaded. RSG is timed per complete scan after text and vectors are loaded onto the
selected device. Graph construction is included; indexing, retrieval, and QA are excluded.
GPU memory is peak allocated memory during measured execution, including resident inputs
and models.

The reported workload uses 2,100 RSQ retrieval events across the three BGE-M3 systems
and one 128,544-document NQ corpus snapshot for RSG (128,044 clean documents and 500
injected documents). The bundled four-query demo does not represent this workload.
For the RSQ workload, retain each clean query once and each of its six attacked contexts,
then pass the combined context file to `benchmark --mode rsq`. For RSG, use the complete
five-document-per-target snapshot above with all 100 targets and `benchmark --mode rsg`.

The experiments use Ubuntu 22.04, two Intel Xeon Gold 6530 processors (64 physical cores,
128 logical CPUs), 503 GiB RAM, and an NVIDIA GeForce RTX 5090 with 32 GB memory.
