# Released data

`datasets/` contains the three text knowledge bases used in the paper. Each dataset
includes the 1,000 sampled queries, their union corpus, relevance labels, attack targets,
counterfactual payloads, and the fixed 100-query target subset. Dense vectors are not
distributed: `data_preparation.sh` materializes them under the ignored `data/indices/`
directory with the selected retriever.

`demo/` is a small, ready-to-inspect example. It contains four NQ target queries, two
PR-W and two CEM-C attack realizations (five documents per query), prepared top-20 BGE-M3
retrieval contexts, and a 520-document corpus snapshot for RSG. Optimization traces are
not included. The attack examples are evaluation fixtures; the release does not contain
attack-generation code.

## Dataset files

| File | Contents |
|---|---|
| `corpus.jsonl` | Raw corpus documents and stable identifiers. |
| `queries.jsonl` | The 1,000 sampled questions and reference answers. |
| `qrels.jsonl` | Relevance judgments for retrieval evaluation. |
| `targets.jsonl` | Per-query adversarial target answers. |
| `counterfactuals.jsonl` | Five counterfactual payloads per target query. |
| `subset_seed2026_n100.json` | The 100 attacked query identifiers. |
| `manifest.json` | Sampling and corpus metadata. |

The underlying NQ, HotpotQA, and MS MARCO content retains its original license.
