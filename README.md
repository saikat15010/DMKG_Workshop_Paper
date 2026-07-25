# Cost Characterization of Vertically Partitioned Federated Knowledge Graphs

Code for the DMKG 2026 paper *Cost Characterization of Vertically Partitioned
Federated Knowledge Graphs*. We treat the assignment of relations to silos as a
design space and measure what each assignment costs. Four partitioning
strategies are compared across three silo counts on two multi-hop KGQA
benchmarks, with the learning substrate held fixed so that every measured
difference is attributable to the partition.

> Anonymized for double-blind review.

## Repository structure

The two benchmarks were run in separate working directories, kept here as two
folders that share the same pipeline. They differ only in the dataset loaded
and in two training-protocol values (see Protocol note).

metaqa/ MetaQA experiments
pathquestion/ PathQuestion experiments + shared analysis scripts

Each folder contains the full pipeline:

| File | Role |
|------|------|
| `config.py`         | datasets, silo counts, hyperparameters |
| `data_prep.py`      | loading, shared-entity indexing, partition splitting |
| `enrich_metaqa.py`  | offline inverse + property-chain enrichment |
| `partition.py`      | the four strategies and the co-occurrence graph |
| `qa_dataset.py`     | question/answer dataset and collation |
| `kge_and_server.py` | local TransE, server-side fusion, QA scoring head |
| `metrics.py`        | M1-M5 definitions |
| `run_study.py`      | main runner: builds partitions, trains, records M1-M5 |
| `run.txt`           | exact commands used |

The `pathquestion/` folder additionally holds the dataset-agnostic analysis
scripts that produce the paper's final numbers:

| File | Role |
|------|------|
| `m3_heldout.py`      | recomputes M3 on held-out test paths (CPU only) |
| `measure_latency.py` | clean M5 re-measurement, inference only |
| `make_figures.py`    | generates the paper figures from result CSVs |

## Strategies and metrics

Strategies: `dom` (semantic-domain, the inherited default), `freq`
(frequency-balanced), `cut` (co-occurrence graph-cut), `rand` (random baseline).

Metrics: M1 communication, M2 index size, M3 cross-silo crossing rate,
M4 load imbalance, M5 query latency.

## Data

The benchmarks are not redistributed. Place each dataset under the `data/`
folder of its directory:

metaqa/data/
kb.txt one triple per line: head <TAB> relation <TAB> tail
qa_train.txt one example per line: question <TAB> topic_entity <TAB> answer[,answer...]
qa_dev.txt
qa_test.txt

pathquestion/data/
kb.txt
qa_train.txt
qa_dev.txt
qa_test.txt

MetaQA is derived from WikiMovies (2-hop split); PathQuestion from Freebase
(PQ-2H). Both `kb.txt` files are the enriched graphs: inverse and
property-chain axioms are materialized offline before partitioning, so answer
entities are reachable within the two-hop expansion the substrate performs.
`enrich_metaqa.py` reproduces this enrichment for MetaQA. After enrichment,
MetaQA has 43,235 entities and 15 relations; PathQuestion has 75,043 entities
and 13 relations.

## Reproducing the results

```bash
python -m venv venv && source venv/bin/activate
pip install torch transformers matplotlib numpy

# MetaQA
cd metaqa
python run_study.py --dataset metaqa
cd ..

# PathQuestion
cd pathquestion
python run_study.py --dataset pathquestion

# held-out M3, clean latency, figures (run from pathquestion/)
python m3_heldout.py
python measure_latency.py
python make_figures.py
```

`run_study.py` needs one CUDA GPU. The three analysis scripts run on CPU.
Results are written to each folder's `results/`; figures to `figures/`.
See each folder's `run.txt` for the exact commands used.

## Protocol note

The two benchmarks use dataset-specific training budgets, reflecting their
different achievable quality ceilings. MetaQA runs 40 QA rounds to a Hits@3
target of 0.70. PathQuestion runs 100 rounds to a target of 0.55, given its
smaller training set of 1,524 questions. Because of this, M1 (communication to
target) is comparable across strategies within a dataset but not between them.

The `cut` strategy fits its co-occurrence graph on training paths only; M3 is
evaluated on held-out test paths via `m3_heldout.py`, so the reported crossing
rates reflect generalization. `m3_heldout.py` regenerates every partition from
a single code path, so the paper's headline M3 and M4 values are internally
consistent regardless of which folder ran which dataset. The `cut` heuristic is
a greedy weighted clustering with a balance slack of 1.25, not an exact
minimum cut.

## License

MIT. See `LICENSE`.
