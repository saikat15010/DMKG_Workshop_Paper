# DMKG 2026 — Vertical Partitioning Cost Characterization

Implementation for "Vertical Partitioning Strategies for Federated Knowledge
Graphs: A Data-Management Cost Characterization". Reuses the FedV-KGQA
(ISWC 2026) TransE + frozen BERT substrate exactly, generalized from a fixed
3-silo setup to arbitrary K, with four pluggable partitioning strategies.

## Install

```bash
cd /home/islamm9/DMKG
pip install torch transformers --break-system-packages   # if not already present
```

## Data layout expected

The loaders **auto-detect** each dataset's native format, so you drop your
existing files in as-is — no reformatting:

- MetaQA: KB is pipe-delimited (`head|relation|tail`); QA is
  `question [Topic]\tans1|ans2` (topic in brackets, pipe-separated answers).
- PathQuestion: KB is tab-delimited (`head\trelation\ttail`); QA is
  `question\ttopic_entity\tanswer` (3 explicit columns).

```
/home/islamm9/DMKG/data/
  metaqa/
    kb.txt         # original unpartitioned MetaQA KB (pipe-delimited)
    qa_train.txt   qa_dev.txt   qa_test.txt      (2-hop split)
  pathquestion/
    kb.txt         # = your PQ2H kb_enriched.txt (tab-delimited)
    qa_train.txt   qa_dev.txt   qa_test.txt
```

Copy from your existing ISWC trees:

```bash
mkdir -p /home/islamm9/DMKG/data/metaqa /home/islamm9/DMKG/data/pathquestion

# MetaQA — original (NOT enriched, NOT pre-split) KB + 2-hop QA
cp /path/to/your/metaqa/kb.txt              /home/islamm9/DMKG/data/metaqa/kb.txt
cp /path/to/your/metaqa/qa/2-hop/qa_*.txt   /home/islamm9/DMKG/data/metaqa/

# PathQuestion — reuse the enriched KB + QA that process_pq2h.py produced
cp /home/islamm9/ISWC/Dataset/PQ2H/data/kb_enriched.txt \
   /home/islamm9/DMKG/data/pathquestion/kb.txt
cp /home/islamm9/ISWC/Dataset/PQ2H/data/qa/2-hop/qa_*.txt \
   /home/islamm9/DMKG/data/pathquestion/
```

Point `config.DATASET_PATHS[...]["kb"]` at the **unpartitioned** KB for each
dataset — partitioning now happens dynamically per (strategy, K), so the
pre-split `silos/kb_silo_*.txt` files are not used here. Using PathQuestion's
`kb_enriched.txt` (with the family-relation inverses already materialized) is
correct and matches your ISWC substrate; just don't feed in the per-silo files.

## What's new vs. FedV-KGQA

| File | Role |
|---|---|
| `partition.py` | The 4 strategies: Π_dom, Π_freq, Π_cut, Π_rand |
| `metrics.py` | M1 (communication), M2 (index size), M3 (CSPL), M4 (imbalance), M5 (latency) |
| `data_prep.py` | Dataset-agnostic KB/QA loading + partition-driven silo split (generalizes `split_kb.py`) |
| `kge_and_server.py` | TransE + FedServer generalized from fixed 3 silos to any K |
| `run_cell.py` | Runs one (dataset, strategy, K) cell end-to-end |
| `run_all.py` | Sweeps the full grid, writes `results/<dataset>_cost_table.csv` |

## Design notes / where this deviates from FedV-KGQA

- **No OWL enrichment.** The DMKG cost model measures partitioning effects on
  the raw candidate index and communication volume; the enrichment layer
  (`ontology_enrich_kb.py`) was movie-domain-specific and orthogonal to the
  question being asked here. If you want it back for MetaQA specifically,
  run your existing `ontology_enrich_kb.py` once and point `DATASET_PATHS`
  at `kb_enriched_owl.txt` instead — everything downstream is agnostic to
  which KB file it's given.
- **Gold relation paths for Π_cut** are recovered via BFS over the *global*
  (pre-partition) KG from each topic entity to its gold answers, since
  neither MetaQA nor PathQuestion ships explicit hop-by-hop relation labels
  in the QA file. This is the co-occurrence signal Eq. (1) in the paper is
  built from — reasonable and consistent with "aggregate co-occurrence
  counts over training paths," but flag it in your limitations paragraph
  since it's a proxy for the true gold path when a query is ambiguous
  (multiple paths reach the same answer; BFS returns the shortest one).
- **M1 (communication to target)** is measured by actually fine-tuning
  each cell and counting optimizer steps until dev Hits@3 crosses
  `config.QUALITY_TARGET_HITS3[dataset]`, capped by `--quick`/`max_qa_steps`
  for smoke testing. Full runs should drop the cap.

## Running

```bash
# smoke test first — a few minutes, checks the whole pipeline wires together
python run_all.py --quick

# one dataset, one strategy, one K, for debugging a specific cell
python -c "from run_cell import run_cell; print(run_cell('metaqa','cut',3))"

# full grid (this is the actual paper run — budget real compute time)
python run_all.py
```

Output: `results/metaqa_cost_table.csv`, `results/pathquestion_cost_table.csv`,
one row per (strategy, K[, seed for rand]) — directly fillable into Table 2.
