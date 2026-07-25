# config.py — Global configuration for the DMKG 2026 cost-characterization study
#
# Reuses the FedV-KGQA fixed substrate (TransE + frozen BERT) exactly as
# accepted at ISWC. The only things that vary across experiment cells are:
#   DATASET   in {"metaqa", "pathquestion"}
#   STRATEGY  in {"dom", "freq", "cut", "rand"}
#   K         in {3, 5, 7}

import os

BASE_DIR        = "/home/islamm9/DMKG"          # ← target path on your machine
DATA_DIR        = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR         = os.path.join(BASE_DIR, "logs")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")

# ── Raw per-dataset paths ───────────────────────────────────────────────────
# Each dataset needs: a global KB (triples) + train/dev/test QA files.
# Same "h|r|t" triple format and MetaQA-style QA format ("question [Topic]\tans1|ans2")
# that dataset.py already parses — PathQuestion should be converted to this
# format once (see data/README_DATA.md).
DATASET_PATHS = {
    "metaqa": {
        # enriched KB (output of enrich_metaqa.py) — matches the ISWC substrate.
        # Run `python enrich_metaqa.py` once before training.
        "kb":       os.path.join(DATA_DIR, "metaqa", "kb_enriched.txt"),
        "qa_train": os.path.join(DATA_DIR, "metaqa", "qa_train.txt"),
        "qa_dev":   os.path.join(DATA_DIR, "metaqa", "qa_dev.txt"),
        "qa_test":  os.path.join(DATA_DIR, "metaqa", "qa_test.txt"),
    },
    "pathquestion": {
        "kb":       os.path.join(DATA_DIR, "pathquestion", "kb.txt"),
        "qa_train": os.path.join(DATA_DIR, "pathquestion", "qa_train.txt"),
        "qa_dev":   os.path.join(DATA_DIR, "pathquestion", "qa_dev.txt"),
        "qa_test":  os.path.join(DATA_DIR, "pathquestion", "qa_test.txt"),
    },
}

# ── Experiment grid ──────────────────────────────────────────────────────────
DATASETS   = ["metaqa", "pathquestion"]
STRATEGIES = ["dom", "freq", "cut", "rand"]      # Π_dom, Π_freq, Π_cut, Π_rand
SILO_COUNTS = [3, 5, 7]                          # K
RAND_SEEDS  = [1, 2, 3]                          # seeds averaged for Π_rand

# Semantic domain labels used only by Π_dom (dataset-specific, coarse).
# Any relation not listed falls back to a hash-based domain bucket so Π_dom
# still works on datasets we haven't manually labeled (e.g. PathQuestion).
RELATION_DOMAINS = {
    "metaqa": {
        # original relations
        "directed_by": "production", "written_by": "production",
        "starred_actors": "cast", "has_tags": "cast",
        "release_year": "classification", "in_language": "classification",
        "has_genre": "classification", "has_imdb_rating": "classification",
        "has_imdb_votes": "classification",
        # entailed relations from enrich_metaqa.py (grouped with their domain)
        "directed": "production", "wrote": "production",
        "associated_actor": "cast",
        "associated_genre": "classification",
        "associated_year": "classification",
        "associated_language": "classification",
    },
    # PathQuestion / Freebase13 — same coherent grouping as the ISWC PQ2H
    # silo split (split_kb_silos.py): family / person-attributes / biographical.
    "pathquestion": {
        "parents": "family", "children": "family", "spouse": "family",
        "gender": "attributes", "nationality": "attributes",
        "ethnicity": "attributes", "religion": "attributes",
        "cause_of_death": "attributes",
        "profession": "biographical", "institution": "biographical",
        "place_of_birth": "biographical", "place_of_death": "biographical",
        "location": "biographical",
    },
}

# ── TransE KGE (identical to FedV-KGQA / ISWC) ───────────────────────────────
KGE_EMBED_DIM   = 256
KGE_MARGIN      = 1.0
KGE_NORM        = 2
KGE_LR          = 1e-3
KGE_EPOCHS      = 40
KGE_BATCH_SIZE  = 512
KGE_NEG_SAMPLES = 10

# ── Question Encoder (BERT + MLP), identical to FedV-KGQA ───────────────────
BERT_MODEL      = "bert-base-uncased"
MLP_HIDDEN_DIMS = [768, 512]
MLP_DROPOUT     = 0.1

# ── Federated QA training ────────────────────────────────────────────────────
QA_LR           = 1e-4
QA_EPOCHS       = 40
QA_BATCH_SIZE   = 64
QA_MARGIN       = 1.0
USE_TOPIC_ANCHORING = True


QUALITY_TARGET = {
    "metaqa":       0.70,
    "pathquestion": 0.50,
}
# kept for backward-compat with older code paths
QUALITY_TARGET_HITS3 = {"metaqa": 0.70, "pathquestion": 0.50}

# ── Candidate index (M2) ─────────────────────────────────────────────────────
MAX_NEIGHBORS       = 100
CANDIDATE_HOP1_CAP  = 50
CANDIDATE_HOP2_CAP  = 20

# ── Misc ──────────────────────────────────────────────────────────────────────
SEED   = 42
DEVICE = "cuda"
