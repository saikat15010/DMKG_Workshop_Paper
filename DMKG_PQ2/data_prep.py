# data_prep.py — dataset-agnostic KB/QA loading + partition-driven silo split
#
# Generalizes FedV-KGQA's dataset.py (build_index, build_shared_entity_index)
# and split_kb.py (hardcoded Silo A/B/C) to: any dataset, any K, any Π.

import re
from collections import defaultdict


# ── Global KB loading ─────────────────────────────────────────────────────────

def _split_triple_line(line):
    """Auto-detect delimiter: MetaQA uses '|', PathQuestion uses TAB."""
    line = line.rstrip("\n")
    if "\t" in line:
        parts = line.split("\t")
    else:
        parts = line.split("|")
    return parts


def load_global_kb(kb_path):
    """Read 'h<delim>r<delim>t' lines (delim = tab or pipe) -> list of (h, r, t)."""
    triples = []
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = _split_triple_line(line)
            if len(parts) == 3:
                triples.append(tuple(p.strip() for p in parts))
    return triples


def relation_vocabulary(triples):
    return sorted({r for _, r, _ in triples})


def triple_counts_by_relation(triples):
    counts = defaultdict(int)
    for _, r, _ in triples:
        counts[r] += 1
    return dict(counts)


def entity_vocabulary(triples):
    ents = set()
    for h, _, t in triples:
        ents.add(h); ents.add(t)
    return sorted(ents)


def build_shared_entity_index(triples):
    return {e: i for i, e in enumerate(entity_vocabulary(triples))}


# ── QA file parsing — auto-detects the two dataset formats ──────────────────
#
#   MetaQA        : "question [Topic]\tans1|ans2|..."     (2 tab columns)
#   PathQuestion  : "question\ttopic_entity\tanswer"       (3 tab columns)
#
# Both resolve to the same (question, topic_entity, [answers]) tuple.

def parse_qa_file(qa_path):
    samples = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")

            if len(parts) == 3:
                # PathQuestion: explicit topic + single answer columns
                question, topic_entity, answer = (p.strip() for p in parts)
                if question and topic_entity and answer:
                    samples.append((question, topic_entity, [answer]))

            elif len(parts) == 2:
                # MetaQA: topic entity embedded in [brackets], pipe-sep answers
                question_raw, answers_raw = parts
                match = re.search(r"\[(.+?)\]", question_raw)
                if not match:
                    continue
                topic_entity = match.group(1)
                question_clean = re.sub(r"\[(.+?)\]", r"\1", question_raw).strip()
                answers = [a.strip() for a in answers_raw.split("|") if a.strip()]
                if answers:
                    samples.append((question_clean, topic_entity, answers))

    return samples


# ── Partition-driven silo split ───────────────────────────────────────────────

def split_by_partition(triples, assignment, K):
    """
    assignment: dict relation -> silo_id (0..K-1), from partition.py
    Returns: list of length K, each a list of (h, r, t) triples for that silo.
    Entities are shared (VFL): every silo keeps the full shared entity index,
    it just may have zero triples touching some entities.
    """
    silos = [[] for _ in range(K)]
    for h, r, t in triples:
        silo_id = assignment.get(r)
        if silo_id is not None:
            silos[silo_id].append((h, r, t))
    return silos


def build_neighbor_index(silo_triples, shared_entity2id, max_neighbors=100):
    """Bidirectional neighbor index for one silo's triples (see FedV dataset.py)."""
    neighbors = defaultdict(set)
    for h, r, t in silo_triples:
        h_id = shared_entity2id.get(h)
        t_id = shared_entity2id.get(t)
        if h_id is not None and t_id is not None:
            neighbors[h_id].add(t_id)
            neighbors[t_id].add(h_id)
    return {k: list(v)[:max_neighbors] for k, v in neighbors.items()}


def two_hop_candidates(entity_id, neighbor_index, hop1_cap=50, hop2_cap=20):
    hop1 = list(neighbor_index.get(entity_id, []))[:hop1_cap]
    hop2 = set()
    for nb in hop1:
        hop2.update(neighbor_index.get(nb, [])[:hop2_cap])
    return {entity_id} | set(hop1) | hop2
