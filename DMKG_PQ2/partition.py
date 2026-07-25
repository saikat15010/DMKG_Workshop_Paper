# partition.py — The vertical partitioning design space (Section 3 of the paper)
#
# A partition is Π : R → {0, ..., K-1}, represented here as a plain dict
# relation -> silo_id. All four strategies take the same signature so the
# experiment driver can call them uniformly.

import hashlib
import random
from collections import defaultdict, Counter


# ── Π_rand — optimize-nothing baseline ───────────────────────────────────────

def partition_random(relations, K, seed=0):
    rng = random.Random(seed)
    rels = list(relations)
    rng.shuffle(rels)
    return {r: i % K for i, r in enumerate(rels)}


# ── Π_dom — semantic-domain grouping ─────────────────────────────────────────

def _hash_bucket(relation, K):
    """Deterministic fallback domain for relations without a manual label."""
    h = int(hashlib.md5(relation.encode()).hexdigest(), 16)
    return h % K


def partition_domain(relations, K, domain_labels=None):
    """
    domain_labels: optional dict relation -> domain string (from config).
    Domains are mapped to silos round-robin by descending domain size, which
    approximates "each org gets a semantically coherent slice" while still
    covering all K silos when #domains != K.
    """
    domain_labels = domain_labels or {}
    groups = defaultdict(list)
    for r in relations:
        dom = domain_labels.get(r)
        if dom is None:
            # unlabeled relation: bucket by hash so behavior is still
            # deterministic and "domain-coherent" in the absence of metadata
            dom = f"_auto{_hash_bucket(r, K)}"
        groups[dom].append(r)

    # assign whole domains to silos, largest domain first, round-robin
    ordered_domains = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    assignment = {}
    for i, (_, rels) in enumerate(ordered_domains):
        silo = i % K
        for r in rels:
            assignment[r] = silo
    return assignment


# ── Π_freq — frequency-balanced (greedy longest-processing-time) ────────────

def partition_frequency(relations, K, triple_counts):
    """
    triple_counts: dict relation -> |T_r| (number of triples with that relation)
    Sort relations by triple count descending, always place the next relation
    into the currently lightest silo (by triple count).
    """
    rels_sorted = sorted(relations, key=lambda r: -triple_counts.get(r, 0))
    load = [0] * K
    assignment = {}
    for r in rels_sorted:
        silo = min(range(K), key=lambda k: load[k])
        assignment[r] = silo
        load[silo] += triple_counts.get(r, 0)
    return assignment


# ── Π_cut — co-occurrence graph-cut (locality-optimal) ───────────────────────

def build_cooccurrence_weights(gold_paths):
    """
    gold_paths: list of relation sequences, one per training query, e.g.
                [["directed_by", "starred_actors"], ["written_by", "has_genre"], ...]
    Returns dict {(r_i, r_j): weight} for consecutive relation pairs (Eq. 1).
    Pairs are stored unordered (min, max) since the cut is undirected.
    """
    weights = Counter()
    for path in gold_paths:
        # normalize reverse-traversal relations (rel + "_inv") back to their
        # base relation, since the partition assigns BASE relations to silos.
        # The co-occurrence graph, the cut, and CSPL must all operate on the
        # same vocabulary or the cut optimizes a different graph than the one
        # CSPL measures (this mismatch made Pi_cut look worst on M3).
        norm = [r[:-4] if r.endswith("_inv") else r for r in path]
        for a, b in zip(norm, norm[1:]):
            if a == b:
                continue
            key = tuple(sorted((a, b)))
            weights[key] += 1
    return weights


def partition_cut(relations, K, cooccurrence_weights, balance_slack=1.25):
    """
    Balanced K-way min-cut over the relation co-occurrence graph, via a
    greedy weighted-clustering heuristic (no external graph-partitioning
    dependency needed):

      1. Sort relations by total incident co-occurrence weight, descending.
      2. Place each relation into the silo that maximizes the sum of edge
         weights to relations already placed there, subject to a soft
         balance cap (no silo may exceed balance_slack * |R|/K relations
         before the cap is relaxed).

    This keeps frequently-chained relations together (minimizing cross-silo
    hops, Eq. 4) while still bounding imbalance, matching the qualitative
    behavior described for Π_cut in the paper.
    """
    relations = list(relations)
    incident = defaultdict(float)
    for (a, b), w in cooccurrence_weights.items():
        incident[a] += w
        incident[b] += w
    rels_sorted = sorted(relations, key=lambda r: -incident.get(r, 0))

    cap = max(1, int(balance_slack * len(relations) / K))
    silo_members = [[] for _ in range(K)]
    assignment = {}

    def affinity(r, silo_id):
        return sum(
            cooccurrence_weights.get(tuple(sorted((r, other))), 0)
            for other in silo_members[silo_id]
        )

    for r in rels_sorted:
        candidates = [k for k in range(K) if len(silo_members[k]) < cap] or list(range(K))
        best_silo = max(candidates, key=lambda k: (affinity(r, k), -len(silo_members[k])))
        assignment[r] = best_silo
        silo_members[best_silo].append(r)

    return assignment


# ── Gold-path extraction (needed to build the co-occurrence graph) ──────────

def extract_gold_relation_paths(qa_samples, triple_index, max_hops=3):
    """
    Recovers a relation chain for each training question via BFS over the
    global (pre-partition) KG, from the topic entity to each gold answer
    entity. This gives Π_cut a query workload to read (Section 3, "only
    Π_cut consults the query workload ... through aggregate co-occurrence
    counts over training paths") without requiring a dataset that ships
    hop-by-hop relation annotations.

    qa_samples:   list of (question, topic_entity, [answer_entities])
    triple_index: dict entity -> list of (relation, neighbor_entity),
                  built over BOTH directions (h->t and t->h, using rel and
                  rel+"_inv" respectively) so BFS can traverse either way.
    Returns: list of relation-sequences (one per resolved question).
    """
    paths = []
    for _, topic, answers in qa_samples:
        for ans in answers:
            path = _bfs_relation_path(topic, ans, triple_index, max_hops)
            if path:
                paths.append(path)
    return paths


def _bfs_relation_path(src, dst, triple_index, max_hops):
    if src == dst:
        return []
    frontier = [(src, [])]
    visited = {src}
    for _ in range(max_hops):
        next_frontier = []
        for node, path in frontier:
            for rel, nbr in triple_index.get(node, []):
                if nbr == dst:
                    return path + [rel]
                if nbr not in visited:
                    visited.add(nbr)
                    next_frontier.append((nbr, path + [rel]))
        frontier = next_frontier
        if not frontier:
            break
    return None


def build_bidirectional_triple_index(triples):
    """triples: list of (h, r, t) -> dict entity -> list of (relation, neighbor)."""
    index = defaultdict(list)
    for h, r, t in triples:
        index[h].append((r, t))
        index[t].append((r + "_inv", h))
    return index


# ── Registry ──────────────────────────────────────────────────────────────────

STRATEGY_NAMES = {"dom": "Semantic-domain", "freq": "Frequency-balanced",
                   "cut": "Co-occurrence graph-cut", "rand": "Random"}
