# enrich_metaqa.py — OWL2-grounded MetaQA KB enrichment for the DMKG study.
#
# Ported from the ISWC FedV-KGQA `ontology_enrich_kb.py`, adapted to:
#   - read  data/metaqa/kb.txt            (pipe-delimited, unsplit)
#   - write data/metaqa/kb_enriched.txt   (pipe-delimited, + entailed triples)
#
# This is the SAME enrichment used in the accepted ISWC substrate. It adds:
#   Layer 1 (owl:inverseOf):        directed, wrote
#   Layer 2 (owl:propertyChainAxiom): associated_actor / _genre / _year / _language
# so that person entities gain head-role embeddings and multi-hop answers
# become reachable in the local candidate sets.
#
# PathQuestion does NOT need this script — its kb_enriched.txt already has
# the parents/children/spouse inverse triples materialized (same relation
# names, so its vocabulary stays at 13).

import os
from collections import defaultdict

import config

KB_IN  = config.DATASET_PATHS["metaqa"]["kb"].replace("kb_enriched.txt", "kb.txt")
KB_OUT = os.path.join(os.path.dirname(KB_IN), "kb_enriched.txt")

# ── T-Box ────────────────────────────────────────────────────────────────────
INVERSE_OF = {
    "directed_by": "directed",
    "written_by":  "wrote",
}
EXTRA_INVERSE = {  # internal only, used to build actor chains; not emitted
    "starred_actors": "starred_actors_inv",
}
PROPERTY_CHAINS = [
    ("directed", "starred_actors", "associated_actor"),
    ("wrote",    "starred_actors", "associated_actor"),
    ("directed", "has_genre",      "associated_genre"),
    ("wrote",    "has_genre",      "associated_genre"),
    ("directed", "release_year",   "associated_year"),
    ("wrote",    "release_year",   "associated_year"),
    ("directed", "in_language",    "associated_language"),
    ("wrote",    "in_language",    "associated_language"),
    ("starred_actors_inv", "starred_actors", "associated_actor"),
    ("starred_actors_inv", "has_genre",      "associated_genre"),
    ("starred_actors_inv", "release_year",   "associated_year"),
    ("starred_actors_inv", "in_language",    "associated_language"),
]


def load_kb(path):
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples


def materialise(triples):
    rel_index = defaultdict(set)
    for h, r, t in triples:
        rel_index[r].add((h, t))

    entailed = []
    seen = set(triples)

    def add(h, r, t):
        tr = (h, r, t)
        if tr not in seen:
            seen.add(tr)
            entailed.append(tr)
            rel_index[r].add((h, t))

    # Layer 1: owl:inverseOf (+ internal starred_actors_inv)
    for orig, inv in list(INVERSE_OF.items()) + list(EXTRA_INVERSE.items()):
        for (h, t) in list(rel_index[orig]):
            add(t, inv, h)

    # Layer 2: owl:propertyChainAxiom
    for (p1, p2, q) in PROPERTY_CHAINS:
        mid_to_heads = defaultdict(set)
        for (x, m) in rel_index[p1]:
            mid_to_heads[m].add(x)
        for (m, y) in rel_index[p2]:
            for x in mid_to_heads[m]:
                if x != y:
                    add(x, q, y)

    # drop internal-only relation from output
    internal = set(EXTRA_INVERSE.values())
    entailed = [(h, r, t) for h, r, t in entailed if r not in internal]
    return triples + entailed, len(entailed)


def main():
    if not os.path.exists(KB_IN):
        raise SystemExit(f"MetaQA KB not found: {KB_IN}")
    triples = load_kb(KB_IN)
    all_triples, n_entailed = materialise(triples)
    with open(KB_OUT, "w", encoding="utf-8") as f:
        for h, r, t in all_triples:
            f.write(f"{h}|{r}|{t}\n")
    rels = sorted({r for _, r, _ in all_triples})
    print(f"Original : {len(triples):,}")
    print(f"Entailed : {n_entailed:,}")
    print(f"Total    : {len(all_triples):,}")
    print(f"Relations: {len(rels)}  ->  {rels}")
    print(f"Written  : {KB_OUT}")


if __name__ == "__main__":
    main()
