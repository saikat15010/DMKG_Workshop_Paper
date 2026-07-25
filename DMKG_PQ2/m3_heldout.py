#!/usr/bin/env python3
"""
m3_heldout.py — recompute M3 on HELD-OUT test paths.

The main run built Pi_cut's co-occurrence graph from training paths and then
measured M3 on those same paths. This script separates them:
    co-occurrence graph  <- TRAIN paths (as before)
    M3 measurement       <- TEST paths  (held out)

It also reports:
  * crossing rate over all recovered test paths
  * crossing rate conditioned on paths with >= 1 relation transition
  * Pi_rand over 20 seeds instead of 3 (structural metrics need no training)

CPU only. Run from a project folder containing data/:
    python m3_heldout.py
Writes results/m3_heldout.csv
"""

import os, csv, statistics as st
import config
from data_prep import (load_global_kb, relation_vocabulary,
                       triple_counts_by_relation, build_shared_entity_index,
                       split_by_partition)
from partition import (partition_domain, partition_frequency, partition_cut,
                       partition_random, build_cooccurrence_weights,
                       extract_gold_relation_paths,
                       build_bidirectional_triple_index)
from metrics import imbalance
from qa_dataset import QADataset

RAND_SEEDS = list(range(1, 21))     # 20 seeds for structural metrics


def crossing_rate(paths, assign, multi_only=False):
    sel = [p for p in paths if len(p) >= 2] if multi_only else paths
    if not sel:
        return 0.0, 0
    tot = 0
    for p in sel:
        for a, b in zip(p, p[1:]):
            sa = assign.get(a.replace("_inv", ""))
            sb = assign.get(b.replace("_inv", ""))
            if sa is not None and sb is not None and sa != sb:
                tot += 1
    return tot / len(sel), len(sel)


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    rows = []
    for d in config.DATASETS:
        p = config.DATASET_PATHS[d]
        triples = load_global_kb(p["kb"])
        e2id = build_shared_entity_index(triples)
        id2e = {i: e for e, i in e2id.items()}
        rels = relation_vocabulary(triples)
        tc = triple_counts_by_relation(triples)
        dom = config.RELATION_DOMAINS.get(d, {})
        bi = build_bidirectional_triple_index(triples)

        def paths_for(split, cap=None):
            qa = QADataset(p[split], e2id).samples
            if cap:
                qa = qa[:cap]
            return extract_gold_relation_paths(
                [(q, id2e[t], [id2e[a] for a in ai]) for q, t, ai in qa], bi)

        train_paths = paths_for("qa_train", cap=2000)   # same as main run
        test_paths = paths_for("qa_test")               # held out
        cw = build_cooccurrence_weights(train_paths)    # fit on TRAIN only

        n_multi = sum(1 for x in test_paths if len(x) >= 2)
        print(f"\n{d}: {len(test_paths)} test paths, "
              f"{n_multi} multi-relation "
              f"({100*n_multi/max(len(test_paths),1):.1f}%)")

        for K in config.SILO_COUNTS:
            for s in ["dom", "freq", "cut", "rand"]:
                seeds = RAND_SEEDS if s == "rand" else [0]
                cr_all, cr_multi, imb = [], [], []
                for sd in seeds:
                    a = (partition_domain(rels, K, domain_labels=dom) if s == "dom"
                         else partition_frequency(rels, K, triple_counts=tc) if s == "freq"
                         else partition_cut(rels, K, cooccurrence_weights=cw) if s == "cut"
                         else partition_random(rels, K, seed=sd))
                    v1, _ = crossing_rate(test_paths, a)
                    v2, _ = crossing_rate(test_paths, a, multi_only=True)
                    cr_all.append(v1); cr_multi.append(v2)
                    imb.append(imbalance([len(x) for x in split_by_partition(triples, a, K)]))
                r = dict(dataset=d, K=K, strategy=s, n_seeds=len(seeds),
                         cspl_test=round(st.mean(cr_all), 4),
                         cspl_test_sd=round(st.pstdev(cr_all), 4) if len(cr_all) > 1 else 0.0,
                         cspl_test_multi=round(st.mean(cr_multi), 4),
                         imbalance=round(st.mean(imb), 4),
                         imbalance_sd=round(st.pstdev(imb), 4) if len(imb) > 1 else 0.0)
                rows.append(r)
                print(f"  K={K} {s:5s} cspl_test={r['cspl_test']:.4f} "
                      f"multi={r['cspl_test_multi']:.4f} imb={r['imbalance']:.4f}")

    out = os.path.join(config.RESULTS_DIR, "m3_heldout.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
