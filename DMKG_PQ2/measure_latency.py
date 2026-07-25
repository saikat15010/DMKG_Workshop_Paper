#!/usr/bin/env python3
# measure_latency.py — clean M5 re-measurement, inference only.
#
# Rebuilds each (dataset, strategy, K) partition and its candidate index
# exactly as run_study.py does, then times the inference path only:
#   candidate lookup -> question encoding -> topic anchoring -> scoring.
#
# No training, so this runs in minutes. Entity embeddings are randomly
# initialized: latency depends on tensor shapes and candidate-set sizes,
# not on embedding values, so timing is unaffected by skipping training.
#
# Run on an IDLE GPU:
#   CUDA_VISIBLE_DEVICES=0 python measure_latency.py
#
# Writes results/latency_table.csv

import os, csv, time, statistics
import torch

import config
from data_prep import (load_global_kb, relation_vocabulary, triple_counts_by_relation,
                       build_shared_entity_index, split_by_partition,
                       build_neighbor_index, two_hop_candidates)
from partition import (partition_domain, partition_frequency, partition_cut,
                       partition_random, build_cooccurrence_weights,
                       extract_gold_relation_paths, build_bidirectional_triple_index)
from kge_and_server import FedServer
from qa_dataset import QADataset, make_collate_fn
from torch.utils.data import DataLoader

WARMUP = 20
REPEATS = 200          # more repeats than the main run, for tighter estimates


def get_assignment(strategy, relations, K, tc, dom, cw, seed):
    if strategy == "dom":  return partition_domain(relations, K, domain_labels=dom)
    if strategy == "freq": return partition_frequency(relations, K, triple_counts=tc)
    if strategy == "cut":  return partition_cut(relations, K, cooccurrence_weights=cw)
    if strategy == "rand": return partition_random(relations, K, seed=seed)
    raise ValueError(strategy)


def measure(dataset_name, strategy, K, seed, device):
    paths = config.DATASET_PATHS[dataset_name]
    triples = load_global_kb(paths["kb"])
    relations = relation_vocabulary(triples)
    tc = triple_counts_by_relation(triples)
    e2id = build_shared_entity_index(triples)
    dom = config.RELATION_DOMAINS.get(dataset_name, {})

    id2e = {i: e for e, i in e2id.items()}
    train_qa = QADataset(paths["qa_train"], e2id).samples
    bi = build_bidirectional_triple_index(triples)
    gold = extract_gold_relation_paths(
        [(q, id2e[t], [id2e[a] for a in ai]) for q, t, ai in train_qa[:2000]], bi)
    cw = build_cooccurrence_weights(gold)

    assign = get_assignment(strategy, relations, K, tc, dom, cw, seed)
    silo_triples = split_by_partition(triples, assign, K)

    pooled = build_neighbor_index([t for k in range(K) for t in silo_triples[k]],
                                  e2id, max_neighbors=config.MAX_NEIGHBORS)
    dev_ds = QADataset(paths["qa_dev"], e2id)
    topic_ids = sorted({t for _, t, _ in dev_ds.samples})
    cand_map = {eid: two_hop_candidates(eid, pooled,
                                        config.CANDIDATE_HOP1_CAP,
                                        config.CANDIDATE_HOP2_CAP)
                for eid in topic_ids}

    server = FedServer(embed_dim=config.KGE_EMBED_DIM, K=K).to(device).eval()
    N = len(e2id)
    with torch.no_grad():
        h_joint = torch.nn.functional.normalize(
            torch.randn(N, K * config.KGE_EMBED_DIM, device=device), dim=-1)

    loader = DataLoader(dev_ds, batch_size=1, shuffle=False,
                        collate_fn=make_collate_fn(cand_map))
    batches = [b for _, b in zip(range(WARMUP + REPEATS), loader)]

    def one(b):
        q, tid, _, cand = b
        tid = tid.to(device); cand = cand.to(device)
        with torch.no_grad():
            qe = server.question_encoder(q, device)
            qf = qe + h_joint[tid]
            server.score_candidates(qf, h_joint, cand)

    for b in batches[:WARMUP]:
        one(b)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for b in batches[WARMUP:]:
        t0 = time.perf_counter()
        one(b)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    s = sorted(times)
    return {
        "dataset": dataset_name, "strategy": strategy, "K": K, "seed": seed,
        "latency_mean_ms": round(statistics.mean(s), 4),
        "latency_p95_ms": round(s[min(len(s) - 1, int(0.95 * len(s)))], 4),
        "latency_median_ms": round(statistics.median(s), 4),
        "n_timed": len(s),
    }


def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    rows = []
    for d in config.DATASETS:
        for K in config.SILO_COUNTS:
            for s in config.STRATEGIES:
                seeds = config.RAND_SEEDS if s == "rand" else [0]
                for sd in seeds:
                    r = measure(d, s, K, sd, device)
                    print(r, flush=True)
                    rows.append(r)
    out = os.path.join(config.RESULTS_DIR, "latency_table.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
