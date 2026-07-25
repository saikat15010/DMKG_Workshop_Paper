#!/usr/bin/env python3
# run_study.py
#
# Unified DMKG cost-characterization runner for BOTH datasets (MetaQA and
# PathQuestion), using ONE codepath so the iso-quality control is provably
# identical across datasets. Built on the proven FedV-KGQA training recipe:
#
#   * KGE trained to convergence per silo (100 epochs), as in train_kge.py
#   * QA loop identical to train_fedv.py: grad clipping on MLP + all silo
#     embeddings, unit-sphere renorm each step, best-checkpoint by dev metric
#   * POOLED bidirectional candidate index (100% answer coverage)
#   * quality control: Hits@3 >= 0.70 for BOTH datasets (same convention)
#   * emits the same full metric set (MRR/Hits@1/3/5/10) for both
#
# Run from the DMKG project root:
#   CUDA_VISIBLE_DEVICES=1 python run_study.py                      # both datasets, full grid
#   CUDA_VISIBLE_DEVICES=1 python run_study.py --dataset metaqa      # one dataset
#   CUDA_VISIBLE_DEVICES=1 python run_study.py --dataset pathquestion --strategies cut --silo-counts 3

import os, sys, csv, argparse, copy, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from data_prep import (load_global_kb, relation_vocabulary, triple_counts_by_relation,
                        build_shared_entity_index, split_by_partition,
                        build_neighbor_index, two_hop_candidates)
from partition import (partition_domain, partition_frequency, partition_cut,
                        partition_random, build_cooccurrence_weights,
                        extract_gold_relation_paths, build_bidirectional_triple_index)
from kge_and_server import train_one_silo_kge, FedServer
from qa_dataset import QADataset, make_collate_fn
from metrics import (communication_to_target, index_size, cross_silo_path_length,
                      imbalance, summarize_latency, build_cost_row)

# ── Control metric — SAME for both datasets ──────────────────────────────────
QUALITY_METRIC = "hits@3"
QUALITY_TARGET = 0.70


# ── Evaluation: full MRR + Hits@1/3/5/10 (matches MetaQA / ISWC evaluate.py) ──

def evaluate_full(server, silo_models, loader, device):
    server.eval()
    for m in silo_models: m.eval()
    hits1 = hits3 = hits5 = hits10 = 0
    mrr = 0.0
    total = 0
    with torch.no_grad():
        h_joint = server.fuse([m.get_entity_embeddings().to(device) for m in silo_models])
        for questions, topic_ids, answer_ids_batch, cand_ids in loader:
            topic_ids = topic_ids.to(device)
            cand_ids = cand_ids.to(device)
            q_embed = server.question_encoder(questions, device)
            q_final = q_embed + h_joint[topic_ids] if config.USE_TOPIC_ANCHORING else q_embed
            sim = server.score_candidates(q_final, h_joint, cand_ids)
            for i, answer_ids in enumerate(answer_ids_batch):
                cands = cand_ids[i]
                valid = cands >= 0
                cand_list = cands[valid].tolist()
                scores = sim[i][valid].tolist()
                ranked = sorted(zip(cand_list, scores), key=lambda x: x[1], reverse=True)
                answer_set = set(answer_ids)
                rank = None
                for pos, (eid, _) in enumerate(ranked):
                    if eid in answer_set:
                        rank = pos + 1
                        break
                total += 1
                if rank is not None:
                    mrr += 1.0 / rank
                    hits1 += int(rank == 1)
                    hits3 += int(rank <= 3)
                    hits5 += int(rank <= 5)
                    hits10 += int(rank <= 10)
    n = max(total, 1)
    return {"mrr": mrr/n, "hits@1": hits1/n, "hits@3": hits3/n,
            "hits@5": hits5/n, "hits@10": hits10/n}


def get_assignment(strategy, relations, K, *, triple_counts, domain_labels,
                    cooc_weights, seed):
    if strategy == "dom":
        return partition_domain(relations, K, domain_labels=domain_labels)
    if strategy == "freq":
        return partition_frequency(relations, K, triple_counts=triple_counts)
    if strategy == "cut":
        return partition_cut(relations, K, cooccurrence_weights=cooc_weights)
    if strategy == "rand":
        return partition_random(relations, K, seed=seed)
    raise ValueError(strategy)


def run_cell(dataset_name, strategy, K, seed=0, device=None, kge_epochs=None, verbose=True):
    device = device or torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    kge_epochs = kge_epochs or config.KGE_EPOCHS
    log = (lambda s: print(s, flush=True)) if verbose else (lambda s: None)

    paths = config.DATASET_PATHS[dataset_name]
    triples = load_global_kb(paths["kb"])
    relations = relation_vocabulary(triples)
    triple_counts = triple_counts_by_relation(triples)
    shared_entity2id = build_shared_entity_index(triples)
    domain_labels = config.RELATION_DOMAINS.get(dataset_name, {})

    id2e = {i: e for e, i in shared_entity2id.items()}
    train_qa = QADataset(paths["qa_train"], shared_entity2id).samples
    bidir = build_bidirectional_triple_index(triples)
    gold_paths = extract_gold_relation_paths(
        [(q, id2e[tid], [id2e[a] for a in aids]) for q, tid, aids in train_qa],
        bidir)
    cooc = build_cooccurrence_weights(gold_paths)

    assignment = get_assignment(strategy, relations, K, triple_counts=triple_counts,
                                domain_labels=domain_labels, cooc_weights=cooc, seed=seed)
    silo_triples = split_by_partition(triples, assignment, K)
    silo_sizes = [len(s) for s in silo_triples]
    log(f"[{dataset_name}/{strategy}/K={K}] silos={silo_sizes}")

    m4_imbalance = imbalance(silo_sizes)
    m3_cspl = cross_silo_path_length(gold_paths, assignment)

    # ── KGE per silo (proven recipe: full epochs, unit-norm) ─────────────────
    silo_models = []
    for k in range(K):
        log(f"  KGE silo {k} ({silo_sizes[k]} triples)")
        m, _ = train_one_silo_kge(silo_triples[k], shared_entity2id, device,
                                   epochs=kge_epochs, log=log)
        silo_models.append(m)

    # ── Candidate index: POOLED for coverage, per-silo for storage (M2/M4) ──
    pooled = build_neighbor_index([t for k in range(K) for t in silo_triples[k]],
                                  shared_entity2id, max_neighbors=config.MAX_NEIGHBORS)
    per_silo = [build_neighbor_index(silo_triples[k], shared_entity2id,
                                     max_neighbors=config.MAX_NEIGHBORS) for k in range(K)]
    topic_ids = sorted({tid for _, tid, _ in train_qa} |
                       {tid for _, tid, _ in QADataset(paths["qa_dev"], shared_entity2id).samples} |
                       {tid for _, tid, _ in QADataset(paths["qa_test"], shared_entity2id).samples})
    candidate_map = {}
    per_silo_contrib = [0]*K
    for eid in topic_ids:
        candidate_map[eid] = two_hop_candidates(eid, pooled,
                                                config.CANDIDATE_HOP1_CAP, config.CANDIDATE_HOP2_CAP)
        for k in range(K):
            per_silo_contrib[k] += len(two_hop_candidates(eid, per_silo[k],
                                       config.CANDIDATE_HOP1_CAP, config.CANDIDATE_HOP2_CAP))
    idx_mean, idx_total = index_size(candidate_map)
    m4_index_imbalance = imbalance(per_silo_contrib)

    # ── QA fine-tune: FedV-KGQA recipe ────────────────────────────────────────
    server = FedServer(embed_dim=config.KGE_EMBED_DIM, K=K).to(device)
    server_opt = torch.optim.Adam(server.question_encoder.mlp.parameters(), lr=config.QA_LR)
    silo_opts = [torch.optim.Adam(m.ent_embed.parameters(), lr=config.QA_LR) for m in silo_models]

    collate = make_collate_fn(candidate_map)
    train_loader = DataLoader(QADataset(paths["qa_train"], shared_entity2id),
                              batch_size=config.QA_BATCH_SIZE, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(QADataset(paths["qa_dev"], shared_entity2id),
                            batch_size=config.QA_BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(QADataset(paths["qa_test"], shared_entity2id),
                             batch_size=config.QA_BATCH_SIZE, shuffle=False, collate_fn=collate)

    rounds = 0
    rounds_at_target = None
    best_metric = 0.0
    best_state = None
    reached = False

    for epoch in range(1, config.QA_EPOCHS + 1):
        server.train()
        for m in silo_models: m.train()
        for questions, tids, ans, cand in train_loader:
            tids = tids.to(device)
            server_opt.zero_grad()
            for o in silo_opts: o.zero_grad()
            embeds = [m.ent_embed.weight for m in silo_models]
            loss, _ = server(questions, tids, embeds, ans, cand, device, margin=config.QA_MARGIN)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(server.question_encoder.mlp.parameters(), 1.0)
            for m in silo_models:
                torch.nn.utils.clip_grad_norm_(m.ent_embed.parameters(), 1.0)
            server_opt.step()
            for o in silo_opts: o.step()
            with torch.no_grad():
                for m in silo_models:
                    m.ent_embed.weight.data = F.normalize(m.ent_embed.weight.data, p=2, dim=-1)
            rounds += 1

        dev = evaluate_full(server, silo_models, dev_loader, device)
        q = dev[QUALITY_METRIC]
        log(f"  epoch {epoch} dev {QUALITY_METRIC}={q:.3f} (target {QUALITY_TARGET})")
        if q > best_metric:
            best_metric = q
            best_state = (copy.deepcopy(server.state_dict()),
                          [copy.deepcopy(m.state_dict()) for m in silo_models])
        if q >= QUALITY_TARGET and rounds_at_target is None:
            rounds_at_target = rounds
            reached = True

    if best_state is not None:
        server.load_state_dict(best_state[0])
        for m, st in zip(silo_models, best_state[1]):
            m.load_state_dict(st)
    if not reached:
        rounds_at_target = rounds
        log(f"  [warn] target not reached; best dev {QUALITY_METRIC}={best_metric:.3f}")

    comm_bytes = communication_to_target(rounds_at_target, K, len(shared_entity2id),
                                         config.KGE_EMBED_DIM)

    # ── M5 latency (best model) ──────────────────────────────────────────────
    server.eval()
    for m in silo_models: m.eval()
    with torch.no_grad():
        h_joint = server.fuse([m.get_entity_embeddings().to(device) for m in silo_models])
    qb = next(iter(dev_loader))
    questions, tids, ans, cand = qb
    tids = tids.to(device); cand = cand.to(device)
    timings = []
    for _ in range(5):  # warmup
        with torch.no_grad():
            qf = server.question_encoder(questions[:1], device) + h_joint[tids[:1]]
            server.score_candidates(qf, h_joint, cand[:1])
    if torch.cuda.is_available(): torch.cuda.synchronize()
    for _ in range(50):
        t0 = time.perf_counter()
        with torch.no_grad():
            qf = server.question_encoder(questions[:1], device) + h_joint[tids[:1]]
            server.score_candidates(qf, h_joint, cand[:1])
        if torch.cuda.is_available(): torch.cuda.synchronize()
        timings.append((time.perf_counter()-t0)*1000)
    latency = summarize_latency(timings)

    # final quality = best dev; also report test metrics for the record
    test = evaluate_full(server, silo_models, test_loader, device)

    row = build_cost_row(strategy, K, comm_bytes, idx_mean, idx_total, m3_cspl,
                         m4_imbalance, latency)
    row.update({"dataset": dataset_name, "seed": seed,
                "index_imbalance": m4_index_imbalance,
                "quality_mrr": round(test["mrr"], 4),
                "quality_hits1": round(test["hits@1"], 4),
                "quality_hits3": round(test["hits@3"], 4),
                "quality_hits5": round(test["hits@5"], 4),
                "quality_hits10": round(test["hits@10"], 4),
                "dev_best": round(best_metric, 4)})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=config.DATASETS, default=None,
                     help="run a single dataset; default runs both")
    ap.add_argument("--strategies", nargs="+", default=config.STRATEGIES)
    ap.add_argument("--silo-counts", nargs="+", type=int, default=config.SILO_COUNTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=config.RAND_SEEDS)
    ap.add_argument("--kge-epochs", type=int, default=None)
    args = ap.parse_args()

    datasets = [args.dataset] if args.dataset else config.DATASETS
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    fields = ["dataset","strategy","K","seed","comm_GB","index_mean_entries",
              "index_total_entries","cspl","imbalance","index_imbalance",
              "latency_mean_ms","latency_p95_ms","quality_mrr","quality_hits1",
              "quality_hits3","quality_hits5","quality_hits10","dev_best"]

    for dataset_name in datasets:
        rows = []
        for strategy in args.strategies:
            for K in args.silo_counts:
                seeds = args.seeds if strategy == "rand" else [0]
                for seed in seeds:
                    rows.append(run_cell(dataset_name, strategy, K, seed=seed,
                                         device=device, kge_epochs=args.kge_epochs))
        out = os.path.join(config.RESULTS_DIR, f"{dataset_name}_cost_table.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fields})
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
