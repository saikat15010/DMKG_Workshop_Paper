# metrics.py — The five data-management cost metrics (Section 4)
#
# M1 communication cost per round / to target
# M2 candidate-set / index size
# M3 cross-silo path length (CSPL)
# M4 load balance (imbalance)
# M5 query latency

import time
import statistics

try:
    import torch  # only needed for cuda.synchronize in measure_latency
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ── M1: Communication cost ────────────────────────────────────────────────────

def communication_per_round(K, num_entities, embed_dim, bytes_per_float=4):
    """Eq. 2: C_round = K * |E| * d * 2 * 4 bytes (upload + gradient slice)."""
    return K * num_entities * embed_dim * 2 * bytes_per_float


def communication_to_target(rounds_to_target, K, num_entities, embed_dim):
    """C_total = T * C_round, where T = epochs needed to reach the quality target."""
    return rounds_to_target * communication_per_round(K, num_entities, embed_dim)


# ── M2: Candidate-set / index size ───────────────────────────────────────────

def index_size(candidate_map):
    """
    candidate_map: dict entity_id -> set/list/tensor of candidate ids
    Returns (mean entries per topic entity, total entries summed over all
    topic entities). Eq. 3 uses the mean; total is reported for the storage
    ("Index (entries)") column.
    """
    if not candidate_map:
        return 0.0, 0
    sizes = [len(v) for v in candidate_map.values()]
    return sum(sizes) / len(sizes), sum(sizes)


# ── M3: Cross-silo path length ───────────────────────────────────────────────

def cross_silo_path_length(gold_relation_paths, assignment):
    """
    Eq. 4: CSPL = mean over queries of #consecutive relation pairs whose
    silo assignment differs.
    gold_relation_paths: list of relation sequences (from partition.py's
                          extract_gold_relation_paths, run on train OR eval
                          queries as appropriate).
    assignment: dict relation -> silo_id
    """
    if not gold_relation_paths:
        return 0.0
    def base(r):
        return r[:-4] if r.endswith("_inv") else r
    total = 0
    for path in gold_relation_paths:
        crossings = 0
        for a, b in zip(path, path[1:]):
            sa = assignment.get(base(a))
            sb = assignment.get(base(b))
            if sa is not None and sb is not None and sa != sb:
                crossings += 1
        total += crossings
    return total / len(gold_relation_paths)


# ── M4: Load balance ──────────────────────────────────────────────────────────

def imbalance(per_silo_values):
    """
    Eq. 5: coefficient of variation, sigma/mu. Lower = more balanced.
    per_silo_values: list of length K (e.g. triple counts, or candidate-index
    contribution counts).
    """
    vals = [v for v in per_silo_values]
    if not vals or statistics.mean(vals) == 0:
        return 0.0
    mu = statistics.mean(vals)
    sigma = statistics.pstdev(vals)
    return sigma / mu


# ── M5: Query latency ────────────────────────────────────────────────────────

def measure_latency(score_fn, num_queries=None, warmup=5):
    """
    score_fn: zero-arg callable that runs one query's candidate lookup +
              forward pass + scoring (wrap your inference call in a lambda).
    Returns dict with mean_ms and p95_ms over `num_queries` calls.
    Caller is responsible for looping score_fn over the actual eval set;
    pass num_queries=None to just time a single externally-looped call and
    aggregate the returned list of raw timings instead (see run_cell.py).
    """
    timings = []
    for _ in range(warmup):
        score_fn()
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()
    n = num_queries or 1
    for _ in range(n):
        t0 = time.perf_counter()
        score_fn()
        if _HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000)
    return summarize_latency(timings)


def summarize_latency(timings_ms):
    if not timings_ms:
        return {"mean_ms": 0.0, "p95_ms": 0.0}
    s = sorted(timings_ms)
    p95_idx = min(len(s) - 1, int(0.95 * len(s)))
    return {"mean_ms": statistics.mean(s), "p95_ms": s[p95_idx]}


# ── Aggregation helper for the per-cell result row ───────────────────────────

def build_cost_row(strategy, K, comm_bytes, idx_mean, idx_total, cspl,
                    imbalance_score, latency_stats):
    return {
        "strategy": strategy,
        "K": K,
        "comm_GB": comm_bytes / (1024 ** 3),
        "index_mean_entries": idx_mean,
        "index_total_entries": idx_total,
        "cspl": cspl,
        "imbalance": imbalance_score,
        "latency_mean_ms": latency_stats["mean_ms"],
        "latency_p95_ms": latency_stats["p95_ms"],
    }
