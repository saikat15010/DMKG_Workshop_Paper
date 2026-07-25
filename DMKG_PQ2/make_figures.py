#!/usr/bin/env python3
"""
make_figures.py — results figures for the DMKG paper.

Reads:
    results/m3_heldout.csv              M3 (held-out test paths) and M4
    results/metaqa_cost_table.csv       quality control
    results/pathquestion_cost_table.csv quality control

Writes (PNG at 300 dpi):
    figures/fig_cost_curves.png     M3 and M4 against K, both datasets
    figures/fig_tradeoff.png        locality vs balance at K=3
    figures/fig_quality.png         answer-quality control

Crossing rates come from m3_heldout.csv so that the figures match Table 2:
Pi_cut's co-occurrence graph is fitted on training paths and M3 is measured
on held-out test paths. Pi_rand is averaged over twenty seeds for the
structural metrics and three for quality.

Usage:
    python make_figures.py
    python make_figures.py --results-dir results --out-dir figures
"""

import os
import csv
import argparse
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.6,
    "lines.markersize": 5.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

STRATEGIES = ["dom", "freq", "cut", "rand"]
LABEL = {"dom": r"$\Pi_{\mathrm{dom}}$", "freq": r"$\Pi_{\mathrm{freq}}$",
         "cut": r"$\Pi_{\mathrm{cut}}$", "rand": r"$\Pi_{\mathrm{rand}}$"}
COLOR = {"dom": "#B45309", "freq": "#1D4ED8", "cut": "#047857", "rand": "#6B7280"}
MARKER = {"dom": "o", "freq": "s", "cut": "^", "rand": "D"}
LINESTYLE = {"dom": "-", "freq": "-", "cut": "-", "rand": "--"}
KS = [3, 5, 7]


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def struct(rows, dataset, metric, strategy, K):
    """Held-out structural metric: returns (value, sd) from m3_heldout.csv."""
    for r in rows:
        if (r["dataset"] == dataset and r["strategy"] == strategy
                and int(r["K"]) == K):
            sd_key = {"cspl_test": "cspl_test_sd",
                      "imbalance": "imbalance_sd"}.get(metric)
            sd = float(r[sd_key]) if sd_key and r.get(sd_key) else 0.0
            return float(r[metric]), sd
    return None, None


def quality(rows, strategy, K):
    """Mean and sd of test Hits@3 from a cost table."""
    vals = [float(r["quality_hits3"]) for r in rows
            if r["strategy"] == strategy and int(r["K"]) == K]
    if not vals:
        return None, None
    return st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)


def panel(ax, hel, dataset, metric, ylabel, title):
    for s in STRATEGIES:
        ms, sds = [], []
        for K in KS:
            m, sd = struct(hel, dataset, metric, s, K)
            ms.append(m); sds.append(sd)
        ax.errorbar(KS, ms, yerr=sds, label=LABEL[s], color=COLOR[s],
                    marker=MARKER[s], linestyle=LINESTYLE[s],
                    capsize=2.5, elinewidth=0.8, markeredgewidth=0,
                    alpha=0.95, zorder=3)
    ax.set_xticks(KS)
    ax.set_xlabel("number of silos $K$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=6)
    ax.grid(True, alpha=0.28, linestyle=":", zorder=0)
    ax.set_axisbelow(True)


# ── Figure: cost curves ──────────────────────────────────────────────────────

def fig_cost_curves(hel, out):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    panel(axes[0, 0], hel, "metaqa", "cspl_test",
          "cross-silo crossing rate", "(a) MetaQA: locality (M3)")
    panel(axes[0, 1], hel, "metaqa", "imbalance",
          "imbalance (CV)", "(b) MetaQA: balance (M4)")
    panel(axes[1, 0], hel, "pathquestion", "cspl_test",
          "cross-silo crossing rate", "(c) PathQuestion: locality (M3)")
    panel(axes[1, 1], hel, "pathquestion", "imbalance",
          "imbalance (CV)", "(d) PathQuestion: balance (M4)")

    handles = [Line2D([0], [0], color=COLOR[s], marker=MARKER[s],
                      linestyle=LINESTYLE[s], label=LABEL[s], markeredgewidth=0)
               for s in STRATEGIES]
    fig.legend(handles=handles, loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


# ── Figure: trade-off at K = 3 ───────────────────────────────────────────────

def fig_tradeoff(hel, out, K=3):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

    for ax, ds, name in [(axes[0], "metaqa", "MetaQA"),
                         (axes[1], "pathquestion", "PathQuestion")]:
        pts = {}
        for s in STRATEGIES:
            x, _ = struct(hel, ds, "imbalance", s, K)
            y, _ = struct(hel, ds, "cspl_test", s, K)
            pts[s] = (x, y)

        xs = [p[0] for p in pts.values()]
        ys = [p[1] for p in pts.values()]
        xr = max(xs) - min(xs)
        yr = max(ys) - min(ys)

        ypad = 0.10 * yr
        ylo = min(ys) - 0.22 * yr
        ax.set_xlim(-0.06 * xr, max(xs) + 0.26 * xr)
        ax.set_ylim(-ypad if ylo < 0 else ylo, max(ys) + 0.22 * yr)

        for s, (x, y) in pts.items():
            ax.scatter([x], [y], s=150, color=COLOR[s], marker=MARKER[s],
                       edgecolors="white", linewidths=1.1, zorder=4)
            ax.annotate(LABEL[s], xy=(x, y),
                        xytext=(x + 0.045 * xr, y + 0.055 * yr),
                        fontsize=9, color=COLOR[s], fontweight="bold",
                        zorder=5)

        ax.axvline(min(xs), color="#9CA3AF", linestyle="--",
                   linewidth=0.7, alpha=0.8, zorder=1)
        ax.axhline(min(ys), color="#9CA3AF", linestyle="--",
                   linewidth=0.7, alpha=0.8, zorder=1)
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        ax.annotate("best balance", xy=(min(xs), y1),
                    xytext=(min(xs) + 0.02 * xr, y1 - 0.05 * (y1 - y0)),
                    fontsize=7, color="#6B7280", style="italic", zorder=5)
        ax.annotate("best locality", xy=(x0, min(ys)),
                    xytext=(x0 + 0.03 * (x1 - x0), min(ys) + 0.05 * (y1 - y0)),
                    fontsize=7, color="#6B7280", style="italic", zorder=5)

        ax.set_xlabel("imbalance (M4)  $\\longrightarrow$ worse balance")
        ax.set_ylabel("crossing rate (M3)\n$\\longrightarrow$ worse locality")
        ax.set_title(f"{name}  ($K={K}$)", pad=6)
        ax.grid(True, alpha=0.25, linestyle=":", zorder=0)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}  (K={K})")


# ── Figure: quality control ──────────────────────────────────────────────────

def fig_quality(mq, pq, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    width = 0.2

    for ax, rows, name in [(axes[0], mq, "MetaQA"),
                           (axes[1], pq, "PathQuestion")]:
        for i, s in enumerate(STRATEGIES):
            ms, sds = [], []
            for K in KS:
                m, sd = quality(rows, s, K)
                ms.append(m); sds.append(sd)
            xs = [k + (i - 1.5) * width for k in range(len(KS))]
            ax.bar(xs, ms, width=width, yerr=sds, label=LABEL[s],
                   color=COLOR[s], alpha=0.88, capsize=2, zorder=3,
                   error_kw={"elinewidth": 0.8})
        ax.set_xticks(range(len(KS)))
        ax.set_xticklabels([f"$K$={k}" for k in KS])
        ax.set_ylim(0.55, 0.92)
        ax.set_title(name, pad=6)
        ax.grid(True, axis="y", alpha=0.28, linestyle=":", zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("answer quality (Hits@3)")

    handles = [Line2D([0], [0], color=COLOR[s], marker="s", linestyle="",
                      label=LABEL[s], markeredgewidth=0) for s in STRATEGIES]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    hel = load(os.path.join(args.results_dir, "m3_heldout.csv"))
    mq = load(os.path.join(args.results_dir, "metaqa_cost_table.csv"))
    pq = load(os.path.join(args.results_dir, "pathquestion_cost_table.csv"))

    fig_cost_curves(hel, os.path.join(args.out_dir, "fig_cost_curves.png"))
    fig_tradeoff(hel, os.path.join(args.out_dir, "fig_tradeoff.png"))
    fig_quality(mq, pq, os.path.join(args.out_dir, "fig_quality.png"))


if __name__ == "__main__":
    main()