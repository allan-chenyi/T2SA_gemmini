#!/usr/bin/env python3
"""
Plot Single-GEMM Compute-Only Cycle Sweep (combined figure).

(a) Cycle count bar chart — Baseline vs T^3, vary Q at M/D=1
(b) T^3 speedup vs M/D — theory curves + measured scatter

Theory formula (from paper):
    baseline = M + (Q-1)*max(M,D) + 3D - 2
    twist    = M + (Q-1)*max(M,D) + 2D - 1
    saving   = D - 1 (constant, independent of M and Q)

Uses compute_only.csv (pure compute pipeline cycles) for measured data.

Usage:
    python3 scripts/plot_exbusy.py
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASEDIR = Path(__file__).resolve().parent.parent
DATADIR = BASEDIR / "data"
FIGDIR  = BASEDIR / "figures"


def load_data():
    """Load compute_only.csv."""
    data = {}
    D = 32

    p = DATADIR / "compute_only.csv"
    if p.exists():
        with open(p) as f:
            for row in csv.DictReader(f):
                D = int(row["DIM"])
                data[(row["config"], int(row["M"]), int(row["Q"]))] = int(row["compute_only"])
        print("  Loaded compute_only.csv")
        return data, D

    # Fallback: try mesh_pipeline.csv (old format)
    p = DATADIR / "mesh_pipeline.csv"
    if p.exists():
        with open(p) as f:
            for row in csv.DictReader(f):
                D = int(row["DIM"])
                data[(row["config"], int(row["M"]), int(row["Q"]))] = int(row["pipeline_cycles"])
        print("  Fallback: using mesh_pipeline.csv")
    return data, D


def theoretical_speedup(M, Q, D):
    """Paper formula: speedup = baseline/twist - 1."""
    baseline = M + (Q - 1) * max(M, D) + 3 * D - 2
    twist    = M + (Q - 1) * max(M, D) + 2 * D - 1
    return 100.0 * (baseline / twist - 1.0)


Q_STYLES = {
    1:  {"color": "#7b2d8e", "marker": "o"},
    2:  {"color": "#3a5fcd", "marker": "v"},
    4:  {"color": "#1874cd", "marker": "s"},
    8:  {"color": "#00868b", "marker": "D"},
    16: {"color": "#cd3333", "marker": "^"},
    32: {"color": "#cd6600", "marker": "p"},
    64: {"color": "#00bfff", "marker": "h"},
}

FS_LABEL = 14
FS_TICK = 12
FS_LEGEND = 11
FS_CAPTION = 14


def plot_combined(data, D):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    M_vals = sorted({m for (c, m, q) in data if c == "baseline"})
    Q_vals = sorted({q for (c, m, q) in data if c == "baseline"})

    # -- (a) Cycle bar chart: vary Q at M/D=1 --
    M_fix = D
    Q_show = [q for q in [1, 2, 4, 8, 16, 32, 64] if q in Q_vals]

    labels, base_cyc, twist_cyc = [], [], []
    for Q in Q_show:
        bk = ("baseline", M_fix, Q)
        tk = ("twist", M_fix, Q)
        if bk in data and tk in data:
            labels.append(str(Q))
            base_cyc.append(data[bk])
            twist_cyc.append(data[tk])

    x = np.arange(len(labels))
    w = 0.32

    ax1.bar(x - w / 2, base_cyc, w, color="#5b9bd5", edgecolor="black",
            linewidth=0.5, label="Baseline")
    ax1.bar(x + w / 2, twist_cyc, w, color="#ed7d31", edgecolor="black",
            linewidth=0.5, label=r"$T^3$")

    ax1.set_xlabel("Tile count  Q", fontsize=FS_LABEL)
    ax1.set_ylabel("Compute-Only Cycles", fontsize=FS_LABEL)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=FS_TICK)
    ax1.tick_params(axis="y", labelsize=FS_TICK)
    ax1.legend(fontsize=FS_LEGEND, loc="upper left")
    ax1.grid(True, alpha=0.3, axis="y")

    ax1.text(0.5, -0.20, "(a) Compute-Only Cycles at M/D=1",
             transform=ax1.transAxes, ha="center", fontsize=FS_CAPTION)

    # -- (b) Speedup vs M/D --
    md_t = np.logspace(-4, np.log2(32), 500, base=2)
    for Q in Q_vals:
        sty = Q_STYLES.get(Q, {"color": "gray"})
        sp_t = [theoretical_speedup(md * D, Q, D) for md in md_t]
        ax2.plot(md_t, sp_t, "-", color=sty["color"], lw=1.5, alpha=0.5,
                 label="Q=%d theory" % Q)

    for Q in Q_vals:
        sty = Q_STYLES.get(Q, {"color": "gray", "marker": "x"})
        md_m, sp_m = [], []
        for M in M_vals:
            bk = ("baseline", M, Q)
            tk = ("twist", M, Q)
            if bk in data and tk in data and data[tk] > 0:
                md_val = M / D
                if md_val >= 2**-4:
                    md_m.append(md_val)
                    sp_m.append(100.0 * (data[bk] / data[tk] - 1.0))
        if md_m:
            ax2.scatter(md_m, sp_m, marker=sty["marker"], color=sty["color"],
                        s=50, zorder=5, label="Q=%d measured" % Q,
                        edgecolors="black", linewidths=0.4)

    ax2.set_xscale("log", base=2)
    ax2.set_xlim(left=2**-4.5, right=2**5.5)
    ax2.set_xlabel("Normalized tile height  M / D", fontsize=FS_LABEL)
    ax2.set_ylabel("Speedup (%)", fontsize=FS_LABEL)
    ax2.tick_params(axis="both", labelsize=FS_TICK)
    ax2.legend(fontsize=FS_LEGEND - 2, ncol=2, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.axvline(1, color="gray", lw=0.5, ls=":")

    ax2.text(0.5, -0.20, r"(b) Gemmini Single-GEMM $T^3$ Compute-Only Speedup",
             transform=ax2.transAxes, ha="center", fontsize=FS_CAPTION)

    fig.tight_layout(h_pad=5.0)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / ("gemmini_exbusy_combined.%s" % ext),
                    dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    print("  Saved gemmini_exbusy_combined.pdf/png")
    plt.close(fig)


def print_summary(data, D):
    M_vals = sorted({m for (c, m, q) in data if c == "baseline"})
    Q_vals = sorted({q for (c, m, q) in data if c == "baseline"})
    print("\n  Compute-only cycle saving (baseline - twist):")
    header = "  %6s" % "M\\Q"
    for Q in Q_vals:
        header += " %6d" % Q
    print(header)
    for M in M_vals:
        row = "  %6d" % M
        for Q in Q_vals:
            bk = ("baseline", M, Q)
            tk = ("twist", M, Q)
            if bk in data and tk in data:
                row += " %6d" % (data[bk] - data[tk])
            else:
                row += "      -"
        print(row)


def main():
    FIGDIR.mkdir(exist_ok=True)
    print("Loading data...")
    data, D = load_data()
    n = sum(1 for (c, m, q) in data if c == "baseline")
    print("  %d entries per config  (DIM=%d)" % (n, D))
    if not data:
        print("  No data found! Run simulations first.")
        return
    print("\nPlotting...")
    plot_combined(data, D)
    print_summary(data, D)
    print("\nDone. Figures in:", FIGDIR)


if __name__ == "__main__":
    main()
