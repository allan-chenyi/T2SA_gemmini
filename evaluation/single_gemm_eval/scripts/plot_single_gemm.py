#!/usr/bin/env python3
"""
Plot Single-GEMM latency results for the Pipette paper.

Generates:
  1. E2E speedup vs M/D with Q curves       (main paper figure)
  2. Compute-only speedup vs Q               (D-1 theory validation)
  3. Absolute cycle comparison                (supplementary)
  4. Combined compute-only + e2e comparison   (shows DMA dilution)

Usage:
    python3 scripts/plot_single_gemm.py
"""

import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASEDIR = Path(__file__).resolve().parent.parent
DATADIR = BASEDIR / "data"
FIGDIR = BASEDIR / "figures"

# ── helpers ──────────────────────────────────────────────────────────


def load_e2e():
    """Return dict keyed by (config, M, Q) -> cycles."""
    data = {}
    p = DATADIR / "e2e.csv"
    if not p.exists():
        return data
    with open(p) as f:
        for row in csv.DictReader(f):
            data[(row["config"], int(row["M"]), int(row["Q"]))] = int(row["cycles"])
    return data


def load_compute_ind():
    """Return dict keyed by (config, Q) -> {cycles, per_tile}."""
    data = {}
    p = DATADIR / "compute_ind.csv"
    if not p.exists():
        return data
    with open(p) as f:
        for row in csv.DictReader(f):
            data[(row["config"], int(row["Q"]))] = {
                "cycles": int(row["cycles"]),
                "per_tile": int(row["per_tile"]),
            }
    return data


def load_compute_mr():
    """Return dict keyed by (config, M_eff, Q) -> cycles."""
    data = {}
    p = DATADIR / "compute_mr.csv"
    if not p.exists():
        return data
    with open(p) as f:
        for row in csv.DictReader(f):
            data[(row["config"], int(row["M_eff"]), int(row["Q"]))] = int(
                row["cycles"]
            )
    return data


def theoretical_speedup(M, Q, D=32):
    """Theoretical speedup % for double-buffered WS."""
    C = M + (Q - 1) * max(M, D) + 3 * D - 2
    return 100.0 * (D - 1) / C


# ── plot functions ───────────────────────────────────────────────────


def plot_e2e_speedup(e2e, D=32):
    """Main paper figure: 2-panel E2E speedup."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    M_vals = sorted({m for (c, m, q) in e2e if c == "baseline"})
    Q_vals = sorted({q for (c, m, q) in e2e if c == "baseline"})
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(Q_vals)))

    # (a) Measured speedup, all Q curves
    for i, Q in enumerate(Q_vals):
        md, sp = [], []
        for M in M_vals:
            bk, tk = ("baseline", M, Q), ("twist", M, Q)
            if bk in e2e and tk in e2e and e2e[bk] > 0:
                md.append(M / D)
                sp.append(100.0 * (e2e[bk] - e2e[tk]) / e2e[bk])
        if md:
            ax1.plot(md, sp, "o-", color=colors[i], label=f"Q={Q}", ms=4, lw=1.5)

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("M / D")
    ax1.set_ylabel("Speedup (%)")
    ax1.set_title(f"(a) Measured E2E Speedup (D={D})")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="gray", lw=0.5)

    # (b) Theory vs measured for selected Q values
    for Q_sel, color, marker in [(1, "tab:blue", "o"), (4, "tab:orange", "s"),
                                  (16, "tab:green", "^"), (64, "tab:red", "D")]:
        # theory
        md_t = np.logspace(np.log2(1 / D), np.log2(32), 200, base=2)
        sp_t = [theoretical_speedup(m * D, Q_sel, D) for m in md_t]
        ax2.plot(md_t, sp_t, "--", color=color, lw=1, alpha=0.5)
        # measured
        md_m, sp_m = [], []
        for M in M_vals:
            bk, tk = ("baseline", M, Q_sel), ("twist", M, Q_sel)
            if bk in e2e and tk in e2e and e2e[bk] > 0:
                md_m.append(M / D)
                sp_m.append(100.0 * (e2e[bk] - e2e[tk]) / e2e[bk])
        if md_m:
            ax2.plot(md_m, sp_m, marker + "-", color=color,
                     label=f"Q={Q_sel}", ms=5, lw=1.5)

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("M / D")
    ax2.set_ylabel("Speedup (%)")
    ax2.set_title(f"(b) Theory (dashed) vs Measured (D={D})")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"single_gemm_e2e_speedup.{ext}",
                    dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    print(f"  Saved single_gemm_e2e_speedup.pdf/png")
    plt.close(fig)


def plot_compute_speedup(ind, D=32):
    """Compute-only speedup vs Q — validates D-1 theory."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    Q_vals = sorted({q for (c, q) in ind if c == "baseline"})

    # measured
    Q_m, sp_m = [], []
    for Q in Q_vals:
        b, t = ind.get(("baseline", Q)), ind.get(("twist", Q))
        if b and t and b["cycles"] > 0:
            Q_m.append(Q)
            sp_m.append(100.0 * (b["cycles"] - t["cycles"]) / b["cycles"])

    # theory (M=D, independent tiles, double-buffered)
    Qq = np.linspace(1, 70, 200)
    sp_t = [theoretical_speedup(D, q, D) for q in Qq]

    ax.plot(Qq, sp_t, "--", color="gray", lw=1.5, label="Theoretical")
    ax.plot(Q_m, sp_m, "s-", color="tab:red", ms=6, lw=2, label="Measured (compute-only)")

    ax.set_xlabel("Tile Count Q")
    ax.set_ylabel("Speedup (%)")
    ax.set_title(f"Compute-Only Speedup vs Tile Count (D={D})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"single_gemm_compute_speedup.{ext}",
                    dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    print(f"  Saved single_gemm_compute_speedup.pdf/png")
    plt.close(fig)


def plot_absolute_cycles(e2e, D=32):
    """Absolute cycle counts for selected Q values."""
    fig, ax = plt.subplots(figsize=(8, 5))

    M_vals = sorted({m for (c, m, q) in e2e if c == "baseline"})

    for Q, ls_b, ls_t in [(1, "o--", "o-"), (4, "s--", "s-"),
                           (16, "^--", "^-"), (64, "D--", "D-")]:
        md_b, cyc_b, md_t, cyc_t = [], [], [], []
        for M in M_vals:
            bk, tk = ("baseline", M, Q), ("twist", M, Q)
            if bk in e2e:
                md_b.append(M / D); cyc_b.append(e2e[bk])
            if tk in e2e:
                md_t.append(M / D); cyc_t.append(e2e[tk])
        if md_b:
            ax.plot(md_b, cyc_b, ls_b, label=f"Baseline Q={Q}", ms=3, alpha=0.6)
            ax.plot(md_t, cyc_t, ls_t, label=f"Pipette Q={Q}", ms=3, alpha=0.8)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("M / D")
    ax.set_ylabel("Cycles")
    ax.set_title(f"Absolute Cycle Counts (D={D})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"single_gemm_absolute_cycles.{ext}",
                    dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    print(f"  Saved single_gemm_absolute_cycles.pdf/png")
    plt.close(fig)


def plot_compute_vs_e2e(ind, e2e, D=32):
    """Compare compute-only vs e2e speedup at M=DIM to show DMA dilution."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    Q_vals = sorted({q for (c, q) in ind if c == "baseline"})

    # compute-only
    Q_c, sp_c = [], []
    for Q in Q_vals:
        b, t = ind.get(("baseline", Q)), ind.get(("twist", Q))
        if b and t and b["cycles"] > 0:
            Q_c.append(Q)
            sp_c.append(100.0 * (b["cycles"] - t["cycles"]) / b["cycles"])

    # e2e at M=DIM
    Q_e, sp_e = [], []
    for Q in Q_vals:
        bk, tk = ("baseline", D, Q), ("twist", D, Q)
        if bk in e2e and tk in e2e and e2e[bk] > 0:
            Q_e.append(Q)
            sp_e.append(100.0 * (e2e[bk] - e2e[tk]) / e2e[bk])

    # theory
    Qq = np.linspace(1, 70, 200)
    sp_t = [theoretical_speedup(D, q, D) for q in Qq]

    ax.plot(Qq, sp_t, "--", color="gray", lw=1.5, label="Theoretical")
    ax.plot(Q_c, sp_c, "s-", color="tab:red", ms=6, lw=2, label="Compute-only")
    ax.plot(Q_e, sp_e, "o-", color="tab:blue", ms=6, lw=2, label="E2E (M=DIM)")

    ax.set_xlabel("Tile Count Q")
    ax.set_ylabel("Speedup (%)")
    ax.set_title(f"Compute-only vs E2E Speedup (M=D={D})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"compute_vs_e2e.{ext}",
                    dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    print(f"  Saved compute_vs_e2e.pdf/png")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────


def main():
    FIGDIR.mkdir(exist_ok=True)

    print("Loading data...")
    e2e = load_e2e()
    ind = load_compute_ind()
    mr = load_compute_mr()
    print(f"  E2E: {len(e2e)} entries, Compute IND: {len(ind)}, Compute MR: {len(mr)}")

    print("Plotting...")
    if e2e:
        plot_e2e_speedup(e2e)
        plot_absolute_cycles(e2e)
    if ind:
        plot_compute_speedup(ind)
    if ind and e2e:
        plot_compute_vs_e2e(ind, e2e)

    print("Done. Figures in:", FIGDIR)


if __name__ == "__main__":
    main()
