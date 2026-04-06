#!/usr/bin/env python3
"""
Generate the Baseline vs T^3 comparison table in the format used by the paper.

Table layout (one row per workload):
    Workload | DIM | Shape (GEMM dims / counts) | Warm % | Cold % | General %

Only the three improvement percentages are reported — raw cycle counts are
intentionally *not* emitted, so the table cannot be used to back out per-op
cycle budgets.

Output:
    figures/conv_table.{pdf,png}
    figures/conv_table.tex
    stdout summary
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASEDIR = Path(__file__).resolve().parent.parent
DATADIR = BASEDIR / "data"
FIGDIR  = BASEDIR / "figures"

# Overridable paths — lets the same script serve 16x16 and 32x32 runs.
CSV_IN     = Path(os.environ.get("CONV_CSV",        DATADIR / "conv_results.csv"))
OUT_PREFIX = os.environ.get("CONV_OUT_PREFIX", "conv_table")


# ----------------------------------------------------------------------
# Per-workload metadata (BibTeX key from figures/current.tex + shape
# template). Shape strings use {H} = 2·DIM, {D} = DIM so that the same
# entry describes the workload regardless of which DIM row it ends up on.
# ----------------------------------------------------------------------
WORKLOAD_SPEC = {
    "lstm_lm_spad": {
        "pretty": "LSTM-LM",
        "cite":   "sak_long_2014",
        "shape_fn": lambda D: "H=128, proj=%d, T=32, 160 spad GEMMs (K=%d)" % (D, 128//D),
    },
    "dlrm_mlp_spad": {
        "pretty": "DLRM MLP",
        "cite":   "naumov_deep_2019",
        "shape_fn": lambda D: "512\\textrightarrow 256\\textrightarrow 64\\textrightarrow %d, 3 DRAM GEMMs" % D,
    },
    "kws_lstm_spad": {
        "pretty": "KWS LSTM",
        "cite":   "arik_convolutional_2017",
        "shape_fn": lambda D: (
            "L=2, H=32, proj=16, T=16, 160 spad GEMMs (2-tile)" if D <= 16
            else "L=2, H=128, proj=64, T=16, 160 spad GEMMs (K=%d)" % (128//D)
        ),
    },
    "bilstm_spad": {
        "pretty": "BiLSTM-CRF NER",
        "cite":   "lample_neural_2016",
        "shape_fn": lambda D: "H=96, proj=%d, T=16, 160 spad (K=%d) + 1 DRAM" % (D, 96//D),
    },
}

# Row order in the rendered table.
ORDER = [
    "lstm_lm_spad",
    "kws_lstm_spad",
    "bilstm_spad",
    "dlrm_mlp_spad",
]


def load_data():
    data = {}
    p = CSV_IN
    if not p.exists():
        return data
    with open(p) as f:
        for row in csv.DictReader(f):
            data[(row["config"], row["test"])] = row
    return data


def _i(row, key):
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        return int(v)
    except ValueError:
        return None


def pct(b, t):
    """Improvement of T^3 over Baseline, as a percentage."""
    return 100.0 * (b / t - 1.0) if (t and b is not None) else 0.0


def shape_for(test, dim):
    spec = WORKLOAD_SPEC.get(test)
    if spec is None or dim is None:
        return ""
    return spec["shape_fn"](dim)


def build_rows(data):
    tests = [t for t in ORDER
             if ("baseline", t) in data and ("twist", t) in data]
    rows = []
    for test in tests:
        br, tr = data[("baseline", test)], data[("twist", test)]

        dim = _i(br, "DIM") or _i(tr, "DIM")

        g_b, g_t = _i(br, "general_cycles"), _i(tr, "general_cycles")
        c_b, c_t = _i(br, "cold_cycles"),    _i(tr, "cold_cycles")
        w_b, w_t = _i(br, "warm_cycles"),    _i(tr, "warm_cycles")

        if g_b is None or g_t is None:
            continue

        rows.append({
            "test":  test,
            "dim":   dim,
            "shape": shape_for(test, dim),
            "w_p":   pct(w_b, w_t) if (w_b is not None and w_t is not None) else None,
            "c_p":   pct(c_b, c_t) if (c_b is not None and c_t is not None) else None,
            "g_p":   pct(g_b, g_t),
        })
    return rows


# ----------------------------------------------------------------------
# LaTeX output — matches figures/current.tex format
# ----------------------------------------------------------------------
def write_latex(rows):
    if not rows:
        return

    def pnum(v):
        return "--" if v is None else r"%.2f\%%" % v

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Speedup of \pipette{} over Baseline on Gemmini across three "
        r"timing modes. Only relative improvements are reported; absolute "
        r"cycle counts are intentionally withheld. Shapes are parameterised "
        r"in the array dimension~$D$.}",
        r"\label{tab:benchmark_results}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{@{} l c l rrr @{}}",
        r"\toprule",
        r"\textbf{Workload} & \textbf{$D$} & \textbf{Shape}"
        r" & \shortstack{Warm$^{a}$\\(\%)}"
        r" & \shortstack{Cold$^{b}$\\(\%)}"
        r" & \shortstack{General$^{c}$\\(\%)} \\",
        r"\midrule",
    ]
    for r in rows:
        spec = WORKLOAD_SPEC.get(r["test"], {})
        label = spec.get("pretty", r["test"].replace("_", r"\_"))
        cite  = spec.get("cite")
        if cite:
            label = "%s~\\cite{%s}" % (label, cite)
        dim_str = "--" if r["dim"] is None else str(r["dim"])
        lines.append(
            r"%s & %s & %s & %s & %s & %s \\" % (
                label, dim_str, r["shape"],
                pnum(r["w_p"]), pnum(r["c_p"]), pnum(r["g_p"]),
            )
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"$^{a}$~\textbf{Warm}: compute-only cycles, measured after weight "
        r"preload completes; isolates pure hardware execution latency.\quad \\",
        r"$^{b}$~\textbf{Cold}: includes initial weight transfer "
        r"DRAM\,$\to$\,scratchpad (\texttt{mvin}); reflects first-inference "
        r"latency. Load cycles are identical for Baseline and \pipette{}.\quad \\",
        r"$^{c}$~\textbf{General}: full program duration from "
        r"\texttt{gemmini\_flush} through all compute phases.",
        r"\end{minipage}",
        r"\end{table*}",
    ]

    (FIGDIR / ("%s.tex" % OUT_PREFIX)).write_text("\n".join(lines) + "\n")
    print("  Saved %s.tex" % OUT_PREFIX)


# ----------------------------------------------------------------------
# PNG/PDF rendering — mirror of the LaTeX table, no raw cycles.
# ----------------------------------------------------------------------
def render_figure(rows):
    if not rows:
        print("  No data to render!")
        return

    col_labels = ["Workload", "D", "Shape", "Warm %", "Cold %", "General %"]

    def f(v):
        return "—" if v is None else "%.2f%%" % v

    cell_text = []
    for r in rows:
        cell_text.append([
            r["test"],
            "—" if r["dim"] is None else str(r["dim"]),
            r["shape"],
            f(r["w_p"]), f(r["c_p"]), f(r["g_p"]),
        ])

    fig_w = 14
    fig_h = 1.3 + 0.55 * len(rows)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.7)

    n_cols = len(col_labels)
    header_colors = ["#4472c4", "#4472c4", "#4472c4",
                     "#ed7d31", "#70ad47", "#5b9bd5"]
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor(header_colors[j])
        cell.set_text_props(color="white", fontweight="bold")

    for i in range(len(rows)):
        color = "#f2f2f2" if i % 2 == 0 else "white"
        for j in range(n_cols):
            table[i + 1, j].set_facecolor(color)

    ax.set_title(
        r"Gemmini Workloads: Baseline vs $T^3$ — speedup only" + "\n"
        "warm = post-preload compute   cold = incl. preload   "
        "general = full program",
        fontsize=11, fontweight="bold", pad=18,
    )

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / ("%s.%s" % (OUT_PREFIX, ext)),
                    dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
    print("  Saved %s.pdf/png" % OUT_PREFIX)
    plt.close(fig)


def print_summary(rows):
    print("\n  Results (improvement %% of T^3 over Baseline):")
    hdr = "  %-22s %4s  %-50s  %8s %8s %8s" % (
        "Workload", "D", "Shape", "warm%", "cold%", "gen%")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        def f(v):
            return "  —  " if v is None else ("%6.2f" % v)
        print("  %-22s %4s  %-50s  %8s %8s %8s" % (
            r["test"][:22],
            "—" if r["dim"] is None else str(r["dim"]),
            r["shape"][:50],
            f(r["w_p"]), f(r["c_p"]), f(r["g_p"]),
        ))
    print()


def main():
    FIGDIR.mkdir(exist_ok=True)
    print("  CSV_IN     = %s" % CSV_IN)
    print("  OUT_PREFIX = %s" % OUT_PREFIX)
    print("Loading data...")
    data = load_data()
    if not data:
        print("  No data found! Run simulations first.")
        return
    rows = build_rows(data)
    print("  %d workload entries" % len(rows))
    if not rows:
        print("  No valid rows!")
        return
    print("\nGenerating table...")
    render_figure(rows)
    write_latex(rows)
    print_summary(rows)
    print("Done. Output in:", FIGDIR)


if __name__ == "__main__":
    main()
