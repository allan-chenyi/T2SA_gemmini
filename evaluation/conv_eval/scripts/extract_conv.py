#!/usr/bin/env python3
"""
Extract E2E cycle data from all conv/spad benchmark logs.

Every benchmark (`conv_perf`, `conv_dw_perf`, `lstm_cell_perf`, and every
`*_spad` variant) emits exactly one unified line near the end of its run:

    BENCH_E2E name=<test> general=<G> cold=<C> warm=<W>

where:
  general — full CPU wall time of the benchmark (from very start of main to
            very end, including setup, data prep, all compute, teardown).
  cold    — cycles from first gemmini command to last gemmini command's
            completion, INCLUDING any DRAM→scratchpad weight preload.
  warm    — same as cold but EXCLUDING any preload phase (compute only).
            For conv-family tests with no explicit preload, warm == cold.

This script parses ONLY that line — no RTL GEMMINI_TRACE post-processing.
Output CSV columns are identical for all workloads:

    config, test, general_cycles, cold_cycles, warm_cycles
    + workload-parameter columns (IN_DIM, HIDDEN, ...) when present.

Usage:
    python3 scripts/extract_conv.py
"""

import os
import re
import csv
from pathlib import Path

BASEDIR = Path(__file__).resolve().parent.parent
# Log directory and output CSV are overridable so the same script can serve
# multiple independent runs (e.g. 16x16 default vs 32x32 sweep) without
# clobbering each other's data.
LOGDIR  = Path(os.environ.get("CONV_LOGDIR", BASEDIR / "logs"))
DATADIR = BASEDIR / "data"
CSV_OUT = Path(os.environ.get("CONV_CSV",    DATADIR / "conv_results.csv"))

# The one line every test must emit.
PAT_BENCH = re.compile(
    r"BENCH_E2E\s+name=(\S+)\s+general=(\d+)\s+cold=(\d+)\s+warm=(\d+)"
)

PAT_PARAMS = {
    "BATCH_SIZE":  re.compile(r"BATCH_SIZE = (\d+)"),
    "IN_DIM":      re.compile(r"IN_DIM = (\d+)"),
    "IN_CHANNELS": re.compile(r"IN_CHANNELS = (\d+)"),
    "OUT_CHANNELS":re.compile(r"OUT_CHANNELS = (\d+)"),
    "CHANNELS":    re.compile(r"CHANNELS = (\d+)"),
    "KERNEL_DIM":  re.compile(r"KERNEL_DIM = (\d+)"),
    "PADDING":     re.compile(r"PADDING = (\d+)"),
    "STRIDE":      re.compile(r"STRIDE = (\d+)"),
    "HIDDEN":      re.compile(r"HIDDEN = (\d+)"),
    "INPUT_SIZE":  re.compile(r"INPUT_SIZE = (\d+)"),
    "BATCH":       re.compile(r"BATCH = (\d+)"),
    # DIM appears in every *_SPAD printf banner (e.g. "LSTM_LM_SPAD DIM=16 ...")
    # — used by the LaTeX table generator to populate the DIM column.
    "DIM":         re.compile(r"_SPAD\s+DIM=(\d+)"),
}


def discover_variants():
    """Scan LOGDIR for baseline_<v>.log with a matching twist_<v>.log."""
    variants = []
    if not LOGDIR.exists():
        return variants
    for bl in sorted(LOGDIR.glob("baseline_*.log")):
        variant = bl.name[len("baseline_"):-len(".log")]
        if variant.startswith("build_") or variant.startswith("mt_"):
            continue
        tw = LOGDIR / ("twist_%s.log" % variant)
        if tw.exists():
            variants.append(variant)
    return variants


def parse_log(path):
    """Return dict with general_cycles, cold_cycles, warm_cycles, and any
    matched workload parameters. Returns None if log is missing."""
    if not path.exists():
        return None

    text = path.read_bytes().decode("utf-8", errors="replace")

    result = {
        "bench_name": None,
        "general_cycles": None,
        "cold_cycles": None,
        "warm_cycles": None,
    }

    m = PAT_BENCH.search(text)
    if m:
        result["bench_name"] = m.group(1)
        result["general_cycles"] = int(m.group(2))
        result["cold_cycles"]    = int(m.group(3))
        result["warm_cycles"]    = int(m.group(4))

    for name, pat in PAT_PARAMS.items():
        pm = pat.search(text)
        if pm:
            result[name] = int(pm.group(1))

    return result


def main():
    DATADIR.mkdir(exist_ok=True)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    print("  LOGDIR = %s" % LOGDIR)
    print("  CSV    = %s" % CSV_OUT)

    variants = discover_variants()
    print("Discovered variants: %s" % (", ".join(variants) if variants
                                        else "none"))

    rows = []
    for variant in variants:
        for config in ("baseline", "twist"):
            logf = LOGDIR / ("%s_%s.log" % (config, variant))
            res = parse_log(logf)
            if res is None:
                print("  %s/%s: log not found" % (config, variant))
                continue
            if res.get("general_cycles") is None:
                print("  %s/%s: no BENCH_E2E line in log — skipping" % (
                    config, variant))
                continue
            print("  %s/%s: general=%d cold=%d warm=%d" % (
                config, variant,
                res["general_cycles"], res["cold_cycles"], res["warm_cycles"],
            ))
            row = {"config": config, "test": variant}
            row.update(res)
            rows.append(row)

    if not rows:
        print("\nNo data extracted.")
        return

    fieldnames = [
        "config", "test", "bench_name",
        "general_cycles", "cold_cycles", "warm_cycles",
        "DIM",
        "BATCH_SIZE", "IN_DIM", "IN_CHANNELS", "OUT_CHANNELS", "CHANNELS",
        "KERNEL_DIM", "PADDING", "STRIDE",
        "HIDDEN", "INPUT_SIZE", "BATCH",
    ]
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("\n  Wrote %d rows -> %s" % (len(rows), CSV_OUT))
    print("Done.")


if __name__ == "__main__":
    main()
