#!/usr/bin/env python3
"""
Extract Single-GEMM Compute Sweep data from 10-run simulation logs.

Parses COMPUTE_SWEEP lines from logs/sweep/baseline_*.log and twist_*.log.
Averages cycles across runs for each (config, M, Q).

Output: data/sweep.csv  (config, DIM, M, Q, cycles, std, n_runs)

Usage:
    python3 scripts/extract_sweep.py
"""

import re
import csv
from collections import defaultdict
from pathlib import Path
import math

BASEDIR = Path(__file__).resolve().parent.parent
LOGDIR = BASEDIR / "logs" / "sweep"
DATADIR = BASEDIR / "data"

PAT = re.compile(
    r"COMPUTE_SWEEP\s+DIM=(\d+)\s+M=(\d+)\s+Q=(\d+)\s+cycles=(\d+)"
)


def parse_log(path):
    entries = []
    try:
        text = path.read_bytes().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return entries
    for m in PAT.finditer(text):
        entries.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))))
    return entries


def main():
    DATADIR.mkdir(exist_ok=True)

    # collect: (config, DIM, M, Q) -> [cycles_run1, cycles_run2, ...]
    all_data = defaultdict(list)

    for config in ("baseline", "twist"):
        logs = sorted(LOGDIR.glob("%s_*.log" % config))
        print("  %s: found %d log files" % (config, len(logs)))
        for logf in logs:
            entries = parse_log(logf)
            print("    %s: %d entries" % (logf.name, len(entries)))
            for dim, m, q, cyc in entries:
                all_data[(config, dim, m, q)].append(cyc)

    if not all_data:
        print("No data found! Check logs in %s" % LOGDIR)
        return

    # average and write
    out = DATADIR / "sweep.csv"
    rows = []
    for (config, dim, m, q), cyc_list in sorted(all_data.items()):
        avg = sum(cyc_list) / len(cyc_list)
        if len(cyc_list) > 1:
            variance = sum((c - avg) ** 2 for c in cyc_list) / (len(cyc_list) - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0
        rows.append({
            "config": config,
            "DIM": dim,
            "M": m,
            "Q": q,
            "cycles": int(round(avg)),
            "std": "%.1f" % std,
            "n_runs": len(cyc_list),
        })

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "DIM", "M", "Q", "cycles", "std", "n_runs"])
        w.writeheader()
        w.writerows(rows)
    print("\n  Wrote %d rows to %s (averaged from %d runs)" % (
        len(rows), out, max(len(v) for v in all_data.values())))


if __name__ == "__main__":
    main()
