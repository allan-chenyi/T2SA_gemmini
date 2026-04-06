#!/usr/bin/env python3
"""
Extract Single-GEMM EX_BUSY data from simulation logs.

Parses EXBUSY_SWEEP lines from logs/exbusy/baseline.log and twist.log.
RTL simulation is deterministic — single run, no averaging needed.

Output: data/exbusy.csv  (config, DIM, M, Q, cycles)

Usage:
    python3 scripts/extract_exbusy.py
"""

import re
import csv
from pathlib import Path

BASEDIR = Path(__file__).resolve().parent.parent
LOGDIR = BASEDIR / "logs" / "exbusy"
DATADIR = BASEDIR / "data"

PAT = re.compile(
    r"EXBUSY_SWEEP\s+DIM=(\d+)\s+M=(\d+)\s+Q=(\d+)\s+cycles=(\d+)"
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

    rows = []
    for config in ("baseline", "twist"):
        logf = LOGDIR / ("%s.log" % config)
        entries = parse_log(logf)
        print("  %s: %d entries" % (config, len(entries)))
        for dim, m, q, cyc in entries:
            rows.append({
                "config": config,
                "DIM": dim,
                "M": m,
                "Q": q,
                "cycles": cyc,
            })

    if not rows:
        print("No data found! Check logs in %s" % LOGDIR)
        return

    out = DATADIR / "exbusy.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "DIM", "M", "Q", "cycles"])
        w.writeheader()
        w.writerows(rows)
    print("\n  Wrote %d rows to %s" % (len(rows), out))


if __name__ == "__main__":
    main()
