#!/usr/bin/env python3
"""
Extract compute-only cycle data from simulation logs.

Parses MESH_PIPELINE START/END pairs → data/compute_only.csv

The MESH_PIPELINE counter measures from first mesh.io.req.fire to
!matmul_in_progress — this is the pure compute pipeline time, excluding
DMA, instruction decode, and RoCC overhead.

Usage:
    python3 scripts/extract_exbusy_debug.py
"""

import re
import csv
from pathlib import Path

BASEDIR = Path(__file__).resolve().parent.parent
LOGDIR_DEBUG = BASEDIR / "logs" / "exbusy_debug"
LOGDIR_ORIG = BASEDIR / "logs" / "exbusy"
DATADIR = BASEDIR / "data"

PAT_SWEEP = re.compile(
    r"EXBUSY_SWEEP\s+DIM=(\d+)\s+M=(\d+)\s+Q=(\d+)\s+cycles=(\d+)"
)
PAT_MESH_PIPELINE = re.compile(
    r"MESH_PIPELINE\s+cycle=\s*(\d+)\s+(START|END)\s*(?:\(first req\.fire\))?\s*(?:counter=\s*(\d+))?"
)


def parse_sweep(path):
    """Parse EXBUSY_SWEEP lines to get (DIM, M, Q) index."""
    entries = []
    try:
        text = path.read_bytes().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return entries
    for m in PAT_SWEEP.finditer(text):
        entries.append((int(m.group(1)), int(m.group(2)),
                        int(m.group(3)), int(m.group(4))))
    return entries


def parse_pipeline_pairs(path):
    """Parse MESH_PIPELINE START/END pairs, return list of cycle counts."""
    pairs = []
    try:
        text = path.read_bytes().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return pairs
    start_cycle = None
    for m in PAT_MESH_PIPELINE.finditer(text):
        if m.group(2) == "START":
            start_cycle = int(m.group(1))
        elif m.group(2) == "END" and start_cycle is not None:
            counter_val = m.group(3)
            if counter_val:
                pairs.append(int(counter_val))
            else:
                pairs.append(int(m.group(1)) - start_cycle)
            start_cycle = None
    return pairs


def main():
    DATADIR.mkdir(exist_ok=True)

    logdir = LOGDIR_DEBUG if LOGDIR_DEBUG.exists() else LOGDIR_ORIG
    print("Using logs from: %s" % logdir)

    rows = []

    for config in ("baseline", "twist"):
        logf = logdir / ("%s.log" % config)

        entries = parse_sweep(logf)
        pairs = parse_pipeline_pairs(logf)
        print("  %s: %d sweep entries, %d pipeline pairs" % (
            config, len(entries), len(pairs)))

        if len(pairs) == len(entries):
            for (dim, m, q, _), pcyc in zip(entries, pairs):
                rows.append({
                    "config": config, "DIM": dim, "M": m, "Q": q,
                    "compute_only": pcyc,
                })
        elif pairs:
            print("  WARNING: %d pipeline pairs vs %d sweep entries — skipping"
                  % (len(pairs), len(entries)))

    if rows:
        out = DATADIR / "compute_only.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "config", "DIM", "M", "Q", "compute_only"])
            w.writeheader()
            w.writerows(rows)
        print("  Wrote %d rows → %s" % (len(rows), out))

    print("\nDone. Data in: %s" % DATADIR)


if __name__ == "__main__":
    main()
