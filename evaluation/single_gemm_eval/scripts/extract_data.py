#!/usr/bin/env python3
"""Extract Single-GEMM benchmark data from simulator logs into CSV.

Handles verbose trace corruption (C0: ... DASM(...) lines interspersed
in printf output) by using regex on the raw binary-safe text.

Usage:
    python3 scripts/extract_data.py
"""

import re
import csv
from pathlib import Path

BASEDIR = Path(__file__).resolve().parent.parent
LOGDIR = BASEDIR / "logs"
DATADIR = BASEDIR / "data"


def read_log(path):
    """Read log file, handling potential binary content from verbose trace."""
    with open(path, "rb") as f:
        return f.read().decode("ascii", errors="ignore")


def extract_e2e(logfile, label):
    """Extract GEMM_E2E lines from a log file."""
    text = read_log(logfile)
    rows = []
    for m in re.finditer(
        r"GEMM_E2E\s+DIM=(\d+)\s+M=(\d+)\s+K=(\d+)\s+N=(\d+)\s+Q=(\d+)\s+cycles=(\d+)",
        text,
    ):
        rows.append(
            {
                "config": label,
                "DIM": int(m.group(1)),
                "M": int(m.group(2)),
                "K": int(m.group(3)),
                "N": int(m.group(4)),
                "Q": int(m.group(5)),
                "cycles": int(m.group(6)),
            }
        )
    return rows


def extract_compute_ind(logfile, label):
    """Extract COMPUTE_IND lines."""
    text = read_log(logfile)
    rows = []
    for m in re.finditer(
        r"COMPUTE_IND\s+DIM=(\d+)\s+Q=(\d+)\s+cycles=(\d+)\s+per_tile=(\d+)",
        text,
    ):
        rows.append(
            {
                "config": label,
                "DIM": int(m.group(1)),
                "Q": int(m.group(2)),
                "cycles": int(m.group(3)),
                "per_tile": int(m.group(4)),
            }
        )
    return rows


def extract_compute_mr(logfile, label):
    """Extract COMPUTE_MR lines."""
    text = read_log(logfile)
    rows = []
    for m in re.finditer(
        r"COMPUTE_MR\s+DIM=(\d+)\s+M_eff=(\d+)\s+Q=(\d+)\s+M_mult=(\d+)\s+cycles=(\d+)",
        text,
    ):
        rows.append(
            {
                "config": label,
                "DIM": int(m.group(1)),
                "M_eff": int(m.group(2)),
                "Q": int(m.group(3)),
                "M_mult": int(m.group(4)),
                "cycles": int(m.group(5)),
            }
        )
    return rows


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {path} ({len(rows)} rows)")


def main():
    DATADIR.mkdir(exist_ok=True)

    # --- E2E ---
    e2e_rows = []
    for label in ["baseline", "twist"]:
        lf = LOGDIR / f"{label}_single_gemm_e2e.log"
        if lf.exists():
            batch = extract_e2e(lf, label)
            e2e_rows.extend(batch)
            print(f"  {label} e2e: {len(batch)} entries")
        else:
            print(f"  WARNING: {lf} not found")
    if e2e_rows:
        write_csv(
            DATADIR / "e2e.csv",
            ["config", "DIM", "M", "K", "N", "Q", "cycles"],
            e2e_rows,
        )

    # --- Compute-only independent ---
    ind_rows = []
    for label in ["baseline", "twist"]:
        lf = LOGDIR / f"{label}_single_gemm_compute.log"
        if lf.exists():
            batch = extract_compute_ind(lf, label)
            ind_rows.extend(batch)
            print(f"  {label} compute_ind: {len(batch)} entries")
    if ind_rows:
        write_csv(
            DATADIR / "compute_ind.csv",
            ["config", "DIM", "Q", "cycles", "per_tile"],
            ind_rows,
        )

    # --- Compute-only multi-row ---
    mr_rows = []
    for label in ["baseline", "twist"]:
        lf = LOGDIR / f"{label}_single_gemm_compute.log"
        if lf.exists():
            batch = extract_compute_mr(lf, label)
            mr_rows.extend(batch)
            print(f"  {label} compute_mr: {len(batch)} entries")
    if mr_rows:
        write_csv(
            DATADIR / "compute_mr.csv",
            ["config", "DIM", "M_eff", "Q", "M_mult", "cycles"],
            mr_rows,
        )

    print("\nSummary:")
    print(f"  E2E:          {len(e2e_rows)} rows")
    print(f"  Compute IND:  {len(ind_rows)} rows")
    print(f"  Compute MR:   {len(mr_rows)} rows")


if __name__ == "__main__":
    main()
