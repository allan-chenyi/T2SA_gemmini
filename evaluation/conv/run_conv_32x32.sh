#!/bin/bash
# 32x32 conv workload runner.
#
# Runs conv_perf and conv_dw_perf on Baseline32x32WS and Twist32x32WSSingleOp
# simulators, 16 Verilator threads each, 4 tasks in parallel (64 cores).
#
# Prerequisites:
#   32x32 simulators built:
#     FORCE_CLEAN=1 bash README/evaluate/papers/conv_eval/build_32x32_mt.sh 16
#
# Usage:
#   bash run_conv_32x32.sh             # full flow
#   bash run_conv_32x32.sh --extract   # skip sims, just extract
set -eo pipefail

CHIPYARD=/data2/chenyi9/pipette/pipette-chipyard
EVALDIR="$CHIPYARD/README/evaluate/papers/conv"
cd "$CHIPYARD"
source env.sh

SYSROOT=/data2/chenyi9/pipette/chipyard/.conda-env/x86_64-conda-linux-gnu/sysroot
LDPATH="${SYSROOT}/lib64:${SYSROOT}/../lib:/data2/chenyi9/pipette/chipyard/.conda-env/riscv-tools/lib:/data2/chenyi9/pipette/chipyard/.conda-env/lib"
SIMDIR=sims/verilator
SWDIR=generators/gemmini/software/gemmini-rocc-tests
BINDIR=${SWDIR}/build/bareMetalC
CACHE_DIR=README/evaluate/params_cache
LOGDIR="$EVALDIR/logs"

DIM_VAL=32
B_SIM="${SIMDIR}/simulator-chipyard.harness-Baseline32x32WSRocketConfig"
T_SIM="${SIMDIR}/simulator-chipyard.harness-Twist32x32WSSingleOpRocketConfig"

WORKLOADS=(
    #"conv_perf"
    #"conv_perf_b4"
    #"resnet_bottleneck_spad"
    "lenet5_fc_spad"
    #"roshambo_l5_spad"
    "roshambo_full_spad"
    "mobilenet_v1_spad"
)

TASK_VTHREADS=16
CPU_BUDGET=${CPU_BUDGET:-64}
MAX_PARALLEL=$(( CPU_BUDGET / TASK_VTHREADS ))
if (( MAX_PARALLEL < 1 )); then MAX_PARALLEL=1; fi

# -----------------------------------------------------------------------
# Inline extraction function
# -----------------------------------------------------------------------
extract_results() {
    local LOGDIR="$1"
    local CSVOUT="$2"
    python3 - "$LOGDIR" "$CSVOUT" <<'PYEOF'
import re, csv, sys
from pathlib import Path

logdir = Path(sys.argv[1])
csvout = Path(sys.argv[2])

PAT_E2E = re.compile(
    r"BENCH_E2E\s+name=(\S+)\s+general=(\d+)\s+cold=(\d+)\s+warm=(\d+)"
)
PAT_GEM = re.compile(
    r"COLD_START\s+name=(\S+)\s+cycles=(\d+)"
)

rows = []
for logf in sorted(logdir.glob("*.log")):
    fname = logf.stem  # e.g. baseline_conv_perf
    parts = fname.split("_", 1)
    if len(parts) != 2:
        continue
    config, test = parts[0], parts[1]
    if config not in ("baseline", "twist"):
        continue

    text = logf.read_bytes().decode("utf-8", errors="replace")

    m_e2e = None
    for m in PAT_E2E.finditer(text):
        m_e2e = m
    m_gem = None
    for m in PAT_GEM.finditer(text):
        m_gem = m

    if m_e2e is None:
        print("  WARNING: no BENCH_E2E in %s" % logf.name)
        continue

    row = {
        "config": config,
        "test": test,
        "DIM": 32,
        "general_cycles": int(m_e2e.group(2)),
        "cold_cycles": int(m_e2e.group(3)),
        "warm_cycles": int(m_e2e.group(4)),
        "cold_start": int(m_gem.group(2)) if m_gem else "",
    }
    rows.append(row)

if rows:
    csvout.parent.mkdir(parents=True, exist_ok=True)
    fields = ["config", "test", "DIM", "general_cycles", "cold_cycles",
              "warm_cycles", "cold_start"]
    with open(csvout, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["test"], r["config"])))
    print("  Wrote %d rows -> %s" % (len(rows), csvout))

# Print summary
print("\n  %-16s %-10s %12s %12s %12s %12s" % (
    "test", "config", "general", "cold/warm", "cold_start", ""))
print("  " + "-" * 76)
tests = sorted(set(r["test"] for r in rows))
for test in tests:
    br = next((r for r in rows if r["test"] == test and r["config"] == "baseline"), None)
    tr = next((r for r in rows if r["test"] == test and r["config"] == "twist"), None)
    if br and tr:
        g_b, g_t = br["general_cycles"], tr["general_cycles"]
        c_b, c_t = br["cold_cycles"], tr["cold_cycles"]
        ge_b = br["cold_start"] if br["cold_start"] != "" else None
        ge_t = tr["cold_start"] if tr["cold_start"] != "" else None
        print("  %-16s %-10s %12d %12d %12s" % (
            test, "baseline", g_b, c_b,
            str(ge_b) if ge_b else "-"))
        print("  %-16s %-10s %12d %12d %12s" % (
            test, "twist", g_t, c_t,
            str(ge_t) if ge_t else "-"))
        g_pct = 100.0 * (g_b / g_t - 1.0) if g_t else 0
        c_pct = 100.0 * (c_b / c_t - 1.0) if c_t else 0
        ge_pct = 100.0 * (ge_b / ge_t - 1.0) if (ge_b and ge_t) else 0
        print("  %-16s %-10s %11.2f%% %11.2f%% %11.2f%%" % (
            "", "speedup", g_pct, c_pct, ge_pct))
        print()
PYEOF
}

# -----------------------------------------------------------------------
# Extract-only mode
# -----------------------------------------------------------------------
if [[ "$1" == "--extract" ]]; then
    echo "=== Extracting conv 32x32 results ==="
    extract_results "$LOGDIR" "$EVALDIR/data/conv_32x32.csv"
    exit 0
fi

# -----------------------------------------------------------------------
# 1. Verify simulators exist
# -----------------------------------------------------------------------
for SIM in "$B_SIM" "$T_SIM"; do
    if [[ ! -f "$SIM" ]]; then
        echo "ERROR: Simulator not found: $SIM"
        echo "Run: FORCE_CLEAN=1 bash README/evaluate/papers/conv_eval/build_32x32_mt.sh 16"
        exit 1
    fi
done

for SIM in "$B_SIM" "$T_SIM"; do
    INTERP=$(patchelf --print-interpreter "$SIM" 2>/dev/null || echo "")
    if [[ "$INTERP" != *"sysroot"* ]]; then
        patchelf --set-interpreter ${SYSROOT}/lib64/ld-linux-x86-64.so.2 \
          --set-rpath "${SYSROOT}/lib64:/data2/chenyi9/pipette/chipyard/.conda-env/riscv-tools/lib:/data2/chenyi9/pipette/chipyard/.conda-env/lib" \
          "$SIM" 2>/dev/null || true
    fi
done

# -----------------------------------------------------------------------
# 2. Build RISC-V binaries with DIM=32 params
# -----------------------------------------------------------------------
PARAMS_H="${SWDIR}/include/gemmini_params.h"
CACHED="${CACHE_DIR}/gemmini_params_dim${DIM_VAL}.h"

if [[ ! -f "$CACHED" ]]; then
    echo "ERROR: Params cache not found: $CACHED"
    exit 1
fi

cp "$PARAMS_H" "$PARAMS_H.bak"
cp "$CACHED" "$PARAMS_H"

ACTUAL_DIM=$(grep "^#define DIM " "$PARAMS_H" | awk '{print $3}')
if [ "$ACTUAL_DIM" != "$DIM_VAL" ]; then
    echo "ERROR: gemmini_params.h DIM=$ACTUAL_DIM, expected $DIM_VAL"
    cp "$PARAMS_H.bak" "$PARAMS_H"
    exit 1
fi

echo "=== Building RISC-V binaries (DIM=$ACTUAL_DIM) ==="
cd "${SWDIR}/build"
mkdir -p bareMetalC
for TEST in "${WORKLOADS[@]}"; do
    echo "  Building: ${TEST}-baremetal"
    rm -f "bareMetalC/${TEST}-baremetal"
    make -C bareMetalC -f "$(pwd)/../bareMetalC/Makefile" \
        abs_top_srcdir="$(pwd)/.." \
        src_dir="$(pwd)/../bareMetalC" \
        "${TEST}-baremetal"
done
cd "$CHIPYARD"

cp "$PARAMS_H.bak" "$PARAMS_H"
rm -f "$PARAMS_H.bak"
echo "  Restored gemmini_params.h"

for TEST in "${WORKLOADS[@]}"; do
    BIN="$BINDIR/${TEST}-baremetal"
    if [[ ! -f "$BIN" ]]; then
        echo "ERROR: Binary not found: $BIN"
        exit 1
    fi
done

# -----------------------------------------------------------------------
# 3. Launch 4 sims in parallel
# -----------------------------------------------------------------------
mkdir -p "$LOGDIR"

TOTAL_TASKS=$(( ${#WORKLOADS[@]} * 2 ))
echo ""
echo "============================================"
echo " 32x32 conv parallel run"
echo "   Workloads      : ${#WORKLOADS[@]}  (x 2 configs = $TOTAL_TASKS sims)"
echo "   Per-sim threads: $TASK_VTHREADS"
echo "   CPU budget     : $CPU_BUDGET"
echo "   Concurrency    : $MAX_PARALLEL"
echo "============================================"
echo ""

declare -A PIDS LOGS
LIVE_PIDS=()

TASK_KEYS=()
declare -A TASK_SIM TASK_LOG TASK_CMD
for TEST in "${WORKLOADS[@]}"; do
    for CFG in baseline twist; do
        if [[ "$CFG" == "baseline" ]]; then SIM="$B_SIM"; else SIM="$T_SIM"; fi
        KEY="${CFG}_${TEST}"
        LOGF="$LOGDIR/${KEY}.log"
        TASK_KEYS+=("$KEY")
        TASK_SIM[$KEY]="$SIM"
        TASK_LOG[$KEY]="$LOGF"
        TASK_CMD[$KEY]="$BINDIR/${TEST}-baremetal"
    done
done

gate_wait() {
    while :; do
        local alive=()
        for pid in "${LIVE_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then alive+=("$pid"); fi
        done
        LIVE_PIDS=("${alive[@]}")
        if (( ${#LIVE_PIDS[@]} < MAX_PARALLEL )); then return; fi
        sleep 5
    done
}

LAUNCHED=0
for KEY in "${TASK_KEYS[@]}"; do
    gate_wait
    SIM="${TASK_SIM[$KEY]}"
    LOGF="${TASK_LOG[$KEY]}"
    CMD="${TASK_CMD[$KEY]}"
    LD_LIBRARY_PATH="$LDPATH" "$SIM" \
        +permissive +max-cycles=100000000000 +permissive-off \
        $CMD > "$LOGF" 2>&1 &
    PID=$!
    PIDS[$KEY]=$PID
    LOGS[$KEY]="$LOGF"
    LIVE_PIDS+=("$PID")
    LAUNCHED=$((LAUNCHED + 1))
    printf "  [%2d/%d] launched %-34s PID=%s  (live=%d/%d)\n" \
        "$LAUNCHED" "$TOTAL_TASKS" "$KEY" "$PID" "${#LIVE_PIDS[@]}" "$MAX_PARALLEL"
done

echo ""
echo "  All $TOTAL_TASKS tasks launched. Waiting..."

# -----------------------------------------------------------------------
# 4. Progress monitor
# -----------------------------------------------------------------------
while true; do
    RUNNING=0
    for K in "${!PIDS[@]}"; do
        if kill -0 "${PIDS[$K]}" 2>/dev/null; then RUNNING=$((RUNNING + 1)); fi
    done
    if [[ $RUNNING -eq 0 ]]; then break; fi
    DONE_COUNT=0
    for K in "${!LOGS[@]}"; do
        if grep -qa 'BENCH_E2E' "${LOGS[$K]}" 2>/dev/null; then
            DONE_COUNT=$((DONE_COUNT + 1))
        fi
    done
    printf "\r  progress: %d/%d done, %d still running " \
        "$DONE_COUNT" "${#PIDS[@]}" "$RUNNING"
    sleep 30
done
wait
echo ""
echo ""

# -----------------------------------------------------------------------
# 5. Per-task quick results
# -----------------------------------------------------------------------
echo "=== Per-task results ==="
for K in "${TASK_KEYS[@]}"; do
    LOGF="${LOGS[$K]}"
    E2E=$(grep -aE 'BENCH_E2E' "$LOGF" 2>/dev/null | head -1 || echo "NO RESULT")
    GEM=$(grep -aE 'COLD_START' "$LOGF" 2>/dev/null | head -1 || echo "")
    printf "  %-34s  %s  %s\n" "$K" "$E2E" "$GEM"
done

# -----------------------------------------------------------------------
# 6. Extract CSV and print summary
# -----------------------------------------------------------------------
echo ""
echo "=== Extracting results ==="
extract_results "$LOGDIR" "$EVALDIR/data/conv_32x32.csv"

echo ""
echo "============================================"
echo " DONE."
echo "============================================"
echo "  Logs: $LOGDIR/"
echo "  CSV:  $EVALDIR/data/conv_32x32.csv"
