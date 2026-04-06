#!/bin/bash
# 16x16 spad-only parallel runner.
#
# Runs spad workloads on Baseline16x16WS and Twist16x16WSSingleOp
# simulators, 16 Verilator threads each, several in parallel.
#
# Prerequisites:
#   1. 16x16 simulators built:
#        FORCE_CLEAN=1 bash build_16x16_mt.sh 16
#   2. params_cache/gemmini_params_dim16.h exists
#
# Usage:
#   bash run_spad_16x16_parallel.sh             # full flow
#   bash run_spad_16x16_parallel.sh --extract   # skip sims, just extract
set -eo pipefail

CHIPYARD=/data2/chenyi9/pipette/pipette-chipyard
EVALDIR="$CHIPYARD/README/evaluate/papers/conv_eval"
cd "$CHIPYARD"
source env.sh

SYSROOT=/data2/chenyi9/pipette/chipyard/.conda-env/x86_64-conda-linux-gnu/sysroot
LDPATH="${SYSROOT}/lib64:${SYSROOT}/../lib:/data2/chenyi9/pipette/chipyard/.conda-env/riscv-tools/lib:/data2/chenyi9/pipette/chipyard/.conda-env/lib"
SIMDIR=sims/verilator
SWDIR=generators/gemmini/software/gemmini-rocc-tests
BINDIR=${SWDIR}/build/bareMetalC
CACHE_DIR=README/evaluate/params_cache
LOGDIR="$EVALDIR/logs_16x16_spad"

DIM_VAL=16
B_SIM="${SIMDIR}/simulator-chipyard.harness-Baseline16x16WSRocketConfig"
T_SIM="${SIMDIR}/simulator-chipyard.harness-Twist16x16WSSingleOpRocketConfig"

WORKLOADS=(
    "lstm_lm_spad||lstm_lm_spad"
    "kws_lstm_spad||kws_lstm_spad"
    "dlrm_mlp_spad||dlrm_mlp_spad"
    "bilstm_spad||bilstm_spad"
)

TASK_VTHREADS=16
CPU_BUDGET=${CPU_BUDGET:-64}
MAX_PARALLEL=$(( CPU_BUDGET / TASK_VTHREADS ))
if (( MAX_PARALLEL < 1 )); then MAX_PARALLEL=1; fi

# -----------------------------------------------------------------------
# Extract-only mode
# -----------------------------------------------------------------------
if [[ "$1" == "--extract" ]]; then
    echo "=== Extracting 16x16 spad results ==="
    CONV_LOGDIR="$LOGDIR" \
    CONV_CSV="$EVALDIR/data/conv_results_16x16.csv" \
        python3 "$EVALDIR/scripts/extract_conv.py"
    CONV_CSV="$EVALDIR/data/conv_results_16x16.csv" \
    CONV_OUT_PREFIX="conv_table_16x16" \
        python3 "$EVALDIR/scripts/plot_conv.py"
    exit 0
fi

# -----------------------------------------------------------------------
# 1. Verify simulators exist
# -----------------------------------------------------------------------
for SIM in "$B_SIM" "$T_SIM"; do
    if [[ ! -f "$SIM" ]]; then
        echo "ERROR: Simulator not found: $SIM"
        echo "Run: FORCE_CLEAN=1 bash build_16x16_mt.sh 16"
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
# 2. Swap DIM=16 params header, rebuild RISC-V binaries, restore
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
UNIQ_BINS=()
for W in "${WORKLOADS[@]}"; do
    IFS='|' read -r TEST ARGS SUFFIX <<< "$W"
    [[ " ${UNIQ_BINS[*]} " =~ " $TEST " ]] || UNIQ_BINS+=("$TEST")
done

cd "${SWDIR}/build"
mkdir -p bareMetalC
for TEST in "${UNIQ_BINS[@]}"; do
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

for TEST in "${UNIQ_BINS[@]}"; do
    BIN="$BINDIR/${TEST}-baremetal"
    if [[ ! -f "$BIN" ]]; then
        echo "ERROR: Binary not found: $BIN"
        exit 1
    fi
done

# -----------------------------------------------------------------------
# 3. Launch sims in parallel (gated by MAX_PARALLEL)
# -----------------------------------------------------------------------
mkdir -p "$LOGDIR"

TOTAL_TASKS=$(( ${#WORKLOADS[@]} * 2 ))
echo ""
echo "============================================"
echo " 16x16 spad parallel run"
echo "   Workloads      : ${#WORKLOADS[@]}  (× 2 configs = $TOTAL_TASKS sims)"
echo "   Per-sim threads: $TASK_VTHREADS"
echo "   CPU budget     : $CPU_BUDGET"
echo "   Concurrency    : $MAX_PARALLEL"
echo "============================================"
echo ""

declare -A PIDS LOGS
LIVE_PIDS=()

TASK_KEYS=()
declare -A TASK_SIM TASK_LOG TASK_CMD
for W in "${WORKLOADS[@]}"; do
    IFS='|' read -r TEST ARGS SUFFIX <<< "$W"
    for CFG in baseline twist; do
        if [[ "$CFG" == "baseline" ]]; then SIM="$B_SIM"; else SIM="$T_SIM"; fi
        KEY="${CFG}_${TEST}"
        LOGF="$LOGDIR/${KEY}.log"
        TASK_KEYS+=("$KEY")
        TASK_SIM[$KEY]="$SIM"
        TASK_LOG[$KEY]="$LOGF"
        TASK_CMD[$KEY]="$BINDIR/${TEST}-baremetal $ARGS"
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
echo "  All $TOTAL_TASKS tasks enqueued. Waiting for the last wave..."

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
# 5. Results
# -----------------------------------------------------------------------
echo "=== Per-task results ==="
for K in "${!LOGS[@]}"; do
    RES=$(grep -aE 'BENCH_E2E' "${LOGS[$K]}" 2>/dev/null | head -1 || echo "NO RESULT")
    printf "  %-34s  %s\n" "$K" "$RES"
done

# -----------------------------------------------------------------------
# 6. Extract CSV and generate plots
# -----------------------------------------------------------------------
echo ""
echo "=== Extracting 16x16 results ==="
CONV_LOGDIR="$LOGDIR" \
CONV_CSV="$EVALDIR/data/conv_results_16x16.csv" \
    python3 "$EVALDIR/scripts/extract_conv.py"

echo ""
echo "=== Generating 16x16 table & figures ==="
CONV_CSV="$EVALDIR/data/conv_results_16x16.csv" \
CONV_OUT_PREFIX="conv_table_16x16" \
    python3 "$EVALDIR/scripts/plot_conv.py"

echo ""
echo "============================================"
echo " DONE."
echo "============================================"
echo "  Logs:    $LOGDIR/"
echo "  CSV:     $EVALDIR/data/conv_results_16x16.csv"
echo "  Figures: $EVALDIR/figures/conv_table_16x16.{tex,pdf,png}"
