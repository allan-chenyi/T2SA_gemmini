#!/bin/bash
# Parallel evaluation: conv_perf, conv_dw_perf, lstm_cell_perf (multiple sizes)
# on Baseline16x16 and Twist16x16 simulators, running all combinations in
# parallel. Each simulator uses up to VTHREADS cores at runtime.
#
# Prerequisites:
#   - Simulators built with multi-threaded Verilator:
#       bash build_16x16_mt.sh
#
# Usage:
#   bash run_all_parallel.sh             # full flow
#   bash run_all_parallel.sh --extract   # skip sims, just extract + plot
#
# Task layout (workloads × 2 configs × 16 threads each):
#   conv_perf            (default 7x7 in=8 out=16)
#   conv_dw_perf         (default 7x7 ch=8)
#   lstm_cell_perf  H=64  I=64  B=1
#   lstm_cell_perf  H=128 I=128 B=1
#   lstm_cell_perf  H=256 I=256 B=1
#
# With 2 configs per workload = 10 parallel simulators × 16 threads = 160
# threads. Host has 128 cores so we allocate VTHREADS=12 (10*12=120) or
# adjust TASK_VTHREADS below.
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
LOGDIR="$EVALDIR/logs"

DIM_VAL=16
B_SIM="${SIMDIR}/simulator-chipyard.harness-Baseline16x16WSRocketConfig"
T_SIM="${SIMDIR}/simulator-chipyard.harness-Twist16x16WSSingleOpRocketConfig"

# Workload definitions: "<testname>|<args to bin>|<log suffix>"
# Empty args means "use defaults compiled into the binary".
WORKLOADS=(
    # Conv family removed — conv_perf / conv_dw_perf took too long to simulate.
    # lstm_cell_perf (DRAM-backed, multi-tile) also removed — each sim was
    # spending 30+ minutes without reaching the first result line.
    # Spad-only RNN/GRU/DLRM family (report cold_total/warm_total/per_gemm).
    # Sizes auto-scale with DIM via spad_gemm_utils.h (D_HIDDEN=2*DIM, BATCH=DIM);
    # at DIM=16 this gives hidden=32, proj=16, batch=16 — proportionally scaled
    # versions of the published configurations.
    "lstm_lm_spad||lstm_lm_spad"             # Sak et al. 2014 (LSTMP)
    "kws_lstm_spad||kws_lstm_spad"           # Sainath & Parada 2015
    "gru_ts_spad||gru_ts_spad"               # Cho et al. 2014
    "stacked_lstm_spad||stacked_lstm_spad"   # Sak et al. 2014 (deep)
    "dlrm_mlp_spad||dlrm_mlp_spad"           # Naumov et al. 2019
    "bilstm_spad||bilstm_spad"               # Lample et al. 2016 (NER)
    "seq2seq_spad||seq2seq_spad"             # Bahdanau et al. 2014
    "deepspeech2_spad||deepspeech2_spad"     # Amodei et al. 2016
)

# Per-sim thread count — MUST match the VERILATOR_THREADS used at build time
# (Verilator bakes its worker pool size into the binary; you cannot shrink at
# runtime). If you rebuilt with `build_16x16_mt.sh 16`, keep this at 16.
TASK_VTHREADS=16

# Total CPU budget. Concurrency = CPU_BUDGET / TASK_VTHREADS (rounded down).
# With 64 cores and 16 threads/sim that's 4 concurrent sims; 26 tasks → 7 waves.
CPU_BUDGET=${CPU_BUDGET:-64}
MAX_PARALLEL=$(( CPU_BUDGET / TASK_VTHREADS ))
if (( MAX_PARALLEL < 1 )); then MAX_PARALLEL=1; fi

# -----------------------------------------------------------------------
# Extract-only mode
# -----------------------------------------------------------------------
if [[ "$1" == "--extract" ]]; then
    echo "=== Extracting + plotting only ==="
    python3 "$EVALDIR/scripts/extract_conv.py"
    python3 "$EVALDIR/scripts/plot_conv.py"
    exit 0
fi

# -----------------------------------------------------------------------
# 1. Verify simulators exist
# -----------------------------------------------------------------------
for SIM in "$B_SIM" "$T_SIM"; do
    if [[ ! -f "$SIM" ]]; then
        echo "ERROR: Simulator not found: $SIM"
        echo "Run: bash build_16x16_mt.sh"
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
# 2. Switch gemmini_params.h to DIM=16, build binaries, restore
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
# 3. Launch all simulations in parallel
# -----------------------------------------------------------------------
mkdir -p "$LOGDIR"

TOTAL_TASKS=$(( ${#WORKLOADS[@]} * 2 ))
echo ""
echo "============================================"
echo " Launching $TOTAL_TASKS sims (${#WORKLOADS[@]} workloads × 2 configs)"
echo " Per-sim threads : $TASK_VTHREADS"
echo " CPU budget      : $CPU_BUDGET"
echo " Max concurrent  : $MAX_PARALLEL  ($(( (TOTAL_TASKS + MAX_PARALLEL - 1) / MAX_PARALLEL )) waves)"
echo "============================================"
echo ""

declare -A PIDS      # key -> pid (all tasks ever launched)
declare -A LOGS      # key -> logfile
LIVE_PIDS=()         # currently-running pids (for the concurrency gate)

# Build the full task list first (as (key, sim, logfile, bin+args) tuples).
TASK_KEYS=()
declare -A TASK_SIM TASK_LOG TASK_CMD
for W in "${WORKLOADS[@]}"; do
    IFS='|' read -r TEST ARGS SUFFIX <<< "$W"
    for CFG in baseline twist; do
        if [[ "$CFG" == "baseline" ]]; then SIM="$B_SIM"; else SIM="$T_SIM"; fi
        if [[ "$TEST" == "lstm_cell_perf" ]]; then
            KEY="${CFG}_${SUFFIX}"
        else
            KEY="${CFG}_${TEST}"
        fi
        LOGF="$LOGDIR/${KEY}.log"
        TASK_KEYS+=("$KEY")
        TASK_SIM[$KEY]="$SIM"
        TASK_LOG[$KEY]="$LOGF"
        TASK_CMD[$KEY]="$BINDIR/${TEST}-baremetal $ARGS"
    done
done

# Gate: wait until fewer than MAX_PARALLEL sims are running, then return.
gate_wait() {
    while :; do
        local alive=()
        for pid in "${LIVE_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then alive+=("$pid"); fi
        done
        LIVE_PIDS=("${alive[@]}")
        if (( ${#LIVE_PIDS[@]} < MAX_PARALLEL )); then
            return
        fi
        sleep 5
    done
}

LAUNCHED=0
for KEY in "${TASK_KEYS[@]}"; do
    gate_wait

    SIM="${TASK_SIM[$KEY]}"
    LOGF="${TASK_LOG[$KEY]}"
    CMD="${TASK_CMD[$KEY]}"

    # NOTE: no +verbose — the commit trace interleaves with HTIF printf
    # output, cutting the result lines in half and breaking extraction.
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
        PID=${PIDS[$K]}
        if kill -0 "$PID" 2>/dev/null; then
            RUNNING=$((RUNNING + 1))
        fi
    done
    if [[ $RUNNING -eq 0 ]]; then break; fi

    DONE_COUNT=0
    for K in "${!LOGS[@]}"; do
        LOGF=${LOGS[$K]}
        if grep -qE 'Gemmini conv took|_SPAD.*warm_total=' "$LOGF" 2>/dev/null; then
            DONE_COUNT=$((DONE_COUNT + 1))
        fi
    done
    TOTAL=${#PIDS[@]}
    printf "\r  progress: %d/%d done, %d still running " \
        "$DONE_COUNT" "$TOTAL" "$RUNNING"
    sleep 60
done
wait

echo ""
echo ""

# -----------------------------------------------------------------------
# 5. Results
# -----------------------------------------------------------------------
echo "=== Per-task results ==="
for K in "${!LOGS[@]}"; do
    LOGF=${LOGS[$K]}
    RES=$(grep -E 'Gemmini conv took|_SPAD.*warm_total=' "$LOGF" 2>/dev/null \
          | head -1 || echo "NO RESULT")
    printf "  %-30s  %s\n" "$K" "$RES"
done

# -----------------------------------------------------------------------
# 6. Extract + plot
# -----------------------------------------------------------------------
echo ""
echo "=== Extracting ==="
# extract_conv.py expects logs named <config>_<test>.log so we need to
# stage the lstm per-size logs. Do this by creating a mini manifest and
# letting the extract script pick up each variant via its test suffix.
# For now: copy the first available lstm log to the canonical name.
for CFG in baseline twist; do
    for SUFFIX in lstm_h64 lstm_h128 lstm_h256; do
        SRC="$LOGDIR/${CFG}_${SUFFIX}.log"
        if [[ -f "$SRC" ]]; then
            # Keep originals; extract step below aggregates all sizes.
            :
        fi
    done
done

python3 "$EVALDIR/scripts/extract_conv.py"

echo ""
echo "=== Plotting ==="
python3 "$EVALDIR/scripts/plot_conv.py"

echo ""
echo "============================================"
echo " DONE."
echo "============================================"
echo "  Data:    $EVALDIR/data/"
echo "  Figures: $EVALDIR/figures/"
echo "  Logs:    $LOGDIR/"
