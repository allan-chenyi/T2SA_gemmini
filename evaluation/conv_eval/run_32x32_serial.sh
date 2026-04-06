#!/bin/bash
# Serial 32x32 evaluation of the 5 DRAM-bound / attention-structure workloads
# that perform worse at 16x16:
#
#   - bert_tiny        (d_head=64 needs 2*DIM>=64 i.e. DIM>=32 to satisfy the
#                       2-tile attention path; at DIM=16 the 0.83% gain from
#                       32x32 disappears entirely)
#   - mlperf_tiny_ad   ) DRAM-bound: more tiles at smaller DIM => LD controller
#   - ds_cnn_kws       )  dominates => Twist pipeline saving gets buried
#   - mobilenet_v1     )
#   - mnist_diffusion  )
#
# Each sim uses 16 Verilator worker threads (must match the build-time
# VERILATOR_THREADS — the 32x32 binaries MUST be rebuilt with
#   bash build_32x32_mt.sh 16
# before running this script for the first time).
#
# Only ONE sim runs at a time (strict serial). With 5 workloads × 2 configs
# this is 10 runs back-to-back.
#
# Usage:
#   bash run_32x32_serial.sh             # full flow: build RISC-V bins + run
#   bash run_32x32_serial.sh --extract   # skip sims, just parse logs
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
LOGDIR="$EVALDIR/logs_32x32"

DIM_VAL=32
B_SIM="${SIMDIR}/simulator-chipyard.harness-Baseline32x32WSRocketConfig"
T_SIM="${SIMDIR}/simulator-chipyard.harness-Twist32x32WSSingleOpRocketConfig"

# Workload list: "<testname>|<args>|<log suffix>". All 5 take no argv.
WORKLOADS=(
    "bert_tiny||bert_tiny"             # attention 2-tile path needs DIM>=32
    "mlperf_tiny_ad||mlperf_tiny_ad"   # small MLP, DRAM-bound
    "ds_cnn_kws||ds_cnn_kws"           # depthwise+pointwise, DRAM-bound
    "mobilenet_v1||mobilenet_v1"       # pointwise stack, DRAM-bound
    "mnist_diffusion||mnist_diffusion" # U-Net style, DRAM-bound
)

mkdir -p "$LOGDIR"

# -----------------------------------------------------------------------
# Extract-only mode (reuses extract_conv.py + plot_conv.py on 32x32 logs)
# -----------------------------------------------------------------------
if [[ "$1" == "--extract" ]]; then
    echo "=== Extracting 32x32 results ==="
    # Point the extractor at our 32x32 logs dir by symlinking (extract script
    # hardcodes ../logs). Simpler: just print the cold_total/warm_total lines.
    for K in baseline twist; do
        for W in "${WORKLOADS[@]}"; do
            IFS='|' read -r TEST ARGS SUFFIX <<< "$W"
            LOGF="$LOGDIR/${K}_${TEST}.log"
            if [[ -f "$LOGF" ]]; then
                printf "  %-30s  " "${K}/${TEST}"
                grep -E "cold_total=" "$LOGF" | head -1 || echo "NO RESULT"
            fi
        done
    done
    exit 0
fi

# -----------------------------------------------------------------------
# 1. Verify 32x32 simulators exist AND were built with threads=16
# -----------------------------------------------------------------------
for SIM in "$B_SIM" "$T_SIM"; do
    if [[ ! -f "$SIM" ]]; then
        echo "ERROR: Simulator not found: $SIM"
        echo "Run: bash build_32x32_mt.sh 16"
        exit 1
    fi
done

for CFG in Baseline32x32WSRocketConfig Twist32x32WSSingleOpRocketConfig; do
    MARKER="$SIMDIR/generated-src/chipyard.harness.TestHarness.${CFG}/.vthreads_marker"
    if [[ ! -f "$MARKER" ]]; then
        echo "WARNING: $CFG has no VTHREADS marker — likely built single-threaded."
        echo "         Run 'bash build_32x32_mt.sh 16' first if you want real 16-thread sims."
        echo "         (Continuing anyway — sim will still run, just slower.)"
    elif [[ "$(cat "$MARKER")" != "16" ]]; then
        echo "WARNING: $CFG built with VTHREADS=$(cat "$MARKER"), not 16."
    fi
done

# Patch interpreter so the sim finds the right libc (same as the parallel script)
for SIM in "$B_SIM" "$T_SIM"; do
    INTERP=$(patchelf --print-interpreter "$SIM" 2>/dev/null || echo "")
    if [[ "$INTERP" != *"sysroot"* ]]; then
        patchelf --set-interpreter ${SYSROOT}/lib64/ld-linux-x86-64.so.2 \
          --set-rpath "${SYSROOT}/lib64:/data2/chenyi9/pipette/chipyard/.conda-env/riscv-tools/lib:/data2/chenyi9/pipette/chipyard/.conda-env/lib" \
          "$SIM" 2>/dev/null || true
    fi
done

# -----------------------------------------------------------------------
# 2. Switch gemmini_params.h to DIM=32, rebuild RISC-V binaries, restore
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
# 3. Run serially (one at a time)
# -----------------------------------------------------------------------
TOTAL=$(( ${#WORKLOADS[@]} * 2 ))
echo ""
echo "============================================"
echo " Serial 32x32 run: $TOTAL tasks"
echo " Per-sim threads : 16  (requires build_32x32_mt.sh 16)"
echo " Concurrency     : 1   (strict serial)"
echo "============================================"
echo ""

run_one() {
    local CFG="$1"
    local SIM="$2"
    local TEST="$3"
    local ARGS="$4"
    local IDX="$5"
    local LOGF="$LOGDIR/${CFG}_${TEST}.log"

    local T0=$(date +%s)
    printf "[%2d/%d] %-8s %-18s  starting..." "$IDX" "$TOTAL" "$CFG" "$TEST"

    LD_LIBRARY_PATH="$LDPATH" "$SIM" \
        +permissive +verbose +max-cycles=100000000000 +permissive-off \
        "$BINDIR/${TEST}-baremetal" $ARGS \
        > "$LOGF" 2>&1

    local RC=$?
    local T1=$(date +%s)
    local DT=$((T1 - T0))

    if [[ $RC -ne 0 ]]; then
        printf " FAILED (rc=%d, %ds) log=%s\n" "$RC" "$DT" "$(basename $LOGF)"
        return 1
    fi

    # Pull the summary line for quick visibility.
    local SUMMARY
    SUMMARY=$(grep -E "cold_total=" "$LOGF" | head -1 || echo "<no result line>")
    printf " done (%4ds)  %s\n" "$DT" "$SUMMARY"
}

IDX=0
for W in "${WORKLOADS[@]}"; do
    IFS='|' read -r TEST ARGS SUFFIX <<< "$W"
    for CFG in baseline twist; do
        IDX=$((IDX + 1))
        if [[ "$CFG" == "baseline" ]]; then SIM="$B_SIM"; else SIM="$T_SIM"; fi
        run_one "$CFG" "$SIM" "$TEST" "$ARGS" "$IDX" || true
    done
done

echo ""
echo "============================================"
echo " Serial 32x32 run complete."
echo "============================================"
echo "  Logs: $LOGDIR/"
echo ""
echo "Quick summary (re-run extract with --extract):"
for W in "${WORKLOADS[@]}"; do
    IFS='|' read -r TEST ARGS SUFFIX <<< "$W"
    for CFG in baseline twist; do
        LOGF="$LOGDIR/${CFG}_${TEST}.log"
        if [[ -f "$LOGF" ]]; then
            printf "  %-30s  " "${CFG}/${TEST}"
            grep -E "cold_total=" "$LOGF" | head -1 || echo "NO RESULT"
        fi
    done
done
