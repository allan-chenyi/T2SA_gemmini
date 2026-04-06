#!/bin/bash
# Conv E2E: run conv_perf and conv_dw_perf on 16x16 Baseline vs Twist.
#
# Prerequisites:
#   - Simulators built: bash build_16x16.sh
#
# Usage:
#   bash run_conv.sh          # full flow (build binary + run sims + extract)
#   bash run_conv.sh --plot   # skip sims, just re-extract
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

# Tests to run
TESTS=("conv_perf" "conv_dw_perf")

# ──────────────────────────────────────────────────────────
# Skip sims if --plot flag
# ──────────────────────────────────────────────────────────
if [[ "$1" == "--plot" ]]; then
    echo "=== Skipping simulations, re-extracting ==="
    python3 "$EVALDIR/scripts/extract_conv.py"
    python3 "$EVALDIR/scripts/plot_conv.py"
    exit 0
fi

# ──────────────────────────────────────────────────────────
# 1. Verify simulators
# ──────────────────────────────────────────────────────────
for SIM in "$B_SIM" "$T_SIM"; do
    if [[ ! -f "$SIM" ]]; then
        echo "ERROR: Simulator not found: $SIM"
        echo "Run build_16x16.sh first."
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

# ──────────────────────────────────────────────────────────
# 2. Switch gemmini_params.h to DIM=16, build, then restore
# ──────────────────────────────────────────────────────────
PARAMS_H="${SWDIR}/include/gemmini_params.h"
CACHED="${CACHE_DIR}/gemmini_params_dim${DIM_VAL}.h"

if [[ ! -f "$CACHED" ]]; then
    echo "ERROR: Params cache not found: $CACHED"
    exit 1
fi

# Save current params, switch to DIM=16
cp "$PARAMS_H" "$PARAMS_H.bak"
cp "$CACHED" "$PARAMS_H"

ACTUAL_DIM=$(grep "^#define DIM " "$PARAMS_H" | awk '{print $3}')
echo "=== Building conv binaries (DIM=$ACTUAL_DIM) ==="
if [ "$ACTUAL_DIM" != "$DIM_VAL" ]; then
    echo "ERROR: gemmini_params.h DIM=$ACTUAL_DIM, expected $DIM_VAL"
    cp "$PARAMS_H.bak" "$PARAMS_H"
    exit 1
fi

# Remove old binaries to force recompile
for TEST in "${TESTS[@]}"; do
    rm -f "$BINDIR/${TEST}-baremetal"
done

cd "${SWDIR}/build"
mkdir -p bareMetalC
for TEST in "${TESTS[@]}"; do
    echo "  Building: ${TEST}-baremetal"
    make -C bareMetalC -f "$(pwd)/../bareMetalC/Makefile" \
        abs_top_srcdir="$(pwd)/.." \
        src_dir="$(pwd)/../bareMetalC" \
        "${TEST}-baremetal"
done
cd "$CHIPYARD"

# Restore original params
cp "$PARAMS_H.bak" "$PARAMS_H"
rm -f "$PARAMS_H.bak"
echo "  Restored gemmini_params.h to original"

for TEST in "${TESTS[@]}"; do
    if [[ ! -f "$BINDIR/${TEST}-baremetal" ]]; then
        echo "ERROR: Binary not found: $BINDIR/${TEST}-baremetal"
        exit 1
    fi
done

# ──────────────────────────────────────────────────────────
# 3. Run simulations
# ──────────────────────────────────────────────────────────
mkdir -p "$LOGDIR"

echo ""
echo "============================================"
echo " Conv E2E: Baseline vs Twist (16x16)"
echo " Tests: ${TESTS[*]}"
echo "============================================"
echo ""

for TEST in "${TESTS[@]}"; do
    BIN="$BINDIR/${TEST}-baremetal"

    echo "--- ${TEST} ---"

    echo "  Launching: baseline ..."
    LD_LIBRARY_PATH="$LDPATH" \
      "$B_SIM" \
      +permissive +verbose +max-cycles=100000000000 +permissive-off \
      "$BIN" > "$LOGDIR/baseline_${TEST}.log" 2>&1 &
    PID_B=$!

    echo "  Launching: twist ..."
    LD_LIBRARY_PATH="$LDPATH" \
      "$T_SIM" \
      +permissive +verbose +max-cycles=100000000000 +permissive-off \
      "$BIN" > "$LOGDIR/twist_${TEST}.log" 2>&1 &
    PID_T=$!

    echo "  PIDs: baseline=$PID_B  twist=$PID_T"
    echo "  Waiting for completion..."

    while kill -0 $PID_B 2>/dev/null || kill -0 $PID_T 2>/dev/null; do
        B_DONE=0
        T_DONE=0
        grep -q 'Gemmini conv took' "$LOGDIR/baseline_${TEST}.log" 2>/dev/null && B_DONE=1
        grep -q 'Gemmini conv took' "$LOGDIR/twist_${TEST}.log" 2>/dev/null && T_DONE=1
        printf "\r  baseline: %s    twist: %s" \
            "$([ $B_DONE -eq 1 ] && echo 'DONE' || echo 'running')" \
            "$([ $T_DONE -eq 1 ] && echo 'DONE' || echo 'running')"
        sleep 30
    done
    wait

    echo ""
    echo "  Results:"
    grep 'Gemmini conv took' "$LOGDIR/baseline_${TEST}.log" 2>/dev/null || echo "    baseline: no result"
    grep 'Gemmini conv took' "$LOGDIR/twist_${TEST}.log" 2>/dev/null || echo "    twist: no result"
    echo ""
done

# ──────────────────────────────────────────────────────────
# 4. Extract + summarize
# ──────────────────────────────────────────────────────────
echo "=== Extracting data ==="
python3 "$EVALDIR/scripts/extract_conv.py"

echo ""
echo "=== Plotting ==="
python3 "$EVALDIR/scripts/plot_conv.py"

echo ""
echo "============================================"
echo " DONE. Output files:"
echo "============================================"
echo "  Data:    $EVALDIR/data/"
echo "  Figures: $EVALDIR/figures/"
echo "  Logs:    $LOGDIR/"
