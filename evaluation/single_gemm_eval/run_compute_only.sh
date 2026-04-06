#!/bin/bash
# Single-GEMM compute-only cycle sweep: run simulations, extract data, plot.
#
# Prerequisites:
#   - Simulators built: FORCE_CLEAN=1 bash build_32x32_mt.sh 16
#
# What it does:
#   1. Build single_gemm_exbusy test binary (RISC-V bare-metal)
#   2. Run baseline + twist simulations in parallel
#   3. Extract compute-only cycle data from logs → CSV
#   4. Plot speedup figure
#
# Output:
#   logs/exbusy_debug/{baseline,twist}.log
#   data/compute_only.csv
#   figures/gemmini_exbusy_combined.{pdf,png}
#
# Usage:
#   bash run_exbusy_debug.sh          # full flow
#   bash run_exbusy_debug.sh --plot   # skip sims, just re-extract+plot
set -eo pipefail

CHIPYARD=/data2/chenyi9/pipette/pipette-chipyard
EVALDIR="$CHIPYARD/README/evaluate/papers/single_gemm_eval"
cd "$CHIPYARD"
source env.sh

SYSROOT=/data2/chenyi9/pipette/chipyard/.conda-env/x86_64-conda-linux-gnu/sysroot
LDPATH="${SYSROOT}/lib64:${SYSROOT}/../lib:/data2/chenyi9/pipette/chipyard/.conda-env/riscv-tools/lib:/data2/chenyi9/pipette/chipyard/.conda-env/lib"
SIMDIR=sims/verilator
SWDIR=generators/gemmini/software/gemmini-rocc-tests
BINDIR=${SWDIR}/build/bareMetalC
CACHE_DIR=README/evaluate/params_cache
LOGDIR="$EVALDIR/logs/exbusy_debug"

DIM_VAL=32
B_SIM="${SIMDIR}/simulator-chipyard.harness-Baseline32x32WSRocketConfig"
T_SIM="${SIMDIR}/simulator-chipyard.harness-Twist32x32WSSingleOpRocketConfig"
BIN="${BINDIR}/single_gemm_exbusy-baremetal"

TOTAL_SWEEPS=140   # 20 M values × 7 Q values

# ──────────────────────────────────────────────────────────
# Skip sims if --plot flag
# ──────────────────────────────────────────────────────────
if [[ "$1" == "--plot" ]]; then
    echo "=== Skipping simulations, re-extracting + plotting ==="
    cd "$CHIPYARD"
    python3 "$EVALDIR/scripts/extract_exbusy_debug.py"
    python3 "$EVALDIR/scripts/plot_exbusy.py"
    exit 0
fi

# ──────────────────────────────────────────────────────────
# 1. Verify simulators exist
# ──────────────────────────────────────────────────────────
for SIM in "$B_SIM" "$T_SIM"; do
    if [[ ! -f "$SIM" ]]; then
        echo "ERROR: Simulator not found: $SIM"
        echo "Run build_rtl.sh first."
        exit 1
    fi
done

# Patch interpreters if needed
for SIM in "$B_SIM" "$T_SIM"; do
    INTERP=$(patchelf --print-interpreter "$SIM" 2>/dev/null || echo "")
    if [[ "$INTERP" != *"sysroot"* ]]; then
        patchelf --set-interpreter ${SYSROOT}/lib64/ld-linux-x86-64.so.2 \
          --set-rpath "${SYSROOT}/lib64:/data2/chenyi9/pipette/chipyard/.conda-env/riscv-tools/lib:/data2/chenyi9/pipette/chipyard/.conda-env/lib" \
          "$SIM" 2>/dev/null || true
    fi
done

# ──────────────────────────────────────────────────────────
# 2. Switch gemmini_params.h to DIM=32, build, then restore
# ──────────────────────────────────────────────────────────
PARAMS_H="${SWDIR}/include/gemmini_params.h"
CACHED="${CACHE_DIR}/gemmini_params_dim${DIM_VAL}.h"

if [[ ! -f "$CACHED" ]]; then
    echo "ERROR: Params cache not found: $CACHED"
    exit 1
fi

cp "$PARAMS_H" "$PARAMS_H.bak"
cp "$CACHED" "$PARAMS_H"

ACTUAL_DIM=$(grep "^#define DIM " "$PARAMS_H" | awk '{print $3}')
echo "=== Building single_gemm_exbusy (DIM=$ACTUAL_DIM) ==="
if [ "$ACTUAL_DIM" != "$DIM_VAL" ]; then
    echo "ERROR: gemmini_params.h DIM=$ACTUAL_DIM, expected $DIM_VAL"
    cp "$PARAMS_H.bak" "$PARAMS_H"
    exit 1
fi

rm -f "$BIN"
cd "${SWDIR}/build"
mkdir -p bareMetalC
make -C bareMetalC -f "$(pwd)/../bareMetalC/Makefile" \
    abs_top_srcdir="$(pwd)/.." \
    src_dir="$(pwd)/../bareMetalC" \
    single_gemm_exbusy-baremetal
cd "$CHIPYARD"

cp "$PARAMS_H.bak" "$PARAMS_H"
rm -f "$PARAMS_H.bak"
echo "  Restored gemmini_params.h to original"

if [[ ! -f "$BIN" ]]; then
    echo "ERROR: Binary not found: $BIN"
    exit 1
fi

# ──────────────────────────────────────────────────────────
# 3. Run simulations
# ──────────────────────────────────────────────────────────
mkdir -p "$LOGDIR"

echo ""
echo "============================================"
echo " Single-GEMM Pipeline Cycle Sweep"
echo " Binary: $BIN"
echo " Expect $TOTAL_SWEEPS EXBUSY_SWEEP lines per config"
echo "============================================"
echo ""

echo "  Launching: baseline ..."
LD_LIBRARY_PATH="$LDPATH" \
  "$B_SIM" \
  +permissive +verbose +max-cycles=50000000000 +permissive-off \
  "$BIN" > "$LOGDIR/baseline.log" 2>&1 &
PID_B=$!

echo "  Launching: twist ..."
LD_LIBRARY_PATH="$LDPATH" \
  "$T_SIM" \
  +permissive +verbose +max-cycles=50000000000 +permissive-off \
  "$BIN" > "$LOGDIR/twist.log" 2>&1 &
PID_T=$!

echo ""
echo "  PIDs: baseline=$PID_B  twist=$PID_T"
echo "  Monitoring progress..."

# Progress monitor
while kill -0 $PID_B 2>/dev/null || kill -0 $PID_T 2>/dev/null; do
    NB=$(grep -c 'EXBUSY_SWEEP' "$LOGDIR/baseline.log" 2>/dev/null) || NB=0
    NT=$(grep -c 'EXBUSY_SWEEP' "$LOGDIR/twist.log" 2>/dev/null) || NT=0
    printf "\r  baseline: %3d/%d    twist: %3d/%d" "$NB" "$TOTAL_SWEEPS" "$NT" "$TOTAL_SWEEPS"
    sleep 30
done
wait

echo ""
echo ""

# Verify
NB=$(grep -c 'EXBUSY_SWEEP' "$LOGDIR/baseline.log" 2>/dev/null) || NB=0
NT=$(grep -c 'EXBUSY_SWEEP' "$LOGDIR/twist.log" 2>/dev/null) || NT=0
echo "  baseline: $NB EXBUSY_SWEEP lines"
echo "  twist:    $NT EXBUSY_SWEEP lines"

if [[ "$NB" -ne "$TOTAL_SWEEPS" || "$NT" -ne "$TOTAL_SWEEPS" ]]; then
    echo "WARNING: Expected $TOTAL_SWEEPS lines per config!"
fi

# ──────────────────────────────────────────────────────────
# 4. Extract data + plot
# ──────────────────────────────────────────────────────────
echo ""
echo "=== Extracting data ==="
python3 "$EVALDIR/scripts/extract_exbusy_debug.py"

echo ""
echo "=== Plotting ==="
python3 "$EVALDIR/scripts/plot_exbusy.py"

echo ""
echo "============================================"
echo " DONE. Output files:"
echo "============================================"
echo "  Data:    $EVALDIR/data/compute_only.csv"
echo "  Figure:  $EVALDIR/figures/gemmini_exbusy_combined.pdf"
echo "  Logs:    $LOGDIR/{baseline,twist}.log"
