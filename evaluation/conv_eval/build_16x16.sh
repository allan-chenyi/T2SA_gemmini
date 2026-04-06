#!/bin/bash
# Build 16x16 Verilator simulators (Baseline + Twist).
# Usage: bash build_16x16.sh
set -eo pipefail

CHIPYARD=/data2/chenyi9/pipette/pipette-chipyard
cd "$CHIPYARD"
source env.sh

SYSROOT=/data2/chenyi9/pipette/chipyard/.conda-env/x86_64-conda-linux-gnu/sysroot
SIMDIR=sims/verilator
LOGDIR=README/evaluate/papers/conv_eval/logs

mkdir -p "$LOGDIR"

CONFIGS=(
    "Baseline16x16WSRocketConfig"
    "Twist16x16WSSingleOpRocketConfig"
)

patchelf_sim() {
    local SIM_BIN="$1"
    if [ ! -f "$SIM_BIN" ]; then return; fi
    local INTERP=$(patchelf --print-interpreter "$SIM_BIN" 2>/dev/null || echo "")
    if [[ "$INTERP" != *"sysroot"* ]]; then
        patchelf --set-interpreter ${SYSROOT}/lib64/ld-linux-x86-64.so.2 \
          --set-rpath "${SYSROOT}/lib64:/data2/chenyi9/pipette/chipyard/.conda-env/riscv-tools/lib:/data2/chenyi9/pipette/chipyard/.conda-env/lib" \
          "$SIM_BIN"
        echo "  Patched: $(basename $SIM_BIN)"
    fi
}

echo "============================================"
echo " Building 16x16 simulators"
echo "============================================"

for CFG in "${CONFIGS[@]}"; do
    SIM_BIN="${SIMDIR}/simulator-chipyard.harness-${CFG}"

    # Force rebuild: remove old binary so make regenerates RTL
    if [ -f "$SIM_BIN" ]; then
        echo "  Removing old: $(basename $SIM_BIN)"
        rm -f "$SIM_BIN"
    fi

    echo "  Building: $CFG ..."
    make -C "$SIMDIR" CONFIG="$CFG" 2>&1 | tee "${LOGDIR}/build_${CFG}.log"
    patchelf_sim "$SIM_BIN"
    echo "  Done: $CFG"
    echo ""
done

echo "============================================"
echo " Build complete. Simulators:"
echo "============================================"
for CFG in "${CONFIGS[@]}"; do
    ls -lh "${SIMDIR}/simulator-chipyard.harness-${CFG}" 2>/dev/null || echo "  MISSING: $CFG"
done
