#!/bin/bash
# Build and run Single-GEMM E2E + Compute-Only benchmarks
# 2 benchmarks × 2 configs = 4 parallel simulations
set -eo pipefail

CHIPYARD=/data2/chenyi9/pipette/pipette-chipyard
cd "$CHIPYARD"
source env.sh

SYSROOT=/data2/chenyi9/pipette/chipyard/.conda-env/x86_64-conda-linux-gnu/sysroot
LDPATH="${SYSROOT}/lib64:${SYSROOT}/../lib:/data2/chenyi9/pipette/chipyard/.conda-env/riscv-tools/lib:/data2/chenyi9/pipette/chipyard/.conda-env/lib"
DRAMSIM=generators/testchipip/src/main/resources/dramsim2_ini
SIMDIR=sims/verilator
BINDIR=generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC
LOGDIR=README/evaluate/papers/single_gemm_eval/logs

B_SIM="${SIMDIR}/simulator-chipyard.harness-Baseline32x32WSRocketConfig"
T_SIM="${SIMDIR}/simulator-chipyard.harness-Twist32x32WSSingleOpRocketConfig"

patchelf_sim() {
    local SIM_BIN="$1"
    local INTERP=$(patchelf --print-interpreter "$SIM_BIN" 2>/dev/null || echo "")
    if [[ "$INTERP" != *"sysroot"* ]]; then
        patchelf --set-interpreter ${SYSROOT}/lib64/ld-linux-x86-64.so.2 \
          --set-rpath "${SYSROOT}/lib64:/data2/chenyi9/pipette/chipyard/.conda-env/riscv-tools/lib:/data2/chenyi9/pipette/chipyard/.conda-env/lib" \
          "$SIM_BIN" 2>/dev/null || true
    fi
}

patchelf_sim "$B_SIM"
patchelf_sim "$T_SIM"
mkdir -p "$LOGDIR"

# Build
echo "Building single_gemm benchmarks..."
cd generators/gemmini/software/gemmini-rocc-tests
make -C build/bareMetalC -f ../../../../bareMetalC/Makefile \
    src_dir=../../../../bareMetalC abs_top_srcdir=../.. \
    single_gemm_e2e-baremetal single_gemm_compute-baremetal
cd "$CHIPYARD"

echo "============================================"
echo " Single-GEMM: 2 benchmarks x 2 configs = 4 sims"
echo "============================================"

run_sim() {
    local LABEL="$1" SIM="$2" BENCH="$3"
    local BIN="${BINDIR}/${BENCH}-baremetal"
    local LOG="${LOGDIR}/${LABEL}_${BENCH}.log"
    if [ ! -f "$BIN" ]; then
        echo "  SKIP: $BIN not found"
        return
    fi
    echo "  Launching: ${LABEL} ${BENCH}"
    LD_LIBRARY_PATH="$LDPATH" \
      "$SIM" \
      +permissive +dramsim +dramsim_ini_dir="$DRAMSIM" \
      +verbose +max-cycles=50000000000 +permissive-off \
      "$BIN" > "$LOG" 2>&1 &
}

run_sim "baseline" "$B_SIM" "single_gemm_e2e"
run_sim "twist"    "$T_SIM" "single_gemm_e2e"
run_sim "baseline" "$B_SIM" "single_gemm_compute"
run_sim "twist"    "$T_SIM" "single_gemm_compute"

echo ""
echo "  4 simulations launched. Waiting..."
wait

echo ""
echo "============================================"
echo "                 DONE"
echo "============================================"
echo "Logs in: $LOGDIR"
ls -lh "$LOGDIR"/*.log 2>/dev/null || echo "(no logs found)"
