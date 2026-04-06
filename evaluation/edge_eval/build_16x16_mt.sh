#!/bin/bash
# Build 16x16 Verilator simulators with multi-threaded runtime.
#
# Each built simulator will be capable of running with up to N worker threads
# at runtime (--threads N passed to Verilator at verilate time).
#
# Usage:
#   bash build_16x16_mt.sh             # default VERILATOR_THREADS=16
#   bash build_16x16_mt.sh 8           # 8 runtime threads per sim
#
# Output simulators live in sims/verilator/ and override any prior single-
# threaded build of the same CONFIG (same filename, different internals).
set -eo pipefail

CHIPYARD=/data2/chenyi9/pipette/pipette-chipyard
cd "$CHIPYARD"
source env.sh

VTHREADS=${1:-16}
BUILD_JOBS=16                # parallelism for the C++ compile stage

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
    [ -f "$SIM_BIN" ] || return
    local INTERP
    INTERP=$(patchelf --print-interpreter "$SIM_BIN" 2>/dev/null || echo "")
    if [[ "$INTERP" != *"sysroot"* ]]; then
        patchelf --set-interpreter ${SYSROOT}/lib64/ld-linux-x86-64.so.2 \
          --set-rpath "${SYSROOT}/lib64:/data2/chenyi9/pipette/chipyard/.conda-env/riscv-tools/lib:/data2/chenyi9/pipette/chipyard/.conda-env/lib" \
          "$SIM_BIN"
        echo "  Patched: $(basename $SIM_BIN)"
    fi
}

echo "============================================"
echo " Multi-threaded 16x16 Build"
echo " VERILATOR_THREADS = $VTHREADS"
echo " Make parallelism   = $BUILD_JOBS"
echo " FORCE_CLEAN        = ${FORCE_CLEAN:-0}  (set to 1 to force full re-verilate)"
echo "============================================"

build_cfg() {
    local CFG="$1"
    local SIM_BIN="${SIMDIR}/simulator-chipyard.harness-${CFG}"
    local LOGF="${LOGDIR}/build_mt_${CFG}.log"
    local GEN_SRC="${SIMDIR}/generated-src/chipyard.harness.TestHarness.${CFG}"
    local MARKER="${GEN_SRC}/.vthreads_marker"

    # Detect VTHREADS change: Verilator's thread pool size is baked into the
    # generated C++ (compiled with --threads N), so changing $VTHREADS does
    # nothing unless we force a re-verilate. We keep a marker file with the
    # last used VTHREADS; if it differs (or is missing), we wipe generated-src
    # for this config so Make regenerates it.
    local NEED_CLEAN=0
    if [[ "$FORCE_CLEAN" == "1" ]]; then
        NEED_CLEAN=1
    elif [[ -d "$GEN_SRC" ]]; then
        if [[ ! -f "$MARKER" ]] || [[ "$(cat "$MARKER" 2>/dev/null)" != "$VTHREADS" ]]; then
            NEED_CLEAN=1
        fi
    fi

    if [[ $NEED_CLEAN -eq 1 && -d "$GEN_SRC" ]]; then
        echo "  [$CFG] VTHREADS changed (was: $(cat "$MARKER" 2>/dev/null || echo none), now: $VTHREADS) — wiping generated-src"
        rm -rf "$GEN_SRC"
    fi

    echo "  [$CFG] removing old binary"
    rm -f "$SIM_BIN"

    echo "  [$CFG] verilating + compiling (log: $LOGF)"
    make -C "$SIMDIR" \
        CONFIG="$CFG" \
        VERILATOR_THREADS="$VTHREADS" \
        -j${BUILD_JOBS} \
        > "$LOGF" 2>&1

    # Record the VTHREADS we just built with so the next invocation can
    # detect a change.
    if [[ -d "$GEN_SRC" ]]; then
        echo "$VTHREADS" > "$MARKER"
    fi

    patchelf_sim "$SIM_BIN"
    echo "  [$CFG] done"
}

# Build both configs in parallel (each uses $BUILD_JOBS cores for C++ compile)
PIDS=()
for CFG in "${CONFIGS[@]}"; do
    build_cfg "$CFG" &
    PIDS+=($!)
done

FAIL=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAIL=1
    fi
done

if [[ $FAIL -ne 0 ]]; then
    echo ""
    echo "BUILD FAILED — check ${LOGDIR}/build_mt_*.log"
    exit 1
fi

echo ""
echo "============================================"
echo " Build complete. Simulators:"
echo "============================================"
for CFG in "${CONFIGS[@]}"; do
    SIM_BIN="${SIMDIR}/simulator-chipyard.harness-${CFG}"
    if [[ -f "$SIM_BIN" ]]; then
        ls -lh "$SIM_BIN"
    else
        echo "  MISSING: $CFG"
    fi
done

echo ""

# -----------------------------------------------------------------------
# Update gemmini_params cache for DIM=16
# -----------------------------------------------------------------------
CACHE_DIR=README/evaluate/params_cache
GEN_BASE="${SIMDIR}/generated-src"
for CFG in "${CONFIGS[@]}"; do
    GEN_PARAMS="${GEN_BASE}/chipyard.harness.TestHarness.${CFG}/gemmini_params.h"
    if [[ -f "$GEN_PARAMS" ]]; then
        NEW_BANK=$(grep "^#define BANK_NUM " "$GEN_PARAMS" | awk '{print $3}')
        NEW_DIM=$(grep "^#define DIM " "$GEN_PARAMS" | awk '{print $3}')
        CACHED="${CACHE_DIR}/gemmini_params_dim${NEW_DIM}.h"
        cp "$GEN_PARAMS" "$CACHED"
        echo "  Updated cache: $CACHED (BANK_NUM=$NEW_BANK, DIM=$NEW_DIM)"
        break
    fi
done
