// LSTM Language Model — Spad-Only
//
// Projection LSTM (LSTMP, Sak et al. 2014):
//   p_t = h_{t-1} × W_proj         (hidden→proj, narrow)
//   gate_i = p_t × W_i             (proj→hidden, wide)
//   gate_f = p_t × W_f             (proj→hidden, wide)
//   gate_o = p_t × W_o             (proj→hidden, wide)
//   gate_g = p_t × W_g  [RELU]     (proj→hidden, wide)
//   [element-wise: c,h update — CPU, not timed]
//
// hidden=128 (fixed), proj=DIM, batch=DIM, timesteps=32
// 5 GEMMs/step × 32 steps = 160 GEMMs, all spad-only.

#include "spad_gemm_utils.h"

#define TIMESTEPS 32
#define HIDDEN    128
#define PROJ_DIM  DIM

#define H_TILES   (HIDDEN / DIM)    // 8 for DIM=16, 4 for DIM=32
#define P_TILES   (PROJ_DIM / DIM)  // always 1

// Tile counts for LSTMP GEMMs
#define PROJ_K    H_TILES   // proj: [DIM×HIDDEN] × [HIDDEN×PROJ_DIM]
#define PROJ_J    P_TILES   // = 1
#define GATE_K    P_TILES   // gate: [DIM×PROJ_DIM] × [PROJ_DIM×HIDDEN]
#define GATE_J    H_TILES

// Spad rows per weight block
#define W_PROJ_ROWS  TILE_ROWS(PROJ_K, PROJ_J)
#define W_GATE_ROWS  TILE_ROWS(GATE_K, GATE_J)

// DRAM weight pointers
#define W_PROJ_DRAM  (DRAM_BASE)
#define W_I_DRAM     (DRAM_BASE + 0x10000)
#define W_F_DRAM     (DRAM_BASE + 0x20000)
#define W_O_DRAM     (DRAM_BASE + 0x30000)
#define W_G_DRAM     (DRAM_BASE + 0x40000)
#define H0_DRAM      (DRAM_BASE + 0x50000)

// Spad layout (all computed from tile counts)
#define SP_WPROJ  0
#define SP_WI     (SP_WPROJ + W_PROJ_ROWS)
#define SP_WF     (SP_WI + W_GATE_ROWS)
#define SP_WO     (SP_WF + W_GATE_ROWS)
#define SP_WG     (SP_WO + W_GATE_ROWS)
#define SP_H0     (SP_WG + W_GATE_ROWS)
#define SP_H1     (SP_H0 + H_TILES * DIM)
#define SP_PROJ   (SP_H1 + H_TILES * DIM)
#define SP_GATE   (SP_PROJ + P_TILES * DIM)
// Total used: SP_GATE + H_TILES * DIM

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); exit(1); }
#endif
    gemmini_flush(0);
    unsigned long bench_general_start = read_cycles();

    printf("=== LSTM-LM Spad-Only (DIM=%d, hidden=%d, proj=%d, T=%d) ===\n",
           DIM, HIDDEN, PROJ_DIM, TIMESTEPS);

    unsigned long cold_start = read_cycles();

    // Pre-load weights
    load_tiles((elem_t *)W_PROJ_DRAM, PROJ_DIM, SP_WPROJ, PROJ_K, PROJ_J);
    load_tiles((elem_t *)W_I_DRAM, HIDDEN, SP_WI, GATE_K, GATE_J);
    load_tiles((elem_t *)W_F_DRAM, HIDDEN, SP_WF, GATE_K, GATE_J);
    load_tiles((elem_t *)W_O_DRAM, HIDDEN, SP_WO, GATE_K, GATE_J);
    load_tiles((elem_t *)W_G_DRAM, HIDDEN, SP_WG, GATE_K, GATE_J);
    // Load initial h [DIM × HIDDEN]
    load_act((elem_t *)H0_DRAM, HIDDEN, SP_H0, H_TILES);
    gemmini_fence();

    gemmini_config_ex(WS, 0, 0);

    unsigned long start = read_cycles();

    for (int t = 0; t < TIMESTEPS; t++) {
        uint32_t h_cur = (t % 2 == 0) ? SP_H0 : SP_H1;

        // proj = h × W_proj: [DIM×HIDDEN] × [HIDDEN×PROJ_DIM], K=H_TILES, J=1
        SPAD_GEMM(PROJ_J, PROJ_K, h_cur, SP_WPROJ, SP_PROJ, NO_ACTIVATION);
        gemmini_fence();

        // 4 gates: proj × W_gate: [DIM×PROJ_DIM] × [PROJ_DIM×HIDDEN], K=1, J=H_TILES
        SPAD_GEMM(GATE_J, GATE_K, SP_PROJ, SP_WI, SP_GATE, NO_ACTIVATION);
        gemmini_fence();
        SPAD_GEMM(GATE_J, GATE_K, SP_PROJ, SP_WF, SP_GATE, NO_ACTIVATION);
        gemmini_fence();
        SPAD_GEMM(GATE_J, GATE_K, SP_PROJ, SP_WO, SP_GATE, NO_ACTIVATION);
        gemmini_fence();
        SPAD_GEMM(GATE_J, GATE_K, SP_PROJ, SP_WG, SP_GATE, RELU);
        gemmini_fence();
    }

    unsigned long end = read_cycles();
    unsigned long total = end - start;

    printf("LSTM_LM_SPAD DIM=%d hidden=%d proj=%d timesteps=%d gemms=%d cold_total=%lu warm_total=%lu load_cyc=%lu per_gemm=%lu\n",
           DIM, HIDDEN, PROJ_DIM, TIMESTEPS, 5*TIMESTEPS,
           end - cold_start, total, start - cold_start, total / (5*TIMESTEPS));
    unsigned long bench_general_end = read_cycles();
    BENCH_E2E_PRINT("lstm_lm_spad", bench_general_end - bench_general_start, end - cold_start, total);

    printf("=== LSTM-LM Done ===\n");
    return 0;
}
