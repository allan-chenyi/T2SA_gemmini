// Stacked LSTM for Speech — 5-Layer Projection LSTM, Spad-Only
//
// Based on Sak et al. 2014 deep LSTMP for speech recognition.
// 5 layers × 5 GEMMs/step × 16 timesteps = 400 GEMMs, all 2-tile, spad-only.
//
// Each layer: projection (64→32) + 4 gates (32→64)
// hidden=64, proj=32, batch=32, timesteps=16

#include "spad_gemm_utils.h"

#define TIMESTEPS  16
#define NUM_LAYERS 5

// DRAM
#define W_BASE(layer, idx)  (DRAM_BASE + ((layer) * 5 + (idx)) * 0x10000UL)
#define H0_DRAM             (DRAM_BASE + 0x200000UL)

// Spad layout: 5 layers × 5 weights = 25 weight blocks (64 rows each)
// Layer 0
#define SP_L0_WPROJ  0
#define SP_L0_WI     64
#define SP_L0_WF     128
#define SP_L0_WO     192
#define SP_L0_WG     256
// Layer 1
#define SP_L1_WPROJ  320
#define SP_L1_WI     384
#define SP_L1_WF     448
#define SP_L1_WO     512
#define SP_L1_WG     576
// Layer 2
#define SP_L2_WPROJ  640
#define SP_L2_WI     704
#define SP_L2_WF     768
#define SP_L2_WO     832
#define SP_L2_WG     896
// Layer 3
#define SP_L3_WPROJ  960
#define SP_L3_WI     1024
#define SP_L3_WF     1088
#define SP_L3_WO     1152
#define SP_L3_WG     1216
// Layer 4
#define SP_L4_WPROJ  1280
#define SP_L4_WI     1344
#define SP_L4_WF     1408
#define SP_L4_WO     1472
#define SP_L4_WG     1536
// Activation buffers
#define SP_H0     1600    // h buf0 [32×64]: 64 rows
#define SP_H1     1664    // h buf1 [32×64]: 64 rows
#define SP_PROJ   1728    // proj [32×32]: 32 rows
#define SP_GATE   1760    // gate [32×64]: 64 rows
// Total: 1824 rows / 32768 (32x32 spad)

static void run_lstm_layer(int timesteps,
                           uint32_t w_proj, uint32_t w_i, uint32_t w_f,
                           uint32_t w_o, uint32_t w_g) {
    for (int t = 0; t < timesteps; t++) {
        uint32_t h_cur = (t % 2 == 0) ? SP_H0 : SP_H1;

        SPAD_NARROW(h_cur, w_proj + 2*DIM, SP_PROJ, NO_ACTIVATION);
        gemmini_fence();
        SPAD_WIDE(SP_PROJ, w_i + 2*DIM, SP_GATE, NO_ACTIVATION);
        gemmini_fence();
        SPAD_WIDE(SP_PROJ, w_f + 2*DIM, SP_GATE, NO_ACTIVATION);
        gemmini_fence();
        SPAD_WIDE(SP_PROJ, w_o + 2*DIM, SP_GATE, NO_ACTIVATION);
        gemmini_fence();
        SPAD_WIDE(SP_PROJ, w_g + 2*DIM, SP_GATE, RELU);
        gemmini_fence();
    }
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); exit(1); }
#endif
    gemmini_flush(0);
    unsigned long bench_general_start = read_cycles();

    printf("=== Stacked 5-Layer LSTM Spad-Only (DIM=%d, layers=%d, T=%d) ===\n",
           DIM, NUM_LAYERS, TIMESTEPS);

    unsigned long cold_start = read_cycles();

    // Pre-load all weights (25 blocks)
    uint32_t sp_addrs[] = {
        SP_L0_WPROJ, SP_L0_WI, SP_L0_WF, SP_L0_WO, SP_L0_WG,
        SP_L1_WPROJ, SP_L1_WI, SP_L1_WF, SP_L1_WO, SP_L1_WG,
        SP_L2_WPROJ, SP_L2_WI, SP_L2_WF, SP_L2_WO, SP_L2_WG,
        SP_L3_WPROJ, SP_L3_WI, SP_L3_WF, SP_L3_WO, SP_L3_WG,
        SP_L4_WPROJ, SP_L4_WI, SP_L4_WF, SP_L4_WO, SP_L4_WG
    };
    for (int i = 0; i < 25; i++) {
        if (i % 5 == 0)
            load_narrow((elem_t *)(W_BASE(i/5, i%5)), DIM, sp_addrs[i]);
        else
            load_wide((elem_t *)(W_BASE(i/5, i%5)), D_HIDDEN, sp_addrs[i]);
    }
    load_wide((elem_t *)H0_DRAM, D_HIDDEN, SP_H0);
    gemmini_fence();

    gemmini_config_ex(WS, 0, 0);

    unsigned long start = read_cycles();

    run_lstm_layer(TIMESTEPS, SP_L0_WPROJ, SP_L0_WI, SP_L0_WF, SP_L0_WO, SP_L0_WG);
    run_lstm_layer(TIMESTEPS, SP_L1_WPROJ, SP_L1_WI, SP_L1_WF, SP_L1_WO, SP_L1_WG);
    run_lstm_layer(TIMESTEPS, SP_L2_WPROJ, SP_L2_WI, SP_L2_WF, SP_L2_WO, SP_L2_WG);
    run_lstm_layer(TIMESTEPS, SP_L3_WPROJ, SP_L3_WI, SP_L3_WF, SP_L3_WO, SP_L3_WG);
    run_lstm_layer(TIMESTEPS, SP_L4_WPROJ, SP_L4_WI, SP_L4_WF, SP_L4_WO, SP_L4_WG);

    unsigned long end = read_cycles();
    unsigned long total = end - start;
    int total_gemms = 5 * TIMESTEPS * NUM_LAYERS;

    printf("STACKED_LSTM_SPAD DIM=%d layers=%d timesteps=%d gemms=%d cold_total=%lu warm_total=%lu load_cyc=%lu per_gemm=%lu\n",
           DIM, NUM_LAYERS, TIMESTEPS, total_gemms,
           end - cold_start, total, start - cold_start, total / total_gemms);
    unsigned long bench_general_end = read_cycles();
    BENCH_E2E_PRINT("stacked_lstm_spad", bench_general_end - bench_general_start, end - cold_start, total);

    printf("=== Stacked LSTM Done ===\n");
    return 0;
}
