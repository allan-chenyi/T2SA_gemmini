// Keyword Spotting — 2-Layer Projection LSTM (Arık et al. 2017)
//
// KWS LSTM: scaled to array dimension.
//   DIM=16 → H=32,  proj=16  (2-tile sweet spot)
//   DIM=32 → H=128, proj=64  (4-tile along K)
//
// 2 layers × 5 GEMMs/step × 16 timesteps = 160 GEMMs, all spad.

#include "spad_gemm_utils.h"

#define TIMESTEPS  16
#define NUM_LAYERS 2

#if DIM <= 16
#define HIDDEN     32
#define PROJ_DIM   16
#else
#define HIDDEN     128
#define PROJ_DIM   64
#endif
#define H_TILES    (HIDDEN / DIM)     // 4
#define P_TILES    (PROJ_DIM / DIM)   // 2

// Tile counts for LSTMP GEMMs
#define PROJ_K  H_TILES   // 4: proj = h(128) × W(128×64)
#define PROJ_J  P_TILES   // 2
#define GATE_K  P_TILES   // 2: gate = proj(64) × W(64×128)
#define GATE_J  H_TILES   // 4

// Spad rows per weight block
#define W_PROJ_ROWS  TILE_ROWS(PROJ_K, PROJ_J)   // 4×2×32 = 256
#define W_GATE_ROWS  TILE_ROWS(GATE_K, GATE_J)   // 2×4×32 = 256

// DRAM
#define W_BASE(layer, idx)  (DRAM_BASE + ((layer) * 5 + (idx)) * 0x40000UL)
#define H0_DRAM             (DRAM_BASE + 0x300000UL)

// Spad layout: 2 layers × (1 proj + 4 gate) weights
// Layer 0
#define SP_L0_WPROJ  0                                          // 256 rows
#define SP_L0_WI     (SP_L0_WPROJ + W_PROJ_ROWS)               // 256
#define SP_L0_WF     (SP_L0_WI + W_GATE_ROWS)                  // 512
#define SP_L0_WO     (SP_L0_WF + W_GATE_ROWS)                  // 768
#define SP_L0_WG     (SP_L0_WO + W_GATE_ROWS)                  // 1024
// Layer 1
#define SP_L1_WPROJ  (SP_L0_WG + W_GATE_ROWS)                  // 1280
#define SP_L1_WI     (SP_L1_WPROJ + W_PROJ_ROWS)               // 1536
#define SP_L1_WF     (SP_L1_WI + W_GATE_ROWS)                  // 1792
#define SP_L1_WO     (SP_L1_WF + W_GATE_ROWS)                  // 2048
#define SP_L1_WG     (SP_L1_WO + W_GATE_ROWS)                  // 2304
// Activation buffers
#define SP_H0     (SP_L1_WG + W_GATE_ROWS)                     // 2560, 128 rows
#define SP_H1     (SP_H0 + H_TILES * DIM)                      // 2688, 128 rows
#define SP_PROJ   (SP_H1 + H_TILES * DIM)                      // 2816, 64 rows
#define SP_GATE   (SP_PROJ + P_TILES * DIM)                     // 2880, 128 rows
// Total: 3008 rows / 32768

static void run_lstm_layer(int timesteps,
                           uint32_t w_proj, uint32_t w_i, uint32_t w_f,
                           uint32_t w_o, uint32_t w_g) {
    for (int t = 0; t < timesteps; t++) {
        uint32_t h_cur = (t % 2 == 0) ? SP_H0 : SP_H1;

        // proj = h(128) × W_proj(128×64) → (64), I=1,J=2,K=4
        SPAD_GEMM(PROJ_J, PROJ_K, h_cur, w_proj, SP_PROJ, NO_ACTIVATION);
        gemmini_fence();
        // 4 gates: proj(64) × W_gate(64×128) → (128), I=1,J=4,K=2
        SPAD_GEMM(GATE_J, GATE_K, SP_PROJ, w_i, SP_GATE, NO_ACTIVATION);
        gemmini_fence();
        SPAD_GEMM(GATE_J, GATE_K, SP_PROJ, w_f, SP_GATE, NO_ACTIVATION);
        gemmini_fence();
        SPAD_GEMM(GATE_J, GATE_K, SP_PROJ, w_o, SP_GATE, NO_ACTIVATION);
        gemmini_fence();
        SPAD_GEMM(GATE_J, GATE_K, SP_PROJ, w_g, SP_GATE, RELU);
        gemmini_fence();
    }
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); exit(1); }
#endif
    gemmini_flush(0);
    unsigned long bench_general_start = read_cycles();

    printf("=== KWS 2-Layer LSTM (DIM=%d, H=%d, proj=%d, T=%d) ===\n",
           DIM, HIDDEN, PROJ_DIM, TIMESTEPS);

    unsigned long cold_start = read_cycles();

    // Pre-load all weights using generalized tile loader
    uint32_t sp_wproj[] = { SP_L0_WPROJ, SP_L1_WPROJ };
    uint32_t sp_gates[][4] = {
        { SP_L0_WI, SP_L0_WF, SP_L0_WO, SP_L0_WG },
        { SP_L1_WI, SP_L1_WF, SP_L1_WO, SP_L1_WG }
    };

    for (int l = 0; l < NUM_LAYERS; l++) {
        // W_proj [128×64]: K=4, J=2
        load_tiles((elem_t *)(W_BASE(l, 0)), PROJ_DIM, sp_wproj[l], PROJ_K, PROJ_J);
        // 4 gate weights [64×128]: K=2, J=4
        for (int g = 0; g < 4; g++)
            load_tiles((elem_t *)(W_BASE(l, 1+g)), HIDDEN, sp_gates[l][g], GATE_K, GATE_J);
    }
    // Load initial h [32×128]
    load_act((elem_t *)H0_DRAM, HIDDEN, SP_H0, H_TILES);
    gemmini_fence();

    gemmini_config_ex(WS, 0, 0);

    unsigned long start = read_cycles();

    run_lstm_layer(TIMESTEPS, SP_L0_WPROJ, SP_L0_WI, SP_L0_WF, SP_L0_WO, SP_L0_WG);
    run_lstm_layer(TIMESTEPS, SP_L1_WPROJ, SP_L1_WI, SP_L1_WF, SP_L1_WO, SP_L1_WG);

    unsigned long end = read_cycles();
    unsigned long total = end - start;
    int total_gemms = 5 * TIMESTEPS * NUM_LAYERS;

    printf("KWS_LSTM_SPAD DIM=%d H=%d proj=%d layers=%d timesteps=%d gemms=%d cold_total=%lu warm_total=%lu load_cyc=%lu per_gemm=%lu\n",
           DIM, HIDDEN, PROJ_DIM, NUM_LAYERS, TIMESTEPS, total_gemms,
           end - cold_start, total, start - cold_start, total / total_gemms);
    unsigned long bench_general_end = read_cycles();
    BENCH_E2E_PRINT("kws_lstm_spad", bench_general_end - bench_general_start, end - cold_start, total);

    printf("=== KWS LSTM Done ===\n");
    return 0;
}
