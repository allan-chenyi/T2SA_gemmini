// BiLSTM-CRF Sequence Labeling — 3-Tile Spad BiLSTM + DRAM Output
//
// Lample et al. 2016, "Neural Architectures for Named Entity Recognition"
// Original H=100 → H=96 (nearest 32-aligned), proj=32 (=DIM).
//
// On 32×32 SA:
//   Proj GEMM: (32×96) × (96×32) → I=1, J=1, K=3   (3 tiles)
//   Gate GEMM: (32×32) × (32×96) → I=1, J=3, K=1   (3 tiles)
//
// 2 directions × 5 GEMMs/step × 16 steps = 160 spad GEMMs + 1 DRAM output.
// K=3 → between 2-tile sweet spot and DRAM-bound. Expect ~6-8% improvement.

#include "spad_gemm_utils.h"

#define TIMESTEPS  16
#define HIDDEN     96
#define PROJ_DIM   DIM      // 32
#define H_TILES    (HIDDEN / DIM)    // 3
#define OUT_DIM    512
#define CONCAT_H   (2 * HIDDEN)     // 192: forward + backward

// Tile counts
#define PROJ_K  H_TILES   // 3: proj = h(96) × W(96×32)
#define PROJ_J  1         // 1
#define GATE_K  1         // 1: gate = proj(32) × W(32×96)
#define GATE_J  H_TILES   // 3

#define W_PROJ_ROWS  TILE_ROWS(PROJ_K, PROJ_J)   // 3×1×32 = 96
#define W_GATE_ROWS  TILE_ROWS(GATE_K, GATE_J)   // 1×3×32 = 96

// DRAM
#define W_BASE(dir, idx)  (DRAM_BASE + ((dir) * 5 + (idx)) * 0x20000UL)
#define H0_DRAM           (DRAM_BASE + 0x200000UL)

// DRAM workspace for output projection
#define OUT_H_DRAM   ((elem_t *)(DRAM_BASE + 0x300000UL))
#define OUT_W_DRAM   ((elem_t *)(DRAM_BASE + 0x400000UL))
#define OUT_Y_DRAM   ((elem_t *)(DRAM_BASE + 0x600000UL))

// Spad layout: 2 directions × 5 weights
// Forward
#define SP_FW_WPROJ  0                                        // 96 rows
#define SP_FW_WI     (SP_FW_WPROJ + W_PROJ_ROWS)             // 96
#define SP_FW_WF     (SP_FW_WI + W_GATE_ROWS)                // 192
#define SP_FW_WO     (SP_FW_WF + W_GATE_ROWS)                // 288
#define SP_FW_WG     (SP_FW_WO + W_GATE_ROWS)                // 384
// Backward
#define SP_BW_WPROJ  (SP_FW_WG + W_GATE_ROWS)                // 480
#define SP_BW_WI     (SP_BW_WPROJ + W_PROJ_ROWS)             // 576
#define SP_BW_WF     (SP_BW_WI + W_GATE_ROWS)                // 672
#define SP_BW_WO     (SP_BW_WF + W_GATE_ROWS)                // 768
#define SP_BW_WG     (SP_BW_WO + W_GATE_ROWS)                // 864
// Activation buffers
#define SP_H0     (SP_BW_WG + W_GATE_ROWS)                   // 960, H_TILES*DIM=96 rows
#define SP_H1     (SP_H0 + H_TILES * DIM)                    // 1056, 96 rows
#define SP_PROJ   (SP_H1 + H_TILES * DIM)                    // 1152, DIM=32 rows
#define SP_GATE   (SP_PROJ + DIM)                             // 1184, H_TILES*DIM=96 rows
// Total: 1280 rows / 32768

static void run_lstm_layer(int timesteps,
                           uint32_t w_proj, uint32_t w_i, uint32_t w_f,
                           uint32_t w_o, uint32_t w_g) {
    for (int t = 0; t < timesteps; t++) {
        uint32_t h_cur = (t % 2 == 0) ? SP_H0 : SP_H1;
        // proj = h(96) × W_proj(96×32) → (32), I=1,J=1,K=3
        SPAD_GEMM(PROJ_J, PROJ_K, h_cur, w_proj, SP_PROJ, NO_ACTIVATION);
        gemmini_fence();
        // 4 gates: proj(32) × W_gate(32×96) → (96), I=1,J=3,K=1
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

    printf("=== BiLSTM-CRF NER (DIM=%d, H=%d, proj=%d, T=%d, out=%d) ===\n",
           DIM, HIDDEN, PROJ_DIM, TIMESTEPS, OUT_DIM);

    unsigned long cold_start = read_cycles();

    // Pre-load forward + backward weights
    uint32_t sp_wproj[] = { SP_FW_WPROJ, SP_BW_WPROJ };
    uint32_t sp_gates[][4] = {
        { SP_FW_WI, SP_FW_WF, SP_FW_WO, SP_FW_WG },
        { SP_BW_WI, SP_BW_WF, SP_BW_WO, SP_BW_WG }
    };

    for (int d = 0; d < 2; d++) {
        // W_proj [96×32]: K=3, J=1
        load_tiles((elem_t *)(W_BASE(d, 0)), PROJ_DIM, sp_wproj[d], PROJ_K, PROJ_J);
        // 4 gate weights [32×96]: K=1, J=3
        for (int g = 0; g < 4; g++)
            load_tiles((elem_t *)(W_BASE(d, 1+g)), HIDDEN, sp_gates[d][g], GATE_K, GATE_J);
    }
    // Load initial h [32×96]
    load_act((elem_t *)H0_DRAM, HIDDEN, SP_H0, H_TILES);
    gemmini_fence();

    gemmini_config_ex(WS, 0, 0);

    unsigned long start = read_cycles();

    // Forward LSTM: spad
    run_lstm_layer(TIMESTEPS, SP_FW_WPROJ, SP_FW_WI, SP_FW_WF, SP_FW_WO, SP_FW_WG);

    unsigned long fw_end = read_cycles();

    // Backward LSTM: spad
    run_lstm_layer(TIMESTEPS, SP_BW_WPROJ, SP_BW_WI, SP_BW_WF, SP_BW_WO, SP_BW_WG);

    unsigned long bw_end = read_cycles();

    // Output projection: DRAM via tiled_matmul_auto
    // concat_h(32×192) × W_out(192×512) → (32×512)
    tiled_matmul_auto(BATCH, OUT_DIM, CONCAT_H,
        OUT_H_DRAM, OUT_W_DRAM, NULL, OUT_Y_DRAM,
        CONCAT_H, OUT_DIM, OUT_DIM, OUT_DIM,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
        false, false, false, false, 0, WS);

    unsigned long end = read_cycles();

    unsigned long fw_cyc  = fw_end - start;
    unsigned long bw_cyc  = bw_end - fw_end;
    unsigned long out_cyc = end - bw_end;
    unsigned long lstm_cyc = bw_end - start;
    unsigned long total   = end - start;

    int spad_gemms = 5 * TIMESTEPS * 2;
    int dram_gemms = 1;

    printf("BILSTM_SPAD DIM=%d H=%d proj=%d T=%d out=%d spad_gemms=%d dram_gemms=%d fw=%lu bw=%lu out=%lu cold_total=%lu warm_total=%lu load_cyc=%lu lstm_frac=%lu%%\n",
           DIM, HIDDEN, PROJ_DIM, TIMESTEPS, OUT_DIM, spad_gemms, dram_gemms,
           fw_cyc, bw_cyc, out_cyc,
           end - cold_start, total, start - cold_start, lstm_cyc * 100 / total);
    unsigned long bench_general_end = read_cycles();
    BENCH_E2E_PRINT("bilstm_spad", bench_general_end - bench_general_start, end - cold_start, total);

    printf("=== BiLSTM Done ===\n");
    return 0;
}
