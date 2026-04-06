// DeepSpeech2 — Real Parameters (Amodei et al. 2016)
//
// Original H=1760 (already 32-aligned), proj=880
// On 32×32 SA: K=55 tiles per proj GEMM (extremely DRAM-bound)
//
// Conv front-end (DRAM) + 3-layer BiGRU (all DRAM) + FC output (DRAM)
// Conv: 2 GEMMs, BiGRU: 3×2×16×4=384 GEMMs, FC: 1 GEMM
// Total: 387 GEMMs. Expect ~0% improvement.

#include "spad_gemm_utils.h"

#define TIMESTEPS  16
#define NUM_LAYERS 3
#define HIDDEN     1760
#define PROJ_DIM   880

// Conv dimensions
#define CONV1_IN   256
#define CONV1_OUT  HIDDEN
#define CONV2_OUT  HIDDEN

// FC output
#define FC_OUT_DIM 1024

// DRAM layout (2MB per weight block for large matrices)
#define W_STRIDE     0x200000UL

// Conv weights
#define CONV_W1_DRAM ((elem_t *)(DRAM_BASE))
#define CONV_W2_DRAM ((elem_t *)(DRAM_BASE + 1 * W_STRIDE))

// BiGRU weights: 3 layers × 2 directions × 4 (proj + 3 gates) = 24 blocks
#define GRU_W_BASE   (DRAM_BASE + 2 * W_STRIDE)
#define GRU_W(set, idx) ((elem_t *)(GRU_W_BASE + ((set) * 4 + (idx)) * W_STRIDE))
// set = layer*2 + dir, idx = 0:proj, 1:r, 2:z, 3:c

// FC weight
#define FC_W_DRAM    ((elem_t *)(GRU_W_BASE + 24 * W_STRIDE))

// Activation buffers (after all weights)
#define ACT_BASE     (GRU_W_BASE + 25 * W_STRIDE)
#define CONV_IN_DRAM ((elem_t *)(ACT_BASE))
#define CONV_MID     ((elem_t *)(ACT_BASE + 0x100000UL))
#define CONV_OUT_D   ((elem_t *)(ACT_BASE + 0x200000UL))
#define H_DRAM       ((elem_t *)(ACT_BASE + 0x400000UL))
#define PROJ_D       ((elem_t *)(ACT_BASE + 0x600000UL))
#define GATE_DRAM    ((elem_t *)(ACT_BASE + 0x800000UL))
#define FC_OUT_DRAM  ((elem_t *)(ACT_BASE + 0xA00000UL))

// GRU cell: proj + 3 gates = 4 GEMMs (DRAM)
static void gru_step_dram(elem_t *h, elem_t *proj, elem_t *gate,
                          elem_t *w_proj, elem_t *w_r, elem_t *w_z, elem_t *w_c) {
    tiled_matmul_auto(BATCH, PROJ_DIM, HIDDEN,
        h, w_proj, NULL, proj,
        HIDDEN, PROJ_DIM, PROJ_DIM, PROJ_DIM,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
        false, false, false, false, 0, WS);

    tiled_matmul_auto(BATCH, HIDDEN, PROJ_DIM,
        proj, w_r, NULL, gate,
        PROJ_DIM, HIDDEN, HIDDEN, HIDDEN,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
        false, false, false, false, 0, WS);

    tiled_matmul_auto(BATCH, HIDDEN, PROJ_DIM,
        proj, w_z, NULL, gate,
        PROJ_DIM, HIDDEN, HIDDEN, HIDDEN,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
        false, false, false, false, 0, WS);

    tiled_matmul_auto(BATCH, HIDDEN, PROJ_DIM,
        proj, w_c, NULL, gate,
        PROJ_DIM, HIDDEN, HIDDEN, HIDDEN,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        RELU, ACC_SCALE_IDENTITY, 0, false,
        false, false, false, false, 0, WS);
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); exit(1); }
#endif
    gemmini_flush(0);
    unsigned long bench_general_start = read_cycles();

    printf("=== DeepSpeech2 Real (DIM=%d, H=%d, proj=%d, layers=%d, T=%d) ===\n",
           DIM, HIDDEN, PROJ_DIM, NUM_LAYERS, TIMESTEPS);

    gemmini_config_ex(WS, 0, 0);

    unsigned long cold_start = read_cycles();
    unsigned long start = cold_start;

    // ---- Conv front-end (DRAM) ----
    // Conv1: (32×256) × (256×1760) → (32×1760)
    tiled_matmul_auto(BATCH, CONV1_OUT, CONV1_IN,
        CONV_IN_DRAM, CONV_W1_DRAM, NULL, CONV_MID,
        CONV1_IN, CONV1_OUT, CONV1_OUT, CONV1_OUT,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        RELU, ACC_SCALE_IDENTITY, 0, false,
        false, false, false, false, 0, WS);

    // Conv2: (32×1760) × (1760×1760) → (32×1760)
    tiled_matmul_auto(BATCH, CONV2_OUT, HIDDEN,
        CONV_MID, CONV_W2_DRAM, NULL, CONV_OUT_D,
        HIDDEN, CONV2_OUT, CONV2_OUT, CONV2_OUT,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        RELU, ACC_SCALE_IDENTITY, 0, false,
        false, false, false, false, 0, WS);

    unsigned long conv_end = read_cycles();

    // ---- 3-layer BiGRU (all DRAM) ----
    for (int l = 0; l < NUM_LAYERS; l++) {
        // Forward
        int fwd_set = l * 2;
        for (int t = 0; t < TIMESTEPS; t++)
            gru_step_dram(H_DRAM, PROJ_D, GATE_DRAM,
                          GRU_W(fwd_set, 0), GRU_W(fwd_set, 1),
                          GRU_W(fwd_set, 2), GRU_W(fwd_set, 3));
        // Backward
        int bwd_set = l * 2 + 1;
        for (int t = 0; t < TIMESTEPS; t++)
            gru_step_dram(H_DRAM, PROJ_D, GATE_DRAM,
                          GRU_W(bwd_set, 0), GRU_W(bwd_set, 1),
                          GRU_W(bwd_set, 2), GRU_W(bwd_set, 3));
    }

    unsigned long lstm_end = read_cycles();

    // ---- FC output (DRAM) ----
    // h(32×1760) × W(1760×1024) → (32×1024)
    tiled_matmul_auto(BATCH, FC_OUT_DIM, HIDDEN,
        H_DRAM, FC_W_DRAM, NULL, FC_OUT_DRAM,
        HIDDEN, FC_OUT_DIM, FC_OUT_DIM, FC_OUT_DIM,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
        false, false, false, false, 0, WS);

    unsigned long end = read_cycles();

    unsigned long conv_cyc = conv_end - start;
    unsigned long gru_cyc  = lstm_end - conv_end;
    unsigned long fc_cyc   = end - lstm_end;
    unsigned long total    = end - start;

    int gru_gemms  = 4 * TIMESTEPS * NUM_LAYERS * 2;
    int dram_gemms = 2 + 1;  // 2 conv + 1 FC

    printf("DS2_SPAD DIM=%d H=%d proj=%d layers=%d T=%d gru_gemms=%d dram_gemms=%d conv=%lu gru=%lu fc=%lu cold_total=%lu warm_total=%lu gru_frac=%lu%%\n",
           DIM, HIDDEN, PROJ_DIM, NUM_LAYERS, TIMESTEPS, gru_gemms, dram_gemms,
           conv_cyc, gru_cyc, fc_cyc,
           end - cold_start, total, gru_cyc * 100 / total);
    unsigned long bench_general_end = read_cycles();
    BENCH_E2E_PRINT("deepspeech2_spad", bench_general_end - bench_general_start, end - cold_start, total);

    printf("=== DeepSpeech2 Done ===\n");
    return 0;
}
