// DLRM Bottom-MLP — Real 3-Layer MLP (Naumov et al. 2019)
//
// Original DLRM bottom MLP: 512 → 256 → 64 (→ 32 output projection)
// On 32×32 SA:
//   Layer 0: (32×512) × (512×256) → DRAM, 128 tiles along K  (DRAM-bound)
//   Layer 1: (32×256) × (256×64)  → DRAM,  16 tiles           (DRAM-bound)
//   Layer 2: (32×64)  × (64×32)   → DRAM,   2 tiles           (2-tile sweet spot)
//
// 3 GEMMs total, all via tiled_matmul_auto (DRAM).
// Expected: <1% improvement (dominated by large DRAM-bound layers).

#include "spad_gemm_utils.h"

#define NUM_LAYERS 3

// MLP dimensions
#define DIM0 512
#define DIM1 256
#define DIM2 64
#define DIM3 DIM    // output = 32

// DRAM layout (2MB spacing per weight, plenty of room)
#define W_STRIDE   0x100000UL
#define W0_DRAM    ((elem_t *)(DRAM_BASE))                        // [512×256]
#define W1_DRAM    ((elem_t *)(DRAM_BASE + 1 * W_STRIDE))        // [256×64]
#define W2_DRAM    ((elem_t *)(DRAM_BASE + 2 * W_STRIDE))        // [64×32]
#define X_DRAM     ((elem_t *)(DRAM_BASE + 3 * W_STRIDE))        // input  [32×512]
#define A0_DRAM    ((elem_t *)(DRAM_BASE + 4 * W_STRIDE))        // after L0 [32×256]
#define A1_DRAM    ((elem_t *)(DRAM_BASE + 5 * W_STRIDE))        // after L1 [32×64]
#define Y_DRAM     ((elem_t *)(DRAM_BASE + 6 * W_STRIDE))        // output [32×32]

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); exit(1); }
#endif
    gemmini_flush(0);
    unsigned long bench_general_start = read_cycles();

    printf("=== DLRM Bottom-MLP Real (DIM=%d, %d->%d->%d->%d, batch=%d) ===\n",
           DIM, DIM0, DIM1, DIM2, DIM3, BATCH);

    gemmini_config_ex(WS, 0, 0);

    unsigned long cold_start = read_cycles();
    unsigned long start = cold_start;  // no spad preload

    // Layer 0: (32×512) × (512×256) → (32×256), RELU
    tiled_matmul_auto(BATCH, DIM1, DIM0,
        X_DRAM, W0_DRAM, NULL, A0_DRAM,
        DIM0, DIM1, DIM1, DIM1,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        RELU, ACC_SCALE_IDENTITY, 0, false,
        false, false, false, false, 0, WS);

    // Layer 1: (32×256) × (256×64) → (32×64), RELU
    tiled_matmul_auto(BATCH, DIM2, DIM1,
        A0_DRAM, W1_DRAM, NULL, A1_DRAM,
        DIM1, DIM2, DIM2, DIM2,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        RELU, ACC_SCALE_IDENTITY, 0, false,
        false, false, false, false, 0, WS);

    // Layer 2: (32×64) × (64×32) → (32×32)
    tiled_matmul_auto(BATCH, DIM3, DIM2,
        A1_DRAM, W2_DRAM, NULL, Y_DRAM,
        DIM2, DIM3, DIM3, DIM3,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
        false, false, false, false, 0, WS);

    unsigned long end = read_cycles();
    unsigned long total = end - start;

    printf("DLRM_MLP_SPAD DIM=%d layers=%d dims=%d/%d/%d/%d gemms=%d cold_total=%lu warm_total=%lu per_gemm=%lu\n",
           DIM, NUM_LAYERS, DIM0, DIM1, DIM2, DIM3, NUM_LAYERS,
           end - cold_start, total, total / NUM_LAYERS);
    unsigned long bench_general_end = read_cycles();
    BENCH_E2E_PRINT("dlrm_mlp_spad", bench_general_end - bench_general_start, end - cold_start, total);

    printf("=== DLRM MLP Done ===\n");
    return 0;
}
