// GRU Time Series Forecasting — Real Parameters (Cho et al. 2014)
//
// Original H=1000 → H=1024 (nearest 32-aligned), proj=512
// On 32×32 SA:
//   Proj GEMM: (32×1024) × (1024×512) → K=32 tiles (DRAM-bound)
//   Gate GEMM: (32×512)  × (512×1024) → K=16 tiles (DRAM-bound)
//
// All GEMMs via tiled_matmul_auto (too large for spad preload).
// 4 GEMMs/step × 32 steps = 128 GEMMs. Expect <1% improvement.

#include "spad_gemm_utils.h"

#define TIMESTEPS  32
#define HIDDEN     1024
#define PROJ_DIM   512

// DRAM layout (2MB per weight block)
#define W_STRIDE   0x200000UL
#define W_PROJ_DRAM  ((elem_t *)(DRAM_BASE))                        // [1024×512]
#define W_R_DRAM     ((elem_t *)(DRAM_BASE + 1 * W_STRIDE))        // [512×1024]
#define W_Z_DRAM     ((elem_t *)(DRAM_BASE + 2 * W_STRIDE))        // [512×1024]
#define W_C_DRAM     ((elem_t *)(DRAM_BASE + 3 * W_STRIDE))        // [512×1024]

// Activation buffers in DRAM
#define ACT_BASE     (DRAM_BASE + 4 * W_STRIDE)
#define H_DRAM       ((elem_t *)(ACT_BASE))                         // [32×1024]
#define PROJ_DRAM    ((elem_t *)(ACT_BASE + 0x100000UL))            // [32×512]
#define GATE_DRAM    ((elem_t *)(ACT_BASE + 0x200000UL))            // [32×1024]

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); exit(1); }
#endif
    gemmini_flush(0);
    unsigned long bench_general_start = read_cycles();

    printf("=== GRU Time-Series Real (DIM=%d, H=%d, proj=%d, T=%d) ===\n",
           DIM, HIDDEN, PROJ_DIM, TIMESTEPS);

    gemmini_config_ex(WS, 0, 0);

    unsigned long cold_start = read_cycles();
    unsigned long start = cold_start;  // no spad preload

    for (int t = 0; t < TIMESTEPS; t++) {
        // proj = h(1024) × W_proj(1024×512) → (512)
        tiled_matmul_auto(BATCH, PROJ_DIM, HIDDEN,
            H_DRAM, W_PROJ_DRAM, NULL, PROJ_DRAM,
            HIDDEN, PROJ_DIM, PROJ_DIM, PROJ_DIM,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false, false, false, 0, WS);

        // reset gate
        tiled_matmul_auto(BATCH, HIDDEN, PROJ_DIM,
            PROJ_DRAM, W_R_DRAM, NULL, GATE_DRAM,
            PROJ_DIM, HIDDEN, HIDDEN, HIDDEN,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false, false, false, 0, WS);

        // update gate
        tiled_matmul_auto(BATCH, HIDDEN, PROJ_DIM,
            PROJ_DRAM, W_Z_DRAM, NULL, GATE_DRAM,
            PROJ_DIM, HIDDEN, HIDDEN, HIDDEN,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false, false, false, 0, WS);

        // candidate (RELU)
        tiled_matmul_auto(BATCH, HIDDEN, PROJ_DIM,
            PROJ_DRAM, W_C_DRAM, NULL, GATE_DRAM,
            PROJ_DIM, HIDDEN, HIDDEN, HIDDEN,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            RELU, ACC_SCALE_IDENTITY, 0, false,
            false, false, false, false, 0, WS);
    }

    unsigned long end = read_cycles();
    unsigned long total = end - start;
    int total_gemms = 4 * TIMESTEPS;

    printf("GRU_TS_SPAD DIM=%d H=%d proj=%d timesteps=%d gemms=%d cold_total=%lu warm_total=%lu per_gemm=%lu\n",
           DIM, HIDDEN, PROJ_DIM, TIMESTEPS, total_gemms,
           end - cold_start, total, total / total_gemms);
    unsigned long bench_general_end = read_cycles();
    BENCH_E2E_PRINT("gru_ts_spad", bench_general_end - bench_general_start, end - cold_start, total);

    printf("=== GRU Time-Series Done ===\n");
    return 0;
}
