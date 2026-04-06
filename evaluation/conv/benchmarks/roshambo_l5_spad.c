// RoshamboNet Layer 5: 1x1 pointwise convolution (spad-only compute).
//
// Architecture from Table V of:
//   Urbach & Bhatt, 2019, "RoshamboNet: Training a Single Classifier
//   for Multiple Rock-Paper-Scissors Datasets"
//
// Layer 5 is a 1x1 conv with C_in=128, C_out=128, the only layer
// whose im2col K dimension (K=128) is large enough to fill the
// systolic array efficiently:
//
//   DIM=32:  K_tiles=4, J_tiles=4  (SA utilization ~100%)
//   DIM=64:  K_tiles=2, J_tiles=2  (SA utilization ~100%, sweet spot)
//
// Matmul: [BATCH x 128] x [128 x 128] -> [BATCH x 128]
//
// Multiple iterations of this single layer simulate a deeper
// pointwise conv stack, accumulating T^3 drain savings.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"
#include "bench_e2e.h"

// Layer 5 parameters (from RoshamboNet Table V)
#define CHANNELS    128
#define BATCH       DIM
#define NUM_ITERS   16

// Tile counts
#define C_TILES     (CHANNELS / DIM)    // 4 for DIM=32, 2 for DIM=64

// Spad layout:
//   buf0: [0, C_TILES*DIM)                     activation buffer 0
//   buf1: [C_TILES*DIM, 2*C_TILES*DIM)         activation buffer 1
//   weights: [2*C_TILES*DIM, ...]               weight tiles (J-outer, K-inner)
#define SP_BUF0     0
#define SP_BUF1     (C_TILES * DIM)
#define SP_WEIGHTS  (2 * C_TILES * DIM)
#define W_TILES     (C_TILES * C_TILES)     // K_tiles * J_tiles

// DRAM buffers
static elem_t init_act[BATCH * CHANNELS];
static elem_t weight_mat[CHANNELS * CHANNELS];

int main(int argc, char *argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall failed");
        exit(1);
    }
#endif
    uint64_t bench_general_start = read_cycles();

    printf("RoshamboNet L5 spad (DIM=%d)\n", DIM);
    printf("  1x1 conv: [%d x %d] x [%d x %d] -> [%d x %d]  I=1, J=%d, K=%d\n",
           BATCH, CHANNELS, CHANNELS, CHANNELS, BATCH, CHANNELS,
           C_TILES, C_TILES);
    printf("  Iters: %d  GEMMs: %d\n", NUM_ITERS, NUM_ITERS);
    printf("  Spad: buf0=%d buf1=%d weights=%d total=%d / %d\n",
           C_TILES * DIM, C_TILES * DIM, W_TILES * DIM,
           SP_WEIGHTS + W_TILES * DIM, BANK_NUM * BANK_ROWS);

    gemmini_flush(0);

    // ---- Cold: mvin activation + weights ----
    uint64_t cold_start = read_cycles();

    // mvin initial activation to buf0
    gemmini_extended_config_ld(CHANNELS * sizeof(elem_t), MVIN_SCALE_IDENTITY);
    for (int t = 0; t < C_TILES; t++)
        gemmini_extended_mvin(init_act + t * DIM,
                              SP_BUF0 + t * DIM, DIM, DIM);

    // mvin weights [CHANNELS x CHANNELS]: K=C_TILES, J=C_TILES
    gemmini_extended_config_ld(CHANNELS * sizeof(elem_t), MVIN_SCALE_IDENTITY);
    for (int j = 0; j < C_TILES; j++)
        for (int k = 0; k < C_TILES; k++)
            gemmini_extended_mvin(
                weight_mat + k * DIM * CHANNELS + j * DIM,
                SP_WEIGHTS + (j * C_TILES + k) * DIM,
                DIM, DIM);

    gemmini_fence();
    uint64_t warm_start = read_cycles();

    // ---- Warm: spad-only compute, LD idle ----
    // Alternate buf0 -> buf1 -> buf0 -> ... to simulate depth
    for (int iter = 0; iter < NUM_ITERS; iter++) {
        uint32_t src = (iter % 2 == 0) ? SP_BUF0 : SP_BUF1;
        uint32_t dst = (iter % 2 == 0) ? SP_BUF1 : SP_BUF0;

        gemmini_loop_ws_spad(1, C_TILES, C_TILES, 0, 0, 0,
            src,
            SP_WEIGHTS + W_TILES * DIM,
            0, dst,
            0, 0, 0, 0, 0,
            NO_ACTIVATION, 0, 0, 0,
            0x38);

        gemmini_fence();
    }

    uint64_t end = read_cycles();

    uint64_t cold_cycles = end - cold_start;
    uint64_t warm_cycles = end - warm_start;

    printf("RoshamboNet L5 cold=%llu warm=%llu (iters=%d)\n",
           (unsigned long long)cold_cycles, (unsigned long long)warm_cycles,
           NUM_ITERS);
    printf("COLD_START name=roshambo_l5_spad cycles=%llu\n",
           (unsigned long long)cold_cycles);

    uint64_t bench_general_end = read_cycles();
    BENCH_E2E_PRINT("roshambo_l5_spad",
                    bench_general_end - bench_general_start,
                    cold_cycles, warm_cycles);

    return 0;
}
