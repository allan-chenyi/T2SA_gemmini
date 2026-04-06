// ResNet-style bottleneck block benchmark (spad-only compute).
//
// Each block does two 1x1 pointwise convolutions:
//   reduce: [BATCH x C_WIDE] x [C_WIDE x C_NARROW] -> [BATCH x C_NARROW]
//   expand: [BATCH x C_NARROW] x [C_NARROW x C_WIDE] -> [BATCH x C_WIDE]
//
// Multiple blocks chain layer-by-layer with spad-resident activations,
// creating multiple drains to accumulate T³ savings.
//
// References:
//   - He et al. 2016, "Deep Residual Learning for Image Recognition",
//     CVPR, arXiv:1512.03385 — 4:1 bottleneck ratio

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

// Bottleneck parameters (He et al. 2016 ratio 4:1)
#define C_WIDE       (4 * DIM)   // 128 for DIM=32
#define C_NARROW     DIM         // 32 for DIM=32
#define BATCH        DIM         // 32 rows of activation
#define NUM_BLOCKS   8           // 8 blocks = 16 GEMMs

// Tile dimensions
#define W_TILES      (C_WIDE / DIM)    // 4
#define N_TILES      (C_NARROW / DIM)  // 1

// Reduce weight: [C_WIDE x C_NARROW] = W_TILES x N_TILES tiles = 4 tiles
// Expand weight: [C_NARROW x C_WIDE] = N_TILES x W_TILES tiles = 4 tiles
// Weight tiles per block = 4 + 4 = 8
#define REDUCE_TILES  (W_TILES * N_TILES)   // 4
#define EXPAND_TILES  (N_TILES * W_TILES)   // 4
#define TILES_PER_BLK (REDUCE_TILES + EXPAND_TILES) // 8
#define ROWS_PER_BLK  (TILES_PER_BLK * DIM)         // 256

// Spad layout:
//   buf0: [0, W_TILES*DIM)            wide activation (128 rows)
//   buf1: [W_TILES*DIM, (W+N)*DIM)    narrow activation (32 rows)
//   weights: [(W+N)*DIM, ...]          all block weights stacked
#define SP_BUF0   0
#define SP_BUF1   (W_TILES * DIM)
#define SP_WBASE  ((W_TILES + N_TILES) * DIM)

// Total spad needed: 160 + 8*256 = 2208 rows (out of 32768)

// DRAM buffers for initial activation and all weights
static elem_t init_act[BATCH * C_WIDE];

// All weights: NUM_BLOCKS * (reduce_weight + expand_weight)
// reduce: [C_WIDE x C_NARROW], expand: [C_NARROW x C_WIDE]
static elem_t reduce_weights[NUM_BLOCKS][C_WIDE * C_NARROW];
static elem_t expand_weights[NUM_BLOCKS][C_NARROW * C_WIDE];

int main(int argc, char *argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall failed");
        exit(1);
    }
#endif
    uint64_t bench_general_start = read_cycles();

    printf("ResNet bottleneck spad (DIM=%d)\n", DIM);
    printf("  C_WIDE=%d C_NARROW=%d BATCH=%d NUM_BLOCKS=%d\n",
           C_WIDE, C_NARROW, BATCH, NUM_BLOCKS);
    printf("  Reduce: [%d x %d] x [%d x %d] -> [%d x %d]  (I=1, J=%d, K=%d)\n",
           BATCH, C_WIDE, C_WIDE, C_NARROW, BATCH, C_NARROW, N_TILES, W_TILES);
    printf("  Expand: [%d x %d] x [%d x %d] -> [%d x %d]  (I=1, J=%d, K=%d)\n",
           BATCH, C_NARROW, C_NARROW, C_WIDE, BATCH, C_WIDE, W_TILES, N_TILES);
    printf("  Spad: buf0=%d buf1=%d wbase=%d total=%d / %d\n",
           W_TILES * DIM, N_TILES * DIM, SP_WBASE,
           SP_WBASE + NUM_BLOCKS * ROWS_PER_BLK, BANK_NUM * BANK_ROWS);
    printf("  GEMMs: %d blocks x 2 = %d total\n", NUM_BLOCKS, NUM_BLOCKS * 2);

    gemmini_flush(0);

    // ---- Cold: mvin initial activation + all weights ----
    uint64_t cold_start = read_cycles();

    // mvin initial wide activation to buf0
    gemmini_extended_config_ld(C_WIDE * sizeof(elem_t), MVIN_SCALE_IDENTITY);
    for (int t = 0; t < W_TILES; t++)
        gemmini_extended_mvin(init_act + t * DIM, SP_BUF0 + t * DIM, DIM, DIM);

    // mvin all weights
    for (int blk = 0; blk < NUM_BLOCKS; blk++) {
        uint32_t w_sp = SP_WBASE + blk * ROWS_PER_BLK;

        // Reduce weight [C_WIDE x C_NARROW]: K=W_TILES, J=N_TILES
        // Spad B layout: J-outer, K-inner
        gemmini_extended_config_ld(C_NARROW * sizeof(elem_t), MVIN_SCALE_IDENTITY);
        for (int j = 0; j < N_TILES; j++)
            for (int k = 0; k < W_TILES; k++)
                gemmini_extended_mvin(
                    reduce_weights[blk] + k * DIM * C_NARROW + j * DIM,
                    w_sp + (j * W_TILES + k) * DIM,
                    DIM, DIM);

        // Expand weight [C_NARROW x C_WIDE]: K=N_TILES, J=W_TILES
        uint32_t exp_sp = w_sp + REDUCE_TILES * DIM;
        gemmini_extended_config_ld(C_WIDE * sizeof(elem_t), MVIN_SCALE_IDENTITY);
        for (int j = 0; j < W_TILES; j++)
            for (int k = 0; k < N_TILES; k++)
                gemmini_extended_mvin(
                    expand_weights[blk] + k * DIM * C_WIDE + j * DIM,
                    exp_sp + (j * N_TILES + k) * DIM,
                    DIM, DIM);
    }

    gemmini_fence();
    uint64_t warm_start = read_cycles();

    // ---- Warm: spad-only compute, LD idle ----
    for (int blk = 0; blk < NUM_BLOCKS; blk++) {
        uint32_t w_sp = SP_WBASE + blk * ROWS_PER_BLK;
        uint32_t reduce_b_sp = w_sp;
        uint32_t expand_b_sp = w_sp + REDUCE_TILES * DIM;

        // Reduce: buf0[BATCH x C_WIDE] x W_reduce -> buf1[BATCH x C_NARROW]
        // I=1, J=N_TILES(1), K=W_TILES(4)
        // b_end = b_sp + K*J*DIM
        gemmini_loop_ws_spad(1, N_TILES, W_TILES, 0, 0, 0,
            SP_BUF0,
            reduce_b_sp + W_TILES * N_TILES * DIM,
            0, SP_BUF1,
            0, 0, 0, 0, 0,
            NO_ACTIVATION, 0, 0, 0,
            0x38);

        gemmini_fence();

        // Expand: buf1[BATCH x C_NARROW] x W_expand -> buf0[BATCH x C_WIDE]
        // I=1, J=W_TILES(4), K=N_TILES(1)
        // b_end = b_sp + K*J*DIM
        gemmini_loop_ws_spad(1, W_TILES, N_TILES, 0, 0, 0,
            SP_BUF1,
            expand_b_sp + N_TILES * W_TILES * DIM,
            0, SP_BUF0,
            0, 0, 0, 0, 0,
            NO_ACTIVATION, 0, 0, 0,
            0x38);

        gemmini_fence();
    }

    uint64_t end = read_cycles();

    uint64_t cold_cycles = end - cold_start;
    uint64_t warm_cycles = end - warm_start;

    printf("ResNet bottleneck cold=%llu warm=%llu (blocks=%d, gemms=%d)\n",
           (unsigned long long)cold_cycles, (unsigned long long)warm_cycles,
           NUM_BLOCKS, NUM_BLOCKS * 2);
    printf("COLD_START name=resnet_bottleneck_spad cycles=%llu\n",
           (unsigned long long)cold_cycles);

    uint64_t bench_general_end = read_cycles();
    BENCH_E2E_PRINT("resnet_bottleneck_spad",
                    bench_general_end - bench_general_start,
                    cold_cycles, warm_cycles);

    return 0;
}
