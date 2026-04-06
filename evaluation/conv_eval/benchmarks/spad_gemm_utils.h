// Shared utilities for spad-only GEMM benchmarks
#ifndef SPAD_GEMM_UTILS_H
#define SPAD_GEMM_UTILS_H

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

#define DRAM_BASE  0x80800000UL
#define D_HIDDEN   (2 * DIM)   // 64
#define BATCH      DIM         // 32

// Load [2*DIM × DIM] narrow weight into spad (2 tiles stacked)
static void load_narrow(const elem_t *base, int stride, uint32_t sp) {
    gemmini_extended_config_ld(stride * sizeof(elem_t), MVIN_SCALE_IDENTITY);
    gemmini_extended_mvin(base, sp, DIM, DIM);
    gemmini_extended_mvin(base + DIM * stride, sp + DIM, DIM, DIM);
}

// Load [DIM × 2*DIM] wide matrix into spad (2 tiles side by side)
static void load_wide(const elem_t *base, int stride, uint32_t sp) {
    gemmini_extended_config_ld(stride * sizeof(elem_t), MVIN_SCALE_IDENTITY);
    gemmini_extended_mvin(base, sp, DIM, DIM);
    gemmini_extended_mvin(base + DIM, sp + DIM, DIM, DIM);
}

// Spad-only GEMM: (BATCH×2D) × (2D×D) → (BATCH×D), I=1,J=1,K=2
#define SPAD_NARROW(a_sp, b_sp_end, c_sp, act) \
    gemmini_loop_ws_spad(1, 1, 2, 0, 0, 0, \
        (a_sp), (b_sp_end), 0, (c_sp), \
        0, 0, 0, 0, 0, (act), 0, 0, 0, 0x38)

// Spad-only GEMM: (BATCH×D) × (D×2D) → (BATCH×2D), I=1,J=2,K=1
#define SPAD_WIDE(a_sp, b_sp_end, c_sp, act) \
    gemmini_loop_ws_spad(1, 2, 1, 0, 0, 0, \
        (a_sp), (b_sp_end), 0, (c_sp), \
        0, 0, 0, 0, 0, (act), 0, 0, 0, 0x38)

// ---- Generalized spad GEMM for arbitrary tile counts ----

// Load [tile_K*DIM × tile_J*DIM] weight matrix from DRAM into spad.
// Spad layout: J-outer, K-inner (matches gemmini_loop_ws_spad B ordering).
static void load_tiles(const elem_t *base, int stride, uint32_t sp,
                       int tile_K, int tile_J) {
    gemmini_extended_config_ld(stride * sizeof(elem_t), MVIN_SCALE_IDENTITY);
    for (int j = 0; j < tile_J; j++)
        for (int k = 0; k < tile_K; k++)
            gemmini_extended_mvin(base + k*DIM*stride + j*DIM,
                                  sp + (j*tile_K + k)*DIM,
                                  DIM, DIM);
}

// Load [DIM × num_tiles*DIM] activation from DRAM into spad (K=1 special case).
static void load_act(const elem_t *base, int stride, uint32_t sp, int num_tiles) {
    gemmini_extended_config_ld(stride * sizeof(elem_t), MVIN_SCALE_IDENTITY);
    for (int t = 0; t < num_tiles; t++)
        gemmini_extended_mvin(base + t*DIM, sp + t*DIM, DIM, DIM);
}

// General spad GEMM: (DIM × K*DIM) × (K*DIM × J*DIM) → (DIM × J*DIM)
// b_sp is the START of B in spad; hardware needs b_end = b_sp + K*J*DIM.
#define SPAD_GEMM(J, K, a_sp, b_sp, c_sp, act) \
    gemmini_loop_ws_spad(1, (J), (K), 0, 0, 0, \
        (a_sp), (b_sp) + (K)*(J)*DIM, 0, (c_sp), \
        0, 0, 0, 0, 0, (act), 0, 0, 0, 0x38)

// Spad rows occupied by a K×J tile block
#define TILE_ROWS(K, J) ((K) * (J) * DIM)

#endif
