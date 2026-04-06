// RoshamboNet full 5-layer CNN (spad-only compute).
//
// Architecture from Table V of:
//   Urbach & Bhatt, 2019, "RoshamboNet: Training a Single Classifier
//   for Multiple Rock-Paper-Scissors Datasets"
//
// Each layer's im2col activation is pre-computed on CPU and pre-loaded
// to spad along with all weights. Warm phase is 5 sequential GEMMs
// (LD idle), each producing one drain for T^3 savings.
//
// Layer params (Table V):
//   L1: C_in=1,   C_out=16,  5x5  -> im2col K= 1*25 = 25
//   L2: C_in=16,  C_out=32,  3x3  -> im2col K=16* 9 =144
//   L3: C_in=32,  C_out=64,  3x3  -> im2col K=32* 9 =288
//   L4: C_in=64,  C_out=128, 3x3  -> im2col K=64* 9 =576
//   L5: C_in=128, C_out=128, 1x1  -> im2col K=128*1 =128
//
// DIM=32 tile counts:
//   L1: K=1, J=1   (1 tile weight)
//   L2: K=5, J=1   (5 tiles)
//   L3: K=9, J=2   (18 tiles)
//   L4: K=18,J=4   (72 tiles)
//   L5: K=4, J=4   (16 tiles)
//
// 5 GEMMs -> expected T^3 saving = 5 x (DIM-1) cycles

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

#define NUM_LAYERS  5
#define BATCH       DIM

// Per-layer im2col K and output channels (from Table V)
static const int IM2COL_K[NUM_LAYERS] = {25, 144, 288, 576, 128};
static const int OUT_CH[NUM_LAYERS]   = {16, 32, 64, 128, 128};

// ceil division
static int cdiv(int a, int b) { return (a + b - 1) / b; }

// Max DRAM buffer: layer 4 activation = BATCH * 576, weight = 576 * 128
#define MAX_K_PAD  (18 * DIM)   // 576 for DIM=32
#define MAX_J_PAD  (4 * DIM)    // 128 for DIM=32
static elem_t act_dram[NUM_LAYERS][BATCH * MAX_K_PAD];
static elem_t wgt_dram[NUM_LAYERS][MAX_K_PAD * MAX_J_PAD];

int main(int argc, char *argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall failed");
        exit(1);
    }
#endif
    uint64_t bench_general_start = read_cycles();

    // Compute per-layer tile counts and spad offsets
    int kt[NUM_LAYERS], jt[NUM_LAYERS];
    int k_pad[NUM_LAYERS], j_pad[NUM_LAYERS];
    int a_rows[NUM_LAYERS], b_rows[NUM_LAYERS];
    uint32_t sp_a[NUM_LAYERS], sp_b[NUM_LAYERS];

    for (int i = 0; i < NUM_LAYERS; i++) {
        kt[i] = cdiv(IM2COL_K[i], DIM);
        jt[i] = cdiv(OUT_CH[i], DIM);
        k_pad[i] = kt[i] * DIM;
        j_pad[i] = jt[i] * DIM;
        a_rows[i] = kt[i] * DIM;       // I=1, so A = K_tiles tiles
        b_rows[i] = kt[i] * jt[i] * DIM; // B = K_tiles * J_tiles tiles
    }

    // Spad layout: all A regions, then all B regions
    //   A[0], A[1], ..., A[4], B[0], B[1], ..., B[4]
    // C output for each layer overlaps its own A region (consumed before overwritten)
    uint32_t offset = 0;
    for (int i = 0; i < NUM_LAYERS; i++) {
        sp_a[i] = offset;
        offset += a_rows[i];
    }
    for (int i = 0; i < NUM_LAYERS; i++) {
        sp_b[i] = offset;
        offset += b_rows[i];
    }

    int total_spad = offset;

    printf("RoshamboNet full 5-layer spad (DIM=%d)\n", DIM);
    for (int i = 0; i < NUM_LAYERS; i++) {
        printf("  L%d: K=%d J=%d (K_tiles=%d, J_tiles=%d)  "
               "sp_a=%d(%d rows) sp_b=%d(%d rows)\n",
               i+1, IM2COL_K[i], OUT_CH[i], kt[i], jt[i],
               sp_a[i], a_rows[i], sp_b[i], b_rows[i]);
    }
    printf("  Total spad: %d / %d rows\n", total_spad, BANK_NUM * BANK_ROWS);
    printf("  GEMMs: %d  expected T3 saving: %d cycles\n",
           NUM_LAYERS, NUM_LAYERS * (DIM - 1));

    gemmini_flush(0);

    // ---- Cold: mvin all activations + all weights ----
    uint64_t cold_start = read_cycles();

    for (int i = 0; i < NUM_LAYERS; i++) {
        // mvin A[i]: I=1, K=kt[i] tiles, stride = k_pad[i]
        gemmini_extended_config_ld(k_pad[i] * sizeof(elem_t), MVIN_SCALE_IDENTITY);
        for (int k = 0; k < kt[i]; k++)
            gemmini_extended_mvin(
                act_dram[i] + k * DIM,
                sp_a[i] + k * DIM,
                DIM, DIM);

        // mvin B[i]: J-outer, K-inner layout, stride = j_pad[i]
        gemmini_extended_config_ld(j_pad[i] * sizeof(elem_t), MVIN_SCALE_IDENTITY);
        for (int j = 0; j < jt[i]; j++)
            for (int k = 0; k < kt[i]; k++)
                gemmini_extended_mvin(
                    wgt_dram[i] + k * DIM * j_pad[i] + j * DIM,
                    sp_b[i] + (j * kt[i] + k) * DIM,
                    DIM, DIM);
    }

    gemmini_fence();
    uint64_t warm_start = read_cycles();

    // ---- Warm: 5 sequential spad-only GEMMs, LD idle ----
    for (int i = 0; i < NUM_LAYERS; i++) {
        // GEMM: A[i] x B[i] -> C at sp_a[i] (overlaps A, consumed before overwrite)
        // I=1, J=jt[i], K=kt[i]
        // b_end = sp_b[i] + kt[i]*jt[i]*DIM
        gemmini_loop_ws_spad(1, jt[i], kt[i], 0, 0, 0,
            sp_a[i],
            sp_b[i] + b_rows[i],
            0, sp_a[i],
            0, 0, 0, 0, 0,
            NO_ACTIVATION, 0, 0, 0,
            0x38);

        gemmini_fence();
    }

    uint64_t end = read_cycles();

    uint64_t cold_cycles = end - cold_start;
    uint64_t warm_cycles = end - warm_start;

    printf("RoshamboNet full cold=%llu warm=%llu (layers=%d)\n",
           (unsigned long long)cold_cycles, (unsigned long long)warm_cycles,
           NUM_LAYERS);
    printf("COLD_START name=roshambo_full_spad cycles=%llu\n",
           (unsigned long long)cold_cycles);

    uint64_t bench_general_end = read_cycles();
    BENCH_E2E_PRINT("roshambo_full_spad",
                    bench_general_end - bench_general_start,
                    cold_cycles, warm_cycles);

    return 0;
}
