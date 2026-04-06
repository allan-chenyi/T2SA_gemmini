// LeNet-5 fully-connected layers F6 + Output (spad-only compute).
//
// Original architecture from:
//   LeCun et al. 1998, "Gradient-Based Learning Applied to Document
//   Recognition", Proc. IEEE, vol. 86, no. 11, pp. 2278-2324.
//
// Layer parameters (directly from Section II-B):
//   F6:  "contains 84 units ... fully connected to C5"  → 120 -> 84
//   Out: "10 RBF units ... 84 inputs each"              → 84 -> 10
//
// Matmul equivalence (DIM=32):
//   F6:  [32 x 120] x [120 x  84] -> [32 x  84]  I=1, J=3, K=4
//   Out: [32 x  84] x [ 84 x  10] -> [32 x  10]  I=1, J=1, K=3
//
// Multiple forward passes accumulate T^3 drain savings:
//   NUM_PASSES passes x 2 GEMMs/pass = 2*NUM_PASSES drains total.

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

// Layer dimensions (from LeCun et al. 1998, Section II-B)
#define IN_F6       120
#define OUT_F6      84
#define OUT_FINAL   10
#define BATCH       DIM
#define NUM_PASSES  8

// Tile counts (padded to DIM boundary)
#define K_F6    ((IN_F6 + DIM - 1) / DIM)       // ceil(120/32)=4
#define J_F6    ((OUT_F6 + DIM - 1) / DIM)      // ceil(84/32)=3
#define K_OUT   ((OUT_F6 + DIM - 1) / DIM)      // ceil(84/32)=3
#define J_OUT   ((OUT_FINAL + DIM - 1) / DIM)   // ceil(10/32)=1

// Padded dimensions
#define IN_F6_PAD    (K_F6 * DIM)
#define OUT_F6_PAD   (J_F6 * DIM)
#define OUT_FINAL_PAD (J_OUT * DIM)

// Spad layout:
//   buf_in:  [0, K_F6*DIM)                   F6 input activation
//   buf_mid: [K_F6*DIM, ...]                 F6 output / Out input
//   buf_out: [(K_F6+J_F6)*DIM, ...]          final output
//   W_F6:   after buffers                     F6 weight tiles
//   W_Out:  after W_F6                        Out weight tiles
#define SP_BUF_IN    0
#define SP_BUF_MID   (K_F6 * DIM)
#define SP_BUF_OUT   ((K_F6 + J_F6) * DIM)
#define SP_W_F6      ((K_F6 + J_F6 + J_OUT) * DIM)
#define W_F6_TILES   (K_F6 * J_F6)
#define SP_W_OUT     (SP_W_F6 + W_F6_TILES * DIM)
#define W_OUT_TILES  (K_OUT * J_OUT)

// DRAM buffers
static elem_t init_act[BATCH * IN_F6_PAD];
static elem_t w_f6[IN_F6_PAD * OUT_F6_PAD];
static elem_t w_out[OUT_F6_PAD * OUT_FINAL_PAD];

int main(int argc, char *argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall failed");
        exit(1);
    }
#endif
    uint64_t bench_general_start = read_cycles();

    printf("LeNet-5 FC spad (DIM=%d)\n", DIM);
    printf("  F6:  [%d x %d] x [%d x %d] -> [%d x %d]  I=1, J=%d, K=%d\n",
           BATCH, IN_F6, IN_F6_PAD, OUT_F6_PAD, BATCH, OUT_F6, J_F6, K_F6);
    printf("  Out: [%d x %d] x [%d x %d] -> [%d x %d]  I=1, J=%d, K=%d\n",
           BATCH, OUT_F6, OUT_F6_PAD, OUT_FINAL_PAD, BATCH, OUT_FINAL, J_OUT, K_OUT);
    printf("  Passes: %d  GEMMs: %d\n", NUM_PASSES, NUM_PASSES * 2);
    printf("  Spad: buf_in=%d buf_mid=%d buf_out=%d w_f6=%d w_out=%d total=%d / %d\n",
           K_F6 * DIM, J_F6 * DIM, J_OUT * DIM,
           W_F6_TILES * DIM, W_OUT_TILES * DIM,
           SP_W_OUT + W_OUT_TILES * DIM, BANK_NUM * BANK_ROWS);

    gemmini_flush(0);

    // ---- Cold: mvin activation + all weights ----
    uint64_t cold_start = read_cycles();

    // mvin input activation to buf_in
    gemmini_extended_config_ld(IN_F6_PAD * sizeof(elem_t), MVIN_SCALE_IDENTITY);
    for (int k = 0; k < K_F6; k++)
        gemmini_extended_mvin(init_act + k * DIM,
                              SP_BUF_IN + k * DIM, DIM, DIM);

    // mvin F6 weights [IN_F6_PAD x OUT_F6_PAD]: K=K_F6, J=J_F6
    gemmini_extended_config_ld(OUT_F6_PAD * sizeof(elem_t), MVIN_SCALE_IDENTITY);
    for (int j = 0; j < J_F6; j++)
        for (int k = 0; k < K_F6; k++)
            gemmini_extended_mvin(
                w_f6 + k * DIM * OUT_F6_PAD + j * DIM,
                SP_W_F6 + (j * K_F6 + k) * DIM,
                DIM, DIM);

    // mvin Out weights [OUT_F6_PAD x OUT_FINAL_PAD]: K=K_OUT, J=J_OUT
    gemmini_extended_config_ld(OUT_FINAL_PAD * sizeof(elem_t), MVIN_SCALE_IDENTITY);
    for (int j = 0; j < J_OUT; j++)
        for (int k = 0; k < K_OUT; k++)
            gemmini_extended_mvin(
                w_out + k * DIM * OUT_FINAL_PAD + j * DIM,
                SP_W_OUT + (j * K_OUT + k) * DIM,
                DIM, DIM);

    gemmini_fence();
    uint64_t warm_start = read_cycles();

    // ---- Warm: spad-only compute, LD idle ----
    for (int pass = 0; pass < NUM_PASSES; pass++) {
        // F6: buf_in -> buf_mid
        gemmini_loop_ws_spad(1, J_F6, K_F6, 0, 0, 0,
            SP_BUF_IN,
            SP_W_F6 + W_F6_TILES * DIM,
            0, SP_BUF_MID,
            0, 0, 0, 0, 0,
            NO_ACTIVATION, 0, 0, 0,
            0x38);

        gemmini_fence();

        // Out: buf_mid -> buf_out
        gemmini_loop_ws_spad(1, J_OUT, K_OUT, 0, 0, 0,
            SP_BUF_MID,
            SP_W_OUT + W_OUT_TILES * DIM,
            0, SP_BUF_OUT,
            0, 0, 0, 0, 0,
            NO_ACTIVATION, 0, 0, 0,
            0x38);

        gemmini_fence();
    }

    uint64_t end = read_cycles();

    uint64_t cold_cycles = end - cold_start;
    uint64_t warm_cycles = end - warm_start;

    printf("LeNet-5 FC cold=%llu warm=%llu (passes=%d, gemms=%d)\n",
           (unsigned long long)cold_cycles, (unsigned long long)warm_cycles,
           NUM_PASSES, NUM_PASSES * 2);
    printf("COLD_START name=lenet5_fc_spad cycles=%llu\n",
           (unsigned long long)cold_cycles);

    uint64_t bench_general_end = read_cycles();
    BENCH_E2E_PRINT("lenet5_fc_spad",
                    bench_general_end - bench_general_start,
                    cold_cycles, warm_cycles);

    return 0;
}
