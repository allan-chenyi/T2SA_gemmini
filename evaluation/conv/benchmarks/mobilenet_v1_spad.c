// MobileNetV1 pointwise 1x1 conv layers pw1-pw7 (spad compute).
//
// Architecture from Table 1 of:
//   Howard et al. 2017, "MobileNets: Efficient Convolutional Neural
//   Networks for Mobile Vision Applications", arXiv:1704.04861
//
// Only pointwise (1x1) layers pw1-pw7 included.
// Depthwise 3x3 layers are DRAM-bound and excluded.
//
// DIM=32 summary:
//   pw1:  32->64    K=1  J=2
//   pw2:  64->128   K=2  J=4   ** 2-tile sweet spot **
//   pw3: 128->128   K=4  J=4
//   pw4: 128->256   K=4  J=8
//   pw5: 256->256   K=8  J=8
//   pw6: 256->512   K=8  J=16
//   pw7: 512->512   K=16 J=16  x5 reps, double-buffer
//
// Total GEMMs: 6 + 5 = 11
// Expected T^3 saving: 11 x (DIM-1) cycles

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

#define BATCH  DIM

// Layer definitions (from Table 1, pw1-pw7)
#define NUM_LAYER_TYPES 7
static const int LAYER_CIN[NUM_LAYER_TYPES]  = {32, 64, 128, 128, 256, 256, 512};
static const int LAYER_COUT[NUM_LAYER_TYPES] = {64, 128, 128, 256, 256, 512, 512};
static const int LAYER_REPS[NUM_LAYER_TYPES] = {1,  1,   1,   1,   1,   1,   5};
static const char *LAYER_NAME[NUM_LAYER_TYPES] = {
    "pw1","pw2","pw3","pw4","pw5","pw6","pw7"};

static int cdiv(int a, int b) { return (a + b - 1) / b; }

// DRAM pool — content irrelevant, just need valid addresses for mvin.
// Sized for pw9/FC: max offset = (K-1)*DIM*J_pad + (DIM-1)*J_pad + DIM
// = 31*32*1024 + 31*1024 + 32 = 1048608, round up
#define DRAM_POOL_SIZE (512 * 512)
static elem_t dram_pool[DRAM_POOL_SIZE];

// Helper: mvin A tiles [BATCH x K_pad] to spad
static void mvin_act(int kt, int kp, uint32_t sp_a) {
    gemmini_extended_config_ld(kp * sizeof(elem_t), MVIN_SCALE_IDENTITY);
    for (int k = 0; k < kt; k++)
        gemmini_extended_mvin(dram_pool + k * DIM,
                              sp_a + k * DIM, DIM, DIM);
}

// Helper: mvin B tiles [K_pad x J_pad] to spad, J-outer K-inner
static void mvin_wgt(int kt, int jt, int jp, uint32_t sp_b) {
    gemmini_extended_config_ld(jp * sizeof(elem_t), MVIN_SCALE_IDENTITY);
    for (int j = 0; j < jt; j++)
        for (int k = 0; k < kt; k++)
            gemmini_extended_mvin(
                dram_pool + k * DIM * jp + j * DIM,
                sp_b + (j * kt + k) * DIM,
                DIM, DIM);
}

int main(int argc, char *argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall failed");
        exit(1);
    }
#endif
    uint64_t bench_general_start = read_cycles();

    int spad_cap = BANK_NUM * BANK_ROWS;
    int gemm_total = 0;

    printf("MobileNetV1 pw1-pw7 (DIM=%d, spad=%d)\n", DIM, spad_cap);
    for (int i = 0; i < NUM_LAYER_TYPES; i++) {
        int kt = cdiv(LAYER_CIN[i], DIM);
        int jt = cdiv(LAYER_COUT[i], DIM);
        int a_r = kt * DIM;
        int b_r = kt * jt * DIM;
        int c_r = jt * DIM;
        int gemms = LAYER_REPS[i];
        gemm_total += gemms;
        printf("  %-4s: %4d->%-4d  K=%2d J=%2d  A=%5d B=%5d C=%4d  (%d gemm%s)\n",
               LAYER_NAME[i], LAYER_CIN[i], LAYER_COUT[i],
               kt, jt, a_r, b_r, c_r, gemms, gemms > 1 ? "s" : "");
    }
    printf("  Total GEMMs: %d  expected T3 saving: %d cycles\n",
           gemm_total, gemm_total * (DIM - 1));

    gemmini_flush(0);
    uint64_t cold_start = read_cycles();

    int gemm_count = 0;

    for (int li = 0; li < NUM_LAYER_TYPES; li++) {
        int cin  = LAYER_CIN[li];
        int cout = LAYER_COUT[li];
        int reps = LAYER_REPS[li];
        int kt = cdiv(cin, DIM);
        int jt = cdiv(cout, DIM);
        int kp = kt * DIM;
        int jp = jt * DIM;
        int a_rows = kt * DIM;
        int b_rows = kt * jt * DIM;
        int c_rows = jt * DIM;

        if (reps > 1) {
            // ---- pw7-style: load B once, double-buffer A for reps ----
            uint32_t sp_b = 0;
            uint32_t sp_a0 = b_rows;
            uint32_t sp_cbuf = b_rows + a_rows;

            mvin_wgt(kt, jt, jp, sp_b);
            mvin_act(kt, kp, sp_a0);
            gemmini_fence();

            for (int rep = 0; rep < reps; rep++) {
                uint32_t src = (rep % 2 == 0) ? sp_a0 : sp_cbuf;
                uint32_t dst = (rep % 2 == 0) ? sp_cbuf : sp_a0;

                gemmini_loop_ws_spad(1, jt, kt, 0, 0, 0,
                    src, sp_b + b_rows, 0, dst,
                    0, 0, 0, 0, 0,
                    NO_ACTIVATION, 0, 0, 0, 0x38);
                gemmini_fence();
                gemm_count++;
            }

        } else {
            // ---- Single layer fits in spad ----
            uint32_t sp_a = 0;
            uint32_t sp_c = a_rows;
            uint32_t sp_b = a_rows + c_rows;

            mvin_act(kt, kp, sp_a);
            mvin_wgt(kt, jt, jp, sp_b);
            gemmini_fence();

            gemmini_loop_ws_spad(1, jt, kt, 0, 0, 0,
                sp_a, sp_b + b_rows, 0, sp_c,
                0, 0, 0, 0, 0,
                NO_ACTIVATION, 0, 0, 0, 0x38);
            gemmini_fence();
            gemm_count++;
        }
    }

    uint64_t end = read_cycles();
    uint64_t cold_cycles = end - cold_start;

    printf("MobileNetV1 pw cycles=%llu (gemms=%d)\n",
           (unsigned long long)cold_cycles, gemm_count);
    printf("COLD_START name=mobilenet_v1_spad cycles=%llu\n",
           (unsigned long long)cold_cycles);

    uint64_t bench_general_end = read_cycles();
    BENCH_E2E_PRINT("mobilenet_v1_spad",
                    bench_general_end - bench_general_start,
                    cold_cycles, cold_cycles);

    return 0;
}
