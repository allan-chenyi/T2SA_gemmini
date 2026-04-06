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

// Conv parameters (original conv_perf defaults, BATCH=1)
#define BATCH_SIZE   1
#define IN_DIM_VAL   224
#define IN_CHANNELS  3
#define OUT_CHANNELS 32
#define KERNEL_DIM   3
#define PADDING      1
#define STRIDE_VAL   2
#define OUT_DIM_VAL  ((IN_DIM_VAL + 2*PADDING - KERNEL_DIM) / STRIDE_VAL + 1)

// Equivalent matmul dimensions
#define MAT_I   (BATCH_SIZE * OUT_DIM_VAL * OUT_DIM_VAL)
#define MAT_J   OUT_CHANNELS
#define MAT_K   (KERNEL_DIM * KERNEL_DIM * IN_CHANNELS)

// Tile counts
#define I_TILES ((MAT_I + DIM - 1) / DIM)
#define J_TILES ((MAT_J + DIM - 1) / DIM)
#define K_TILES ((MAT_K + DIM - 1) / DIM)
#define I_PAD   (I_TILES * DIM)
#define J_PAD   (J_TILES * DIM)
#define K_PAD   (K_TILES * DIM)

// DRAM buffers
static elem_t input_raw[BATCH_SIZE * IN_DIM_VAL * IN_DIM_VAL * IN_CHANNELS];
static elem_t weights_raw[OUT_CHANNELS * KERNEL_DIM * KERNEL_DIM * IN_CHANNELS];
static elem_t im2col_buf[I_PAD * K_PAD];
static elem_t weight_mat[K_PAD * J_PAD];

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif
    uint64_t bench_general_start = read_cycles();

    printf("Conv spad-only (BATCH=%d, IN=%d)\n", BATCH_SIZE, IN_DIM_VAL);
    printf("  Matmul: I=%d J=%d K=%d\n", MAT_I, MAT_J, MAT_K);
    printf("  Tiles:  I=%d J=%d K=%d  (DIM=%d)\n", I_TILES, J_TILES, K_TILES, DIM);

    // ---- CPU im2col ----
    for (int i = 0; i < I_PAD * K_PAD; i++) im2col_buf[i] = 0;
    for (int i = 0; i < K_PAD * J_PAD; i++) weight_mat[i] = 0;

    int pos = 0;
    for (int b = 0; b < BATCH_SIZE; b++)
      for (int oh = 0; oh < OUT_DIM_VAL; oh++)
        for (int ow = 0; ow < OUT_DIM_VAL; ow++, pos++)
          for (int kh = 0; kh < KERNEL_DIM; kh++)
            for (int kw = 0; kw < KERNEL_DIM; kw++)
              for (int ic = 0; ic < IN_CHANNELS; ic++) {
                int ih = oh * STRIDE_VAL - PADDING + kh;
                int iw = ow * STRIDE_VAL - PADDING + kw;
                int kidx = kh * KERNEL_DIM * IN_CHANNELS + kw * IN_CHANNELS + ic;
                if (ih >= 0 && ih < IN_DIM_VAL && iw >= 0 && iw < IN_DIM_VAL)
                  im2col_buf[pos * K_PAD + kidx] =
                    input_raw[b*IN_DIM_VAL*IN_DIM_VAL*IN_CHANNELS +
                              ih*IN_DIM_VAL*IN_CHANNELS + iw*IN_CHANNELS + ic];
              }

    for (int oc = 0; oc < OUT_CHANNELS; oc++)
      for (int kh = 0; kh < KERNEL_DIM; kh++)
        for (int kw = 0; kw < KERNEL_DIM; kw++)
          for (int ic = 0; ic < IN_CHANNELS; ic++) {
            int row = kh * KERNEL_DIM * IN_CHANNELS + kw * IN_CHANNELS + ic;
            weight_mat[row * J_PAD + oc] =
              weights_raw[oc*KERNEL_DIM*KERNEL_DIM*IN_CHANNELS +
                          kh*KERNEL_DIM*IN_CHANNELS + kw*IN_CHANNELS + ic];
          }

    // Spad layout: A at [0, A_rows), B at [A_rows, A_rows+B_rows)
    // C output overlaps A at [0] (each tile consumed before overwritten)
    int A_rows = I_TILES * K_TILES * DIM;
    int B_rows = K_TILES * J_TILES * DIM;
    uint32_t sp_A = 0;
    uint32_t sp_B = A_rows;

    printf("  Spad: A=%d B=%d total=%d / %d\n",
           A_rows, B_rows, A_rows + B_rows, BANK_NUM * BANK_ROWS);

    gemmini_flush(0);

    // -- Cold: mvin all data to spad --
    uint64_t cold_start = read_cycles();

    gemmini_extended_config_ld(K_PAD * sizeof(elem_t), MVIN_SCALE_IDENTITY);
    for (int i = 0; i < I_TILES; i++)
      for (int k = 0; k < K_TILES; k++)
        gemmini_extended_mvin(
            im2col_buf + i * DIM * K_PAD + k * DIM,
            sp_A + (i * K_TILES + k) * DIM,
            DIM, DIM);

    gemmini_extended_config_ld(J_PAD * sizeof(elem_t), MVIN_SCALE_IDENTITY);
    for (int j = 0; j < J_TILES; j++)
      for (int k = 0; k < K_TILES; k++)
        gemmini_extended_mvin(
            weight_mat + k * DIM * J_PAD + j * DIM,
            sp_B + (j * K_TILES + k) * DIM,
            DIM, DIM);

    gemmini_fence();
    uint64_t warm_start = read_cycles();

    // -- Warm: pure spad compute (LD idle) --
    gemmini_loop_ws_spad(I_TILES, J_TILES, K_TILES,
        0, 0, 0,
        sp_A, sp_B + B_rows, 0, 0,
        0, 0, 0, 0, 0,
        NO_ACTIVATION, 0, 0, 0,
        0x38);

    gemmini_fence();
    uint64_t end = read_cycles();

    uint64_t cold_cycles = end - cold_start;
    uint64_t warm_cycles = end - warm_start;

    printf("Conv spad cold=%llu warm=%llu\n",
           (unsigned long long)cold_cycles, (unsigned long long)warm_cycles);
    printf("COLD_START name=conv_perf cycles=%llu\n",
           (unsigned long long)cold_cycles);

    uint64_t bench_general_end = read_cycles();
    BENCH_E2E_PRINT("conv_perf",
                    bench_general_end - bench_general_start,
                    cold_cycles, warm_cycles);

    return 0;
}
