// Single-GEMM EX_BUSY Sweep
//
// Same sweep as single_gemm_sweep.c but uses Gemmini's hardware
// MAIN_EX_CYCLES counter instead of read_cycles().
// MAIN_EX_CYCLES counts cycles where ex_controller.io.busy is true
// (and ld/st are idle), which includes both the compute state AND
// the mesh pipeline drain time (matmul_in_progress).
//
// Output:  EXBUSY_SWEEP DIM=<D> M=<M> Q=<Q> cycles=<C>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

static elem_t A_data[DIM][DIM] row_align(1);
static elem_t B_data[DIM][DIM] row_align(1);

static void init_data(void) {
    for (int i = 0; i < DIM; i++)
        for (int j = 0; j < DIM; j++) {
            A_data[i][j] = (elem_t)((i + j * 3 + 1) % 5 - 2);
            B_data[i][j] = (elem_t)((i * 2 + j + 7) % 5 - 2);
        }
}

static void bench(int M, int Q) {
    int M_mult = (M <= DIM) ? 1 : M / DIM;
    int M_rows = (M < DIM) ? M : DIM;

    uint32_t A_base = 0;                      // bank 0
    uint32_t B_base = (uint32_t)BANK_ROWS;    // bank 1 — avoid bank conflict with A during mul_pre
    uint32_t C_acc  = 1u << (ADDR_LEN - 1);

    // spad capacity check: A in bank 0, B in bank 1
    if ((long)M_mult * DIM > BANK_ROWS || (long)Q * DIM > BANK_ROWS) {
        printf("EXBUSY_SWEEP DIM=%d M=%d Q=%d cycles=SKIP\n", DIM, M, Q);
        return;
    }

    // load tiles into spad
    gemmini_config_ld(DIM * sizeof(elem_t));
    for (int i = 0; i < M_mult; i++)
        gemmini_mvin(A_data, A_base + (uint32_t)(i * DIM));
    for (int j = 0; j < Q; j++)
        gemmini_mvin(B_data, B_base + (uint32_t)(j * DIM));
    gemmini_fence();

    gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);

    // Reset counter, configure slots for ex_busy decomposition
    // MAIN_EX_CYCLES: cycles where ONLY ex is busy (no ld/st)
    // In compute-only mode (data pre-loaded), ld/st are idle,
    // so MAIN_EX_CYCLES ≈ total ex_busy cycles.
    // Also capture overlap counters for completeness.
    counter_reset();
    counter_configure(0, MAIN_EX_CYCLES);
    counter_configure(1, MAIN_LD_EX_CYCLES);
    counter_configure(2, MAIN_ST_EX_CYCLES);
    counter_configure(3, MAIN_LD_ST_EX_CYCLES);

    int acc_idx = 0;

    // Issue all preload+compute pairs
    for (int q = 0; q < Q; q++) {
        for (int m = 0; m < M_mult; m++) {
            gemmini_extended_preload(
                B_base + (uint32_t)(q * DIM),
                C_acc  + (uint32_t)((acc_idx % 16) * DIM),
                DIM, DIM, DIM, M_rows);
            gemmini_extended_compute_preloaded(
                A_base + (uint32_t)(m * DIM), GARBAGE_ADDR,
                DIM, M_rows, DIM, DIM);
            acc_idx++;
        }
    }
    gemmini_fence();

    // Read counters after fence (all ops done)
    // Total ex_busy = MAIN_EX + MAIN_LD_EX + MAIN_ST_EX + MAIN_LD_ST_EX
    uint32_t ex_only   = counter_read(0);
    uint32_t ld_ex     = counter_read(1);
    uint32_t st_ex     = counter_read(2);
    uint32_t ld_st_ex  = counter_read(3);
    uint32_t ex_cycles = ex_only + ld_ex + st_ex + ld_st_ex;

    printf("EXBUSY_SWEEP DIM=%d M=%d Q=%d cycles=%u\n",
           DIM, M, Q, ex_cycles);
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); exit(1); }
#endif
    gemmini_flush(0);
    init_data();

    printf("=== Single-GEMM EX_ACTIVE Sweep (DIM=%d) ===\n", DIM);

    static const int M_vals[] = {
        1, 2, 3, 4, 6, 8, 12, 16, 24,
        32,
        64, 96, 128, 160, 192, 256,
        384, 512, 768, 1024
    };
    static const int Q_vals[] = {1, 2, 4, 8, 16, 32, 64};

    int nM = sizeof(M_vals) / sizeof(M_vals[0]);
    int nQ = sizeof(Q_vals) / sizeof(Q_vals[0]);

    for (int qi = 0; qi < nQ; qi++)
        for (int mi = 0; mi < nM; mi++)
            bench(M_vals[mi], Q_vals[qi]);

    printf("=== Single-GEMM EX_ACTIVE Sweep Done ===\n");
    return 0;
}
