// Seq2Seq with Bahdanau Attention — Real Parameters
//
// Bahdanau et al. 2014 / Luong et al. 2015
// Original H=1000 → H=1024 (nearest 32-aligned), proj=512
// On 32×32 SA: K=32 tiles per proj GEMM (DRAM-bound)
//
// Encoder: 16 steps × 4 GEMMs + Decoder: 16 steps × (4 GEMMs + 2 attn)
// All via tiled_matmul_auto. Expect <1% improvement.

#include "spad_gemm_utils.h"

#define ENC_STEPS  16
#define DEC_STEPS  16
#define HIDDEN     1024
#define PROJ_DIM   512

// DRAM layout (2MB per weight block)
#define W_STRIDE     0x200000UL
// Encoder weights (proj + 3 gates = 4)
#define EW_PROJ_DRAM ((elem_t *)(DRAM_BASE))
#define EW_R_DRAM    ((elem_t *)(DRAM_BASE + 1 * W_STRIDE))
#define EW_Z_DRAM    ((elem_t *)(DRAM_BASE + 2 * W_STRIDE))
#define EW_C_DRAM    ((elem_t *)(DRAM_BASE + 3 * W_STRIDE))
// Decoder weights
#define DW_PROJ_DRAM ((elem_t *)(DRAM_BASE + 4 * W_STRIDE))
#define DW_R_DRAM    ((elem_t *)(DRAM_BASE + 5 * W_STRIDE))
#define DW_Z_DRAM    ((elem_t *)(DRAM_BASE + 6 * W_STRIDE))
#define DW_C_DRAM    ((elem_t *)(DRAM_BASE + 7 * W_STRIDE))

// Activation buffers in DRAM
#define ACT_BASE     (DRAM_BASE + 8 * W_STRIDE)
#define H_DRAM       ((elem_t *)(ACT_BASE))
#define PROJ_D       ((elem_t *)(ACT_BASE + 0x100000UL))
#define GATE_DRAM    ((elem_t *)(ACT_BASE + 0x200000UL))
// Attention workspace
#define HENC_DRAM    ((elem_t *)(ACT_BASE + 0x400000UL))   // encoder outputs
#define HENC_T_DRAM  ((elem_t *)(ACT_BASE + 0x600000UL))   // transposed
#define SCORE_DRAM   ((elem_t *)(ACT_BASE + 0x800000UL))
#define CTX_DRAM     ((elem_t *)(ACT_BASE + 0x900000UL))

// GRU cell helper: proj + 3 gates = 4 GEMMs
static void gru_step(elem_t *h, elem_t *proj, elem_t *gate,
                     elem_t *w_proj, elem_t *w_r, elem_t *w_z, elem_t *w_c) {
    // proj = h(1024) × W_proj(1024×512) → (512)
    tiled_matmul_auto(BATCH, PROJ_DIM, HIDDEN,
        h, w_proj, NULL, proj,
        HIDDEN, PROJ_DIM, PROJ_DIM, PROJ_DIM,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
        false, false, false, false, 0, WS);
    // reset
    tiled_matmul_auto(BATCH, HIDDEN, PROJ_DIM,
        proj, w_r, NULL, gate,
        PROJ_DIM, HIDDEN, HIDDEN, HIDDEN,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
        false, false, false, false, 0, WS);
    // update
    tiled_matmul_auto(BATCH, HIDDEN, PROJ_DIM,
        proj, w_z, NULL, gate,
        PROJ_DIM, HIDDEN, HIDDEN, HIDDEN,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
        false, false, false, false, 0, WS);
    // candidate
    tiled_matmul_auto(BATCH, HIDDEN, PROJ_DIM,
        proj, w_c, NULL, gate,
        PROJ_DIM, HIDDEN, HIDDEN, HIDDEN,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        RELU, ACC_SCALE_IDENTITY, 0, false,
        false, false, false, false, 0, WS);
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); exit(1); }
#endif
    gemmini_flush(0);
    unsigned long bench_general_start = read_cycles();

    printf("=== Seq2Seq NMT Real (DIM=%d, H=%d, proj=%d, enc=%d, dec=%d) ===\n",
           DIM, HIDDEN, PROJ_DIM, ENC_STEPS, DEC_STEPS);

    gemmini_config_ex(WS, 0, 0);

    unsigned long cold_start = read_cycles();
    unsigned long start = cold_start;

    // Encoder: GRU steps
    for (int t = 0; t < ENC_STEPS; t++)
        gru_step(H_DRAM, PROJ_D, GATE_DRAM,
                 EW_PROJ_DRAM, EW_R_DRAM, EW_Z_DRAM, EW_C_DRAM);

    unsigned long enc_end = read_cycles();

    // Decoder with cross-attention
    for (int t = 0; t < DEC_STEPS; t++) {
        // Decoder GRU step: 4 GEMMs
        gru_step(H_DRAM, PROJ_D, GATE_DRAM,
                 DW_PROJ_DRAM, DW_R_DRAM, DW_Z_DRAM, DW_C_DRAM);

        // Cross-attention score: h_dec(32×1024) × H_enc^T(1024×32) → (32×32)
        tiled_matmul_auto(BATCH, DIM, HIDDEN,
            H_DRAM, HENC_T_DRAM, NULL, SCORE_DRAM,
            HIDDEN, DIM, DIM, DIM,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false, false, false, 0, WS);

        // Context: attn(32×32) × H_enc(32×1024) → (32×1024)
        tiled_matmul_auto(BATCH, HIDDEN, DIM,
            SCORE_DRAM, HENC_DRAM, NULL, CTX_DRAM,
            DIM, HIDDEN, HIDDEN, HIDDEN,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false, false, false, 0, WS);
    }

    unsigned long end = read_cycles();
    unsigned long enc_cyc = enc_end - start;
    unsigned long dec_cyc = end - enc_end;
    unsigned long total = end - start;

    int enc_gemms = 4 * ENC_STEPS;
    int dec_gemms = 4 * DEC_STEPS;
    int attn_gemms = 2 * DEC_STEPS;

    printf("SEQ2SEQ_SPAD DIM=%d H=%d proj=%d enc=%d dec=%d enc_gemms=%d dec_gemms=%d attn_gemms=%d enc_cyc=%lu dec_cyc=%lu cold_total=%lu warm_total=%lu\n",
           DIM, HIDDEN, PROJ_DIM, ENC_STEPS, DEC_STEPS, enc_gemms, dec_gemms, attn_gemms,
           enc_cyc, dec_cyc, end - cold_start, total);
    unsigned long bench_general_end = read_cycles();
    BENCH_E2E_PRINT("seq2seq_spad", bench_general_end - bench_general_start, end - cold_start, total);

    printf("=== Seq2Seq Done ===\n");
    return 0;
}
