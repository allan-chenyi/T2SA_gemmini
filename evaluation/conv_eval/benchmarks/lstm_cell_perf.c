// LSTM Cell-Step Latency Benchmark
//
// Measures the latency of ONE LSTM cell timestep on Gemmini as a single
// fused gate GEMM:
//
//   gates = [W_i | W_f | W_o | W_g] × [x_t ; h_{t-1}]
//         = ( 4H, I+H ) × ( I+H, B )  →  ( 4H, B )
//
// where H=hidden size, I=input size, B=batch size. The 4 gate projections
// (input, forget, output, cell-candidate) are fused into one matrix-multiply
// with M = 4H, K = I+H, N = B. This is the standard fusion used in real
// LSTM inference kernels (cuDNN, oneDNN, TF-Lite).
//
// Default parameters follow "small mobile LSTM" sizes (edge/IoT inference):
//   H = 128, I = 128, B = 1 (single-sample latency)
//
// Workload references:
//   - Hochreiter & Schmidhuber 1997, "Long Short-Term Memory" (original LSTM)
//   - Sak et al. 2014, "LSTM-based Recurrent Neural Network Architectures
//     for Large Scale Acoustic Modeling" (LSTMP, H∈{64,128,256} speech)
//   - Zaremba et al. 2014, "Recurrent Neural Network Regularization"
//     (PTB small/medium/large: H∈{200,650,1500})
//   - Wu et al. 2016, "Google NMT" (H=1024, 8 layers)
//   - Amodei et al. 2016, "DeepSpeech2" (H=1760..2048)
//
// Usage:
//   lstm_cell_perf-baremetal [HIDDEN INPUT_SIZE BATCH]
//
// Output format (single line, parseable):
//   LSTM_CELL HIDDEN=<H> INPUT_SIZE=<I> BATCH=<B> M=<4H> K=<I+H> N=<B> \
//             cycles=<cycles>
//   Gemmini conv took <cycles> cycles         (for extract_conv.py compat)

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

#define HEAP_SIZE (8*1024*1024)

static int str2int(char * str)
{
    int res = 0;
    for (int i = 0; str[i] != '\0'; ++i)
        res = res * 10 + str[i] - '0';
    return res;
}

int main(int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif
    uint64_t bench_general_start = read_cycles();

    int HIDDEN     = 128;
    int INPUT_SIZE = 128;
    int BATCH      = 1;

    if (argc == 4) {
        HIDDEN     = str2int(argv[1]);
        INPUT_SIZE = str2int(argv[2]);
        BATCH      = str2int(argv[3]);
    } else if (argc > 1) {
        printf("HIDDEN INPUT_SIZE BATCH\n");
        exit(1);
    }

    // Fused gate GEMM dimensions
    int M = 4 * HIDDEN;           // 4 gates stacked along M
    int K = INPUT_SIZE + HIDDEN;  // [x_t ; h_{t-1}] stacked along K
    int N = BATCH;

    printf("HIDDEN = %d\n", HIDDEN);
    printf("INPUT_SIZE = %d\n", INPUT_SIZE);
    printf("BATCH = %d\n", BATCH);
    printf("M = %d\n", M);
    printf("K = %d\n", K);
    printf("N = %d\n", N);

    gemmini_flush(0);

    static uint8_t heap[HEAP_SIZE];

    // Layout:
    //   weights W[M][K]  — stacked gate weights
    //   inputs  X[K][N]  — concatenated [x_t ; h_{t-1}] batched
    //   bias    b[M]
    //   output  Y[M][N]  — gate pre-activations
    elem_t * W = (elem_t*)(&heap[0]);
    elem_t * X = (elem_t*)((elem_t*)W + (size_t)M * K);
    acc_t  * B_bias = (acc_t*)((elem_t*)X + (size_t)K * N);
    elem_t * Y = (elem_t*)((acc_t*)B_bias + M);

    {
        uint8_t * end = (uint8_t*)((elem_t*)Y + (size_t)M * N);
        if (end >= &heap[HEAP_SIZE]) {
            printf("problem size is too large to fit in memory\n");
            exit(1);
        }
    }

    printf("LSTM cell...\n");
    uint64_t start = read_cycles();

    tiled_matmul_auto(
        M, N, K,
        W, X, B_bias, Y,
        K, N, N, N,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, true,
        false, false,
        false, false,
        0,
        WS);

    uint64_t end = read_cycles();
    uint64_t cycles = end - start;

    printf("LSTM_CELL HIDDEN=%d INPUT_SIZE=%d BATCH=%d M=%d K=%d N=%d cycles=%llu\n",
           HIDDEN, INPUT_SIZE, BATCH, M, K, N, (unsigned long long)cycles);
    // Compatibility line for legacy extractors
    printf("Gemmini conv took %llu cycles\n", (unsigned long long)cycles);

    uint64_t bench_general_end = read_cycles();
    BENCH_E2E_PRINT("lstm_cell_perf",
                    bench_general_end - bench_general_start,
                    cycles, cycles);

    return 0;
}
