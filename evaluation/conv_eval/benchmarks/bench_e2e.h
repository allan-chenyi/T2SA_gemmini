// Unified E2E cycle reporting for all conv/rnn/gru/mlp/seq2seq benchmarks.
//
// Every benchmark prints exactly one line of the form:
//
//   BENCH_E2E name=<test_name> general=<G> cold=<C> warm=<W>
//
// where:
//   general — full CPU wall time of the benchmark (read_cycles() from
//             very start of main to very end, including setup, data prep,
//             all compute, and teardown)
//   cold    — cycles from the first gemmini command to the last gemmini
//             command's completion, INCLUDING any DRAM→scratchpad weight
//             preload phase at the start
//   warm    — cycles from the first gemmini command to the last gemmini
//             command's completion, EXCLUDING any preload phase (i.e.
//             with all weights already resident in the scratchpad)
//
// For conv-family benchmarks (conv_perf, conv_dw_perf, lstm_cell_perf) which
// have no explicit preload phase, cold == warm == compute window; general is
// slightly larger because it also covers CPU-side im2col and setup printing.
//
// The extract script parses ONLY this line and treats all three metrics
// identically regardless of family. No RTL trace post-processing required.

#ifndef BENCH_E2E_H
#define BENCH_E2E_H

#include <stdio.h>

#define BENCH_E2E_PRINT(test_name, general_cyc, cold_cyc, warm_cyc)          \
    printf("BENCH_E2E name=%s general=%lu cold=%lu warm=%lu\n",              \
           (test_name),                                                      \
           (unsigned long)(general_cyc),                                     \
           (unsigned long)(cold_cyc),                                        \
           (unsigned long)(warm_cyc))

#endif // BENCH_E2E_H
