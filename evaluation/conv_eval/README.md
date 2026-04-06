# Conv + RNN/GRU/DLRM E2E Evaluation (16×16 Baseline vs T³)

End-to-end evaluation of Gemmini on 16×16 systolic arrays, comparing Baseline
vs T³ (Twist SingleOp) configurations. Workloads span two families:

1. **Conv family**: original upstream `conv_perf`, `conv_dw_perf`, and our
   new `lstm_cell_perf` (fused-gate GEMM). Use `tiled_conv_auto` /
   `tiled_matmul_auto` (DRAM-backed, full CPU wall time).
2. **Spad family**: `*_spad.c` benchmarks that preload all weights into the
   scratchpad and run chained 2-tile GEMMs via `gemmini_loop_ws_spad` with
   `skips=0x38`. 100% spad-only compute — the LD controller stays idle so
   the pipeline saving of T³ is fully exposed in the warm window.

All spad benchmarks have literature backing (see table below) and come from
the `optimal_gemm_analysis.md` investigation.

## Configurations

Both configs use identical parameters:

| Parameter | Value |
|-----------|-------|
| Array size | 16×16 |
| Dataflow | Weight-Stationary |
| Scratchpad | 256 KB (4 banks) |
| Accumulator | 64 KB (2 banks) |
| DMA bus | 128-bit |
| Data type | INT8 input, INT32 accumulator |

- **Baseline**: `Baseline16x16WSRocketConfig` (StandardMesh)
- **T³**: `Twist16x16WSSingleOpRocketConfig` (TwistSingleOp)

## Workloads

### Conv family (`tiled_conv_auto` / `tiled_matmul_auto`, DRAM-backed)

| Test | Description | Parameters |
|------|-------------|------------|
| `conv_perf` | ResNet-style first-layer convolution | BATCH=4, 224×224, IN_CH=3, OUT_CH=32, K=3, S=2 (upstream defaults) |
| `conv_dw_perf` | MobileNet-style depthwise separable | BATCH=3, 112×112, CH=17, K=3, S=2 (upstream defaults) |
| `lstm_cell_perf` | Fused LSTM gate GEMM (M=4H, K=I+H, N=B) | H=64 / 128 / 256, I=H, B=1 |

### Spad family (`gemmini_loop_ws_spad` skips=0x38, 100% spad-only)

All spad tests share a single utility header (`spad_gemm_utils.h`) with
`D_HIDDEN = 2 × DIM`, `BATCH = DIM`, so sizes **auto-scale with DIM**. At
DIM=16 we get `hidden=32`, `proj=16`, `batch=16` — proportionally scaled
versions of the published configurations.

| Test | Paper | Architecture | Expected saving on 32×32 |
|------|-------|--------------|--------------------------|
| `lstm_lm_spad` | Sak et al. 2014 ([arXiv:1402.1128](https://arxiv.org/abs/1402.1128)) | Projection LSTM (LSTMP) language model, 32 timesteps × 5 GEMMs | ~11.0% warm |
| `kws_lstm_spad` | Sainath & Parada 2015 (Interspeech) | 2-layer LSTMP for keyword spotting | ~10.9% warm |
| `gru_ts_spad` | Cho et al. 2014 ([arXiv:1406.1078](https://arxiv.org/abs/1406.1078)) | Projection GRU for time-series forecasting | ~11.0% warm |
| `stacked_lstm_spad` | Sak et al. 2014 (deep variant for speech) | 3-layer LSTMP | ~10.9% warm |
| `dlrm_mlp_spad` | Naumov et al. 2019 ([arXiv:1906.00091](https://arxiv.org/abs/1906.00091)) | DLRM bottleneck MLP (alternating narrow/wide) | ~10.8% warm |
| `bilstm_spad` | Lample et al. 2016 (NAACL) | BiLSTM-CRF for Named Entity Recognition | ~9.2% warm |
| `seq2seq_spad` | Bahdanau et al. 2014 ([arXiv:1409.0473](https://arxiv.org/abs/1409.0473)) | Encoder–decoder NMT with attention | ~8.8% warm |
| `deepspeech2_spad` | Amodei et al. 2016 (ICML) | Conv front-end + 3-layer BiLSTM + FC | ~10.6% warm |

> These "expected saving" numbers are from the reference 32×32 runs in
> `README/evaluate/optimal_gemm_analysis.md`. On DIM=16 the per-GEMM saving
> drops from 32 to 16 cycles and the per-GEMM baseline also drops, so the
> *ratio* stays within the same ballpark but absolute cycle counts scale
> down proportionally.

### LSTM cell workload rationale

The original `gemmini-rocc-tests` contains several LSTM-based end-to-end
models (`lstm_lm_spad`, `kws_lstm_spad`, `stacked_lstm_spad`, `awdlstm_spad`,
`bilstm_spad`, `deepspeech2_spad`, `seq2seq_spad`), but **no isolated single-cell
latency benchmark**. For T³ evaluation we want a clean per-cell latency number
that isolates a single fused-gate GEMM, since the cell-step latency is what
bounds real-time inference in streaming RNN workloads.

`lstm_cell_perf.c` measures one LSTM cell step as a single fused 4-gate GEMM:

```
[i; f; g; o] = W · [x_t; h_{t-1}]     with W : (4H) × (I+H), output : 4H × B
```

This is the standard fused-gate formulation used in production LSTM inference
and motivates several workload sizes that map to well-known models in the
literature:

| Size | H | I | B | Reference models |
|------|---|---|---|------------------|
| small  |  64 |  64 | 1 | TIMIT-scale KWS / small acoustic models (Sainath & Parada 2015, Interspeech) |
| medium | 128 | 128 | 1 | PTB medium LSTM (Zaremba et al. 2014, arXiv:1409.2329); AWD-LSTM small (Merity 2018) |
| large  | 256 | 256 | 1 | LSTMP acoustic models (Sak et al. 2014, Interspeech); DeepSpeech2 small RNN tier (Amodei 2016, ICML) |

Additional references motivating fused-gate GEMM as the cell-step primitive:
Hochreiter & Schmidhuber 1997 (original LSTM), Wu et al. 2016 (Google NMT,
arXiv:1609.08144), Lample et al. 2016 (BiLSTM-CRF for NER, NAACL).

## Unified E2E extraction (one printf, three metrics, all tests)

Every benchmark — conv family and spad family alike — emits a single line
from the shared macro in `bareMetalC/bench_e2e.h`:

```
BENCH_E2E name=<test> general=<G> cold=<C> warm=<W>
```

| Metric | Definition |
|--------|-----------|
| `general` | Full CPU wall time of the benchmark: `read_cycles()` from the very start of `main` to the very end, covering setup, allocation, data prep, all compute, and teardown. Equivalent to what `Gemmini conv took` measures for the conv tests, extended to cover the whole `main`. |
| `cold`    | Cycles from the first gemmini command to the last gemmini command's completion, **including** any DRAM→scratchpad weight preload phase at the start. For spad tests this matches the previous `cold_total` (`end - cold_start`). |
| `warm`    | Same as `cold` but **excluding** the preload phase — i.e. with all weights already resident in the scratchpad. Matches the previous `warm_total` (`end - start`) for spad tests. |

**Conv-family tests have no explicit preload phase**, so `cold == warm == the
tiled_conv_auto window`; `general` is slightly larger because it also covers
CPU-side im2col and setup printing. For the spad family all three numbers
are meaningful and distinct.

`extract_conv.py` parses **only** the `BENCH_E2E` line — no RTL trace
post-processing, no family-specific regex. The output CSV has exactly three
cycle columns (`general_cycles`, `cold_cycles`, `warm_cycles`) that mean the
same thing for every workload.

> The conv tests still emit the legacy `Gemmini conv took N cycles` line
> for backwards compatibility with other tooling, but the extractor ignores
> it and uses only the unified line.

## Three metrics, three stories

Each log contains enough information to compute three different "latency"
numbers. Understanding which one answers which question is critical:

| Metric | Source | Measures |
|--------|--------|----------|
| `read_cycles` | C `rdcycle` CSR (`Gemmini conv took N cycles`) | **Full CPU wall time**, including im2col, RoCC dispatch, fence waits. This is what a user would see. |
| `gemmini_span` | `max(END) - min(START)` over `GEMMINI_TRACE LD/EX/ST_{START,END}` within the work fence window | **First moment gemmini starts working → last moment it stops**, *including* idle gaps where it is waiting for CPU to issue the next command. This isolates gemmini's contribution to wall time. |
| `fence_any_busy` | `any_busy` column of `GEMMINI_HW_CYCLE` fence line | Cumulative count of cycles where ≥1 of ld/ex/st controllers is busy. **Excludes** idle gaps. Closest to "gemmini utilization". |

### Why `read_cycles` can be identical across configs

For small workloads (e.g. the 7×7 `conv_perf` default), CPU dispatch + im2col
dominates wall time. Gemmini finishes its ~450-cycle compute inside a ~3283
cycle CPU window, so any T³ speedup on the matmul is **completely hidden
behind CPU setup work**. Both Baseline and T³ report identical `read_cycles`
even though the gemmini compute itself is ~30 cycles faster on T³.

This is why we also extract `gemmini_span`: it shows the savings that
`read_cycles` hides.

**Example from `conv_perf` (7×7, in=8, out=16):**

| | Baseline | T³ | Δ |
|---|----------|-----|---|
| `read_cycles` | 3283 | 3283 | 0 |
| `gemmini_span` | 1012 | 983 | −29 |
| `fence_any_busy` | 1006 | 977 | −29 |

For larger workloads the CPU overhead is amortized and `read_cycles` also
drops — so we report all three metrics to make the scaling visible.

> **Related:** the 6-cycle offset between `io.busy` (EXBUSY) and the
> `MESH_PIPELINE` window used in `single_gemm_eval` is documented in
> `../single_gemm_eval/README.md` ("EXBUSY vs MESH_PIPELINE").

## Multi-threaded build + parallel run

Single-threaded Verilator simulation is too slow for a 13-workload × 2-config
sweep. We build with Verilator's `--threads` flag and launch all 26
simulations in parallel on the 128-core host.

> **Important — Verilator threads are build-time fixed.** Each simulator
> binary allocates a worker-thread pool of size `VERILATOR_THREADS` at
> startup; you cannot shrink it at runtime. So the build-time thread count
> **must equal** the per-task thread budget you want at launch. Passing
> `+verilator+threads+N` at runtime is a no-op.

```bash
# 1. Build both simulators with 4 runtime threads each (13 workloads × 2
#    configs × 4 threads = 104 threads on a 128-core host).
#    If you previously built with a different VTHREADS, the script detects
#    the change via a marker file in generated-src/ and automatically wipes
#    the Verilator output so the new thread count takes effect. Use
#    FORCE_CLEAN=1 bash build_16x16_mt.sh 4  to force a clean rebuild.
bash build_16x16_mt.sh 4

# 2. Launch all 13 workloads × both configs in parallel.
bash run_all_parallel.sh

# 3. Extract + plot only (skip sims)
bash run_all_parallel.sh --extract
```

Thread budget inside `run_all_parallel.sh`: `TASK_VTHREADS=4`, so
`13 × 2 × 4 = 104` threads total. If you change `TASK_VTHREADS`, you **must
rebuild** the simulators with the matching value.

## Output

```
data/conv_results.csv                 # one row per (config, test)
                                      # columns include: read_cycles, gemmini_span,
                                      # fence_wall, fence_any_busy, fence_ex/ld/st,
                                      # plus per-test workload parameters
figures/conv_table.pdf/png            # dual-metric comparison table
figures/conv_table.tex                # LaTeX table source (table* with multicol)
logs/{baseline,twist}_<variant>.log   # raw simulation logs
```

`extract_conv.py` auto-discovers variants by scanning `logs/` for matching
`baseline_*.log` / `twist_*.log` pairs — no hardcoded test list to maintain
when adding new workloads.

## Directory Structure

```
conv_eval/
├── README.md
├── build_16x16.sh            # single-threaded build (legacy)
├── build_16x16_mt.sh         # multi-threaded Verilator build (preferred)
├── run_conv.sh               # serial run (legacy)
├── run_all_parallel.sh       # parallel run: conv + LSTM + spad sweep
├── benchmarks/               # Copies of every .c file used in the sweep,
│                             # for at-a-glance review:
│   ├── conv_perf.c           #   ResNet first layer (upstream defaults)
│   ├── conv_dw_perf.c        #   MobileNet DW (upstream defaults)
│   ├── lstm_cell_perf.c      #   Fused LSTM gate GEMM
│   ├── lstm_lm_spad.c        #   Sak 2014 LSTMP
│   ├── kws_lstm_spad.c       #   Sainath & Parada 2015
│   ├── gru_ts_spad.c         #   Cho 2014
│   ├── stacked_lstm_spad.c   #   Sak 2014 (deep)
│   ├── dlrm_mlp_spad.c       #   Naumov 2019
│   ├── bilstm_spad.c         #   Lample 2016 (NER)
│   ├── seq2seq_spad.c        #   Bahdanau 2014
│   ├── deepspeech2_spad.c    #   Amodei 2016
│   ├── spad_gemm_utils.h     #   Shared spad helpers (2*DIM, DIM macros)
│   └── bench_e2e.h           #   Unified BENCH_E2E_PRINT macro (included by all)
├── scripts/
│   ├── extract_conv.py       # Parse logs → unified CSV (both families)
│   └── plot_conv.py          # CSV → warm/cold/span table figure + LaTeX
├── logs/                     # Simulation logs
├── data/                     # Extracted CSV
└── figures/                  # Generated tables
```

## Notes

- Binaries are compiled with `DIM=16` via the `params_cache` pattern
  (`run_all_parallel.sh` swaps `gemmini_params.h`, compiles, restores).
- `lstm_cell_perf` takes `H I B` as argv, so one binary serves all three
  sizes (`64 64 1`, `128 128 1`, `256 256 1`) via the `WORKLOADS` array.
- For the conv tests, T³ speedup applies to the matmul tiles *inside* each
  conv operation — `gemmini_span` is the right metric to see it.
