# Spad-Only Workload Evaluation (Baseline vs T³)

End-to-end evaluation of T³ on Gemmini, comparing Baseline vs T³ (Twist
SingleOp) on 32×32 and 16×16 systolic arrays. All workloads are spad-only:
weights preloaded into scratchpad, chained GEMMs via `gemmini_loop_ws_spad`
with `skips=0x38`, so the LD controller stays idle and the T³ pipeline
saving is fully exposed.

## Configurations

### 32×32

| Parameter | Value |
|-----------|-------|
| Array size | 32×32 |
| Dataflow | Weight-Stationary |
| Scratchpad | 512 KB (4 banks) |
| Accumulator | 128 KB (2 banks) |

- **Baseline**: `Baseline32x32WSRocketConfig`
- **T³**: `Twist32x32WSSingleOpRocketConfig`

### 16×16

| Parameter | Value |
|-----------|-------|
| Array size | 16×16 |
| Dataflow | Weight-Stationary |
| Scratchpad | 256 KB (4 banks) |
| Accumulator | 64 KB (2 banks) |

- **Baseline**: `Baseline16x16WSRocketConfig`
- **T³**: `Twist16x16WSSingleOpRocketConfig`

## Workloads

All 4 workloads use spad-resident activations with literature backing:

| Test | Paper | Architecture | Notes |
|------|-------|--------------|-------|
| `lstm_lm_spad` | Sak et al. 2014 | Projection LSTM (LSTMP) language model, H=128, T=32 | 160 spad GEMMs per run |
| `kws_lstm_spad` | Arik et al. 2017 | 2-layer LSTMP for keyword spotting | H=128/proj=64 (32×32), H=32/proj=16 (16×16) |
| `bilstm_spad` | Lample et al. 2016 | BiLSTM-CRF for NER, H=96 | 160 spad + 1 DRAM GEMM |
| `dlrm_mlp_spad` | Naumov et al. 2019 | DLRM bottleneck MLP (512→256→64→DIM) | 3 DRAM GEMMs |

## Three Timing Metrics

Every benchmark emits a single line from `bench_e2e.h`:

```
BENCH_E2E name=<test> general=<G> cold=<C> warm=<W>
```

| Metric | Definition |
|--------|-----------|
| **Warm** | Compute-only cycles after weight preload completes; isolates pure hardware execution latency. |
| **Cold** | Includes initial weight transfer DRAM→scratchpad (`mvin`); reflects first-inference latency. |
| **General** | Full program duration from `gemmini_flush` through all compute phases. |

## Usage

### Build simulators

```bash
# 32×32 (multi-threaded Verilator, 16 threads per sim)
FORCE_CLEAN=1 bash build_32x32_mt.sh 16

# 16×16
FORCE_CLEAN=1 bash build_16x16_mt.sh 16
```

### Run workloads

```bash
# 32×32: run all 4 workloads × 2 configs in parallel
bash run_spad_32x32_parallel.sh

# 16×16
bash run_spad_16x16_parallel.sh

# Extract + plot only (skip simulation)
bash run_spad_32x32_parallel.sh --extract
bash run_spad_16x16_parallel.sh --extract
```

## Output

```
data/conv_results_32x32.csv           # 32×32 results
data/conv_results_16x16.csv           # 16×16 results
figures/conv_table_32x32.{tex,pdf,png}
figures/conv_table_16x16.{tex,pdf,png}
logs_32x32_spad/                      # 32×32 simulation logs
logs_16x16_spad/                      # 16×16 simulation logs
```

## Directory Structure

```
conv_eval/
├── README.md
├── build_32x32_mt.sh             # Build 32×32 Verilator simulators
├── build_16x16_mt.sh             # Build 16×16 Verilator simulators
├── run_spad_32x32_parallel.sh    # Run 32×32 spad workloads in parallel
├── run_spad_16x16_parallel.sh    # Run 16×16 spad workloads in parallel
├── benchmarks/                   # Copies of .c files (synced with gemmini-rocc-tests)
│   ├── lstm_lm_spad.c
│   ├── kws_lstm_spad.c
│   ├── bilstm_spad.c
│   ├── dlrm_mlp_spad.c
│   ├── spad_gemm_utils.h         # Shared spad helpers
│   └── bench_e2e.h               # Unified BENCH_E2E_PRINT macro
├── scripts/
│   ├── extract_conv.py           # Parse logs → CSV
│   └── plot_conv.py              # CSV → table figure + LaTeX
├── logs_32x32_spad/              # 32×32 simulation logs
├── logs_16x16_spad/              # 16×16 simulation logs
├── data/                         # Extracted CSVs
└── figures/                      # Generated tables
```

## Notes

- Binaries are compiled with the correct DIM via `params_cache/gemmini_params_dim{16,32}.h`.
  The run scripts swap `gemmini_params.h`, compile, then restore the original.
- `kws_lstm_spad.c` uses `#if DIM <= 16` to select H=32/proj=16 (16×16) vs
  H=128/proj=64 (32×32). All other workloads use fixed parameters.
- `lstm_lm_spad.c` uses fixed H=128 regardless of DIM; tile count K=128/DIM
  scales automatically.
