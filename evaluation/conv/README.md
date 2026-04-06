# Conv & Multi-Layer CNN Evaluation (Baseline vs T^3)

Evaluates T^3 on convolution and multi-layer CNN workloads using spad-only
compute. All benchmarks do CPU-side data preparation, mvin to scratchpad,
then pure spad compute with LD idle — exposing T^3 pipeline savings.

## Workloads

| Test | Description | Parameters | GEMMs | 32×32 | 64×64 |
|------|-------------|------------|-------|-------|-------|
| `conv_perf` | ResNet first-layer conv, spad-only | BATCH=1, 224×224, IN_CH=3, OUT_CH=32, K=3, S=2 | 1 | yes | - |
| `conv_perf_b4` | Same conv, BATCH=4, chunked spad | BATCH=4, 224×224, same conv params, 2 chunks | 2 | yes | - |
| `resnet_bottleneck_spad` | ResNet bottleneck block chain | 8 blocks × 2 pointwise (reduce 4D→D + expand D→4D) | 16 | yes | yes |
| `lenet5_fc_spad` | LeNet-5 FC layers (F6+Out) | 8 passes × 2 layers (120→84, 84→10) | 16 | yes | - |
| `roshambo_l5_spad` | RoshamboNet Layer 5 (1×1 conv) | 16 iters of 128→128 pointwise | 16 | yes | yes |

### resnet_bottleneck_spad

ResNet bottleneck blocks (He et al. 2016, arXiv:1512.03385, 4:1 ratio).
Each block: reduce [BATCH×4D] × [4D×D] + expand [BATCH×D] × [D×4D].
Activations stay spad-resident (buf0 ↔ buf1). 8 blocks = 16 GEMMs.

- DIM=32: reduce I=1,J=1,K=4; expand I=1,J=4,K=1. Expected saving: 16×31 = 496 cycles.
- DIM=64: reduce I=1,J=1,K=4; expand I=1,J=4,K=1. Expected saving: 16×63 = 1008 cycles.

### lenet5_fc_spad

LeNet-5 fully-connected layers (LeCun et al. 1998, Proc. IEEE vol.86 no.11).
Parameters directly from Section II-B:
- **F6**: "contains 84 units ... fully connected to C5" → 120→84 (I=1, J=3, K=4 for DIM=32)
- **Out**: "10 RBF units ... 84 inputs each" → 84→10 (I=1, J=1, K=3 for DIM=32)

8 forward passes × 2 layers = 16 GEMMs. Expected saving: 16×31 = 496 cycles (DIM=32).

### roshambo_l5_spad

RoshamboNet Layer 5 (Urbach & Bhatt 2019, Table V):
1×1 pointwise conv with C_in=128, C_out=128 — the only layer whose K dimension
fills the SA efficiently.

- DIM=32: K_tiles=4, J_tiles=4. 16 iters × 31 = 496 cycles saving (~4%).
- DIM=64: K_tiles=2, J_tiles=2. 16 iters × 63 = 1008 cycles saving. **Sweet spot.**

## Metrics

| Metric | Source | Definition |
|--------|--------|-----------|
| **general** | `BENCH_E2E` | Full CPU wall time of `main` (setup + compute + teardown) |
| **cold** | `BENCH_E2E` | From first mvin to last fence (includes weight loading) |
| **warm** | `BENCH_E2E` | Compute-only window (weights already in spad, LD idle) |
| **cold_start** | `COLD_START` | Same as cold (for compatibility with extraction) |

## Usage

```bash
# 32×32: all 5 workloads
bash README/evaluate/papers/conv/run_conv_32x32.sh

# 64×64: resnet_bottleneck_spad + roshambo_l5_spad
bash README/evaluate/papers/conv/run_conv_64x64.sh

# Re-extract only (skip simulation)
bash README/evaluate/papers/conv/run_conv_32x32.sh --extract
bash README/evaluate/papers/conv/run_conv_64x64.sh --extract
```

## Directory Structure

```
conv/
├── README.md
├── run_conv_32x32.sh             # 32×32: 5 workloads × 2 configs = 10 sims
├── run_conv_64x64.sh             # 64×64: 2 workloads × 2 configs = 4 sims
├── benchmarks/                   # Copies of C files (synced with gemmini-rocc-tests)
│   ├── conv_perf.c
│   ├── conv_perf_b4.c
│   ├── resnet_bottleneck_spad.c
│   ├── lenet5_fc_spad.c
│   └── roshambo_l5_spad.c
├── logs/                         # 32×32 simulation logs
├── logs_64x64/                   # 64×64 simulation logs
└── data/                         # Extracted CSVs
```

## References

- He et al. 2016, "Deep Residual Learning for Image Recognition", CVPR, arXiv:1512.03385
- LeCun et al. 1998, "Gradient-Based Learning Applied to Document Recognition", Proc. IEEE, vol.86, no.11
- Urbach & Bhatt 2019, "RoshamboNet: Training a Single Classifier for Multiple Rock-Paper-Scissors Datasets"
- Sandler et al. 2018, "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR, arXiv:1801.04381

## Notes

- **Why multi-GEMM matters**: A single `gemmini_loop_ws_spad` call pipelines all
  I-tiles, overlapping drain[i] with preload[i+1]. Only the LAST drain is exposed,
  so single-GEMM workloads save only D-1 cycles total. Multi-GEMM workloads
  accumulate D-1 savings per GEMM call.
- 32×32 simulators are from `conv_eval/`. 64×64 simulators are pre-built in `sims/verilator/`.
- Adjust `CPU_BUDGET` env var to control parallelism (default 64 cores).
