# Gemmini ISA Reference

Gemmini is a RISC-V RoCC (Rocket Custom Coprocessor) accelerator for matrix operations.
All instructions are encoded as RISC-V custom instructions using the `custom0` opcode
space, with a 7-bit `funct` field selecting the operation. Each instruction takes two
64-bit source registers (`rs1`, `rs2`) and optionally writes a 64-bit destination (`rd`).

Source files:
- RTL definitions: `src/main/scala/gemmini/GemminiISA.scala`
- Software macros: `software/gemmini-rocc-tests/include/gemmini.h`
- Performance counters: `software/gemmini-rocc-tests/include/gemmini_counter.h`

---

## Table of Contents

1. [Instruction Summary](#1-instruction-summary)
2. [Primitive Instructions (funct 0-7)](#2-primitive-instructions)
3. [Loop MatMul Instructions (funct 8-14)](#3-loop-matmul-instructions)
4. [Loop Conv Instructions (funct 15-21)](#4-loop-conv-instructions)
5. [Spad-Only Instructions (funct 22-25)](#5-spad-only-instructions)
6. [Performance Counter (funct 126)](#6-performance-counter)
7. [CISC Instructions (funct 10-17, alternate mode)](#7-cisc-instructions)
8. [Address Space Layout](#8-address-space-layout)
9. [Performance Counters Reference](#9-performance-counters-reference)

---

## 1. Instruction Summary

| funct | Name | C Macro | Description |
|-------|------|---------|-------------|
| 0 | CONFIG | `k_CONFIG` | Configure accelerator (sub-type in rs1[1:0]) |
| 1 | LOAD2 | `k_MVIN2` | Move-in from DRAM to scratchpad (alternate LD unit) |
| 2 | LOAD | `k_MVIN` | Move-in from DRAM to scratchpad (primary LD unit) |
| 3 | STORE | `k_MVOUT` | Move-out from scratchpad/accumulator to DRAM |
| 4 | COMPUTE_AND_FLIP | `k_COMPUTE_PRELOADED` | Matrix multiply with preloaded B, flip acc bank |
| 5 | COMPUTE_AND_STAY | `k_COMPUTE_ACCUMULATE` | Matrix multiply accumulating into same acc bank |
| 6 | PRELOAD | `k_PRELOAD` | Preload B/D matrix and set output address |
| 7 | FLUSH | `k_FLUSH` | Flush all pending operations |
| 8 | LOOP_WS | `k_LOOP_WS` | Execute weight-stationary matmul loop |
| 9 | LOOP_WS_CONFIG_BOUNDS | `k_LOOP_WS_CONFIG_BOUNDS` | Set loop tile bounds (I, J, K) |
| 10 | LOOP_WS_CONFIG_ADDRS_AB | `k_LOOP_WS_CONFIG_ADDRS_AB` | Set A and B DRAM base addresses |
| 11 | LOOP_WS_CONFIG_ADDRS_DC | `k_LOOP_WS_CONFIG_ADDRS_DC` | Set D (bias) and C (output) DRAM addresses |
| 12 | LOOP_WS_CONFIG_STRIDES_AB | `k_LOOP_WS_CONFIG_STRIDES_AB` | Set A and B DRAM strides |
| 13 | LOOP_WS_CONFIG_STRIDES_DC | `k_LOOP_WS_CONFIG_STRIDES_DC` | Set D and C DRAM strides |
| 14 | LOAD3 | `k_MVIN3` | Move-in from DRAM (third LD unit, acc-capable) |
| 15 | LOOP_CONV_WS | `k_LOOP_CONV_WS` | Execute weight-stationary convolution loop |
| 16 | LOOP_CONV_WS_CONFIG_1 | `k_LOOP_CONV_WS_CONFIG_1` | Conv config: batch, channels, dimensions |
| 17 | LOOP_CONV_WS_CONFIG_2 | `k_LOOP_CONV_WS_CONFIG_2` | Conv config: kernel, pooling |
| 18 | LOOP_CONV_WS_CONFIG_3 | `k_LOOP_CONV_WS_CONFIG_3` | Conv config: kernel tiles, padding |
| 19 | LOOP_CONV_WS_CONFIG_4 | `k_LOOP_CONV_WS_CONFIG_4` | Conv config: output tiles, dilation, strides |
| 20 | LOOP_CONV_WS_CONFIG_5 | `k_LOOP_CONV_WS_CONFIG_5` | Conv config: weights and output pointers |
| 21 | LOOP_CONV_WS_CONFIG_6 | `k_LOOP_CONV_WS_CONFIG_6` | Conv config: bias and input pointers |
| 22 | CLKGATE_EN | — | Clock gating enable/disable |
| 23 | STORE_SPAD | `k_MVOUT_SPAD` | Move data between scratchpad addresses (spad-to-spad) |
| 24 | LOOP_WS_CONFIG_SPAD_AB | `k_LOOP_WS_CONFIG_SPAD_AB` | Set A/B scratchpad addresses for spad-only loop |
| 25 | LOOP_WS_CONFIG_SPAD_C | `k_LOOP_WS_CONFIG_SPAD_C` | Set C scratchpad address for spad-only loop |
| 126 | COUNTER_OP | `k_COUNTER` | Performance counter read/configure |

---

## 2. Primitive Instructions

### 2.1 CONFIG (funct = 0)

A multiplexed configuration instruction. The sub-type is selected by `rs1[1:0]`:

| rs1[1:0] | Sub-Type | Name | Purpose |
|----------|----------|------|---------|
| 0 | CONFIG_EX | Execute config | Dataflow, activation, scaling, strides |
| 1 | CONFIG_LD | Load config | Load stride, scale, state ID |
| 2 | CONFIG_ST | Store config | Store stride, activation, pooling |
| 3 | CONFIG_NORM | Normalization config | LayerNorm/IGELU quantization constants |

#### 2.1.1 CONFIG_EX (rs1[1:0] = 0)

Configures the execution (compute) controller.

**rs1 layout:**
```
[63:32]  acc_scale      — Accumulator output scale factor (32-bit float/fixed)
                          Identity = ACC_SCALE_IDENTITY (0x3F800000 for float)
[31:16]  a_stride       — A matrix stride for weight-stationary preloads (16-bit)
[15:10]  (reserved)
[9]      b_transpose    — Transpose B matrix before compute
[8]      a_transpose    — Transpose A matrix before compute
[7]      set_only_strides — If 1, only update strides (not activation/dataflow)
[6:4]    (reserved)
[3:2]    activation     — Activation function (see Activation Codes below)
[1]      dataflow       — 0 = Output Stationary, 1 = Weight Stationary
[0]      (must be 0)    — CONFIG_EX identifier
```

**rs2 layout:**
```
[63:48]  c_stride       — C (output) matrix stride (16-bit)
[47:32]  relu6_shift    — Right-shift for ReLU6 saturation (16-bit)
[31:0]   in_shift       — Input right-shift / rounding (32-bit)
```

**Activation codes:**
| Value | Name | Description |
|-------|------|-------------|
| 0 | NO_ACTIVATION | Pass through |
| 1 | RELU | Rectified Linear Unit (clamp negatives to 0) |
| 2 | LAYERNORM | Layer normalization |
| 3 | IGELU | Integer-approximated GELU |
| 4 | SOFTMAX | Softmax (row-wise) |

**C macros:**
```c
gemmini_config_ex(dataflow, act, shift)
gemmini_extended_config_ex(dataflow, act, shift, A_stride, A_transpose, B_transpose)
gemmini_extended3_config_ex(dataflow, act, shift, acc_scale, C_stride, A_stride,
                            A_transpose, B_transpose, set_only_strides)
```

#### 2.1.2 CONFIG_LD (rs1[1:0] = 1)

Configures the load (MVIN) controller. Gemmini has up to 3 independent load units
(state_id 0, 1, 2), each configured separately.

**rs1 layout:**
```
[63:32]  scale          — Input scale factor (32-bit, typically MVIN_SCALE_IDENTITY)
[31:16]  stride         — DRAM row stride in bytes (16-bit)
[15:8]   pixel_repeats  — Pixel repetition count for im2col convolution (8-bit)
[7:5]    (reserved)
[4:3]    state_id       — Load unit selector: 0 for MVIN, 1 for MVIN2, 2 for MVIN3
[2]      shrink         — Enable data shrinking (quantization)
[1:0]    (must be 01)   — CONFIG_LD identifier
```

**rs2 layout:**
```
[31:0]   block_mvin_stride — Block MVIN stride (default = DIM)
```

**C macros:**
```c
gemmini_config_ld(stride)
gemmini_extended_config_ld(stride, scale)
gemmini_extended3_config_ld(stride, scale, shrunk, id)
```

#### 2.1.3 CONFIG_ST (rs1[1:0] = 2)

Configures the store (MVOUT) controller, including optional max-pooling.

**rs1 layout:**
```
[63:56]  ocols          — Output columns for tiled store (8-bit)
[55:48]  orows          — Output rows for tiled store (8-bit)
[47:40]  pocols         — Pooled output columns (8-bit)
[39:32]  porows         — Pooled output rows (8-bit)
[31:24]  pool_out_dim   — Pooling output dimension (8-bit)
[23:12]  (reserved)
[11:10]  lpad           — Left zero-padding for pooling (2-bit)
[9:8]    upad           — Upper zero-padding for pooling (2-bit)
[7:6]    pool_size      — Max-pooling window size (2-bit)
[5:4]    pool_stride    — Max-pooling stride (2-bit)
[3:2]    activation     — Activation function applied during store
[1:0]    (must be 10)   — CONFIG_ST identifier
```

**rs2 layout:**
```
[63:32]  acc_scale      — Accumulator-to-output scale (32-bit float/fixed)
[31:0]   stride         — DRAM store stride in bytes (32-bit)
```

**C macros:**
```c
gemmini_config_st(stride)
gemmini_extended_config_st(stride, act, acc_scale)
gemmini_extended2_config_st(stride, act, acc_scale, pool_stride, pool_size,
                            pool_out_dim, porows, pocols, orows, ocols, upad, lpad)
```

#### 2.1.4 CONFIG_NORM (rs1[1:0] = 3)

Configures normalization parameters for LayerNorm and IGELU activations.

**rs1 layout:**
```
[63:32]  q_const        — Quantization constant (32-bit)
[18]     q_const_type   — 0 = fixed-point, 1 = floating-point
[17]     set_stats_id_only — Only update statistics ID, keep other fields
[16]     act_msb        — MSB selection for IGELU activation
[15:8]   norm_stats_id  — Statistics buffer ID for normalization (8-bit)
[7:2]    (reserved)
[1:0]    (must be 11)   — CONFIG_NORM identifier
```

**rs2 layout:**
```
[63:32]  qc             — IGELU quantization constant C (32-bit)
[31:0]   qb             — IGELU quantization constant B (32-bit)
```

**C macro:**
```c
gemmini_config_norm(q_const, q_const_type, set_stats_id_only, act_msb,
                    stat_id, igelu_qb, igelu_qc)
```

---

### 2.2 MVIN / MVIN2 / MVIN3 (funct = 2 / 1 / 14)

Moves a tile of data from DRAM into the scratchpad or accumulator. Three independent
load units allow concurrent loading with different configurations (stride, scale).

**rs1:** DRAM source address (64-bit virtual address). If 0, the command is a no-op.

**rs2 layout:**
```
[63:48]  num_rows       — Number of rows to load (16-bit, max = DIM)
[47:32]  num_cols       — Number of columns to load (16-bit, max = DIM)
[31:0]   local_addr     — Destination scratchpad/accumulator address (32-bit)
                          Bit 31 = 1 → accumulator address
                          Bit 31 = 0 → scratchpad address
```

The load unit is selected by which MVIN variant is used:
- `MVIN` (funct=2) uses load unit 0
- `MVIN2` (funct=1) uses load unit 1
- `MVIN3` (funct=14) uses load unit 2

Each load unit has its own stride and scale configured via `CONFIG_LD` with the
corresponding `state_id`.

**C macros:**
```c
gemmini_mvin(dram_addr, spad_addr)                         // DIM×DIM tile
gemmini_extended_mvin(dram_addr, spad_addr, cols, rows)    // custom size
gemmini_block_mvin(dram_addr, spad_addr, len)              // len×DIM tiles
gemmini_extended_mvin2(dram_addr, spad_addr, cols, rows)   // via load unit 1
gemmini_extended_mvin3(dram_addr, spad_addr, cols, rows)   // via load unit 2
```

---

### 2.3 MVOUT (funct = 3)

Moves a tile of data from the scratchpad or accumulator to DRAM. Optionally applies
activation, scaling, and max-pooling (configured via CONFIG_ST).

**rs1:** DRAM destination address (64-bit virtual address).

**rs2 layout:**
```
[63:48]  num_rows       — Number of rows to store (16-bit)
[47:32]  num_cols       — Number of columns to store (16-bit)
[31:0]   local_addr     — Source scratchpad/accumulator address (32-bit)
                          Bit 31 = 1 → read from accumulator
                          Bit 31 = 0 → read from scratchpad
```

**C macros:**
```c
gemmini_mvout(dram_addr, spad_addr)                        // DIM×DIM tile
gemmini_extended_mvout(dram_addr, spad_addr, cols, rows)   // custom size
```

---

### 2.4 PRELOAD (funct = 6)

Preloads the B (weight) matrix and sets the output (C) accumulator address for the
next COMPUTE instruction. This is the first half of a PRELOAD+COMPUTE pair.

**rs1 layout (B/D source):**
```
[63:48]  BD_rows        — B matrix rows (16-bit)
[47:32]  BD_cols        — B matrix columns (16-bit)
[31:0]   BD_addr        — B matrix scratchpad address (32-bit)
                          GARBAGE_ADDR (0xFFFFFFFF) = use zeros
```

**rs2 layout (C destination):**
```
[63:48]  C_rows         — Output rows (16-bit)
[47:32]  C_cols         — Output columns (16-bit)
[31:0]   C_addr         — Output accumulator address (32-bit)
```

**C macros:**
```c
gemmini_preload(BD, C)                                     // DIM×DIM
gemmini_extended_preload(BD, C, BD_cols, BD_rows, C_cols, C_rows)
gemmini_preload_zeros(C)                                   // preload zeros
```

---

### 2.5 COMPUTE_PRELOADED / COMPUTE_ACCUMULATE (funct = 4 / 5)

Performs the matrix multiplication `C += A × B` using the systolic array, where B was
set by the preceding PRELOAD instruction.

- **COMPUTE_PRELOADED** (funct=4): Flips the accumulator double-buffer after writing.
  Used for the first tile or when starting a new output.
- **COMPUTE_ACCUMULATE** (funct=5): Accumulates into the same buffer. Used for
  subsequent tiles along the K (reduction) dimension.

**rs1 layout (A source):**
```
[63:48]  A_rows         — A matrix rows (16-bit)
[47:32]  A_cols         — A matrix columns (16-bit)
[31:0]   A_addr         — A matrix scratchpad address (32-bit)
```

**rs2 layout (B/D source for next PRELOAD, pipelined):**
```
[63:48]  BD_rows        — Next B matrix rows (16-bit)
[47:32]  BD_cols        — Next B matrix columns (16-bit)
[31:0]   BD_addr        — Next B matrix scratchpad address (32-bit)
```

The hardware merges adjacent PRELOAD+COMPUTE pairs into a single pipelined operation
(`compute_preloaded` / `mul_pre` in RTL), reducing overhead.

**C macros:**
```c
gemmini_compute_preloaded(A, BD)                           // DIM×DIM
gemmini_compute_accumulated(A, BD)                         // DIM×DIM
gemmini_extended_compute_preloaded(A, BD, A_cols, A_rows, BD_cols, BD_rows)
gemmini_extended_compute_accumulated(A, BD, A_cols, A_rows, BD_cols, BD_rows)
```

---

### 2.6 FLUSH (funct = 7)

Flushes all pending operations in the accelerator pipeline. Typically issued once
at startup to initialize the accelerator state.

**rs1:** Skip flag. If `rs1[0] = 1`, skip (cancel) any pending flush.

**rs2:** Unused (must be 0).

**C macro:**
```c
gemmini_flush(skip)    // skip = 0 for normal flush, 1 to skip
```

---

### 2.7 FENCE

Not a Gemmini-specific instruction — uses the standard RISC-V `fence` instruction
to synchronize the CPU with the accelerator. Ensures all previously issued Gemmini
commands have completed before the CPU proceeds.

**C macro:**
```c
gemmini_fence()        // emits: fence
```

---

## 3. Loop MatMul Instructions

The Loop MatMul (LOOP_WS) subsystem implements a hardware-controlled tiled matrix
multiplication loop in the Weight Stationary dataflow. Instead of issuing individual
PRELOAD/COMPUTE/MVIN/MVOUT commands from software, the hardware FSM generates the
entire sequence automatically, reducing instruction issue overhead.

The loop computes: `C[I×DIM, J×DIM] += A[I×DIM, K×DIM] × B[K×DIM, J×DIM]`

### 3.1 LOOP_WS_CONFIG_BOUNDS (funct = 9)

Sets the tile loop bounds and padding.

**rs1 layout:**
```
[47:32]  pad_K          — Padding tiles along K dimension (16-bit)
[31:16]  pad_J          — Padding tiles along J dimension (16-bit)
[15:0]   pad_I          — Padding tiles along I dimension (16-bit)
```

**rs2 layout:**
```
[47:32]  K              — Total tiles along K (reduction) dimension (16-bit)
[31:16]  J              — Total tiles along J (output columns) dimension (16-bit)
[15:0]   I              — Total tiles along I (output rows) dimension (16-bit)
```

### 3.2 LOOP_WS_CONFIG_ADDRS_AB (funct = 10)

Sets the DRAM base addresses for input matrices A and B.

**rs1:** A matrix DRAM base address (64-bit). Set to 0 to skip A loading.

**rs2:** B matrix DRAM base address (64-bit). Set to 0 to skip B loading.

### 3.3 LOOP_WS_CONFIG_ADDRS_DC (funct = 11)

Sets the DRAM base addresses for bias matrix D and output matrix C.

**rs1:** D (bias) DRAM base address (64-bit). Set to 0 for no bias.

**rs2:** C (output) DRAM base address (64-bit). Set to 0 to skip output store.

### 3.4 LOOP_WS_CONFIG_STRIDES_AB (funct = 12)

Sets the DRAM row strides for A and B matrices (in bytes).

**rs1:** A stride (64-bit, used as bytes per row).

**rs2:** B stride (64-bit, used as bytes per row).

### 3.5 LOOP_WS_CONFIG_STRIDES_DC (funct = 13)

Sets the DRAM row strides for D and C matrices (in bytes).

**rs1:** D stride (64-bit).

**rs2:** C stride (64-bit).

### 3.6 LOOP_WS (funct = 8) — Execute

Triggers the hardware loop execution. Must be preceded by CONFIG_BOUNDS and either
ADDRS_AB/DC + STRIDES_AB/DC (DRAM mode) or SPAD_AB (spad-only mode).

**rs1 layout:**
```
[19:18]  a_spad_id      — Load state ID for A matrix (2-bit, selects MVIN unit)
[17:16]  b_spad_id      — Load state ID for B matrix (2-bit)
[15:8]   act            — Activation function for output (8-bit, see codes above)
[2]      low_D          — D (bias) is in low-precision format
[1]      full_C         — C (output) is in full accumulator precision
[0]      ex_accumulate  — 1 = accumulate into existing C, 0 = overwrite
```

**rs2 layout (DRAM mode):**
```
[2]      is_resadd      — Residual addition mode (element-wise add instead of matmul)
[1]      B_transpose    — Transpose B matrix
[0]      A_transpose    — Transpose A matrix
```

**C macro (DRAM mode):**
```c
gemmini_loop_ws(I, J, K, pad_I, pad_J, pad_K,
                A, B, D, C,                    // DRAM addresses
                A_stride, B_stride, D_stride, C_stride,  // DRAM strides (bytes)
                A_transpose, B_transpose,
                full_C, low_D, ex_accumulate,
                act, a_spad_id, b_spad_id, is_resadd)
```

This macro expands to 6 instructions:
1. `LOOP_WS_CONFIG_BOUNDS` — set I, J, K and padding
2. `LOOP_WS_CONFIG_ADDRS_AB` — set A, B DRAM addresses
3. `LOOP_WS_CONFIG_ADDRS_DC` — set D, C DRAM addresses
4. `LOOP_WS_CONFIG_STRIDES_AB` — set A, B strides
5. `LOOP_WS_CONFIG_STRIDES_DC` — set D, C strides
6. `LOOP_WS` — execute

---

## 4. Loop Conv Instructions

The Loop Conv (LOOP_CONV_WS) subsystem implements hardware-controlled tiled
convolution with im2col transformation, pooling, padding, and strided access.

### 4.1 LOOP_CONV_WS_CONFIG_1 (funct = 16)

**rs1 layout:**
```
[63:48]  out_channels   — Number of output channels (16-bit)
[47:32]  in_channels    — Number of input channels (16-bit)
[31:16]  in_row_dim     — Input spatial row dimension (16-bit)
[15:0]   batch_size     — Batch size (16-bit)
```

**rs2 layout:**
```
[63:56]  padding        — Convolution padding (8-bit)
[55:48]  stride         — Convolution stride (8-bit)
[47:32]  out_col_dim    — Output spatial column dimension (16-bit)
[31:16]  pool_out_row_dim — Pooled output row dimension (16-bit)
[15:0]   out_row_dim    — Output spatial row dimension (16-bit)
```

### 4.2 LOOP_CONV_WS_CONFIG_2 (funct = 17)

**rs1 layout:**
```
[63:48]  kernel_dim     — Kernel spatial dimension (16-bit)
[47:32]  pool_out_col_dim — Pooled output column dimension (16-bit)
[31:16]  pool_size      — Pooling window size (16-bit)
[15:8]   pool_stride    — Pooling stride (8-bit)
[7:0]    pool_padding   — Pooling padding (8-bit)
```

**rs2 layout:**
```
[63:48]  batches        — Number of batch tiles (16-bit)
[47:32]  porows         — Pooled output row tiles (16-bit)
[31:16]  pocols         — Pooled output column tiles (16-bit)
[15:0]   pochs          — Pooled output channel tiles (16-bit)
```

### 4.3 LOOP_CONV_WS_CONFIG_3 (funct = 18)

**rs1 layout:**
```
[63:48]  krows          — Kernel row tiles (16-bit)
[47:32]  kcols          — Kernel column tiles (16-bit)
[31:16]  kchs           — Kernel channel tiles (16-bit)
[15:0]   lpad           — Left padding (16-bit)
```

**rs2 layout:**
```
[63:48]  rpad           — Right padding (16-bit)
[47:32]  upad           — Upper padding (16-bit)
[31:24]  dpad           — Down padding (8-bit)
[23:16]  plpad          — Pooling left padding (8-bit)
[15:0]   in_col_dim     — Input column dimension (16-bit)
```

### 4.4 LOOP_CONV_WS_CONFIG_4 (funct = 19)

**rs1 layout:**
```
[63:48]  orows          — Output row tiles (16-bit)
[47:32]  prpad          — Pooling right padding (16-bit)
[31:21]  pupad          — Pooling upper padding (11-bit)
[20:10]  pdpad          — Pooling down padding (11-bit)
[9:0]    kernel_dilation — Kernel dilation factor (10-bit)
```

**rs2 layout:**
```
[63:48]  in_stride      — Input channel stride (16-bit)
[47:32]  weight_stride  — Weight channel stride (16-bit)
[31:16]  out_stride     — Output channel stride (16-bit)
[15:0]   ocols          — Output column tiles (16-bit)
```

### 4.5 LOOP_CONV_WS_CONFIG_5 (funct = 20)

**rs1:** Weights DRAM base address (64-bit).

**rs2:** Output DRAM base address (64-bit).

### 4.6 LOOP_CONV_WS_CONFIG_6 (funct = 21)

**rs1:** Bias DRAM base address (64-bit). Set to 0 for no bias.

**rs2:** Input DRAM base address (64-bit).

### 4.7 LOOP_CONV_WS (funct = 15) — Execute

**rs1 layout:**
```
[19:18]  a_spad_id      — Load state ID for input (2-bit)
[17:16]  b_spad_id      — Load state ID for weights (2-bit)
[15:8]   max_pixels_per_row — Max pixels per im2col row (8-bit)
[7:6]    dw             — Depthwise convolution mode (2-bit)
[5]      trans_input_3120   — Transpose input (NHWC→NWHC)
[4]      trans_weight_0132  — Transpose weight layout
[3]      trans_weight_1203  — Transpose weight layout
[2]      trans_output_1203  — Transpose output layout
[1]      wrot180        — Rotate weights 180° (for transposed conv)
[0]      no_bias        — Disable bias addition
```

**rs2 layout:**
```
[5:3]    activation     — Activation function (3-bit)
[2]      input_dilated  — Input is dilated (for transposed conv)
[1]      downsample     — Enable downsampling
[0]      no_pool        — Disable pooling
```

**C macro:**
```c
gemmini_loop_conv_ws(batch_size, in_row_dim, in_col_dim, in_channels, out_channels,
    out_row_dim, out_col_dim, pool_out_row_dim, pool_out_col_dim,
    stride, padding, kernel_dim, kernel_dilation,
    pool_size, pool_stride, pool_padding,
    batches, porows, pocols, pochs, krows, kcols, kchs,
    lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad,
    orows, ocols,
    weights, output, bias, input,
    no_bias, no_pool, downsample, wrot180, input_dilated, activation,
    trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120,
    max_pixels_per_row, in_stride, weight_stride, out_stride,
    dw, a_spad_id, b_spad_id)
```

This macro expands to 7 instructions (CONFIG_1 through CONFIG_6 + LOOP_CONV_WS execute).

---

## 5. Spad-Only Instructions

These instructions enable scratchpad-to-scratchpad data movement and spad-only
matrix multiplication loops, bypassing DRAM entirely.

### 5.1 MVOUT_SPAD / STORE_SPAD (funct = 23)

Moves data between scratchpad addresses (accumulator → scratchpad or spad → spad).
Used by the spad-only loop to write compute results back to scratchpad instead of DRAM.

**rs1 layout:**
```
[63:32]  dst_stride     — Destination stride in scratchpad rows (32-bit)
[31:0]   dst_addr       — Destination scratchpad address (32-bit)
```

**rs2 layout:**
```
[63:48]  num_rows       — Number of rows to move (16-bit)
[47:32]  num_cols       — Number of columns to move (16-bit)
[31:0]   src_addr       — Source scratchpad/accumulator address (32-bit)
```

**C macros:**
```c
gemmini_mvout_spad(dst_addr, src_addr)                     // DIM×DIM, stride=1
gemmini_extended_mvout_spad(dst_addr, dst_stride, src_addr, cols, rows)
```

### 5.2 LOOP_WS_CONFIG_SPAD_AB (funct = 24)

Configures scratchpad addresses for A and B matrices in spad-only loop mode.
Replaces LOOP_WS_CONFIG_ADDRS_AB + STRIDES_AB/DC for spad-only operation.

**rs1:** A matrix scratchpad base address (row number).

**rs2:** B matrix scratchpad end address (row number).
Note: B uses `b_addr_end`, not `b_addr_start`. The hardware computes
`b_start = b_end - K * J * DIM`.

### 5.3 LOOP_WS_CONFIG_SPAD_C (funct = 25)

Configures the output scratchpad address for spad-only loop mode.

**rs1:** Unused.

**rs2:** C output scratchpad address (in bits [63:32] of the LOOP_WS rs2 register).

### 5.4 LOOP_WS in Spad-Only Mode (funct = 8, with spad_only flag)

When using spad-only mode, the LOOP_WS execute instruction encodes the C scratchpad
address and skip flags in rs2:

**rs2 layout (spad-only mode):**
```
[63:32]  C_spad_addr    — Output scratchpad address (32-bit)
[9]      spad_only      — Must be 1 (0x200) to enable spad-only mode
[5]      skip_ld_d      — Skip D (bias) load controller
[4]      skip_ld_b      — Skip B load controller
[3]      skip_ld_a      — Skip A load controller
[2]      is_resadd      — Residual addition mode
[1]      B_transpose    — Transpose B
[0]      A_transpose    — Transpose A
```

Setting `skips = 0x38` (bits 3,4,5) disables all three load controllers, so the
execute controller reads A and B directly from scratchpad addresses without any
DRAM access. Combined with `STORE_SPAD_CMD` (funct=23) to write results back to
scratchpad, this enables fully on-chip matrix multiplication chains.

**C macro (spad-only mode):**
```c
gemmini_loop_ws_spad(I, J, K, pad_I, pad_J, pad_K,
                     A_spad, B_spad_end, D_spad, C_spad,
                     A_transpose, B_transpose,
                     full_C, low_D, ex_accumulate,
                     act, a_spad_id, b_spad_id, is_resadd, skips)
```

This macro expands to 3 instructions:
1. `LOOP_WS_CONFIG_BOUNDS` — set I, J, K and padding
2. `LOOP_WS_CONFIG_SPAD_AB` — set A, B scratchpad addresses
3. `LOOP_WS` — execute with spad_only=1, C address and skips in rs2

---

## 6. Performance Counter (funct = 126)

Reads or configures hardware performance counters. Gemmini has 8 configurable
counter slots, each assignable to one of ~50 event signals.

**rs1:** Counter configuration register (bit layout depends on operation).

**rd:** Counter value (for read operations).

### Counter Configuration Register Layout

**Read / Snapshot:**
```
[7:4]    counter_index  — Counter slot (0-7)
[3]      (must be 0)    — Read mode
[2]      snapshot       — If 1, take snapshot of all counters
[1]      snapshot_reset — If 1, reset snapshot buffer
[0]      module_reset   — If 1, reset all counters
```

**Configure:**
```
[31]     non_incremental — 1 = snapshot-style (not cumulative)
[17:12]  counter_code    — Event code to monitor (6-bit, see table below)
[7:4]    counter_index   — Counter slot (0-7)
[3]      (must be 1)     — Configure mode
```

**C functions:**
```c
counter_configure(index, counter_code)  // Assign event to counter slot
counter_read(index)                     // Read counter value
counter_snapshot_take()                 // Snapshot all counters
counter_snapshot_reset()                // Reset snapshot buffer
counter_reset()                         // Reset all counters
```

---

## 7. CISC Instructions

An alternate, higher-level instruction set overlaid on funct codes 10-17. These
provide a simplified interface for basic matrix operations through a separate
command FSM (CmdFSM). Note: funct codes 10-17 overlap with LOOP_WS_CONFIG and
LOOP_CONV_WS commands; the CISC path is selected by a separate CmdFSM module.

| funct | Name | Purpose |
|-------|------|---------|
| 10 | CISC_CONFIG | Set dataflow and activation |
| 11 | ADDR_AB | Set A and B DRAM addresses |
| 12 | ADDR_CD | Set C and D DRAM addresses |
| 13 | SIZE_MN | Set M and N dimensions |
| 14 | SIZE_K | Set K dimension |
| 15 | RPT_BIAS | Set repeating bias flag |
| 16 | RESET | Reset CISC state machine |
| 17 | COMPUTE_CISC | Execute tiled matrix multiply |

---

## 8. Address Space Layout

Gemmini uses a unified 32-bit local address space for scratchpad and accumulator:

```
Bit 31 = 0: Scratchpad address
  [30:0]  Row number within scratchpad banks

Bit 31 = 1: Accumulator address
  [30]    accumulate flag (1 = add to existing, 0 = overwrite)
  [29]    read_full_value flag (1 = read full 32-bit acc, 0 = read truncated)
  [28:0]  Row number within accumulator

Special: 0xFFFFFFFF = GARBAGE_ADDR (use zeros / don't care)
```

**Scratchpad layout (for default 32×32 config):**
- 4 banks × 4096 rows = 16384 total rows
- Each row = DIM = 32 elements
- Element type = int8 (1 byte) → 32 bytes per row
- Total scratchpad = 512 KB

**Accumulator layout:**
- 2 banks × 1024 rows = 2048 total rows (double-buffered: 1024 per buffer)
- Each row = DIM = 32 elements
- Element type = int32 (4 bytes) → 128 bytes per row
- Total accumulator = 256 KB

---

## 9. Performance Counters Reference

### Incremental Counters (counter_code 1-44)

These counters increment each cycle the monitored condition is true.

**Top-Level Pipeline State:**

| Code | Name | Description |
|------|------|-------------|
| 1 | MAIN_LD_CYCLES | Cycles with only LD controller active |
| 2 | MAIN_ST_CYCLES | Cycles with only ST controller active |
| 3 | MAIN_EX_CYCLES | Cycles with only EX controller active |
| 4 | MAIN_LD_ST_CYCLES | Cycles with LD + ST active (no EX) |
| 5 | MAIN_LD_EX_CYCLES | Cycles with LD + EX active (no ST) |
| 6 | MAIN_ST_EX_CYCLES | Cycles with ST + EX active (no LD) |
| 7 | MAIN_LD_ST_EX_CYCLES | Cycles with all three controllers active |

**Load Controller:**

| Code | Name | Description |
|------|------|-------------|
| 8 | LOAD_DMA_WAIT_CYCLE | Cycles LD waiting for DMA response |
| 9 | LOAD_ACTIVE_CYCLE | Cycles LD actively processing commands |
| 10 | LOAD_SCRATCHPAD_WAIT_CYCLE | Cycles LD waiting for scratchpad write port |

**Store Controller:**

| Code | Name | Description |
|------|------|-------------|
| 11 | STORE_DMA_WAIT_CYCLE | Cycles ST waiting for DMA response |
| 12 | STORE_ACTIVE_CYCLE | Cycles ST actively processing commands |
| 13 | STORE_POOLING_CYCLE | Cycles ST performing max-pooling |
| 14 | STORE_SCRATCHPAD_WAIT_CYCLE | Cycles ST waiting for scratchpad read port |

**DMA / TLB:**

| Code | Name | Description |
|------|------|-------------|
| 15 | DMA_TLB_MISS_CYCLE | Cycles spent on TLB misses |
| 16 | DMA_TLB_HIT_REQ | Number of TLB hits (count, not cycles) |
| 17 | DMA_TLB_TOTAL_REQ | Total TLB requests |

**Read DMA:**

| Code | Name | Description |
|------|------|-------------|
| 18 | RDMA_ACTIVE_CYCLE | Cycles read-DMA is active |
| 19 | RDMA_TLB_WAIT_CYCLES | Cycles read-DMA waiting for TLB |
| 20 | RDMA_TL_WAIT_CYCLES | Cycles read-DMA waiting for TileLink response |

**Write DMA:**

| Code | Name | Description |
|------|------|-------------|
| 21 | WDMA_ACTIVE_CYCLE | Cycles write-DMA is active |
| 22 | WDMA_TLB_WAIT_CYCLES | Cycles write-DMA waiting for TLB |
| 23 | WDMA_TL_WAIT_CYCLES | Cycles write-DMA waiting for TileLink response |

**Execute Controller:**

| Code | Name | Description |
|------|------|-------------|
| 24 | EXE_ACTIVE_CYCLE | Cycles EX actively computing (mesh firing) |
| 25 | EXE_FLUSH_CYCLE | Cycles EX flushing the systolic array pipeline |
| 26 | EXE_CONTROL_Q_BLOCK_CYCLE | Cycles EX blocked on control queue |
| 27 | EXE_PRELOAD_HAZ_CYCLE | Cycles EX stalled on preload hazard |
| 28 | EXE_OVERLAP_HAZ_CYCLE | Cycles EX stalled on overlap hazard |

**Scratchpad Arbitration:**

| Code | Name | Description |
|------|------|-------------|
| 29 | SCRATCHPAD_A_WAIT_CYCLE | Cycles waiting for scratchpad A read port |
| 30 | SCRATCHPAD_B_WAIT_CYCLE | Cycles waiting for scratchpad B read port |
| 31 | SCRATCHPAD_D_WAIT_CYCLE | Cycles waiting for scratchpad D read port |

**Accumulator Arbitration:**

| Code | Name | Description |
|------|------|-------------|
| 32 | ACC_A_WAIT_CYCLE | Cycles waiting for accumulator A port |
| 33 | ACC_B_WAIT_CYCLE | Cycles waiting for accumulator B port |
| 34 | ACC_D_WAIT_CYCLE | Cycles waiting for accumulator D port |

**Garbage (No-Op) Cycles:**

| Code | Name | Description |
|------|------|-------------|
| 35 | A_GARBAGE_CYCLES | Cycles A input is garbage (zeros) |
| 36 | B_GARBAGE_CYCLES | Cycles B input is garbage (zeros) |
| 37 | D_GARBAGE_CYCLES | Cycles D input is garbage (zeros) |

**Im2Col:**

| Code | Name | Description |
|------|------|-------------|
| 38 | IM2COL_MEM_CYCLES | Cycles im2col waiting for memory |
| 39 | IM2COL_ACTIVE_CYCLES | Cycles im2col actively transforming |
| 40 | IM2COL_TRANSPOSER_WAIT_CYCLE | Cycles im2col waiting for transposer |

**Reservation Station:**

| Code | Name | Description |
|------|------|-------------|
| 41 | RESERVATION_STATION_FULL_CYCLES | Cycles reservation station is full (backpressure) |
| 42 | RESERVATION_STATION_ACTIVE_CYCLES | Cycles reservation station has entries |

**Miscellaneous:**

| Code | Name | Description |
|------|------|-------------|
| 43 | LOOP_MATMUL_ACTIVE_CYCLES | Cycles loop matmul FSM is active |
| 44 | TRANSPOSE_PRELOAD_UNROLLER_ACTIVE_CYCLES | Cycles transpose unroller is active |

### Non-Incremental Counters (counter_code > 44)

These counters provide cumulative totals rather than per-cycle state.

| Code | Name | Description |
|------|------|-------------|
| 45 | RESERVATION_STATION_LD_COUNT | Total LD commands issued |
| 46 | RESERVATION_STATION_ST_COUNT | Total ST commands issued |
| 47 | RESERVATION_STATION_EX_COUNT | Total EX commands issued |
| 48 | RDMA_BYTES_REC | Total bytes received via read-DMA |
| 49 | WDMA_BYTES_SENT | Total bytes sent via write-DMA |
| 50 | RDMA_TOTAL_LATENCY | Cumulative read-DMA latency |
| 51 | WDMA_TOTAL_LATENCY | Cumulative write-DMA latency |
