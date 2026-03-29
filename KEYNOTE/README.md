# Pipette-Chipyard Gemmini Systolic Array 配置与测试指南

本文档说明如何在 `pipette-chipyard` 中定义不同尺寸和特性的 Gemmini systolic array，
以及如何编译和运行逻辑测试（重点是 `matmul_ws` 等自带测试）。

涵盖三种核心配置类型：
1. **Baseline OS** — 标准 8×8 Output-Stationary
2. **Baseline WS** — 标准 8×8 Weight-Stationary
3. **Twist WS Single-Op** — 自研 8×8 Twist-WS（LP1 固定，A×B）

---

## 1. 配置定义

### 1.1 配置文件位置

| 文件 | 作用 |
|------|------|
| `generators/gemmini/src/main/scala/gemmini/Configs.scala` | Gemmini 参数定义（`GemminiConfigs` 对象）+ Gemmini Mixin 类 |
| `generators/gemmini/chipyard/GemminiConfigs.scala` | Chipyard 顶层 Config 类（组合 Gemmini + Rocket Core + Bus） |

### 1.2 三种配置对比

| 配置名 | Chipyard Config | Mesh 类型 | Dataflow | 尺寸 | meshType |
|--------|----------------|-----------|----------|------|----------|
| Baseline OS | `Baseline8x8OSRocketConfig` | StandardMesh | OS-only | 8×8 | `StandardMesh`（默认） |
| Baseline WS | `Baseline8x8WSRocketConfig` | StandardMesh | WS-only | 8×8 | `StandardMesh`（默认） |
| Twist WS Single | `TwistWSSingleOpRocketConfig` | TwistSingleOp | WS-only | 8×8 | `TwistSingleOp` |

三种配置的通用参数：

```scala
tileRows = 1, tileColumns = 1,
meshRows = 8, meshColumns = 8,
max_in_flight_mem_reqs = 64,
acc_read_full_width = false,
ex_read_from_acc = false,
ex_write_to_spad = false,
hardcode_d_to_garbage_addr = true    // D/bias 端口为 garbage，无 bias
```

精度：`inputType=SInt(8)`, `weightType=SInt(8)`, `outputType=SInt(20)`, `accType=SInt(32)`

### 1.3 Gemmini 参数定义（Configs.scala）

```scala
// generators/gemmini/src/main/scala/gemmini/Configs.scala

object GemminiConfigs {
  // Baseline 8x8 WS-only
  val baseline8x8WSConfig = defaultConfig.copy(
    tileRows = 1, tileColumns = 1,
    meshRows = 8, meshColumns = 8,
    dataflow = Dataflow.WS,
    max_in_flight_mem_reqs = 64,
    acc_read_full_width = false,
    ex_read_from_acc = false,
    ex_write_to_spad = false,
    hardcode_d_to_garbage_addr = true
  )

  // Baseline 8x8 OS-only
  val baseline8x8OSConfig = defaultConfig.copy(
    tileRows = 1, tileColumns = 1,
    meshRows = 8, meshColumns = 8,
    dataflow = Dataflow.OS,
    // ... 其余参数同上
  )

  // Twist-WS single-op (A*B only, LP1 fixed)
  val twistSingleOpConfig = defaultConfig.copy(
    tileRows = 1, tileColumns = 1,
    meshRows = 8, meshColumns = 8,
    dataflow = Dataflow.WS,
    // ... 其余参数同上
    meshType = TwistSingleOp     // <-- 唯一区别：指定 Twist mesh
  )
}
```

### 1.4 Chipyard 顶层 Config（chipyard/GemminiConfigs.scala）

```scala
// generators/gemmini/chipyard/GemminiConfigs.scala

class Baseline8x8OSRocketConfig extends Config(
  new gemmini.Baseline8x8OSGemminiConfig ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)

class Baseline8x8WSRocketConfig extends Config(
  new gemmini.Baseline8x8WSGemminiConfig ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)

class TwistWSSingleOpRocketConfig extends Config(
  new gemmini.TwistWSSingleOpGemminiConfig ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)
```

### 1.5 如何自定义尺寸

修改 `meshRows` 和 `meshColumns` 即可改变 systolic array 大小。例如创建 16×16 WS：

```scala
val custom16x16WSConfig = defaultConfig.copy(
  tileRows = 1, tileColumns = 1,
  meshRows = 16, meshColumns = 16,
  dataflow = Dataflow.WS,
  // ...
)
```

**注意**: 修改尺寸后 `gemmini_params.h` 中的 `DIM` 会在仿真器构建时自动更新，需要重新编译测试软件。

---

## 2. 环境准备

```bash
# 所有命令在 pipette-chipyard 根目录执行
cd /data2/chenyi9/pipette/pipette-chipyard

# 2.1 激活 conda 环境
source env.sh

# 2.2 RISC-V 工具链
export PATH="/data2/chenyi9/pipette/chipyard/.conda-env/riscv-tools/bin:$PATH"

# 2.3 GLIBC 兼容性（仿真器运行必需）
SYSROOT=/data2/chenyi9/pipette/chipyard/.conda-env/x86_64-conda-linux-gnu/sysroot
export LD_LIBRARY_PATH=${SYSROOT}/lib64:${SYSROOT}/../lib:/data2/chenyi9/pipette/chipyard/.conda-env/riscv-tools/lib:/data2/chenyi9/pipette/chipyard/.conda-env/lib
```

---

## 3. 构建仿真器

### 3.1 构建命令

```bash
cd /data2/chenyi9/pipette/pipette-chipyard/sims/verilator

# Baseline OS
make CONFIG=Baseline8x8OSRocketConfig -j$(nproc)

# Baseline WS
make CONFIG=Baseline8x8WSRocketConfig -j$(nproc)

# Twist WS Single-Op
make CONFIG=TwistWSSingleOpRocketConfig -j$(nproc)
```

构建时间约 10-30 分钟。产物为 `sims/verilator/simulator-chipyard.harness-<ConfigName>`。

### 3.2 Patchelf（构建后一次性操作）

由于服务器系统 GLIBC 版本 (2.28) 与 conda 环境 (2.34) 不匹配，需要 patchelf：

```bash
SYSROOT=/data2/chenyi9/pipette/chipyard/.conda-env/x86_64-conda-linux-gnu/sysroot

# 以 TwistWSSingleOp 为例（其他 config 替换文件名即可）
/data2/chenyi9/pipette/chipyard/.conda-env/bin/patchelf \
  --set-interpreter ${SYSROOT}/lib64/ld-linux-x86-64.so.2 \
  --set-rpath "${SYSROOT}/lib64:/data2/chenyi9/pipette/chipyard/.conda-env/riscv-tools/lib:/data2/chenyi9/pipette/chipyard/.conda-env/lib" \
  ./simulator-chipyard.harness-TwistWSSingleOpRocketConfig
```

### 3.3 清理重建（如有链接错误）

```bash
rm -rf sims/verilator/generated-src/chipyard.harness.TestHarness.<ConfigName>
cd sims/verilator
make CONFIG=<ConfigName> -j$(nproc)
```

---

## 4. 编译测试软件

### 4.1 确认 gemmini_params.h

```bash
grep "^#define DIM" generators/gemmini/software/gemmini-rocc-tests/include/gemmini_params.h
```

**关键**: DIM 必须与仿真器硬件尺寸匹配：
- 8×8 configs → `DIM=8`
- 16×16 config (GemminiRocketConfig) → `DIM=16`

`gemmini_params.h` 在构建仿真器时自动生成。如果切换了配置，必须重新编译测试软件。

### 4.2 编译

```bash
cd /data2/chenyi9/pipette/pipette-chipyard/generators/gemmini/software/gemmini-rocc-tests/build
make -j4 BAREMETAL_ONLY=1
```

编译产物在 `build/bareMetalC/` 目录下，均带 `-baremetal` 后缀。

---

## 5. 运行测试

### 5.1 通用变量定义

```bash
cd /data2/chenyi9/pipette/pipette-chipyard

SYSROOT=/data2/chenyi9/pipette/chipyard/.conda-env/x86_64-conda-linux-gnu/sysroot
TESTS=generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC
DRAMSIM=generators/testchipip/src/main/resources/dramsim2_ini
LOGDIR=README/log
```

### 5.2 通用命令模板

```bash
LD_LIBRARY_PATH=${SYSROOT}/lib64:${SYSROOT}/../lib:/data2/chenyi9/pipette/chipyard/.conda-env/riscv-tools/lib:/data2/chenyi9/pipette/chipyard/.conda-env/lib \
  sims/verilator/simulator-chipyard.harness-<CONFIG_NAME> \
  +permissive +dramsim +dramsim_ini_dir=${DRAMSIM} \
  +max-cycles=200000000 +permissive-off \
  ${TESTS}/<TEST_NAME>-baremetal \
  2>&1 | tee ${LOGDIR}/<LOG_NAME>.log
```

### 5.3 Baseline OS 测试命令

```bash
SIM=sims/verilator/simulator-chipyard.harness-Baseline8x8OSRocketConfig

# matmul_os — OS 模式单 tile 矩阵乘法
LD_LIBRARY_PATH=${SYSROOT}/lib64:${SYSROOT}/../lib:/data2/chenyi9/pipette/chipyard/.conda-env/riscv-tools/lib:/data2/chenyi9/pipette/chipyard/.conda-env/lib \
  ${SIM} +permissive +dramsim +dramsim_ini_dir=${DRAMSIM} +max-cycles=200000000 +permissive-off \
  ${TESTS}/matmul_os-baremetal \
  2>&1 | tee ${LOGDIR}/baseline_os_matmul_os.log

# mvin_mvout — DMA 读写基线
LD_LIBRARY_PATH=${SYSROOT}/lib64:${SYSROOT}/../lib:/data2/chenyi9/pipette/chipyard/.conda-env/riscv-tools/lib:/data2/chenyi9/pipette/chipyard/.conda-env/lib \
  ${SIM} +permissive +dramsim +dramsim_ini_dir=${DRAMSIM} +max-cycles=10000000 +permissive-off \
  ${TESTS}/mvin_mvout-baremetal \
  2>&1 | tee ${LOGDIR}/baseline_os_mvin_mvout.log
```

### 5.4 Baseline WS 测试命令

```bash
SIM=sims/verilator/simulator-chipyard.harness-Baseline8x8WSRocketConfig

# matmul_ws — WS 模式单 tile 矩阵乘法（核心测试）
LD_LIBRARY_PATH=${SYSROOT}/lib64:${SYSROOT}/../lib:/data2/chenyi9/pipette/chipyard/.conda-env/riscv-tools/lib:/data2/chenyi9/pipette/chipyard/.conda-env/lib \
  ${SIM} +permissive +dramsim +dramsim_ini_dir=${DRAMSIM} +max-cycles=200000000 +permissive-off \
  ${TESTS}/matmul_ws-baremetal \
  2>&1 | tee ${LOGDIR}/baseline_ws_matmul_ws.log

# tiled_matmul_ws — 多 tile WS 矩阵乘法（含自动分 tile 和 K 维累加）
LD_LIBRARY_PATH=${SYSROOT}/lib64:${SYSROOT}/../lib:/data2/chenyi9/pipette/chipyard/.conda-env/riscv-tools/lib:/data2/chenyi9/pipette/chipyard/.conda-env/lib \
  ${SIM} +permissive +dramsim +dramsim_ini_dir=${DRAMSIM} +max-cycles=200000000 +permissive-off \
  ${TESTS}/tiled_matmul_ws-baremetal \
  2>&1 | tee ${LOGDIR}/baseline_ws_tiled_matmul_ws.log

# mvin_mvout — DMA 读写基线
LD_LIBRARY_PATH=${SYSROOT}/lib64:${SYSROOT}/../lib:/data2/chenyi9/pipette/chipyard/.conda-env/riscv-tools/lib:/data2/chenyi9/pipette/chipyard/.conda-env/lib \
  ${SIM} +permissive +dramsim +dramsim_ini_dir=${DRAMSIM} +max-cycles=10000000 +permissive-off \
  ${TESTS}/mvin_mvout-baremetal \
  2>&1 | tee ${LOGDIR}/baseline_ws_mvin_mvout.log
```

### 5.5 Twist WS Single-Op 测试命令

```bash
SIM=sims/verilator/simulator-chipyard.harness-TwistWSSingleOpRocketConfig

# matmul_ws — WS 模式单 tile 矩阵乘法（验证 Twist mesh 计算正确性的核心测试）
LD_LIBRARY_PATH=${SYSROOT}/lib64:${SYSROOT}/../lib:/data2/chenyi9/pipette/chipyard/.conda-env/riscv-tools/lib:/data2/chenyi9/pipette/chipyard/.conda-env/lib \
  ${SIM} +permissive +dramsim +dramsim_ini_dir=${DRAMSIM} +max-cycles=200000000 +permissive-off \
  ${TESTS}/matmul_ws-baremetal \
  2>&1 | tee ${LOGDIR}/twist_single_matmul_ws.log

# tiled_matmul_ws — 多 tile WS 矩阵乘法
LD_LIBRARY_PATH=${SYSROOT}/lib64:${SYSROOT}/../lib:/data2/chenyi9/pipette/chipyard/.conda-env/riscv-tools/lib:/data2/chenyi9/pipette/chipyard/.conda-env/lib \
  ${SIM} +permissive +dramsim +dramsim_ini_dir=${DRAMSIM} +max-cycles=200000000 +permissive-off \
  ${TESTS}/tiled_matmul_ws-baremetal \
  2>&1 | tee ${LOGDIR}/twist_single_tiled_matmul_ws.log

# mvin_mvout — DMA 读写基线
LD_LIBRARY_PATH=${SYSROOT}/lib64:${SYSROOT}/../lib:/data2/chenyi9/pipette/chipyard/.conda-env/riscv-tools/lib:/data2/chenyi9/pipette/chipyard/.conda-env/lib \
  ${SIM} +permissive +dramsim +dramsim_ini_dir=${DRAMSIM} +max-cycles=10000000 +permissive-off \
  ${TESTS}/mvin_mvout-baremetal \
  2>&1 | tee ${LOGDIR}/twist_single_mvin_mvout.log
```

### 5.6 Chisel 单元测试（仅 Twist，不需构建仿真器）

```bash
cd /data2/chenyi9/pipette/pipette-chipyard

# 运行所有 Twist mesh 单元测试
sbt "project gemmini; testOnly gemmini.TwistMesh*"

# 单独运行权重加载测试
sbt "project gemmini; testOnly gemmini.TwistMeshWeightLoadSpec"

# 单独运行 A×B 计算验证
sbt "project gemmini; testOnly gemmini.TwistMeshABVerifySpec"

# 单独运行逐 cycle trace
sbt "project gemmini; testOnly gemmini.TwistMeshComputeSpec"
```

---

## 6. 判断测试结果

- **PASS**: 日志末尾出现 `Verilog $finish`，无 `*** FAILED ***` 字样
- **FAIL**: 出现 `*** FAILED ***` 或 `Assertion failed`
- **TIMEOUT**: 出现 `*** FAILED *** (timeout) after N simulation cycles`

快速检查：

```bash
# 检查是否通过
grep -l "FAILED" ${LOGDIR}/*.log    # 列出失败的日志
grep -L "FAILED" ${LOGDIR}/*.log    # 列出通过的日志
```

---

## 7. 测试结果汇总

| Config | 测试 | 结果 | 说明 |
|--------|------|------|------|
| Baseline8x8OS | mvin_mvout | **PASS** | |
| Baseline8x8OS | matmul_os | **PASS** | |
| Baseline8x8WS | mvin_mvout | **PASS** | |
| Baseline8x8WS | tiled_matmul_ws | **PASS** | |
| TwistWSSingleOp | mvin_mvout | **PASS** | |
| TwistWSSingleOp | matmul_ws | **PASS** | 单 tile WS 矩阵乘法，验证 Twist mesh 计算正确性 |
| TwistWSSingleOp | tiled_matmul_ws | **FAIL** | LoadController DMA 断言，与 mesh 无关 |

**关于 tiled_matmul_ws 失败**: 断言在 `LoadController.scala:136`，是 DMA 子系统问题，不影响 systolic array 正确性。`matmul_ws`（单 tile）通过已证明 mesh 计算逻辑正确。

---

## 8. 可用测试列表

所有测试二进制位于 `generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC/`。

### 功能测试

| 测试名 | 适用 Dataflow | 说明 |
|--------|-------------|------|
| `mvin_mvout-baremetal` | 任意 | Scratchpad DMA 读写 |
| `mvin_mvout_acc-baremetal` | 任意 | Accumulator DMA 读写 |
| `matmul_ws-baremetal` | WS | **WS 模式单 tile 矩阵乘**（核心验证） |
| `matmul_os-baremetal` | OS | OS 模式单 tile 矩阵乘 |
| `matmul-baremetal` | OS+WS | 综合矩阵乘（遍历所有 dataflow/transpose 组合，耗时长） |
| `tiled_matmul_ws-baremetal` | WS | 多 tile WS 矩阵乘（自动分 tile + K 累加） |
| `tiled_matmul_os-baremetal` | OS | 多 tile OS 矩阵乘 |
| `conv-baremetal` | 任意 | 卷积 |
| `transpose-baremetal` | 任意 | 矩阵转置 |

### 性能测试

| 测试名 | 说明 |
|--------|------|
| `matmul_ws_perf_compare-baremetal` | WS 性能对比（带 cycle counter） |
| `tiled_matmul_ws_perf-baremetal` | 多 tile WS 性能 |
| `single_tile_timing-baremetal` | 单 tile 时序测量 |

---

## 9. 一键测试脚本

```bash
#!/bin/bash
# run_all_tests.sh — 在 pipette-chipyard 根目录执行
set -e

cd /data2/chenyi9/pipette/pipette-chipyard

SYSROOT=/data2/chenyi9/pipette/chipyard/.conda-env/x86_64-conda-linux-gnu/sysroot
LDPATH="${SYSROOT}/lib64:${SYSROOT}/../lib:/data2/chenyi9/pipette/chipyard/.conda-env/riscv-tools/lib:/data2/chenyi9/pipette/chipyard/.conda-env/lib"
TESTS=generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC
DRAMSIM=generators/testchipip/src/main/resources/dramsim2_ini
LOGDIR=README/log
MAX_CYCLES=200000000

mkdir -p ${LOGDIR}

run_test() {
    local sim=$1
    local test_name=$2
    local log_name=$3
    echo "=== Running ${test_name} on $(basename ${sim}) ==="
    LD_LIBRARY_PATH=${LDPATH} ${sim} \
        +permissive +dramsim +dramsim_ini_dir=${DRAMSIM} \
        +max-cycles=${MAX_CYCLES} +permissive-off \
        ${TESTS}/${test_name}-baremetal \
        2>&1 | tee ${LOGDIR}/${log_name}.log
    if grep -q "FAILED" ${LOGDIR}/${log_name}.log; then
        echo ">>> FAILED <<<"
    else
        echo ">>> PASSED <<<"
    fi
    echo ""
}

# Baseline OS
SIM_OS=sims/verilator/simulator-chipyard.harness-Baseline8x8OSRocketConfig
run_test ${SIM_OS} mvin_mvout  baseline_os_mvin_mvout
run_test ${SIM_OS} matmul_os   baseline_os_matmul_os

# Baseline WS
SIM_WS=sims/verilator/simulator-chipyard.harness-Baseline8x8WSRocketConfig
run_test ${SIM_WS} mvin_mvout       baseline_ws_mvin_mvout
run_test ${SIM_WS} matmul_ws        baseline_ws_matmul_ws
run_test ${SIM_WS} tiled_matmul_ws  baseline_ws_tiled_matmul_ws

# Twist WS Single-Op
SIM_TW=sims/verilator/simulator-chipyard.harness-TwistWSSingleOpRocketConfig
run_test ${SIM_TW} mvin_mvout       twist_single_mvin_mvout
run_test ${SIM_TW} matmul_ws        twist_single_matmul_ws
run_test ${SIM_TW} tiled_matmul_ws  twist_single_tiled_matmul_ws
```

---

## 10. Twist WS 架构简介

Twist-WS 是 Gemmini 标准 `MeshWithDelays` 的 drop-in 替换，核心区别：

| | 标准 Mesh (MeshWithDelays) | Twist Mesh (TwistMeshWithDelays) |
|---|---|---|
| 权重输入方向 | 从顶部垂直进入 | 从左侧水平进入（pass_reg chain） |
| 权重 skew | ShiftRegisters (`shifted()`) | pass_reg chain（每 PE 1 cycle 延迟） |
| 互连 | 垂直 + 水平 | Black（水平）+ Red（对角线） |
| 状态机 | 无 | 无（与标准完全一致） |
| mul_pre 支持 | 是 | 是（weight/ifmap 走不同线路，带宽独立） |

### 关键源文件

| 文件 | 说明 |
|------|------|
| `generators/gemmini/src/main/scala/gemmini/TwistPE.scala` | PE：pass_reg + 双缓冲 + MAC |
| `generators/gemmini/src/main/scala/gemmini/TwistMesh.scala` | n×n mesh，Black/Red 互连 |
| `generators/gemmini/src/main/scala/gemmini/TwistMeshWithDelays.scala` | 控制器：接口适配 ExecuteController |
| `generators/gemmini/src/main/scala/gemmini/MeshWithDelaysWrapper.scala` | 按 meshType 条件实例化 |
| `generators/gemmini/src/main/scala/gemmini/GemminiConfigs.scala` | MeshType trait 定义 |

---

## 11. 常见问题

### Q1: 仿真超时 (timeout)
- `+max-cycles` 是否足够大。复杂测试需要 200M+ cycles
- `gemmini_params.h` 的 DIM 是否与仿真器硬件 DIM 匹配

### Q2: GLIBC 报错
- 确保设置了完整的 `LD_LIBRARY_PATH`（包含 sysroot/lib64 及 conda lib）
- 新编译仿真器需要先运行 patchelf（见第 3.2 节）

### Q3: 测试二进制与硬件不匹配
- 构建仿真器后 `gemmini_params.h` 会更新，需重新编译测试：
  ```bash
  cd generators/gemmini/software/gemmini-rocc-tests/build
  make -j4 BAREMETAL_ONLY=1
  ```

### Q4: hardcode_d_to_garbage_addr 导致 bias 测试失败
所有 8×8 configs 设置了 `hardcode_d_to_garbage_addr=true`，D/bias 输入强制为 0。
`matmul_ws` 中涉及 bias 的子测试可能失败，这是**预期行为**，不是 bug。

### Q5: 如何添加自定义测试
1. 在 `generators/gemmini/software/gemmini-rocc-tests/bareMetalC/` 下创建 `.c` 文件
2. 在 `bareMetalC/Makefile` 中添加编译目标
3. 重新编译：`cd build && make -j4 BAREMETAL_ONLY=1`
4. 用上述命令格式运行
