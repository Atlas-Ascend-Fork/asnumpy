# asnumpy.matmul 的 ops-blas 后端

本文说明 `asnumpy.matmul` 可选的 [ops-blas](https://gitcode.com/cann/ops-blas) 后端：设备支持核查
结论、接口与布局映射、后端选择方式、构建方法、验证结果与限制。

## 一、设备支持核查（先核查，后实现）

固定版本与设备（本文件所有"实测"均来自这一组环境）：

| 项 | 值 |
| :--- | :--- |
| 设备 | Ascend 910B3（`npu-smi` 25.5.1，CANN 9.1.0，`ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0`） |
| ops-blas | commit `c5d9e051b93c4d769d15cdf9ea1342a485e31469`（仓库无 tag），按 `--soc=ascend910b3` → arch22 构建 |
| 头文件 | `include/cann_ops_blas.h`、`include/cann_ops_blas_common.h`、`include/cann_ops_blasLt.h` |

**经典 GEMM 在 910B 上不存在**（不是"函数名存在即支持"）：

| 候选接口 | 910B(arch22) 实现 | 证据 |
| :--- | :--- | :--- |
| `aclblasSgemm` | **无** | `blas/gemm/` 只有 `arch35/`；官方 README 标注 Atlas A2 不支持；arch22 构建产物中**无该符号**；`dlsym` 探测 ABSENT |
| `aclblasGemmEx` / `aclblasSgemmEx` | 无 | 同上（`blas/gemm_ex/` 仅 arch35） |
| `aclblasSgemmBatched` / `aclblasGemmBatchedEx` | 无 | `blas/gemm_batched*/` 仅 arch35；README 明确 "Atlas A2 训练/推理系列产品：不支持" |
| `aclblasSgemmStridedBatched` | **有（arch22）** | `blas/gemm_strided_batched/arch22/` 有 kernel/host 实现；README "Atlas A2 训练/推理系列（含 Atlas 800I A2）：支持（arch22），Ascend910B3：1312 项测试、200 项性能全部通过"；arch22 构建产物导出该符号（`gemm_sb22_cube/tiny/vector` 内核） |
| `aclblasLtMatmul`（blasLt） | 无 arch22 kernel | `blasLt/` 下只有 `api/` 与 `matrix_transform/`，无 arch22 GEMM 实现；`test/blasLtMatmul/` 只有 `arch35` |

因此本后端使用 **`aclblasSgemmStridedBatched` 且 `batchCount = 1`**：官方文档对该接口的定义是
`C_i = alpha * op(A_i) * op(B_i) + beta * C_i`，`batchCount = 1` 时就是一次稠密 GEMM。
这不是替代算法，也不是用 aclnnMatmul 或 CPU 计算冒充 ops-blas。

### 最小设备探针（实现前置条件，已通过）

探针（源码见提交信息中的 `opsblas_gemm_probe.cpp` 记录）在同一环境实测：

```
entry points: aclblasSgemm ABSENT, aclblasGemmEx ABSENT, aclblasSgemmBatched ABSENT,
              aclblasSgemmStridedBatched PRESENT
aclblasCreate/aclblasSetStream/aclblasGetVersion -> SUCCESS (version 10000)
non-square dense GEMM (batchCount=1, swap mapping):
  7x5x3      rel err 4.5e-08   inputs_intact=yes
  1x9x4      rel err 4.2e-08   inputs_intact=yes
  64x128x32  rel err 8.0e-08   inputs_intact=yes
  129x257x33 rel err 1.2e-07   inputs_intact=yes
  512x512x512 rel err 4.2e-07  inputs_intact=yes
workspace: 默认（库自带）与 512 B / 64 KiB / 1 MiB 均 OK
batchCount=2 有 strided 批量正确
边界：m=0 -> SUCCESS；batchCount=0 -> SUCCESS；k=0,beta=0 -> C 置 0；alpha=0,beta=1 -> C 不变
SUMMARY: PROBE_OK
```

**异步与同步**：kernel 提交到 handle 绑定的 stream，调用返回时**尚未完成**。探针第一次实现同步了
默认 stream（`aclrtSynchronizeStream(nullptr)`）而在小尺寸上读到未写入的 C —— 那是探针自身的
同步错误，改用 `aclrtSynchronizeDevice()` 后全部正确。本后端沿用在每次调用后
`aclrtSynchronizeDevice()` 的既有约定，保证输入与输出在 kernel 执行期间始终有效。

## 二、接口与布局映射

`aclblasSgemmStridedBatched(handle, transA, transB, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)`

- `alpha`/`beta` 是 **Host 指针**（本实现传 `&alpha`，`alpha=1.0f`、`beta=0.0f`）；
- `A`/`B`/`C` 是 **Device 指针**、**列主序**、`lda/ldb/ldc` 遵循 NETLIB/CBLAS 语义；
- handle 携带 stream（`aclblasSetStream`），本实现传 `nullptr` 即默认 stream，与 asnumpy 其他算子
  提交到 null stream 的既有做法一致；
- workspace 由库默认管理（探针在 512³ 与 1 MiB 用户 workspace 下都正确），本实现不覆盖。

asnumpy 数组是行主序，因此采用**交换操作数**的等价映射（无拷贝、无转置）：

```
C_rm(m x n) = A_rm(m x k) * B_rm(k x n)
   <=>  列主序语义下   C_cm(n x m) = B_cm(n x k) * A_cm(k x m)
```

即调用时 `m_blas = n`、`n_blas = m`、`k_blas = k`，BLAS 的 A 操作数取 `x2` 缓冲区（`lda = n`）、
B 操作数取 `x1` 缓冲区（`ldb = k`）、C 取结果缓冲区（`ldc = n`）。行主序 (M,N) 缓冲同时就是列主序
(N,M) 缓冲（`ldc = n`），行主序 (K,N) 缓冲同时是列主序 (N,K)（`lda = n`），行主序 (M,K) 缓冲同时是
列主序 (K,M)（`ldb = k`），故三个缓冲区可直接复用。该映射由**非方阵**用例（含 1×9、7×5、129×257）
在设备上验证，不是只测方阵。

## 三、后端选择

```python
import asnumpy as ap

ap.matmul(a, b)                      # 默认：原实现（Host NumPy 路径），行为与之前完全一致
```

```bash
ASNUMPY_MATMUL_BACKEND=ops-blas python your_script.py   # 显式改用 ops-blas
```

- 取值只有 `original`（默认）与 `ops-blas`；其他取值立即抛 `ValueError`。
- 每调用读取一次环境变量，便于测试确定性地切换。
- 显式选择 ops-blas 时，**只有 2-D float32 NPU 数组**走该后端；不满足时给出明确异常
  （`NotImplementedError`/`TypeError`/`ValueError`），不静默回退到原路径。
- 用户要求"不能因启用可选后端而让公开 API 功能缩水"：默认仍是原实现，原实现对任意维、
  广播、float64、其他 dtype 的能力不受影响。
- 可观察证据：成功进入 ops-blas 的调用会累加 `asnumpy.linalg.direct.ops_blas_dispatches()`，
  测试据此断言结果确实由 ops-blas 产生，而不是"碰巧通过"。

## 四、构建

```bash
# 1) 编译 ops-blas（910B 只需 gemm_strided_batched；测试目标需要系统 Netlib BLAS/GTest，
#    只编库时用 BUILD_TEST=OFF）
git clone https://gitcode.com/cann/ops-blas.git && cd ops-blas
cmake -S . -B build-lib -DSOC_VERSION=ascend910b3 -DBUILD_TEST=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-lib -j

# 2) 带 ops-blas 构建 asnumpy
pip install -e . --no-build-isolation \
    --config-settings=cmake.define.ASNUMPY_ENABLE_OPS_BLAS=ON \
    --config-settings=cmake.define.ASNUMPY_OPS_BLAS_ROOT=<ops-blas checkout>
```

- `ASNUMPY_ENABLE_OPS_BLAS` 默认 `OFF`：不启用时构建、导入与 `matmul` 完全不受影响。
- 显式启用但找不到依赖时，CMake 配置阶段直接失败并打印构建指引。
- 依赖查找不硬编码个人路径、不自动联网下载、不覆盖共享 CANN；可用
  `ASNUMPY_OPS_BLAS_ROOT`，或直接给 `ASNUMPY_OPS_BLAS_INCLUDE_DIR` + `ASNUMPY_OPS_BLAS_LIBRARY`。
- `libops_blas.so` 所在目录会写入 `_core.so` 的 RUNPATH，安装后从构建目录之外也可加载。

## 五、范围与限制

- 仅支持 **2-D × 2-D、同为 float32、内维匹配**的 NPU 数组；不扩展批量、广播、复数、混合 dtype、
  solver，也不手写 GEMM 内核。
- `m == 0` 或 `n == 0`：ops-blas 文档规定"不计算、直接 SUCCESS 且不写 C"，本后端改为直接返回
  空结果数组，避免把未写入的缓冲区返回给调用方。
- `k == 0`：按文档语义 `C := beta * C` 且 `beta = 0` 时置零（设备实测 `max|C| = 0`）。
  实现注意：`k == 0` 时 `ldb` 必须取 `max(1, k)` 而不是 `0`，否则后端返回
  `ACLBLAS_STATUS_INVALID_VALUE`（实现过程中实测到并在设备测试中固化）。
- **未验证**：Ascend 950（无设备）、float16（后端未提供所需 FP32 累加保证前不接入）、
  `blasLt` 路径（910B 无 kernel）、负 stride、`matmul` 的 >2 维/广播输入在该后端下的行为。
- 每次调用创建一个 handle（不做全局 handle/plan 缓存），避免并发共享可变 stream 状态。

## 六、设备验证与性能

### 正确性（Ascend 910B3，ops-blas `c5d9e05`，CANN 9.1.0，numpy 2.5.3）

- 新增 `tests/asnumpy_tests/linalg_tests/test_matmul_ops_blas.py`：**26 passed**。
  覆盖 1×1、2×3×4、7×5、64×128×32、129×257×33、256³、1×64×128、128×64×1 等非方阵/非对齐
  形状；K=0、M=0/N=0、单位阵与零阵、大动态范围与相消；内维不匹配、非 2-D、非 float32、
  host numpy 操作数、非法后端取值等报错路径；输入不被修改；重复 20 次结果一致；10 次连续
  失败后仍能正确调用；默认后端不触碰 ops-blas（计数不变）。
- linalg 目录回归：**150 passed, 3 xfailed**（既有用例不受影响）。
- 关闭 ops-blas 的构建：可正常构建/导入，`matmul` 默认路径正常，显式选择后端时给出可操作
  `RuntimeError`（含 `-DASNUMPY_ENABLE_OPS_BLAS=ON` 指引），非本后端用例全部通过。
- 数值对照：与 NumPy 及原实现逐元素对照，float32 容差 `rtol=1e-4 / atol=1e-4`；大 K 情形下
  ops-blas 与原实现的 `max|diff|` 见下表（K=2048 时 3.4e-4，相对约 1e-5，属 FP32 累加差异）。

### 性能（public API 延迟，同一进程内交错运行，warm-up 5 / 重复 20，中位数与最小值）

| 形状 (M×K×N) | 原实现 median / min (ms) | ops-blas median / min (ms) | 倍率 | max&#124;diff&#124; |
| :--- | ---: | ---: | ---: | ---: |
| 64×64×64 | 0.211 / 0.208 | 0.143 / 0.140 | 1.47× | 9.54e-06 |
| 1024×1024×1024 | 6.483 / 6.416 | 0.904 / 0.873 | 7.17× | 2.21e-04 |
| 2048×2048×2048 | 103.493 / 99.698 | 2.715 / 2.373 | 38.12× | 3.36e-04 |
| 512×2048×512 | 4.182 / 4.089 | 0.397 / 0.389 | 10.54× | 3.20e-04 |
| 2048×512×512 | 4.687 / 4.644 | 0.631 / 0.622 | 7.43× | 1.30e-04 |

- 计时口径：两侧都测**公开 API 完整调用**，同一设备、同一输入、同一 dtype（float32）、同一
  精度模式，基线与候选在同一次循环中交错执行。
- ops-blas 列**包含每次调用创建一个 handle 的开销**（本阶段不做 handle 缓存）；原实现列
  **包含两次输入 D2H 与一次结果 H2D**（其 NumPy 回退在 Host 上计算）。二者口径一致但开销构成
  不同，因此这里只报告各自的公开 API 延迟，不把差额归因于单一因素。
- 加速仅指 **ops-blas 后端相对 asnumpy 原 matmul 后端**，不是相对 NumPy 或其他 BLAS 库；
  也没有单独的"纯设备执行时间"测量。
- **未验证**：Ascend 950 与 float16 的性能（无设备 / 未接入）。
