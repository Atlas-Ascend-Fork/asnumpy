# asnumpy.fft：基于 ops-fft 的 NPU 离散傅里叶变换

本文说明 `asnumpy.fft` 当前实现的范围、执行路径、精度与硬件限制。数值计算全部由 CANN
[ops-fft](https://gitcode.com/cann/ops-fft) 在 NPU 上完成；NumPy 只用于测试对照，不作为运行时回退。

## 一、已实现接口与支持范围

| 接口 | 输入 dtype | 输出 dtype | 支持范围 |
| :--- | :--- | :--- | :--- |
| `fft(a, n=None, axis=-1, norm=None)` | complex64 | complex64 | 任意维；沿一个 `axis` 变换，其余维为 batch |
| `ifft(a, n=None, axis=-1, norm=None)` | complex64 | complex64 | 同上 |
| `rfft(a, n=None, axis=-1, norm=None)` | float32 | complex64 | 同上；输出轴长度 `n//2+1` |
| `irfft(a, n=None, axis=-1, norm=None)` | complex64 | float32 | 同上；`n` 为**输出**长度 |
| `fft2(a, s=None, axes=(-2,-1), norm=None)` | complex64 | complex64 | 恰好两个轴；其余维为 batch |
| `ifft2(a, s=None, axes=(-2,-1), norm=None)` | complex64 | complex64 | 同上 |

`n` / `s` 的语义与 NumPy 一致：小于输入轴长度时截断，大于时补零。`norm` 支持
`None`/`"backward"`、`"forward"`、`"ortho"`。负 `axis`/`axes` 会按维数规范化，越界抛
`numpy.exceptions.AxisError`（NumPy 对此抛 `IndexError`，`AxisError` 是其子类）。

**已覆盖边界的对照方式**：设备测试对下列输入逐项与 numpy 2.5.3 对照（值、shape、dtype 与
输入是否被修改）：奇偶长度、多规模、`n`/`s` 的截断与补零、正负轴、批次、三种 norm、DC 与
Nyquist 分量、奇数长度显式 `n`、`irfft` 的 `n=None` 推导、非方阵与不同 `axes`、重复轴、空
batch 维、零长度变换轴、非法参数。这不等于"所有边界都与 NumPy 一致"：已知一处行为差异见下。

**空半谱的已知行为差异**：本环境中观察到 NumPy 2.5.3 对空半谱、显式 `n=4` 的输入
（`irfft(zeros(0, complex64), n=4)`）返回不稳定结果（不同进程分别观察到约 `2.03e32` 与
`6.31e19`）；本实现明确按补零规则返回零，作为已知行为差异记录。

**未实现**：`complex128`、`fftn`/`ifftn`（多于两个轴）、`rfft2`/`irfft2` 等其余 NumPy FFT
接口、`hfft`/`ihfft`、`fftfreq`/`fftshift` 辅助函数，以及 ops-blas / ops-solver 的接入。
以上用法都会显式报错，不会静默降级。

## 二、执行路径与额外开销

ops-fft 的公开接口 `aclfftExecC2C` / `aclfftExecR2C` / `aclfftExecC2R` 的参数是 **Host 指针**，
库内部完成 Host→Device、核函数执行、Device→Host 与流同步（`workSize` 恒为 0）。

**传输与同步口径**：asnumpy 包装层每次公开调用做一次输入 D2H、一次输出 H2D；ops-fft 各次
Exec 内部仍执行其自身的传输与同步。`fft2`/`ifft2` 通过两次一维 NPU FFT 完成，中间结果在 Host
上衔接（两次 Exec 之间不发生额外的 asnumpy 侧设备往返）。具体路径：

1. `asnumpy` 侧把整块输入 `aclrtMemcpy` 到 host（1 次 D2H）；
2. host 侧按 `(batch, n)` 行主序整理布局：截断/补零、沿轴打包（`axis` 为最后一维时退化为连续拷贝）；
3. 用 `aclfftPlan1d(n, type, batch, ACLFFT_HORIZONTAL)` 建一次 plan，对**整批**调用一次
   `aclfftExec*`（库内部再做 H2D、计算、D2H，并在返回前同步；本层随后调用一次
   `aclrtSynchronizeDevice` 以保持 asnumpy 既有的算子后置约定）；
4. host 侧按 NumPy 的归一化系数缩放一次，写回目标布局；
5. 结果 `aclrtMemcpy` 到新的 NPU 数组（1 次 H2D），plan 立即 `aclfftDestroy`。

一维变换（`fft`/`ifft`/`rfft`/`irfft`）每调用即上述 1 次 D2H + 1 次 plan + 1 次 Exec + 1 次 H2D；
`fft2`/`ifft2` 为 1 次 D2H + 2 次 plan/Exec（每个轴各一次，中间结果 `buffer1` 留在 host）
+ 1 次 H2D。plan 不做全局缓存（每次调用创建并销毁，符合当前阶段约定）。

## 三、归一化约定（实测）

ops-fft（commit `294cbc5`）在 Ascend 910B + CANN 9.1.0 上实测的约定：

| 方向 | 后端原始结果 | 说明 |
| :--- | :--- | :--- |
| C2C `ACLFFT_FORWARD` | 无缩放 | 与 NumPy `fft` 一致 |
| C2C `ACLFFT_BACKWARD` | 乘 N（未归一化） | 实测 `backward(forward(x)) == N·x` |
| R2C `aclfftExecR2C` | 无缩放 | 与 host DFT 最大误差 3.6e-7（n=8） |
| C2R `aclfftExecC2R` | 乘 N（未归一化） | n=6/7/8 实测原始值 = N × `numpy.irfft` |

因此包装层只做一次缩放：`None`/`"backward"` 正向系数 1、逆向 `1/n`；`"forward"` 正向 `1/n`、
逆向 1；`"ortho"` 双向 `1/sqrt(n)`。`fft2`/`ifft2` 按 NumPy 的方式**逐轴**应用对应轴的系数
（`1/n0`、`1/n1` 或 `1/sqrt(n0)`、`1/sqrt(n1)`），总缩放与二维 NumPy 语义一致。这些系数在
`src/asnumpy/fft/__init__.py::_norm_factor` 中实现，设备测试会逐项对照 NumPy。

## 四、硬件与后端限制

以下结论的证据来源逐条标注：**文档/头文件证据**指 ops-fft commit `294cbc5` 的
`include/cann_ops_fft.h` 与 `docs/zh/API_Reference/FFT_1D.md`、`FFT_2D.md`；**设备实测**指本次
在 Ascend 910B3 + CANN 9.1.0 上用该 commit 构建的镜像实跑。

- 立项目标设备为 Ascend 910B（本次实测 910B3）；**Ascend 950 未验证**（无可用设备）。
- 一维 C2C 横向（文档证据，本次未逐档穷举实测）：910B 上 `nx ≤ 256` 走 DFT（任意正整数），
  `2 的幂且 < 32768` 走 FFTB，`2 的幂且 ≥ 32768`（≤ 2^27）走 FFTN，`256 < nx ≤ 2^27` 且质因子
  ⊆ {2,3,…,47} 走 Mix；超出范围的后端返回 `ACLFFT_NOT_IMPLEMENTED`/`ACLFFT_NOT_SUPPORTED`，
  本层原样抛出。设备实测两例：`n=300`、`n=256` 成功；`n=371=7×53`、`n=257` 返回
  `aclfftResult=14`（Feature not implemented）。
- 一维 R2C/C2R（文档证据 + 设备实测缩放约定）：910B 上 `n ≤ 1024` 走 DFT（任意正整数），
  `1024 < n ≤ 2^27` 需质因子 ⊆ {2,3,…,47}；R2C 仅前向、C2R 仅后向。本次实测 n=6/7/8 均成功。
- ops-fft 在 910B 上横向 FFT 且 `nx ≥ 32768` 为 2 的幂时会**修改输入缓冲区**（文档证据，本次
  未实测该尺寸）；本实现总是先把输入复制到自有 host 缓冲，公开 API 的输入数组不会被修改
  （设备测试对已覆盖的尺寸逐项校验）。
- **二维原生接口受限**（文档证据 + 设备实测一致）：
  - 文档：`aclfftPlan2d`（算子 `fft2_d`）在 910B 仅支持 `nx, ny ∈ {32,64,128}`，其他尺寸执行时
    返回 `ACLFFT_NOT_SUPPORTED`；Ascend 950 不支持二维。
  - 设备实测（commit `294cbc5`）：`aclfftPlan2d` 对 `(64,64)`、`(32,128)` 建 plan 后
    `aclfftExecC2C` 返回 `0`（Success）；对 `(48,40)`、`(16,16)`、`(128,129)` 返回 `16`
    （Feature not supported），库同时打印
    `[ops-fft] FFT2D DD: unsupported size (48x40), only {32,64,128}x{32,64,128} is supported`。
  - 因此 `fft2`/`ifft2` 采用**两次一维 ops-fft 变换**（每个轴一次），而不是原生二维核；两次
    Exec 都真实执行在 NPU 上，中间结果在 host 上衔接。按 NumPy `_raw_fftnd` 的顺序，两次变换
    逆序执行（重复轴时这一点可观测）。
  - 该限制只对本次检查的 ops-fft commit `294cbc5` 成立；更换 ops-fft 版本需重新核对。

## 五、构建与启用

`asnumpy.fft` 依赖可选的 ops-fft：

```bash
# 1) 编译并安装 ops-fft 到私有前缀（需按目标 SoC 选择 --soc）
git clone https://gitcode.com/cann/ops-fft.git && cd ops-fft
bash build.sh --pkg --soc=Ascend910B
./build_out/cann-910b-ops-fft_*.run --full --quiet --install-path=<prefix>

# 2) 带 ops-fft 构建 asnumpy
pip install -e . --no-build-isolation \
    --config-settings=cmake.define.ASNUMPY_ENABLE_FFT=ON \
    --config-settings=cmake.define.ASNUMPY_OPS_FFT_ROOT=<prefix>
```

不开启 `ASNUMPY_ENABLE_FFT` 时 asnumpy 照常构建与导入，调用 `asnumpy.fft.*` 会得到一条
说明如何启用的 `RuntimeError`；开启但找不到依赖时 CMake 配置阶段直接失败并打印同样的指引。
依赖查找不硬编码路径、不自动下载，可用 `ASNUMPY_OPS_FFT_ROOT` 或
`ASNUMPY_OPS_FFT_INCLUDE_DIR` + `ASNUMPY_OPS_FFT_LIBRARY` 指定。

> 已知打包问题：ops-fft 的 `.run` 目前只安装 `libcann_ops_fft.so` 这个符号链接，
> 未安装其指向的 `libcann_ops_fft.so.1` / `libcann_ops_fft.so.9.0.0`，安装后该链接是悬空的。
> 需要从包的 staging 目录补齐，或直接指向 ops-fft 构建目录中的 `libcann_ops_fft.so`。

## 六、示例

```python
import numpy as np
import asnumpy as ap

x = ap.ndarray.from_numpy(np.random.default_rng(0).standard_normal((4, 128)).astype(np.complex64))

# 沿最后一维做 128 点 FFT，4 条向量共享一次批量 plan
X = ap.fft.fft(x, axis=-1)
assert X.shape == (4, 128)

# 补零到 256 点、ortho 归一化
X2 = ap.fft.fft(x, n=256, norm="ortho")

# 实数变换：float32 -> complex64，输出长度 65
r = ap.ndarray.from_numpy(np.random.default_rng(1).standard_normal(128).astype(np.float32))
R = ap.fft.rfft(r)
assert R.shape == (65,)

# 奇数长度的往返需要显式传回原始长度
odd = ap.ndarray.from_numpy(np.random.default_rng(2).standard_normal(7).astype(np.float32))
back = ap.fft.irfft(ap.fft.rfft(odd), n=7)
assert back.shape == (7,)

# 二维变换（两次一维 ops-fft 实现）
img = ap.ndarray.from_numpy(np.random.default_rng(3).standard_normal((8, 16)).astype(np.complex64))
assert ap.fft.ifft2(ap.fft.fft2(img)).shape == (8, 16)
```

## 七、精度与性能

- 后端为 FP32 实现。设备实测（complex64，含脉冲/常量/斜坡/随机输入，长度 1…1000）最坏**相对**
  误差约 `3.1e-7`（约 2.6 个 float32 ulp）；绝对误差随信号幅值增长，例如 n=1000 的斜坡信号
  变换后幅值约 `7e5`，绝对误差约 `1.6e-2`。
- 同一 `(64, 256)` complex64 输入下，批量调用中位耗时 `0.855 ms`，逐条调用 `41.893 ms`，
  本组实验约获得 **49× 加速**。结果与减少调用、plan 创建和传输次数的预期一致，尚未对各项开销
  分别计时，结论不外推到其他尺寸。

  实验口径：Ascend 910B3；输入 `(64, 256)` complex64，沿 `axis=-1` 的 64 条 256 点 FFT；
  warm-up 3 次、重复 10 次；批量口径为 `fft(x, axis=-1)`（一次 plan、一次 Exec、整块一次
  D2H/H2D，中位 0.855 ms / 最小 0.838 ms），逐条口径为 64 次 `fft(row)`（每次各自 plan、Exec、
  D2H/H2D，中位 41.893 ms / 最小 41.693 ms）。两者结果逐元素完全相同（`max|diff| = 0`），
  且都与 NumPy 一致。该加速仅指**批量相对逐条调用**，不是相对 NumPy 或其他 FFT 库。
- 其余参考耗时（public API，含 host 中转与 plan 创建/销毁；NumPy 为 CPU float64，口径不同，
  因此不作为加速结论）：`fft2 (48,40)` median 0.316 ms、`fft2 (48,40)->s=(64,32)` 0.325 ms、
  `rfft n=4096` 1.191 ms。
- 资源：60 轮 × 6 类变换的重复成功调用实测 HBM Δ0 MB、host RSS Δ0.0 MB；50 次连续失败调用
  （`n=371`，后端不支持）实测 HBM 3510→3510 MB、host RSS 582.1→582.1 MB，随后一次正常调用
  数值正确。失败路径上的 plan 由 RAII 在异常展开时销毁、host 缓冲为 `std::vector`、结果
  `NPUArray` 为局部对象（代码审计：该模块内无 `new`/`malloc`/`aclrtMalloc`/裸 `aclCreateTensor`）；
  上述内存数据是该结论的实测支撑，而不是仅凭"报错后仍能调用"推断。
