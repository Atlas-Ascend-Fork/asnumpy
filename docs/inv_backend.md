# asnumpy.linalg.inv 的 ops-solver 后端

本文说明 `asnumpy.linalg.inv` 的可选 [ops-solver](https://gitcode.com/cann/ops-solver) 后端：
真实接口契约（读源码 + 设备实测得出）、为什么需要一处上游修复、构建与选择方式、误差与性能结果、
以及明确的限制。**只影响 `inv`**，其余 linalg 入口与默认后端行为不变。

## 一、接口契约（源码依据 + 设备实测）

基线版本：ops-solver commit `91ad606ffae8800ded9ea8e9143b442f4d69eb31`（master，仓库无 tag），
另有本轮最小修复（见第三节）。设备：Ascend 910B3 / CANN 9.1.0。

| 问题 | 结论 | 依据 |
| :--- | :--- | :--- |
| 输入是原始 A 还是 LU？ | **原始矩阵 A**；接口内部自己完成 LU 分解 | `src/sgetri/sgetri_kernel.cpp`：`pad` → `custom_lu` → `custom_trsm` → `custom_trsm_upper` → `restore`；`docs/zh/sgetri.md` 原文"通常为 Sgetrf 分解后的 L 和 U 矩阵"**是错的**；实测喂 LU 误差 `6.7e-03`（对角占优）/`4.4e+00`（需选主元），喂原始 A 误差 `3.5e-08`/`1.7e-06` |
| ipiv 是否读写？ | 调用方不传 ipiv；主元数组 `W` 由 host 内部申请（初值 -1 作为跨 block 同步哨兵）、kernel 写入主元**下标**，host 不回读 | `src/sgetri/sgetri_host.cpp`（`wHost` 申请后未使用）；`src/utils/kernel/c32/getf2.hpp` `wGlobal.SetValue(offsetM + k, ...)` |
| 布局 / lda / 覆盖 / 输出位置 | **行主序**、`lda` 必须等于 `n`、原地覆盖 A 写入逆矩阵 | 非对称矩阵实测：按行主序解释残差 `7.04e-08`，按列主序 `3.08e+00`；`lda != n` 返回 `ACL_ERROR_INVALID_PARAM` |
| Host 还是 Device 指针？ | **Host**（契约）：实现把 A 当 `uint8_t*` host 指针做 H2D，末尾 D2H 写回 | `sgetri_host.cpp` 的 `reinterpret_cast<uint8_t*>(A)` + `aclrtMemcpy(..., ACL_MEMCPY_HOST_TO_DEVICE)`；官方 `test/sgetri/sgetri_test.cpp` 用 `aclrtMallocHost`。**设备指针实测"偶然可用"（前向误差 6e-8），但那是运行时按指针类型推断的结果，不是契约**（修复版在前置检查处读取非法 host 地址会崩溃，见第五节） |
| stream 与完成语义 | 取 handle 的 stream（未设则为默认流），内部 `aclrtSynchronizeStream` 后才返回 → **返回即完成** | `sgetri_host.cpp`：`aclsolverGetStream` + `aclrtMemcpyAsync(..., stream)` + `aclrtSynchronizeStream(stream)` |
| info 语义 | 入口被置 0；成功保持 0；**修复后**奇异时置 `k`（1-based，LAPACK）并返回非零 | 入口 `*info = 0`；修复前 kernel 从不写 info |
| 零主元处理 | **修复前完全缺失**：`Divide` 直接 `1/pivot`，零主元产生 inf/NaN 且仍返回 `ACL_SUCCESS`；无任何哨兵 | `getf2.hpp` 的 `Divide`（`T(1) / srcLocal.GetValue(...)`）；设备实测 `all_zero`/`duplicate_rows`/`late_zero_pivot` 全部 `ret=0, info=0`、输出非有限；`duplicate_columns` 更隐蔽：输出有限但前向误差 `1.8e16`，而**归一化残差仅 2.6e-08**（故残差阈值判据不可用） |
| 尺寸范围 | `n ∈ [1, 8192]`（超出返回参数错误）；`n=1`、非 16/128 对齐尺寸均可用 | host 校验 `LU_MAX_GETRI_N = 8192`；实测 n=1 前向误差 `1.3e-08`、n=17/33/129 亦正常 |

## 二、与 NumPy 的行为对齐

对同一批矩阵（float32，导出字节完全一致）比较 `numpy.linalg.inv`：

| 用例 | numpy | 修复前 ops-solver | 修复后 ops-solver |
| :--- | :--- | :--- | :--- |
| 常规非对称可逆 | 正常 | `ret=0`，前向误差 1.7e-07 | 同（逐位一致） |
| 零对角需换主元 | 正常 | `ret=0`，1.8e-09 | 同 |
| 全零 | **LinAlgError** | `ret=0`，输出 NaN | `ret≠0`，`info=1`，**A 未被修改** |
| 重复行 | **LinAlgError** | `ret=0`，输出 NaN | `ret≠0`，`info=16`，A 未被修改 |
| 后段零主元 | **LinAlgError** | `ret=0`，输出 NaN | `ret≠0`，`info=16`，A 未被修改 |
| 重复列（cond 9.6e16） | **正常返回有限值** | `ret=0`，有限垃圾 | 同 numpy（有限垃圾，不误报） |
| Hilbert-12 病态 | 正常（结果本身不可信） | `ret=0` | 同 |

即：修复采用了 LAPACK 的**精确零主元**判据，行为与 NumPy 一致；没有引入行列式阈值、残差阈值或
"输出含 NaN"这类判据。asnumpy 侧把"精确奇异"转换为 `numpy.linalg.LinAlgError`。

## 三、ops-solver 最小修复（依赖上游补丁）

> **依赖状态**：本修复位于本地 ops-solver 分支 `fix/issue28-sgetri-contract` = **`e68d690`**（基于
> `91ad606ffae8800ded9ea8e9143b442f4d69eb31`），**尚未合并到上游**。因此 asnumpy 侧属于
> **"依赖上游补丁的接入草稿"**：构建前必须先应用该补丁（或待上游合并后改用对应发布版本）。
> 仅凭存在 `aclsolverSgetri` 符号不能保证满足接入要求——未修复版本对奇异输入会静默返回
> `ACL_SUCCESS`。本项目不做版本自动探测，依赖路径由使用者显式给出。

文件：`src/sgetri/sgetri_host.cpp`（+ 头文件与 `docs/zh/sgetri.md` 契约订正）。

- **做什么**：在求逆 kernel 执行之后、把结果写回调用方 A 之前，用**本库自己的** `aclsolverSgetrf`
  对 A 的副本做一次 LU，若某个 `U(k,k)` **恰为 0.0** 则置 `info = k`（1-based）并返回
  `ACL_ERROR_INTERNAL_ERROR`，不写回结果、不修改调用方的 A；否则照原路径写回。
- **不做什么**：不改主元算法、不改分解/求逆 kernel、不用行列式或残差阈值、不用 CPU 求逆替代。
- **为什么放在求逆之后**：把检查放在求逆之前（额外一次 kernel/大块显存分配先行）时，在 `n=8192`
  （kernel 设计上限）观测到**间歇性非有限输出**：前置版本 9 次运行中 2 次非有限，基线与后置版本
  各自 8 次全部有限。后置放置使求逆 kernel 的执行与改动前完全一致，因此作为最终方案。
- **代价**：多一次 LU（成功路径）与一次 n² 的 host 副本；见第六节实测。奇异输入反而更省（不写回）。
- **未覆盖**：复数路径（`cgetri` 用 `c64/getf2.hpp`，未改动）与 `Sgetrf` 的同类缺口（`info` 同样只置 0）。

## 四、asnumpy 侧

- 公开签名不变：`asnumpy.linalg.inv(a)`；默认后端（`aclnnInverse`）行为不变。
- 后端选择：`ASNUMPY_LINALG_BACKEND=original`（默认）或 `ops-solver`，每调用读取；只作用于 `inv`。
- 范围：仅 **2-D 方阵、float32、NPUArray**；`n ∈ [1, 8192]`，`n == 0` 返回空结果（与 NumPy 一致）。
- 显式选择该后端时，范围外输入**明确报错、不静默回退**：
  非 ndarray → `TypeError`；非 float32 → `TypeError`；非 2-D / 非方阵 → `LinAlgError`；
  精确奇异 → `LinAlgError`；未启用该后端构建 → 带构建指引的 `RuntimeError`。
- 可观察证据：成功进入该后端的调用累加 `asnumpy.linalg._linalg._OPS_SOLVER_CALLS`。
- 输入保护与传输：输入**不会**被覆盖（库要求 host 指针，asnumpy 用自有 pinned host 暂存区
  `aclrtMallocHost`，D2H→调用→H2D 共 **2 次 n² 传输**），结果写入新分配的 NPUArray。
- 资源：handle、host 暂存区均 RAII，异常路径释放；维度与字节数做溢出/范围检查。

## 五、限制与未验证

- **A 必须是 host 指针是库的契约**；本修复在入口用 `aclrtPointerGetAttributes` 显式拒绝设备指针
  （返回 `ACL_ERROR_INVALID_PARAM`，实测 `aclrtMallocHost→HOST`、`aclrtMalloc→DEVICE`、
  `malloc→UNREGISTERED`），修复前该用法会依赖运行时推断而"偶然可用"。asnumpy 后端从不把设备
  指针交给库，故不受影响。
- 未验证：Ascend 950/A3；float16/float64/复数；批量与广播（本后端不支持，显式报错）。
- `n = 8192` 附近存在敏感性：任何在求逆 kernel 之前引入额外 kernel/大块显存的改动都可能使其
  出现非有限输出（本轮观测到 2/9 次）。当前实现为 8/8 次有限，但该上限附近的稳健性建议上游复核。
- 性能对比仅在本设备、本版本上成立，不代表其他代际或与其他 NPU 求解库的比较。
## 六、设备验证与性能

环境：Ascend 910B3 / CANN 9.1.0 / ops-solver `91ad606` + 本轮修复 / numpy 2.5.3。

### 正确性

- 后端用例 `tests/asnumpy_tests/linalg_tests/test_inv_ops_solver.py`：**29 passed，连续 3 次运行结果一致**；
  覆盖 n=1/2/3/7/16/17/33/64/129/257、单位阵/对角阵/零对角需换主元、精确奇异三类（LinAlgError 且输入不被修改）、
  数值奇异与 NumPy 同判、病态矩阵单列记录、0×0 与 1×1、非 2-D/非方阵/非 float32/host 操作数/非法后端取值、
  默认后端不受影响、重复调用稳定、连续失败后恢复。
- linalg 目录回归：**153 passed / 3 xfailed**。
- 误差定义与容差：`forward = max|X − X_ref|`（float64 独立参考）；`res_AX = max|A·X − I|`、
  `res_XA = max|X·A − I|`，归一化 `res/(max|A|·max|X| + 1)`。良态 float32 方阵在 n ≤ 257 上实测归一化残差
  ≤ 2e-6，用例上限取 1e-4（约 50 倍余量，未因不达标而放宽）。
- 病态单列：Hilbert-12（cond ≈ 1.1e9）forward ≈ 1.8e9、res_AX ≈ 1.06e1、归一化 ≈ 6.1e-9 ——
  结果不可信，按"只记录、不放宽容差、不声称通过"处理。
- 内存：n=512，50 次成功调用与 20 次奇异调用前后 RSS 均 652.4 MB（无增长）。
  设备 HBM 无法从本机 sysfs 读取（`-1`），本阶段**未测量 HBM**，仅报告 RSS。
- 独立目录安装：`pip install . --target <dir>`（非 editable）后，在**全新 venv** 中从
  `/tmp` 通过 `PYTHONPATH` 加载成功（`imported from .../asnumpy-s4-install/asnumpy/__init__.py`），
  两种后端结果均正确；`_core.so` 自带 `RPATH=/home/ma-user/work/ops-solver/build-lib`。
- 关闭依赖构建：`ASNUMPY_ENABLE_OPS_SOLVER` 默认 OFF 时可正常构建/导入，`inv` 默认路径正常，
  显式选择 ops-solver 时抛出带 `ASNUMPY_ENABLE_OPS_SOLVER` 指引的 `RuntimeError`，计数器不变，
  其余 linalg 用例 124 passed / 3 xfailed。

### 性能（公开 API 延迟，同进程交错，warm-up 5 / 重复 20，median / min，ms）

| n | 原后端 median / min | ops-solver median / min | 倍率 | 原后端 res_AX | ops-solver res_AX | max\|X_orig − X_ops\| |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1.108 / 1.062 | 0.693 / 0.655 | 1.60× | 2.20e-07 | 2.54e-07 | 4.47e-08 |
| 256 | 8.202 / 8.075 | 2.608 / 2.594 | 3.14× | 1.42e-05 | 9.19e-06 | 5.22e-06 |
| 1024 | 265.353 / 262.963 | 14.302 / 13.686 | 18.55× | 2.58e-04 | 2.51e-04 | 1.63e-04 |
| 2048 | 1962.632 / 1952.342 | 38.121 / 34.186 | 51.48× | 3.38e-04 | 3.37e-04 | 8.79e-05 |

- 口径：两侧均为**公开 API 完整调用**、同一输入/ dtype / 输出语义，交错执行、统一同步边界。
- ops-solver 列**包含**：2 次 n² Host↔Device 传输（库要求 host 指针）、每次调用一次 handle 创建/销毁、
  以及修复引入的额外一次 LU（上游奇异检查）。原后端列**包含**其 `aclnnSlogdet` 奇异判据、
  `aclnnInverse` 与相应传输。两侧开销构成不同，故只报告各自公开 API 延迟。
- 加速仅指**本设备、本版本下 ops-solver 后端相对 asnumpy 原 inv 后端**；未测纯设备执行时间，
  不代表与其他 NPU 求解库的比较。n=2048 的 51× 很大程度上来自原后端基于 slogdet 的奇异检查开销。

