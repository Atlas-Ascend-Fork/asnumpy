# Asnumpy 开发指南

## 目录

1. [项目概述](#一项目概述)
2. [开发流程](#二开发流程)
3. [Branch Management](#三branch-management)
4. [后端开发 (C++)](#四后端开发-c)
5. [绑定层 (Pybind11)](#五绑定层-pybind11)
6. [前端开发 (Python)](#六前端开发-python)
7. [测试编写](#七测试编写)
8. [编译与运行](#八编译与运行)

---

## 一、项目概述

Asnumpy 是一个基于华为昇腾 NPU 的数值计算库，提供与 NumPy 兼容的 API，将计算任务转移到 NPU 上执行。项目采用分层架构：

- **后端层 (C++)**: 使用昇腾 CANN API 实现 NPU 算子
- **绑定层 (Pybind11)**: 将 C++ 函数绑定到 Python 接口
- **前端层 (Python)**: 提供用户友好的 API，与 NumPy 接口保持一致
- **测试层**: 使用 pytest 框架和自定义测试工具验证功能正确性

### 项目目录结构

```
asnumpy/
├── asnumpy/                 # Python 包
│   ├── __init__.py         # 主包初始化
│   ├── math.py             # 数学模块
│   ├── linalg/             # 线性代数模块
│   ├── random/             # 随机数模块
│   └── ...
├── include/                # C++ 头文件
│   └── asnumpy/
│       ├── math/           # 数学模块头文件
│       └── ...
├── src/                    # C++ 源文件
│   ├── math/               # 数学模块实现
│   └── ...
├── python/                 # Pybind11 绑定
│   ├── bind_math.cpp       # 数学模块绑定
│   └── ...
├── tests/                  # 测试文件
│   └── asnumpy_tests/      # 测试用例
└── ...
```

---

## 二、开发流程

开发新功能通常遵循以下步骤：

```
1. 添加函数声明 (include/)
   ↓
2. 实现函数逻辑 (src/)
   ↓
3. 添加 Python 绑定 (python/)
   ↓
4. 添加 Python 包装层 (asnumpy/math.py)
   ↓
5. 导出到主命名空间 (asnumpy/__init__.py)
   ↓
6. 编写测试用例 (tests/)
   ↓
7. 编译并运行测试
```

---

## 三、Branch Management

### 3.1 Overview

Asnumpy uses a simplified branching model based on a single main branch with release branches for stabilization:

```
master ─── PR_A ─── PR_B ─── PR_C ─── PR_D ─── PR_E ─── PR_F
                          │                      │
                   release/v0.2.0          cherry-pick hotfix
                          │                      │
                    tag v0.2.0               tag v0.2.1
                    tag v0.2.1                    │
                                                PR to master
```

### 3.2 Branch Types

| Branch | Purpose | Lifetime |
|--------|---------|----------|
| `master` | Main branch. All feature PRs merge here. | Permanent |
| `feature/<name>` | Develop a new feature or enhancement. | Short-lived |
| `fix/<name>` | Fix a bug. | Short-lived |
| `release/vX.Y.Z` | Stabilize and test a release. | Per release |
| `hotfix/<name>` | Backport a fix from a release branch to master. | Short-lived |

### 3.3 Workflow

#### Feature Development

1. Create a feature branch from `master`:
   ```bash
   git checkout master
   git pull upstream master
   git checkout -b feature/my-feature
   ```
2. Develop, test, and commit.
3. Push and open a Pull Request targeting `master`.
4. Ensure code review and CI pass.
5. Merge the PR.

#### Release Process

1. Cut a release branch from `master`:
   ```bash
   git checkout master
   git checkout -b release/v0.3.0
   ```
2. Run tests and stabilize. Fix any issues directly on the release branch.
3. Tag the release:
   ```bash
   git tag v0.3.0
   ```
4. If additional fixes are needed after tagging, apply them on the release branch and tag a new patch version (e.g., `v0.3.1`).
5. Cherry-pick those fixes back to `master` (see [Hotfix Process](#333-hotfix-process)).
6. Delete the release branch when the release series is no longer maintained:
   ```bash
   git branch -d release/v0.3.0
   ```

#### Hotfix Process

When a bug is fixed on a release branch, the fix must be brought back to `master` so that future releases include it:

1. Fix the bug on the release branch and tag the patch release.
2. Cherry-pick the fix to a new hotfix branch:
   ```bash
   git checkout -b hotfix/backport-fix-xxx master
   git cherry-pick <commit-hash-of-the-fix>
   ```
3. Push the hotfix branch and open a Pull Request targeting `master`.
4. This PR goes through the normal code review and CI pipeline.
5. After merge, delete the hotfix branch:
   ```bash
   git branch -d hotfix/backport-fix-xxx
   ```

> **Why use a separate hotfix branch instead of cherry-picking directly to master?**
> A dedicated branch allows the fix to go through PR review and CI checks, ensuring the same quality standards as any other contribution.

### 3.4 Merge Strategy

**Always use merge commits. Never squash.**

```bash
# When merging PRs on the platform, select "Merge" (not "Squash and merge").
# When merging locally:
git merge --no-ff feature/my-feature
```

**Rationale:** Squash merges create a new commit that discards the original commit history and authorship. This breaks contribution statistics on the repository homepage and makes it harder to trace changes back to their original authors. Merge commits preserve the full history and ensure every contributor is properly credited.

### 3.5 Branch Naming Conventions

| Pattern | Example | Description |
|---------|---------|-------------|
| `feature/<name>` | `feature/add-sinc-function` | New feature or enhancement |
| `fix/<name>` | `fix/incorrect-signbit` | Bug fix |
| `release/vX.Y.Z` | `release/v0.3.0` | Release stabilization |
| `hotfix/<name>` | `hotfix/backport-signbit-fix` | Backported fix from a release branch |

Use lowercase kebab-case for branch names. Keep names concise but descriptive.

### 3.6 CI/CD Pipeline

CI/CD pipelines are triggered by events, not by branch types. Configure the following rules:

| Event | Pipeline | Description |
|-------|----------|-------------|
| PR created/updated targeting `master` | CI | Run tests, linting, and code analysis on every PR |
| Push to `master` (post-merge) | CI | Verify the merged code is healthy |
| Tag created (`vX.Y.Z`) | CD | Build release artifacts, publish packages |

No additional CI configuration is needed for `feature/`, `fix/`, or `hotfix/` branches. As long as the PR targets `master`, CI will run regardless of the source branch name.

### 3.7 Contributor Workflow

#### Team Members (with push access)

Team members create branches directly in the upstream repository:

```
upstream/master ←── PR ─── upstream/fix/xxx
```

```bash
git clone <upstream-url>
cd asnumpy
git checkout master
git pull origin master
git checkout -b fix/signbit-error
# Develop, commit, push
git push origin fix/signbit-error
# Open a PR on the platform: fix/signbit-error → master
```

#### External Contributors (without push access)

External contributors work in their own fork:

```
upstream/master ←── PR ─── fork/fix/xxx
```

```bash
git clone <fork-url>
cd asnumpy
git remote add upstream https://gitcode.com/cann/asnumpy.git
git checkout master
git pull upstream master
git checkout -b fix/signbit-error
# Develop, commit, push
git push origin fix/signbit-error
# Open a PR on the platform: fork/fix/xxx → upstream/master
```

#### Key Points

- **One PR per change.** Always create a PR directly from the working branch to `master`. There is no need for a two-step PR process (e.g., PR to a fix branch, then PR from fix branch to master).
- **All code changes go through PR.** No direct pushes to `master`.
- **All PRs go through CI and code review.** Regardless of whether the contributor is a team member or an external contributor.

### 3.8 Commit Hygiene and Review Norms

#### Commit Quality

Every commit in a PR should represent a single, meaningful change. Before submitting a PR, contributors are expected to clean up their commit history:

**Bad (fragmented commits that pad the count):**
```
fix typo
fix another typo
update import
add sinc function
add sinc test
```

**Good (clean, meaningful commits):**
```
feat: add sinc function
test: add sinc unit tests
```

#### Cleaning Up Commits Before Submitting a PR

Use interactive rebase to squash trivial commits:

```bash
git rebase -i master
# Mark trivial commits with 's' (squash) or 'f' (fixup)
# Keep only meaningful commits as 'p' (pick)
git push --force  # Update the remote branch after rebase
```

#### Reviewer Responsibilities

- Reviewers should check commit quality in addition to code quality.
- If a PR contains fragmented or trivial commits, the reviewer should request the contributor to clean them up before approving the PR.
- A PR with well-organized commit history makes it easier to understand the change, bisect bugs, and revert individual changes if needed.

#### Relationship Between Platform Settings and Commit Hygiene

| Aspect | Who is responsible | What happens |
|--------|--------------------|--------------|
| Disable platform-level squash | Platform setting | Prevents loss of author attribution |
| Merge commit strategy | Platform setting | Preserves full commit history and contribution stats |
| Commit cleanup (squash trivial commits) | Contributor (local) | Ensures meaningful commit history |
| Enforce commit quality | Reviewer | Ensures PRs have clean, organized commits |

> **Summary:** The platform is configured to always use merge commits and disallow squash to preserve authorship. Contributors are responsible for keeping their own commit history clean and meaningful. Reviewers enforce this standard during code review.

---

## 四、后端开发 (C++)

后端开发主要涉及在 C++ 层面实现 NPU 算子调用逻辑。本节以开发 `sinc` 函数为例。

### 4.1 添加函数声明

在对应的头文件中添加函数声明。`sinc` 属于数学模块的特殊函数，声明位于：

**文件位置**: `include/asnumpy/math/other_special_functions.hpp`

```cpp
/**
 * @brief Compute the normalized sinc function element-wise on the input array.
 *
 * Uses NPU operator aclnnSinc to compute:
 *     sinc(x) = sin(pi * x) / (pi * x), with sinc(0) = 1.
 *
 * @param x Input array.
 * @param dtype Optional output dtype. If not specified, uses input dtype.
 * @return NPUArray Output array with sinc applied element-wise.
 * @throws std::runtime_error If the ACL operator or memory allocation fails.
 */
NPUArray Sinc(const NPUArray& x, std::optional<py::dtype> dtype = std::nullopt);
```

### 4.2 实现函数主体

在对应的源文件中实现函数逻辑：

**文件位置**: `src/math/other_special_functions.cpp`

```cpp
NPUArray Sinc(const NPUArray& x, std::optional<py::dtype> dtype) {
    // 1. 确定输出数据类型
    py::dtype py_dtype = x.dtype;
    aclDataType in_dtype = NPUArray::GetACLDataType(py_dtype);
    aclDataType out_dtype = in_dtype;

    py::dtype out_py_dtype = NPUArray::GetPyDtype(out_dtype);
    if (dtype != std::nullopt) {
        out_py_dtype = *dtype;
        out_dtype = NPUArray::GetACLDataType(out_py_dtype);
    }

    // 2. 创建输出数组
    NPUArray out(x.shape, out_py_dtype);

    // 3. 准备 NPU 算子执行所需资源
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 4. 获取 workspace 大小和执行器
    auto error = aclnnSincGetWorkspaceSize(
        x.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string msg = "[other_special_functions.cpp](sinc) aclnnSincGetWorkspaceSize error = "
                          + std::to_string(error);
        const char* detail = aclGetRecentErrMsg();
        if (detail && std::strlen(detail) > 0) msg += " - " + std::string(detail);
        throw std::runtime_error(msg);
    }

    // 5. 分配 workspace 内存（如果需要）
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string msg = "[other_special_functions.cpp](sinc) aclrtMalloc error = "
                              + std::to_string(error);
            throw std::runtime_error(msg);
        }
    }

    // 6. 执行算子
    error = aclnnSinc(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        std::string msg = "[other_special_functions.cpp](sinc) aclnnSinc error = "
                          + std::to_string(error);
        throw std::runtime_error(msg);
    }

    // 7. 同步设备，确保计算完成
    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        std::string msg = "[other_special_functions.cpp](sinc) aclrtSynchronizeDevice error = "
                          + std::to_string(error);
        throw std::runtime_error(msg);
    }

    // 8. 释放 workspace 内存
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}
```

### 4.3 关键实现要点

#### 错误处理

所有 ACL API 调用都需要检查返回值：

```cpp
auto error = aclnnSomeFunction(/* 参数 */);
if (error != ACL_SUCCESS) {
    std::string msg = "[filename](function) aclnnSomeFunction error = "
                      + std::to_string(error);
    const char* detail = aclGetRecentErrMsg();
    if (detail && std::strlen(detail) > 0) {
        msg += " - " + std::string(detail);
    }
    throw std::runtime_error(msg);
}
```

#### NPU 算子执行流程

典型的 NPU 算子执行流程包括：

1. **GetWorkspaceSize**: 获取所需 workspace 大小和执行器
2. **Malloc**: 分配 workspace 内存（如果需要）
3. **Execute**: 执行算子计算
4. **Synchronize**: 同步设备，等待计算完成
5. **Free**: 释放 workspace 内存

---

## 五、绑定层 (Pybind11)

### 5.1 添加函数绑定

在对应的绑定文件中添加函数绑定：

**文件位置**: `python/bind_math.cpp`

```cpp
namespace asnumpy {
    void bind_other_special_functions(py::module_& math);
}

void bind_math(py::module_& math) {
    math.doc() = "math module of asnumpy";
    bind_other_special_functions(math);
}

namespace asnumpy {
    void bind_other_special_functions(py::module_& math){
        math.def("sinc", &Sinc, py::arg("x"), py::arg("dtype") = py::none());
    }
}
```

---

## 六、前端开发 (Python)

前端开发主要涉及将 C++ 函数暴露到 Python 层，并确保 API 与 NumPy 兼容。

### 6.1 添加 Python 包装层

在对应的 Python 模块中导入 C++ 函数，并添加 Python 包装层：

**文件位置**: `asnumpy/math.py`

首先，从编译好的 C++ 扩展导入函数：

```python
from .lib.asnumpy_core.math import (
    sin as _ap_sin,
    cos as _ap_cos,
    sinc as _ap_sinc,
    # ... 其他函数
)
from .utils import ndarray, _convert_dtype
```

然后，为每个函数添加 Python 包装层：

```python
def sinc(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(_ap_sinc(x, _convert_dtype(dtype)))
```

### 6.2 导出到主命名空间

在主包的 `__init__.py` 中添加函数：

**文件位置**: `asnumpy/__init__.py`

```python
from .math import (
    sin,
    sinc,
    # ... 其他数学函数
)

__all__ = [
    # ... 其他导出
    "sin",
    "sinc",
]
```

---

## 七、测试编写

asnumpy 使用 pytest 和自定义测试框架编写测试。

### 7.1 测试文件组织

测试文件按模块组织，位于 `tests/asnumpy_tests/` 目录：

```
tests/
├── conftest.py                    # pytest 配置
└── asnumpy_tests/
    ├── math_tests/
    │   └── test_miscellaneous.py  # 测试用例
    └── ...
```

### 7.2 编写测试用例

**文件位置**: `tests/asnumpy_tests/math_tests/test_miscellaneous.py`

```python
import numpy
import pytest
from asnumpy import testing

def _create_array(xp, data, dtype):
    """辅助函数：创建数组"""
    np_arr = numpy.array(data, dtype=dtype)
    if xp is numpy:
        return np_arr
    return xp.ndarray.from_numpy(np_arr)


@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-5)
def test_sinc_basic(xp, dtype):
    """基础随机测试"""
    numpy.random.seed(42)
    np_a = numpy.random.uniform(low=-5.0, high=5.0, size=(10, 10)).astype(dtype)
    a = _create_array(xp, np_a, dtype)
    return xp.sinc(a)


@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-8)
def test_sinc_at_zero(xp, dtype):
    """测试 sinc 在 x=0 处的行为"""
    data = [0.0, -0.0]
    a = _create_array(xp, data, dtype)
    return xp.sinc(a)
```

### 7.3 装饰器说明

asnumpy 提供了丰富的测试装饰器，用于参数化测试和结果比较。

#### 数据类型装饰器

- `for_dtypes(dtypes)` - 指定数据类型列表
- `for_all_dtypes()` - 所有数据类型（默认排除不支持的 float16、uint32、uint64）
- `for_float_dtypes()` - 浮点类型（float32、float64）
- `for_int_dtypes()` - 整数类型（int8-64、uint8-32）
- `for_signed_dtypes()` - 有符号整数（int8-64）
- `for_unsigned_dtypes()` - 无符号整数（uint8-16）
- `for_complex_dtypes()` - 复数类型（complex64、complex128）

```python
@testing.for_float_dtypes(no_float16=True)
def test_func(xp, dtype):
    a = xp.array([1.0, 2.0, 3.0], dtype=dtype)
    return xp.sinc(a)
```

#### 内存顺序装饰器

- `for_orders(orders)` - 指定内存顺序列表
- `for_cf_orders()` - C 和 F 顺序

```python
@testing.for_orders(['C', 'F'])
def test_func(xp, order):
    return xp.zeros((3, 3), order=order)
```

#### NumPy-Asnumpy 比较装饰器

- `numpy_asnumpy_array_equal()` - 比较数组是否完全相等
- `numpy_asnumpy_allclose(rtol=1e-7, atol=0)` - 比较浮点数组是否接近（设置相对/绝对容差）

```python
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-8)
def test_func(xp, dtype):
    a = xp.array([1.0, 2.0, 3.0], dtype=dtype)
    return xp.sinc(a)
```

#### pytest 集成装饰器

- `parameterize(*params)` - 参数化测试
- `fixture` - pytest fixture
- `skip` - 跳过测试
- `skipif(condition)` - 条件跳过
- `xfail` - 预期失败

```python
@testing.parameterize('n', [1, 2, 4])
def test_func(n):
    assert n > 0
```

---

## 八、编译与运行

### 8.1 编译项目

在开发模式下安装和编译项目：

```bash
pip install -e .
```

### 8.2 运行测试

#### 运行所有测试

```bash
pytest tests/
```

#### 运行特定模块测试

```bash
# 数学模块测试
pytest tests/asnumpy_tests/math_tests/
```

#### 运行单个测试

```bash
# 运行特定测试文件
pytest tests/asnumpy_tests/math_tests/test_miscellaneous.py

# 运行特定测试函数
pytest tests/asnumpy_tests/math_tests/test_miscellaneous.py::test_sinc_basic
```

## 附录：常用命令速查

```bash
# 安装项目
pip install -e .

# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/asnumpy_tests/math_tests/test_miscellaneous.py

# 详细输出
pytest -v

# 显示打印输出
pytest -s

# 清理并重新安装
pip uninstall asnumpy -y && pip install -e .
```

## 参考资源

- [昇腾 CANN 文档](https://www.hiascend.com/document)
- [NumPy 文档](https://numpy.org/doc/stable/)
- [Pybind11 文档](https://pybind11.readthedocs.io/)
- [pytest 文档](https://docs.pytest.org/)
