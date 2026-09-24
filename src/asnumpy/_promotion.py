# *****************************************************************************
# Copyright (c) 2025 ISE Group at Harbin Institute of Technology. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *****************************************************************************

"""NumPy 兼容的输出 dtype 提升.

提升表以本环境 NumPy (2.4.4, NEP 50 weak promotion) 实测行为为锚点:
归约/累积族 (sum/prod/cumsum/...) 将整数提升到平台整数 (int64/uint64),
fabs/sqrt 将整数提升到能容纳其位宽的最小浮点类型 (float16/32/64);
float_power 恒为 float64 (由 math.py 注入, 非本模块职责).

实际生效受 aclnn 算子能力限制 (NPU 实测, 详见
docs/superpowers/specs/2026-09-22-dtype-promotion-design.md):
aclnnReduceSum/ProdDim/Prod/NanToNum 不支持 uint8 以外的 uint,
aclnnCumprod 输入不支持 bool, aclnnCumsum 仅不支持 bool 输出 (提升到
int64 输出后 bool 输入可用); 被拦截的输入回退原 dtype (保持历史行为).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

# 归约/累积族: sum, prod, cumsum, cumprod, nansum, nanprod, nancumsum, nancumprod
# 浮点/complex 不在表中 -> fallback 保持原 dtype
_REDUCTION_TABLE: dict[np.dtype, np.dtype] = {
    np.dtype(np.bool_): np.dtype(np.int64),
    np.dtype(np.int8): np.dtype(np.int64),
    np.dtype(np.int16): np.dtype(np.int64),
    np.dtype(np.int32): np.dtype(np.int64),
    np.dtype(np.int64): np.dtype(np.int64),
    np.dtype(np.uint8): np.dtype(np.uint64),
    np.dtype(np.uint16): np.dtype(np.uint64),
    np.dtype(np.uint32): np.dtype(np.uint64),
    np.dtype(np.uint64): np.dtype(np.uint64),
}

# 一元浮点提升族: fabs, sqrt
# 规则: 能容纳整数位宽的最小浮点类型; 浮点/complex 保持
_UNARY_FLOATING_TABLE: dict[np.dtype, np.dtype] = {
    np.dtype(np.bool_): np.dtype(np.float16),
    np.dtype(np.int8): np.dtype(np.float16),
    np.dtype(np.uint8): np.dtype(np.float16),
    np.dtype(np.int16): np.dtype(np.float32),
    np.dtype(np.uint16): np.dtype(np.float32),
    np.dtype(np.int32): np.dtype(np.float64),
    np.dtype(np.uint32): np.dtype(np.float64),
    np.dtype(np.int64): np.dtype(np.float64),
    np.dtype(np.uint64): np.dtype(np.float64),
}

# aclnn 归约算子仅支持 int64 整数输出 (uint8 以外的 uint 均不支持)
_INT64_ONLY: frozenset[np.dtype] = frozenset({np.dtype(np.int64)})


def is_bool_dtype(dtype: np.dtype) -> bool:
    """能力过滤谓词: aclnnCumprod 输入不支持 bool.

    aclnnCumsum 仅限制 bool 输出 dtype, 提升到 int64 输出后 bool 输入
    可用 (NPU 实测), 故本谓词只应作用于 cumprod/nancumprod 族.
    """
    return bool(np.issubdtype(dtype, np.bool_))


def promote_reduction_dtype(
    dtype: np.dtype,
    *,
    allowed: frozenset[np.dtype] = _INT64_ONLY,
    exclude_input: Callable[[np.dtype], bool] | None = None,
) -> np.dtype:
    """归约/累积族输出 dtype: NumPy 权威表 + aclnn 能力过滤.

    输入被 exclude_input 命中, 或提升目标不在 allowed 中时, 原样返回
    (保持历史行为, 不提升).
    """
    dt = np.dtype(dtype)
    if exclude_input is not None and exclude_input(dt):
        return dt
    target = _REDUCTION_TABLE.get(dt, dt)
    return target if target in allowed else dt


def promote_unary_floating_dtype(dtype: np.dtype) -> np.dtype:
    """fabs/sqrt 族输出 dtype (前端 cast 输入绕开 aclnn 整数限制, 全表生效)."""
    dt = np.dtype(dtype)
    return _UNARY_FLOATING_TABLE.get(dt, dt)


def promote_unary_input(x: object) -> Any:
    """fabs/sqrt 输入预处理: 整数/bool 输入 cast 到提升后的浮点 dtype, 其余直通.

    浮点/complex/无 dtype 属性的输入原样返回; 整数与 bool 输入按
    NumPy 规则 cast (int8/uint8/bool->float16, int16/uint16->float32,
    更宽整数->float64), cast 均可精确表示, 无精度损失.
    """
    dt = getattr(x, "dtype", None)
    if dt is None or np.issubdtype(dt, np.floating) or np.issubdtype(dt, np.complexfloating):
        return x
    target = promote_unary_floating_dtype(dt)
    if hasattr(x, "astype"):
        return x.astype(target)
    return np.asarray(x).astype(target)
