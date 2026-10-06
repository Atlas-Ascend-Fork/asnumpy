# *****************************************************************************
# Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
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

"""逻辑运算函数测试

包含：
1. 归约运算: all, any
2. 无穷/有限检查: isfinite, isinf, isneginf, isposinf
3. 逻辑运算: logical_and, logical_or, logical_not, logical_xor
4. 比较运算: greater, less, equal, not_equal 等

优化维度：
- NaN 传播行为
- 布尔/整数/浮点各 dtype
- 广播
- 标量输入
- 空数组
"""

import numpy

from asnumpy import testing


# ========== 辅助函数 ==========
def _create_array(xp, data, dtype):
    """辅助函数：创建数组"""
    np_arr = numpy.array(data, dtype=dtype)
    if xp is numpy:
        return np_arr
    # asnumpy 环境
    return xp.ndarray.from_numpy(np_arr)


# ==========================================================================
# 1. 归约运算测试 (Reduction): all, any
# ==========================================================================


# ---------- 1.1 基础功能 ----------
@testing.for_dtypes([numpy.bool_, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_all_basic(xp, dtype):
    """测试 all (逻辑与归约)"""
    data = [True, True, False, True]
    a = _create_array(xp, data, dtype)
    return xp.all(a)


@testing.for_dtypes([numpy.bool_, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_any_basic(xp, dtype):
    """测试 any (逻辑或归约)"""
    data = [False, False, True, False]
    a = _create_array(xp, data, dtype)
    return xp.any(a)


@testing.for_dtypes([numpy.bool_, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_all_axis(xp, dtype):
    """测试带 axis 的 all"""
    data = [[True, False], [True, True]]
    a = _create_array(xp, data, dtype)
    return xp.all(a, axis=(0,))


@testing.for_dtypes([numpy.bool_, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_any_axis(xp, dtype):
    """测试带 axis 的 any"""
    data = [[False, False], [True, False]]
    a = _create_array(xp, data, dtype)
    return xp.any(a, axis=(1,))


# ---------- 1.2 多 dtype ----------
@testing.for_dtypes([numpy.bool_, numpy.int32, numpy.int64, numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_all_dtypes(xp, dtype):
    """测试 all 对多种 dtype 的支持"""
    if numpy.issubdtype(dtype, numpy.floating):
        data = [1.0, 2.0, 3.0]
    else:
        data = [1, 2, 3]
    a = _create_array(xp, data, dtype)
    return xp.all(a)


@testing.for_dtypes([numpy.bool_, numpy.int32, numpy.int64, numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_any_dtypes(xp, dtype):
    """测试 any 对多种 dtype 的支持"""
    if numpy.issubdtype(dtype, numpy.floating):
        data = [0.0, 0.0, 1.0]
    else:
        data = [0, 0, 1]
    a = _create_array(xp, data, dtype)
    return xp.any(a)


# ---------- 1.3 keepdims ----------
@testing.for_dtypes([numpy.bool_, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_all_keepdims(xp, dtype):
    """测试 all 的 keepdims 参数"""
    data = [[True, True], [True, False]]
    a = _create_array(xp, data, dtype)
    return xp.all(a, axis=(0,), keepdims=True)


@testing.for_dtypes([numpy.bool_, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_any_keepdims(xp, dtype):
    """测试 any 的 keepdims 参数"""
    data = [[False, False], [True, False]]
    a = _create_array(xp, data, dtype)
    return xp.any(a, axis=(1,), keepdims=True)


@testing.for_dtypes([numpy.bool_, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_all_global_keepdims(xp, dtype):
    """Global reduction retains one singleton dimension per input axis."""
    a = _create_array(xp, [[True, False], [True, True]], dtype)
    result = xp.all(a, keepdims=True)
    assert result.shape == (1, 1)
    return result


@testing.for_dtypes([numpy.bool_, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_any_global_keepdims(xp, dtype):
    """Global reduction retains one singleton dimension per input axis."""
    a = _create_array(xp, [[True, False], [True, True]], dtype)
    result = xp.any(a, keepdims=True)
    assert result.shape == (1, 1)
    return result


# ---------- 1.4 全 True / 全 False ----------
@testing.for_dtypes([numpy.bool_, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_all_all_true(xp, dtype):
    """测试 all: 全为 True 时应返回 True"""
    data = [True, True, True]
    a = _create_array(xp, data, dtype)
    return xp.all(a)


@testing.for_dtypes([numpy.bool_, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_any_all_false(xp, dtype):
    """测试 any: 全为 False 时应返回 False"""
    data = [False, False, False]
    a = _create_array(xp, data, dtype)
    return xp.any(a)


# ---------- 1.5 空数组 ----------
@testing.numpy_asnumpy_array_equal()
def test_all_empty(xp):
    """测试 all: 空数组应返回 True (空归约的数学定义)"""
    a = _create_array(xp, [], numpy.float32)
    return xp.all(a)


@testing.numpy_asnumpy_array_equal()
def test_any_empty(xp):
    """测试 any: 空数组应返回 False (空归约的数学定义)"""
    a = _create_array(xp, [], numpy.float32)
    return xp.any(a)


# ==========================================================================
# 2. 无穷/有限检查测试 (Finite Checks)
# ==========================================================================


# ---------- 2.1 基础功能 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_isfinite(xp, dtype):
    data = [0.0, 1.0, float("inf"), float("-inf"), float("nan")]
    a = _create_array(xp, data, dtype)
    return xp.isfinite(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_isinf(xp, dtype):
    data = [0.0, 1.0, float("inf"), float("-inf"), float("nan")]
    a = _create_array(xp, data, dtype)
    return xp.isinf(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_isposinf(xp, dtype):
    data = [0.0, float("inf"), float("-inf")]
    a = _create_array(xp, data, dtype)
    return xp.isposinf(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_isneginf(xp, dtype):
    data = [0.0, float("inf"), float("-inf")]
    a = _create_array(xp, data, dtype)
    return xp.isneginf(a)


# ---------- 2.2 NaN 行为 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_array_equal()
def test_isfinite_with_nan(xp, dtype):
    """NaN 不是有限数，应返回 False"""
    data = [float("nan"), 1.0, float("inf")]
    a = _create_array(xp, data, dtype)
    return xp.isfinite(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_isinf_with_nan(xp, dtype):
    """NaN 不是无穷，应返回 False"""
    data = [float("nan"), float("inf"), float("-inf"), 1.0]
    a = _create_array(xp, data, dtype)
    return xp.isinf(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_isposinf_with_nan(xp, dtype):
    """NaN 不是正无穷，应返回 False"""
    data = [float("nan"), float("inf"), float("-inf"), 0.0]
    a = _create_array(xp, data, dtype)
    return xp.isposinf(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_isneginf_with_nan(xp, dtype):
    """NaN 不是负无穷，应返回 False"""
    data = [float("nan"), float("inf"), float("-inf"), 0.0]
    a = _create_array(xp, data, dtype)
    return xp.isneginf(a)


# ---------- 2.3 多 dtype ----------
@testing.for_float_dtypes()
@testing.numpy_asnumpy_array_equal()
def test_isfinite_float_dtypes(xp, dtype):
    """isfinite 应支持所有浮点 dtype"""
    data = [0.0, 1.0, float("inf"), float("-inf")]
    a = _create_array(xp, data, dtype)
    return xp.isfinite(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_isinf_float_dtypes(xp, dtype):
    """isinf 应支持 float32"""
    data = [0.0, float("inf"), float("-inf"), 1.0]
    a = _create_array(xp, data, dtype)
    return xp.isinf(a)


# ---------- 2.4 全有限 / 全无穷 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_isfinite_all_finite(xp, dtype):
    """所有元素都是有限数"""
    data = [1.0, 2.0, -3.5, 0.0]
    a = _create_array(xp, data, dtype)
    return xp.isfinite(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_isinf_all_inf(xp, dtype):
    """所有元素都是无穷"""
    data = [float("inf"), float("-inf"), float("inf")]
    a = _create_array(xp, data, dtype)
    return xp.isinf(a)


# ---------- 2.5 空数组 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_isfinite_empty(xp, dtype):
    """isfinite: 空数组"""
    a = _create_array(xp, [], dtype)
    return xp.isfinite(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_isinf_empty(xp, dtype):
    """isinf: 空数组"""
    a = _create_array(xp, [], dtype)
    return xp.isinf(a)


# ==========================================================================
# 3. 逻辑运算测试 (Logical Operators)
# ==========================================================================


# ---------- 3.1 基础功能 ----------
@testing.for_dtypes([numpy.bool_, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_logical_and(xp, dtype):
    data1 = [True, False, True, False]
    data2 = [True, True, False, False]
    x1 = _create_array(xp, data1, dtype)
    x2 = _create_array(xp, data2, dtype)
    return xp.logical_and(x1, x2)


@testing.for_dtypes([numpy.bool_, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_logical_or(xp, dtype):
    data1 = [True, False, True, False]
    data2 = [True, True, False, False]
    x1 = _create_array(xp, data1, dtype)
    x2 = _create_array(xp, data2, dtype)
    return xp.logical_or(x1, x2)


@testing.for_dtypes([numpy.bool_, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_logical_xor(xp, dtype):
    data1 = [True, False, True, False]
    data2 = [True, True, False, False]
    x1 = _create_array(xp, data1, dtype)
    x2 = _create_array(xp, data2, dtype)
    return xp.logical_xor(x1, x2)


@testing.for_dtypes([numpy.bool_, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_logical_not(xp, dtype):
    data = [True, False]
    x = _create_array(xp, data, dtype)
    return xp.logical_not(x)


# ---------- 3.2 多 dtype ----------
@testing.for_dtypes(
    [numpy.bool_, numpy.int8, numpy.int32, numpy.int64, numpy.float32, numpy.float64]
)
@testing.numpy_asnumpy_array_equal()
def test_logical_and_dtypes(xp, dtype):
    """logical_and 对多种 dtype"""
    if numpy.issubdtype(dtype, numpy.floating):
        data1 = [1.0, 0.0, 1.0, 0.0]
        data2 = [1.0, 1.0, 0.0, 0.0]
    else:
        data1 = [1, 0, 1, 0]
        data2 = [1, 1, 0, 0]
    x1 = _create_array(xp, data1, dtype)
    x2 = _create_array(xp, data2, dtype)
    return xp.logical_and(x1, x2)


@testing.for_dtypes(
    [numpy.bool_, numpy.int8, numpy.int32, numpy.int64, numpy.float32, numpy.float64]
)
@testing.numpy_asnumpy_array_equal()
def test_logical_or_dtypes(xp, dtype):
    """logical_or 对多种 dtype"""
    if numpy.issubdtype(dtype, numpy.floating):
        data1 = [1.0, 0.0, 1.0, 0.0]
        data2 = [1.0, 1.0, 0.0, 0.0]
    else:
        data1 = [1, 0, 1, 0]
        data2 = [1, 1, 0, 0]
    x1 = _create_array(xp, data1, dtype)
    x2 = _create_array(xp, data2, dtype)
    return xp.logical_or(x1, x2)


@testing.for_dtypes([numpy.bool_, numpy.int32, numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_logical_not_dtypes(xp, dtype):
    """logical_not 对多种 dtype"""
    if numpy.issubdtype(dtype, numpy.floating):
        data = [1.0, 0.0, -1.0, 0.5]
    else:
        data = [1, 0, -1, 2]
    x = _create_array(xp, data, dtype)
    return xp.logical_not(x)


# ---------- 3.3 NaN 行为 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_array_equal()
def test_logical_and_with_nan(xp, dtype):
    """NaN 在逻辑运算中视为 True (非零)"""
    data1 = [float("nan"), float("nan"), 0.0, float("nan")]
    data2 = [1.0, 0.0, float("nan"), float("nan")]
    x1 = _create_array(xp, data1, dtype)
    x2 = _create_array(xp, data2, dtype)
    return xp.logical_and(x1, x2)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_array_equal()
def test_logical_or_with_nan(xp, dtype):
    """NaN 在逻辑或中视为 True"""
    data1 = [float("nan"), float("nan"), 0.0, 0.0]
    data2 = [0.0, 1.0, float("nan"), 0.0]
    x1 = _create_array(xp, data1, dtype)
    x2 = _create_array(xp, data2, dtype)
    return xp.logical_or(x1, x2)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_array_equal()
def test_logical_not_with_nan(xp, dtype):
    """NaN 的逻辑非应为 False (NaN 视为 True)"""
    data = [float("nan"), 0.0, 1.0]
    x = _create_array(xp, data, dtype)
    return xp.logical_not(x)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_array_equal()
def test_logical_xor_with_nan(xp, dtype):
    """NaN 在逻辑异或中视为 True"""
    data1 = [float("nan"), float("nan"), 0.0, 1.0]
    data2 = [0.0, 1.0, float("nan"), float("nan")]
    x1 = _create_array(xp, data1, dtype)
    x2 = _create_array(xp, data2, dtype)
    return xp.logical_xor(x1, x2)


# ---------- 3.4 广播 ----------
@testing.for_dtypes([numpy.bool_, numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_logical_and_broadcast(xp, dtype):
    """logical_and 广播: (2,3) 与 (3,)"""
    data1 = [[1, 0, 1], [0, 1, 0]]
    data2 = [1, 0, 1]
    if numpy.issubdtype(dtype, numpy.floating):
        data1 = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        data2 = [1.0, 0.0, 1.0]
    x1 = _create_array(xp, data1, dtype)
    x2 = _create_array(xp, data2, dtype)
    return xp.logical_and(x1, x2)


@testing.for_dtypes([numpy.bool_, numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_logical_or_broadcast(xp, dtype):
    """logical_or 广播: (2,1) 与 (1,3)"""
    data1 = [[1], [0]]
    data2 = [[0, 1, 1]]
    if numpy.issubdtype(dtype, numpy.floating):
        data1 = [[1.0], [0.0]]
        data2 = [[0.0, 1.0, 1.0]]
    x1 = _create_array(xp, data1, dtype)
    x2 = _create_array(xp, data2, dtype)
    return xp.logical_or(x1, x2)


@testing.for_dtypes([numpy.bool_, numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_logical_xor_broadcast(xp, dtype):
    """logical_xor 广播: (3,1) 与 (1,3)"""
    data1 = [[1], [0], [1]]
    data2 = [[1, 0, 1]]
    if numpy.issubdtype(dtype, numpy.floating):
        data1 = [[1.0], [0.0], [1.0]]
        data2 = [[1.0, 0.0, 1.0]]
    x1 = _create_array(xp, data1, dtype)
    x2 = _create_array(xp, data2, dtype)
    return xp.logical_xor(x1, x2)


# ---------- 3.5 空数组 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_logical_and_empty(xp, dtype):
    """logical_and: 空数组"""
    x1 = _create_array(xp, [], dtype)
    x2 = _create_array(xp, [], dtype)
    return xp.logical_and(x1, x2)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_logical_not_empty(xp, dtype):
    """logical_not: 空数组"""
    x = _create_array(xp, [], dtype)
    return xp.logical_not(x)


# ==========================================================================
# 4. 比较运算测试 (Comparisons)
# ==========================================================================


# ---------- 4.1 基础功能 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_greater(xp, dtype):
    numpy.random.seed(42)
    np_a = numpy.random.uniform(-10, 10, (5,)).astype(dtype)
    np_b = numpy.random.uniform(-10, 10, (5,)).astype(dtype)
    a = _create_array(xp, np_a, dtype)
    b = _create_array(xp, np_b, dtype)
    return xp.greater(a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_greater_equal(xp, dtype):
    numpy.random.seed(42)
    np_a = numpy.random.uniform(-10, 10, (5,)).astype(dtype)
    np_b = numpy.random.uniform(-10, 10, (5,)).astype(dtype)
    a = _create_array(xp, np_a, dtype)
    b = _create_array(xp, np_b, dtype)
    return xp.greater_equal(a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_less(xp, dtype):
    numpy.random.seed(42)
    np_a = numpy.random.uniform(-10, 10, (5,)).astype(dtype)
    np_b = numpy.random.uniform(-10, 10, (5,)).astype(dtype)
    a = _create_array(xp, np_a, dtype)
    b = _create_array(xp, np_b, dtype)
    return xp.less(a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_less_equal(xp, dtype):
    numpy.random.seed(42)
    np_a = numpy.random.uniform(-10, 10, (5,)).astype(dtype)
    np_b = numpy.random.uniform(-10, 10, (5,)).astype(dtype)
    a = _create_array(xp, np_a, dtype)
    b = _create_array(xp, np_b, dtype)
    return xp.less_equal(a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_equal(xp, dtype):
    data_a = [1.0, 2.0, 3.0]
    data_b = [1.0, 2.0, 4.0]
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.equal(a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_not_equal(xp, dtype):
    data_a = [1.0, 2.0, 3.0]
    data_b = [1.0, 2.0, 4.0]
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.not_equal(a, b)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_array_equal()
def test_greater_scalar(xp, dtype):
    """测试 greater_scalar (API内部强制把输入scalar转为float64)"""
    data = [1.0, 2.0, 3.0]
    scalar = 2.0
    a = _create_array(xp, data, dtype)
    return xp.greater(a, scalar)


# ---------- 4.2 多 dtype ----------
@testing.for_dtypes([numpy.bool_, numpy.int32, numpy.int64, numpy.float32, numpy.float64])
@testing.numpy_asnumpy_array_equal()
def test_greater_dtypes(xp, dtype):
    """greater 对多种 dtype"""
    if numpy.issubdtype(dtype, numpy.floating):
        data_a = [1.0, 3.0, 2.0]
        data_b = [2.0, 2.0, 2.0]
    elif dtype == numpy.bool_:
        data_a = [True, False, True]
        data_b = [False, False, True]
    else:
        data_a = [1, 3, 2]
        data_b = [2, 2, 2]
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.greater(a, b)


@testing.for_dtypes([numpy.int32, numpy.float32, numpy.float64])
@testing.numpy_asnumpy_array_equal()
def test_less_dtypes(xp, dtype):
    """less 对多种 dtype"""
    if numpy.issubdtype(dtype, numpy.floating):
        data_a = [1.0, 3.0, 2.0]
        data_b = [2.0, 2.0, 2.0]
    else:
        data_a = [1, 3, 2]
        data_b = [2, 2, 2]
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.less(a, b)


@testing.for_dtypes([numpy.int32, numpy.float32, numpy.float64])
@testing.numpy_asnumpy_array_equal()
def test_equal_dtypes(xp, dtype):
    """equal 对多种 dtype"""
    if numpy.issubdtype(dtype, numpy.floating):
        data_a = [1.0, 2.0, 3.0]
        data_b = [1.0, 2.0, 4.0]
    else:
        data_a = [1, 2, 3]
        data_b = [1, 2, 4]
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.equal(a, b)


@testing.for_dtypes([numpy.int32, numpy.float32, numpy.float64])
@testing.numpy_asnumpy_array_equal()
def test_not_equal_dtypes(xp, dtype):
    """not_equal 对多种 dtype"""
    if numpy.issubdtype(dtype, numpy.floating):
        data_a = [1.0, 2.0, 3.0]
        data_b = [1.0, 2.0, 4.0]
    else:
        data_a = [1, 2, 3]
        data_b = [1, 2, 4]
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.not_equal(a, b)


@testing.for_dtypes([numpy.int32, numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_greater_equal_dtypes(xp, dtype):
    """greater_equal 对多种 dtype"""
    if numpy.issubdtype(dtype, numpy.floating):
        data_a = [1.0, 2.0, 3.0]
        data_b = [2.0, 2.0, 2.0]
    else:
        data_a = [1, 2, 3]
        data_b = [2, 2, 2]
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.greater_equal(a, b)


@testing.for_dtypes([numpy.int32, numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_less_equal_dtypes(xp, dtype):
    """less_equal 对多种 dtype"""
    if numpy.issubdtype(dtype, numpy.floating):
        data_a = [1.0, 2.0, 3.0]
        data_b = [2.0, 2.0, 2.0]
    else:
        data_a = [1, 2, 3]
        data_b = [2, 2, 2]
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.less_equal(a, b)


# ---------- 4.3 NaN 行为 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_array_equal()
def test_greater_with_nan(xp, dtype):
    """NaN 与任何值比较都返回 False"""
    data_a = [float("nan"), 1.0, float("nan")]
    data_b = [1.0, float("nan"), float("nan")]
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.greater(a, b)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_array_equal()
def test_less_with_nan(xp, dtype):
    """NaN 与任何值比较都返回 False"""
    data_a = [float("nan"), 1.0, float("nan")]
    data_b = [1.0, float("nan"), float("nan")]
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.less(a, b)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_array_equal()
def test_greater_equal_with_nan(xp, dtype):
    """NaN >= x 始终为 False"""
    data_a = [float("nan"), 1.0, 2.0]
    data_b = [1.0, float("nan"), 2.0]
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.greater_equal(a, b)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_array_equal()
def test_less_equal_with_nan(xp, dtype):
    """NaN <= x 始终为 False"""
    data_a = [float("nan"), 1.0, 2.0]
    data_b = [1.0, float("nan"), 2.0]
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.less_equal(a, b)


@testing.suppress_warnings
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_array_equal()
def test_equal_with_nan(xp, dtype):
    """NaN == NaN 为 False (IEEE 754)"""
    data_a = [float("nan"), 1.0]
    data_b = [float("nan"), 1.0]
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.equal(a, b)


@testing.suppress_warnings
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_array_equal()
def test_not_equal_with_nan(xp, dtype):
    """NaN != NaN 为 True (IEEE 754)"""
    data_a = [float("nan"), 1.0]
    data_b = [float("nan"), 1.0]
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.not_equal(a, b)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_array_equal()
def test_greater_scalar_with_nan(xp, dtype):
    """NaN > scalar 应返回 False"""
    data = [float("nan"), 1.0, 3.0]
    a = _create_array(xp, data, dtype)
    return xp.greater(a, 2.0)


# ---------- 4.4 广播 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_greater_broadcast(xp, dtype):
    """greater 广播: (2,3) 与 (3,)"""
    data_a = [[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]]
    data_b = [2.0, 3.0, 4.0]
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.greater(a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_less_broadcast(xp, dtype):
    """less 广播: (3,1) 与 (1,3)"""
    data_a = [[1.0], [5.0], [3.0]]
    data_b = [[2.0, 4.0, 6.0]]
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.less(a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_equal_broadcast(xp, dtype):
    """equal 广播: (2,3) 与 (3,)"""
    data_a = [[1.0, 2.0, 3.0], [1.0, 4.0, 3.0]]
    data_b = [1.0, 2.0, 3.0]
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.equal(a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_not_equal_broadcast(xp, dtype):
    """not_equal 广播: (2,3) 与 (3,)"""
    data_a = [[1.0, 2.0, 3.0], [1.0, 4.0, 3.0]]
    data_b = [1.0, 2.0, 3.0]
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.not_equal(a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_greater_equal_broadcast(xp, dtype):
    """greater_equal 广播: (2,3) 与 (1,3)"""
    data_a = [[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]]
    data_b = [[2.0, 3.0, 4.0]]
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.greater_equal(a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_less_equal_broadcast(xp, dtype):
    """less_equal 广播: (2,3) 与 (3,)"""
    data_a = [[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]]
    data_b = [2.0, 3.0, 4.0]
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.less_equal(a, b)


# ---------- 4.5 标量输入 ----------
@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_array_equal()
def test_greater_scalar_values(xp, dtype):
    """greater 与标量比较"""
    data = [1.0, 2.0, 3.0, 4.0]
    a = _create_array(xp, data, dtype)
    return xp.greater(a, 2.0)


@testing.for_dtypes([numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_greater_scalar_int(xp, dtype):
    """greater: int 数组与标量"""
    data = [1, 2, 3, 4]
    a = _create_array(xp, data, dtype)
    return xp.greater(a, 2)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_array_equal()
def test_greater_equal_scalar(xp, dtype):
    """greater_equal 与标量"""
    data = [1.0, 2.0, 3.0]
    a = _create_array(xp, data, dtype)
    return xp.greater_equal(a, 2.0)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_array_equal()
def test_less_scalar(xp, dtype):
    """less 与标量"""
    data = [1.0, 2.0, 3.0]
    a = _create_array(xp, data, dtype)
    return xp.less(a, 2.0)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_array_equal()
def test_less_equal_scalar(xp, dtype):
    """less_equal 与标量"""
    data = [1.0, 2.0, 3.0]
    a = _create_array(xp, data, dtype)
    return xp.less_equal(a, 2.0)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_array_equal()
def test_equal_scalar(xp, dtype):
    """equal 与标量"""
    data = [1.0, 2.0, 3.0]
    a = _create_array(xp, data, dtype)
    return xp.equal(a, 2.0)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_array_equal()
def test_not_equal_scalar(xp, dtype):
    """not_equal 与标量"""
    data = [1.0, 2.0, 3.0]
    a = _create_array(xp, data, dtype)
    return xp.not_equal(a, 2.0)


# ---------- 4.6 空数组 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_greater_empty(xp, dtype):
    """greater: 空数组"""
    a = _create_array(xp, [], dtype)
    b = _create_array(xp, [], dtype)
    return xp.greater(a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_equal_empty(xp, dtype):
    """equal: 空数组"""
    a = _create_array(xp, [], dtype)
    b = _create_array(xp, [], dtype)
    return xp.equal(a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_less_empty(xp, dtype):
    """less: 空数组"""
    a = _create_array(xp, [], dtype)
    b = _create_array(xp, [], dtype)
    return xp.less(a, b)
