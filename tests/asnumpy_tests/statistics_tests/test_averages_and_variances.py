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

"""平均值与方差相关统计函数测试

包含：
1. 均值函数: mean

优化维度：
- axis / keepdims 参数
- dtype 参数行为
- 空数组输入
- NaN 传播行为
- 非法 axis
"""

import numpy
import pytest


# ========== 辅助函数 ==========
def _create_array(xp, data, dtype):
    """辅助函数：创建数组"""
    np_arr = numpy.array(data, dtype=dtype)
    if xp is numpy:
        return np_arr
    # asnumpy 环境
    return xp.ndarray.from_numpy(np_arr)


def _to_numpy(value):
    """辅助函数：将 asnumpy 结果转换为 NumPy 对象"""
    if hasattr(value, "to_numpy"):
        return value.to_numpy()
    return value


def _assert_mean_allclose(
    data, axis=None, keepdims=False, dtype=None, rtol=1e-5, atol=1e-5, equal_nan=False
):
    """辅助函数：比较 numpy 和 asnumpy 的 mean 结果"""
    import asnumpy as ap

    np_data = numpy.array(data)
    ap_data = ap.ndarray.from_numpy(np_data)

    np_result = numpy.mean(np_data, axis=axis, keepdims=keepdims, dtype=dtype)
    ap_result = ap.mean(ap_data, axis=axis, keepdims=keepdims, dtype=dtype)

    numpy.testing.assert_allclose(
        _to_numpy(ap_result), np_result, rtol=rtol, atol=atol, equal_nan=equal_nan
    )


def _assert_mean_dtype(data, axis=None, keepdims=False, dtype=None):
    """辅助函数：比较 mean 的返回 dtype"""
    import asnumpy as ap

    np_data = numpy.array(data)
    ap_data = ap.ndarray.from_numpy(np_data)

    np_result = numpy.mean(np_data, axis=axis, keepdims=keepdims, dtype=dtype)
    ap_result = ap.mean(ap_data, axis=axis, keepdims=keepdims, dtype=dtype)

    assert numpy.asarray(_to_numpy(ap_result)).dtype == numpy.asarray(np_result).dtype


# ============================================================================
# 1. 均值函数测试 (Mean)
# ============================================================================


# ---------- 1.1 基础功能: axis ----------
def test_mean_basic_global_float32():
    """测试 mean: 全局均值"""
    data = numpy.array([1.0, 2.0, 3.0, 4.0], dtype=numpy.float32)
    _assert_mean_allclose(data)


def test_mean_basic_global_float32_dtype():
    """测试 mean: 全局均值返回 dtype 与 NumPy 一致"""
    data = numpy.array([1.0, 2.0, 3.0, 4.0], dtype=numpy.float32)
    _assert_mean_dtype(data)


def test_mean_axis0():
    """测试 mean: axis=0"""
    data = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
    _assert_mean_allclose(data, axis=0)


def test_mean_axis0_dtype():
    """测试 mean: axis=0 返回 dtype 与 NumPy 一致"""
    data = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
    _assert_mean_dtype(data, axis=0)


def test_mean_axis1():
    """测试 mean: axis=1"""
    data = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
    _assert_mean_allclose(data, axis=1)


def test_mean_negative_axis():
    """测试 mean: 负 axis"""
    data = numpy.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=numpy.float32)
    _assert_mean_allclose(data, axis=-1)


# ---------- 1.2 keepdims 参数 ----------
def test_mean_keepdims_axis0():
    """测试 mean: axis=0 且 keepdims=True"""
    data = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
    _assert_mean_allclose(data, axis=0, keepdims=True)


def test_mean_keepdims_axis1():
    """测试 mean: axis=1 且 keepdims=True"""
    data = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
    _assert_mean_allclose(data, axis=1, keepdims=True)


def test_mean_keepdims_axis_none():
    """测试 mean: axis=None 且 keepdims=True"""
    data = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
    _assert_mean_allclose(data, keepdims=True)


# ---------- 1.3 dtype 参数 ----------
def test_mean_dtype_axis0_float64():
    """测试 mean: axis=0 且 dtype=float64"""
    data = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
    _assert_mean_allclose(data, axis=0, dtype=numpy.float64)


def test_mean_dtype_axis_none_from_int32_value():
    """测试 mean: int32 输入 axis=None 且 dtype=float64 的数值行为"""
    data = numpy.array([1, 2, 4], dtype=numpy.int32)
    _assert_mean_allclose(data, dtype=numpy.float64)


def test_mean_dtype_axis_none_from_int32_dtype():
    """测试 mean: int32 输入 axis=None 且 dtype=float64 的 dtype 一致性"""
    data = numpy.array([1, 2, 4], dtype=numpy.int32)
    _assert_mean_dtype(data, dtype=numpy.float64)


# ---------- 1.4 空数组输入 ----------
def test_mean_empty_1d():
    """测试 mean: 一维空数组"""
    data = numpy.array([], dtype=numpy.float32)
    with numpy.errstate(invalid="ignore", divide="ignore"):
        _assert_mean_allclose(data, equal_nan=True)


def test_mean_empty_axis0():
    """测试 mean: 空维度数组 axis=0"""
    data = numpy.zeros((0, 3), dtype=numpy.float32)
    with numpy.errstate(invalid="ignore", divide="ignore"):
        _assert_mean_allclose(data, axis=0, equal_nan=True)


def test_mean_empty_axis1_keepdims():
    """测试 mean: 空维度数组 axis=1 且 keepdims=True"""
    data = numpy.zeros((2, 0), dtype=numpy.float32)
    with numpy.errstate(invalid="ignore", divide="ignore"):
        _assert_mean_allclose(data, axis=1, keepdims=True, equal_nan=True)


# ---------- 1.5 NaN 传播行为 ----------
def test_mean_nan_global():
    """测试 mean: 全局归约应传播 NaN"""
    data = numpy.array([1.0, numpy.nan, 3.0], dtype=numpy.float32)
    _assert_mean_allclose(data, equal_nan=True)


def test_mean_nan_axis0():
    """测试 mean: axis=0 按切片传播 NaN"""
    data = numpy.array([[1.0, numpy.nan, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
    _assert_mean_allclose(data, axis=0, equal_nan=True)


def test_mean_nan_axis1_keepdims():
    """测试 mean: axis=1 且 keepdims=True 时传播 NaN"""
    data = numpy.array([[1.0, numpy.nan, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
    _assert_mean_allclose(data, axis=1, keepdims=True, equal_nan=True)


# ---------- 1.6 非法 axis ----------
def test_mean_invalid_axis_positive():
    """测试 mean: 正向越界 axis 应抛出异常"""
    import asnumpy as ap

    data = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
    a = ap.ndarray.from_numpy(data)
    with pytest.raises(Exception):
        ap.mean(a, axis=2)


def test_mean_invalid_axis_negative():
    """测试 mean: 负向越界 axis 应抛出异常"""
    import asnumpy as ap

    data = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
    a = ap.ndarray.from_numpy(data)
    with pytest.raises(Exception):
        ap.mean(a, axis=-3)
