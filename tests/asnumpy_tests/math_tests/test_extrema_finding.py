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

import numpy
import pytest

import asnumpy as ap
from asnumpy import nanmin as top_level_nanmin
from asnumpy import testing
from asnumpy.math import nanmin as math_nanmin


@testing.for_all_dtypes(no_complex=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_maximum(xp, dtype):
    """测试 maximum(x1, x2, dtype=None) - 逐元素最大值"""
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42)
    b = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=43)
    return xp.maximum(a, b)


@testing.for_all_dtypes(no_complex=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_minimum(xp, dtype):
    """测试 minimum(x1, x2, dtype=None) - 逐元素最小值"""
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42)
    b = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=43)
    return xp.minimum(a, b)


@testing.for_all_dtypes(no_complex=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_fmax(xp, dtype):
    """测试 fmax(x1, x2, dtype=None) - 逐元素最大值（忽略NaN）"""
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42)
    b = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=43)
    return xp.fmax(a, b)


@testing.for_all_dtypes(no_complex=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_fmin(xp, dtype):
    """测试 fmin(x1, x2, dtype=None) - 逐元素最小值（忽略NaN）"""
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42)
    b = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=43)
    return xp.fmin(a, b)


@testing.for_all_dtypes(no_complex=True, exclude=[numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_max(xp, dtype):
    """测试 max(a) - 数组最大值"""
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42)
    return xp.max(a)


@testing.for_all_dtypes(no_complex=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_max_dim(xp, dtype):
    """测试 max(a, axis, keepdims) - 数组沿特定维度的最大值"""
    a = testing.shaped_random((3, 4, 5), dtype=dtype, xp=xp, seed=42)
    return xp.max(a, axis=1, keepdims=True)


@testing.for_all_dtypes(no_complex=True, exclude=[numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_min(xp, dtype):
    """测试 min(a) - 数组最大值"""
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42)
    return xp.min(a)


@testing.for_all_dtypes(no_complex=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_min_dim(xp, dtype):
    """测试 min(a, axis, keepdims) - 数组沿特定维度的最大值"""
    a = testing.shaped_random((3, 4, 5), dtype=dtype, xp=xp, seed=42)
    return xp.min(a, axis=1, keepdims=True)


# ========== 5. 别名与 NaN 感知入口 (Amax, Amin, Nanmin, Nanmax) ==========


@testing.for_all_dtypes(no_complex=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_amax_dim(xp, dtype):
    """测试 amax(a, axis, keepdims) - 沿特定维度的最大值，axis 路径全 dtype 可用"""
    a = testing.shaped_random((3, 4, 5), dtype=dtype, xp=xp, seed=42)
    return xp.amax(a, axis=1, keepdims=True)


@testing.for_all_dtypes(no_complex=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_amin_dim(xp, dtype):
    """测试 amin(a, axis, keepdims) - 沿特定维度的最小值，axis 路径全 dtype 可用"""
    a = testing.shaped_random((3, 4, 5), dtype=dtype, xp=xp, seed=42)
    return xp.amin(a, axis=1, keepdims=True)


@testing.for_dtypes([numpy.float64])  # float32 全局归约返回 dtype 错误（同 max/min 现状）
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_amax_basic(xp, dtype):
    """测试 amax(a) - 全局最大值"""
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42)
    return xp.amax(a)


@testing.for_dtypes([numpy.float64])  # float32 全局归约返回 dtype 错误（同 max/min 现状）
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_amin_basic(xp, dtype):
    """测试 amin(a) - 全局最小值"""
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42)
    return xp.amin(a)


@pytest.mark.xfail(
    reason="[FIXABLE] aclnnMax/Min global reduce unsupported dtype (int64)", strict=True
)
@testing.for_dtypes([numpy.int64])
def test_amax_amin_int64_global_xfail(xp, dtype):
    """int64 全局归约在 amax/amin 入口不受支持（max/min 入口可用）"""
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42)
    return xp.amax(a), xp.amin(a)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.xfail(
    reason="[FIXABLE] nanmax 全 NaN 列返回 -inf（内部 NanToNum 实现），NumPy 返回 nan",
    strict=True,
)
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_nanmax_nan_ignored(xp, dtype):
    """测试 nanmax 对 NaN 的忽略语义：全 NaN 的列返回 NaN"""
    np_a = numpy.array([[numpy.nan, 1.0], [numpy.nan, numpy.nan]], dtype=dtype)
    a = np_a if xp is numpy else xp.ndarray.from_numpy(np_a)
    return xp.nanmax(a, axis=0)


@pytest.mark.xfail(
    reason="[FIXABLE] aclnnNanToNum unsupported dtype (float64) in nanmax", strict=True
)
@testing.for_dtypes([numpy.float64])
def test_nanmax_float64_xfail(xp, dtype):
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42)
    return xp.nanmax(a)


def _make_nan_array(shape, dtype, xp, seed=42):
    """创建测试数组，浮点类型时注入 NaN。"""
    numpy.random.seed(seed)
    arr = numpy.random.random(shape).astype(dtype)
    if numpy.issubdtype(dtype, numpy.floating):
        arr = arr.copy()
        arr.flat[0] = numpy.nan
        arr.flat[-1] = numpy.nan
    if xp is numpy:
        return arr
    return xp.ndarray.from_numpy(arr)


def test_nanmin_exports():
    """nanmin is importable from both public namespaces and listed in __all__."""
    assert "nanmin" in ap.__all__
    assert ap.nanmin is top_level_nanmin
    assert top_level_nanmin is math_nanmin


def test_nanmax():
    """测试 nanmax(a) 标量归约 - 忽略 float32 输入中的 NaN。"""
    a = numpy.array([1.0, numpy.nan, 3.0], dtype=numpy.float32)
    result = ap.nanmax(ap.ndarray.from_numpy(a))
    assert numpy.isclose(result, numpy.nanmax(a), rtol=1e-5)


@testing.for_dtypes([numpy.float32, numpy.int32])  # float64 不支持（见 xfail）
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_nanmax_dim(xp, dtype):
    """测试 nanmax(a, axis, keepdims) - 数组沿特定维度的最大值（忽略 NaN）"""
    a = _make_nan_array((3, 4, 5), dtype, xp, seed=42)
    return xp.nanmax(a, axis=1, keepdims=True)


def test_nanmin():
    """测试 nanmin(a) 标量归约 - 忽略 float32 输入中的 NaN。"""
    a = numpy.array([1.0, numpy.nan, -2.0], dtype=numpy.float32)
    result = ap.nanmin(ap.ndarray.from_numpy(a))
    assert numpy.isclose(result, numpy.nanmin(a), rtol=1e-5)


@testing.for_float_dtypes(exclude=[numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_nanmin_dim(xp, dtype):
    """测试 nanmin(a, axis, keepdims) - 数组沿特定维度的最小值（忽略 NaN）"""
    a = _make_nan_array((3, 4, 5), dtype, xp, seed=42)
    return xp.nanmin(a, axis=1, keepdims=True)
