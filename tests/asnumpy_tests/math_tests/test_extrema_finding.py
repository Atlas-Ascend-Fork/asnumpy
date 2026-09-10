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

from asnumpy import testing


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


# ========== 5. 别名与 NaN 感知入口 (Amax, Amin, Nanmax) ==========


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


def _array_with_nan(xp, dtype, seed=42):
    """构造含 NaN 的二维数组（numpy 侧与 asnumpy 侧数据一致）"""
    np_a = numpy.asarray(testing.shaped_random((3, 4), dtype=numpy.float32, seed=seed))
    if numpy.issubdtype(dtype, numpy.floating):
        np_a[0, 1] = numpy.nan
        np_a[2, 3] = numpy.nan
    if xp is numpy:
        return np_a.astype(dtype, copy=False)
    return xp.ndarray.from_numpy(np_a.astype(dtype, copy=False))


@testing.for_dtypes([numpy.float32, numpy.int32])  # float64 不支持（见 xfail）
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_nanmax_dim(xp, dtype):
    """测试 nanmax(a, axis, keepdims) - 沿维度忽略 NaN 的最大值"""
    a = _array_with_nan(xp, dtype)
    return xp.nanmax(a, axis=1, keepdims=True)


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
