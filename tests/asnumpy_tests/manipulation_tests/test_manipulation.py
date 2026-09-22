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

"""数组操作 API 测试 (issue #25 的 P0 子集)

包含：
1. concatenate
2. stack
3. where
4. expand_dims
5. squeeze

行为以 numpy 为基准：`numpy_asnumpy_array_equal` 装饰器会同时跑 numpy 与 asnumpy，
因此正常路径不需要手写期望值；错误路径用 pytest.raises 直接约束异常类型。
"""

import numpy
import pytest

import asnumpy as ap
from asnumpy import testing

DTYPES = testing.TEST_DTYPES


def _seq(xp, shapes, dtype, start=0):
    return tuple(
        testing.shaped_arange(shape, dtype=dtype, xp=xp, start=start + i)
        for i, shape in enumerate(shapes)
    )


# ==========================================================================
# 1. concatenate
# ==========================================================================


@testing.for_dtypes(DTYPES)
@testing.numpy_asnumpy_array_equal()
def test_concatenate_1d(xp, dtype):
    a, b = _seq(xp, [(3,), (4,)], dtype)
    return xp.concatenate((a, b))


@testing.for_dtypes(DTYPES)
@testing.numpy_asnumpy_array_equal()
def test_concatenate_accepts_list(xp, dtype):
    a, b = _seq(xp, [(2,), (2,)], dtype)
    return xp.concatenate([a, b])


@testing.for_dtypes(DTYPES)
@testing.numpy_asnumpy_array_equal()
def test_concatenate_axis0(xp, dtype):
    a, b = _seq(xp, [(2, 3), (4, 3)], dtype)
    return xp.concatenate((a, b), axis=0)


@testing.for_dtypes(DTYPES)
@testing.numpy_asnumpy_array_equal()
def test_concatenate_axis1(xp, dtype):
    a, b = _seq(xp, [(2, 3), (2, 4)], dtype)
    return xp.concatenate((a, b), axis=1)


@testing.for_dtypes(DTYPES)
@testing.numpy_asnumpy_array_equal()
def test_concatenate_negative_axis(xp, dtype):
    a, b = _seq(xp, [(2, 3), (2, 4)], dtype)
    return xp.concatenate((a, b), axis=-1)


@testing.for_dtypes(DTYPES)
@testing.numpy_asnumpy_array_equal()
def test_concatenate_three_arrays_axis0(xp, dtype):
    a, b, c = _seq(xp, [(2, 2), (1, 2), (3, 2)], dtype)
    return xp.concatenate((a, b, c), axis=0)


@testing.for_dtypes(DTYPES)
@testing.numpy_asnumpy_array_equal()
def test_concatenate_single_array_is_copy(xp, dtype):
    (a,) = _seq(xp, [(3, 3)], dtype)
    return xp.concatenate((a,))


@testing.for_dtypes(DTYPES)
@testing.numpy_asnumpy_array_equal()
def test_concatenate_axis_none_flattens(xp, dtype):
    a, b = _seq(xp, [(2, 3), (2, 2)], dtype)
    return xp.concatenate((a, b), axis=None)


# ---------- 错误路径 ----------


def test_concatenate_empty_sequence_raises_value_error():
    with pytest.raises(ValueError):
        ap.concatenate([])


def test_concatenate_shape_mismatch_raises_value_error():
    a = ap.ndarray.from_numpy(numpy.zeros((2, 3), dtype=numpy.float32))
    b = ap.ndarray.from_numpy(numpy.zeros((2, 4), dtype=numpy.float32))
    with pytest.raises(ValueError):
        ap.concatenate((a, b), axis=0)


def test_concatenate_axis_out_of_range_raises():
    a = ap.ndarray.from_numpy(numpy.zeros((2, 3), dtype=numpy.float32))
    with pytest.raises((ValueError, IndexError)):
        ap.concatenate((a, a), axis=5)


# ==========================================================================
# 2. stack
# ==========================================================================


@testing.for_dtypes(DTYPES)
@testing.numpy_asnumpy_array_equal()
def test_stack_1d_axis0(xp, dtype):
    a, b = _seq(xp, [(3,), (3,)], dtype)
    return xp.stack((a, b))


@testing.for_dtypes(DTYPES)
@testing.numpy_asnumpy_array_equal()
def test_stack_1d_axis1(xp, dtype):
    a, b = _seq(xp, [(3,), (3,)], dtype)
    return xp.stack((a, b), axis=1)


@testing.for_dtypes(DTYPES)
@testing.numpy_asnumpy_array_equal()
def test_stack_2d_axis0(xp, dtype):
    a, b = _seq(xp, [(2, 3), (2, 3)], dtype)
    return xp.stack((a, b), axis=0)


@testing.for_dtypes(DTYPES)
@testing.numpy_asnumpy_array_equal()
def test_stack_2d_negative_axis(xp, dtype):
    a, b = _seq(xp, [(2, 3), (2, 3)], dtype)
    return xp.stack((a, b), axis=-1)


@testing.for_dtypes(DTYPES)
@testing.numpy_asnumpy_array_equal()
def test_stack_three_arrays(xp, dtype):
    a, b, c = _seq(xp, [(2, 2), (2, 2), (2, 2)], dtype)
    return xp.stack((a, b, c), axis=1)


def test_stack_shape_mismatch_raises_value_error():
    a = ap.ndarray.from_numpy(numpy.zeros((2, 3), dtype=numpy.float32))
    b = ap.ndarray.from_numpy(numpy.zeros((2, 4), dtype=numpy.float32))
    with pytest.raises(ValueError):
        ap.stack((a, b), axis=0)


# ==========================================================================
# 3. where
# ==========================================================================


@testing.for_dtypes(DTYPES)
@testing.numpy_asnumpy_array_equal()
def test_where_array_condition(xp, dtype):
    # asnumpy arrays do not overload Python operators, so build the mask via xp.greater.
    cond = xp.greater(testing.shaped_arange((3, 4), dtype=dtype, xp=xp), 1)
    x = testing.shaped_arange((3, 4), dtype=dtype, xp=xp)
    y = testing.shaped_arange((3, 4), dtype=dtype, xp=xp, start=100)
    return xp.where(cond, x, y)


@testing.for_dtypes(DTYPES)
@testing.numpy_asnumpy_array_equal()
def test_where_broadcasts_scalars(xp, dtype):
    cond = xp.greater(testing.shaped_arange((3, 4), dtype=dtype, xp=xp), 1)
    x = testing.shaped_arange((3, 4), dtype=dtype, xp=xp)
    return xp.where(cond, x, 0)


@testing.for_dtypes(DTYPES)
@testing.numpy_asnumpy_array_equal()
def test_where_broadcasts_shapes(xp, dtype):
    cond = xp.greater(testing.shaped_arange((3, 1), dtype=dtype, xp=xp), 1)
    x = testing.shaped_arange((1, 4), dtype=dtype, xp=xp)
    y = testing.shaped_arange((3, 4), dtype=dtype, xp=xp, start=50)
    return xp.where(cond, x, y)


# ==========================================================================
# 4. expand_dims
# ==========================================================================


@testing.for_dtypes(DTYPES)
@testing.numpy_asnumpy_array_equal()
def test_expand_dims_axis0(xp, dtype):
    a = testing.shaped_arange((3, 4), dtype=dtype, xp=xp)
    return xp.expand_dims(a, axis=0)


@testing.for_dtypes(DTYPES)
@testing.numpy_asnumpy_array_equal()
def test_expand_dims_axis1(xp, dtype):
    a = testing.shaped_arange((3, 4), dtype=dtype, xp=xp)
    return xp.expand_dims(a, axis=1)


@testing.for_dtypes(DTYPES)
@testing.numpy_asnumpy_array_equal()
def test_expand_dims_negative_axis(xp, dtype):
    a = testing.shaped_arange((3, 4), dtype=dtype, xp=xp)
    return xp.expand_dims(a, axis=-1)


def test_expand_dims_out_of_range_axis_raises():
    a = ap.ndarray.from_numpy(numpy.zeros((3, 4), dtype=numpy.float32))
    with pytest.raises((ValueError, IndexError)):
        ap.expand_dims(a, axis=5)


# ==========================================================================
# 5. squeeze
# ==========================================================================


@testing.for_dtypes(DTYPES)
@testing.numpy_asnumpy_array_equal()
def test_squeeze_all_singleton_axes(xp, dtype):
    a = testing.shaped_arange((1, 3, 1, 4), dtype=dtype, xp=xp)
    return xp.squeeze(a)


@testing.for_dtypes(DTYPES)
@testing.numpy_asnumpy_array_equal()
def test_squeeze_specific_axis(xp, dtype):
    a = testing.shaped_arange((1, 3, 4), dtype=dtype, xp=xp)
    return xp.squeeze(a, axis=0)


@testing.for_dtypes(DTYPES)
@testing.numpy_asnumpy_array_equal()
def test_squeeze_negative_axis(xp, dtype):
    a = testing.shaped_arange((3, 4, 1), dtype=dtype, xp=xp)
    return xp.squeeze(a, axis=-1)


def test_squeeze_non_singleton_axis_raises_value_error():
    a = ap.ndarray.from_numpy(numpy.zeros((3, 4), dtype=numpy.float32))
    with pytest.raises(ValueError):
        ap.squeeze(a, axis=0)


@testing.for_dtypes(DTYPES)
@testing.numpy_asnumpy_array_equal()
def test_squeeze_multiple_axes(xp, dtype):
    a = testing.shaped_arange((1, 3, 1), dtype=dtype, xp=xp)
    return xp.squeeze(a, axis=(0, 2))


# ==========================================================================
# 6. dtype promotion with mixed inputs
# ==========================================================================


def test_concatenate_promotes_mixed_dtypes():
    a = ap.ndarray.from_numpy(numpy.array([1.0, 2.0], dtype=numpy.float32))
    b = ap.ndarray.from_numpy(numpy.array([3.0, 4.0], dtype=numpy.float64))
    got = ap.concatenate([a, b], 0).to_numpy()
    expected = numpy.concatenate([a.to_numpy(), b.to_numpy()], 0)
    assert got.dtype == expected.dtype
    numpy.testing.assert_array_equal(got, expected)


def test_stack_promotes_mixed_dtypes():
    a = ap.ndarray.from_numpy(numpy.array([1, 2], dtype=numpy.int32))
    b = ap.ndarray.from_numpy(numpy.array([3.5, 4.5], dtype=numpy.float32))
    got = ap.stack([a, b], 0).to_numpy()
    expected = numpy.stack([a.to_numpy(), b.to_numpy()], 0)
    assert got.dtype == expected.dtype
    numpy.testing.assert_array_equal(got, expected)


def test_where_accepts_non_boolean_condition():
    # NumPy treats a non-boolean condition by truthiness; asnumpy casts it to bool on device.
    cond = ap.ndarray.from_numpy(numpy.array([1, 0, 2], dtype=numpy.int32))
    x = ap.ndarray.from_numpy(numpy.array([10, 20, 30], dtype=numpy.int32))
    y = ap.ndarray.from_numpy(numpy.array([7, 8, 9], dtype=numpy.int32))
    got = ap.where(cond, x, y).to_numpy()
    expected = numpy.where(cond.to_numpy(), x.to_numpy(), y.to_numpy())
    numpy.testing.assert_array_equal(got, expected)
