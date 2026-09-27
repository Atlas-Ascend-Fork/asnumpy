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

"""数组创建函数完整测试

1. zeros (在 test_basic.py)
2. zeros_like
3. ones (在 test_basic.py)
4. ones_like
5. empty (在 test_basic.py)
6. empty_like
7. eye
8. identity
9. full
10. full_like
"""

import numpy

import asnumpy as ap
from asnumpy import testing

# ========== zeros_like 测试 ==========


@testing.for_all_dtypes()
@testing.numpy_asnumpy_array_equal()
def test_zeros_like(xp, dtype):
    """测试zeros_like - 基本功能

    API差异：asnumpy需要显式传递dtype参数
    """
    a = testing.shaped_arange((2, 3), dtype=dtype, xp=xp)
    return xp.zeros_like(a, dtype=dtype)


@testing.for_all_dtypes()
@testing.numpy_asnumpy_array_equal()
def test_zeros_like_1d(xp, dtype):
    """测试zeros_like - 1维数组"""
    a = testing.shaped_arange((10,), dtype=dtype, xp=xp)
    return xp.zeros_like(a, dtype=dtype)


@testing.for_all_dtypes()
@testing.numpy_asnumpy_array_equal()
def test_zeros_like_3d(xp, dtype):
    """测试zeros_like - 3维数组"""
    a = testing.shaped_arange((2, 3, 4), dtype=dtype, xp=xp)
    return xp.zeros_like(a, dtype=dtype)


@testing.for_all_dtypes()
@testing.numpy_asnumpy_array_equal()
def test_zeros_like_scalar_shape(xp, dtype):
    """测试zeros_like - 标量形状 ()"""
    a = testing.shaped_arange((), dtype=dtype, xp=xp)
    return xp.zeros_like(a, dtype=dtype)


# ========== ones_like 测试 ==========


@testing.for_dtypes(
    [
        numpy.float32,
        numpy.float64,
        numpy.int8,
        numpy.int16,
        numpy.int32,
        numpy.int64,
        numpy.uint8,
        numpy.bool_,
    ]
)
@testing.numpy_asnumpy_array_equal()
def test_ones_like(xp, dtype):
    """测试ones_like - 基本功能

    只测试ones支持的dtype
    API差异：asnumpy需要显式传递dtype参数
    """
    a = testing.shaped_arange((3, 4), dtype=dtype, xp=xp)
    return xp.ones_like(a, dtype=dtype)


@testing.for_dtypes(
    [
        numpy.float32,
        numpy.float64,
        numpy.int32,
        numpy.int64,
    ]
)
@testing.numpy_asnumpy_array_equal()
def test_ones_like_large(xp, dtype):
    """测试ones_like - 较大数组"""
    a = testing.shaped_arange((10, 20), dtype=dtype, xp=xp)
    return xp.ones_like(a, dtype=dtype)


# ========== empty_like 测试 ==========


@testing.for_float_dtypes()
def test_empty_like(dtype):
    """测试empty_like - 基本功能

    API差异：asnumpy需要显式传递dtype参数
    """
    numpy_prototype = numpy.zeros((2, 3), dtype=dtype)
    asnumpy_prototype = ap.zeros((2, 3), dtype=dtype)
    expected = numpy.empty_like(numpy_prototype, dtype=dtype)
    actual = ap.empty_like(asnumpy_prototype, dtype=dtype)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype


@testing.for_float_dtypes()
def test_empty_like_shape(dtype):
    """测试empty_like - 验证形状正确性"""
    numpy_prototype = numpy.zeros((5, 6, 7), dtype=dtype)
    asnumpy_prototype = ap.zeros((5, 6, 7), dtype=dtype)
    expected = numpy.empty_like(numpy_prototype, dtype=dtype)
    actual = ap.empty_like(asnumpy_prototype, dtype=dtype)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype


# ========== eye 测试 ==========
# API限制：eye 不支持 float64（CANN: DT_DOUBLE not implemented）


@testing.for_dtypes(
    [
        numpy.float32,  # float64 不支持
        numpy.int8,
        numpy.int16,
        numpy.int32,
        numpy.int64,
        numpy.uint8,  # uint16 也不支持！
        numpy.bool_,
    ]
)
@testing.numpy_asnumpy_array_equal()
def test_eye_square(xp, dtype):
    """测试eye - 方阵（n=5）

    API限制：排除 float64 和 uint16（CANN 限制）
    """
    return xp.eye(5, dtype=dtype)


@testing.for_dtypes([numpy.float32, numpy.int32, numpy.int64])
@testing.numpy_asnumpy_array_equal()
def test_eye_square_small(xp, dtype):
    """测试eye - 小方阵（n=2）"""
    return xp.eye(2, dtype=dtype)


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_eye_square_large(xp, dtype):
    """测试eye - 大方阵（n=10）"""
    return xp.eye(10, dtype=dtype)


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_eye_n1(xp, dtype):
    """测试eye - 最小尺寸（n=1）"""
    return xp.eye(1, dtype=dtype)


# ========== identity 测试 ==========
# API限制：identity 不支持 float64（CANN: DT_DOUBLE not implemented）


@testing.for_dtypes(
    [
        numpy.float32,  # float64 不支持
        numpy.int8,
        numpy.int16,
        numpy.int32,
        numpy.int64,
        numpy.uint8,  # uint16 也不支持！
        numpy.bool_,
    ]
)
@testing.numpy_asnumpy_array_equal()
def test_identity(xp, dtype):
    """测试identity - 基本功能（n=6）

    API限制：排除 float64 和 uint16（CANN 限制）
    """
    return xp.identity(6, dtype=dtype)


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_identity_small(xp, dtype):
    """测试identity - 小矩阵（n=1）"""
    return xp.identity(1, dtype=dtype)


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_identity_large(xp, dtype):
    """测试identity - 大矩阵（n=15）"""
    return xp.identity(15, dtype=dtype)


# ========== full 测试 ==========
# API限制：full 不支持 uint16（CANN aclnnInplaceFillScalarGetWorkspaceSize error）


@testing.for_dtypes(
    [
        numpy.float32,
        numpy.float64,
        numpy.int8,
        numpy.int16,
        numpy.int32,
        numpy.int64,
        numpy.uint8,  # uint16 不支持
        numpy.bool_,
        # complex64/128 也不支持！
    ]
)
@testing.numpy_asnumpy_array_equal()
def test_full(xp, dtype):
    """测试full - 基本功能

    API限制：排除 uint16 和 complex（CANN 限制）
    """
    # 根据dtype选择合适的fill_value
    if dtype == numpy.bool_:
        fill_value = True
    elif numpy.issubdtype(dtype, numpy.integer):
        fill_value = 42
    else:
        fill_value = 3.14
    return xp.full((3, 4), fill_value, dtype=dtype)


@testing.for_dtypes(
    [
        numpy.float32,
        numpy.float64,
        numpy.int8,
        numpy.int16,
        numpy.int32,
        numpy.int64,
        # uint8 会触发溢出检查差异，单独测试
        numpy.bool_,
    ]
)
@testing.numpy_asnumpy_array_equal()
def test_full_1d(xp, dtype):
    """测试full - 1维数组

    使用非负值避免 uint8 溢出检查差异
    """
    if dtype == numpy.bool_:
        fill_value = False
    elif numpy.issubdtype(dtype, numpy.integer):
        fill_value = 5  # 改用非负值
    else:
        fill_value = 2.71
    return xp.full((20,), fill_value, dtype=dtype)


@testing.for_dtypes([numpy.float32, numpy.int32, numpy.int64, numpy.bool_])
@testing.numpy_asnumpy_array_equal()
def test_full_3d(xp, dtype):
    """测试full - 3维数组"""
    if dtype == numpy.bool_:
        fill_value = True
    elif numpy.issubdtype(dtype, numpy.integer):
        fill_value = 100
    else:
        fill_value = 1.414
    return xp.full((2, 3, 4), fill_value, dtype=dtype)


@testing.for_float_dtypes()
@testing.numpy_asnumpy_array_equal()
def test_full_zero_fill(xp, dtype):
    """测试full - 填充值为0"""
    return xp.full((4, 5), 0.0, dtype=dtype)


# ========== full_like 测试 ==========
# API限制：full_like 需要显式 dtype 参数


@testing.for_dtypes(
    [
        numpy.float32,
        numpy.float64,
        numpy.int8,
        numpy.int16,
        numpy.int32,
        numpy.int64,
        numpy.uint8,  # uint16 不支持
        numpy.bool_,
    ]
)
@testing.numpy_asnumpy_array_equal()
def test_full_like(xp, dtype):
    """测试full_like - 基本功能

    API差异：asnumpy需要显式传递dtype参数
    API限制：排除 uint16（CANN 不支持）
    """
    a = testing.shaped_arange((2, 3, 4), dtype=dtype, xp=xp)

    # 根据dtype选择合适的fill_value
    if dtype == numpy.bool_:
        fill_value = False
    elif numpy.issubdtype(dtype, numpy.integer):
        fill_value = 7
    else:
        fill_value = 2.71

    return xp.full_like(a, fill_value, dtype=dtype)


@testing.for_dtypes(
    [
        numpy.float32,
        numpy.float64,
        numpy.int32,
        numpy.int64,
    ]
)
@testing.numpy_asnumpy_array_equal()
def test_full_like_1d(xp, dtype):
    """测试full_like - 1维数组"""
    a = testing.shaped_arange((15,), dtype=dtype, xp=xp)

    if dtype == numpy.bool_:
        fill_value = True
    elif numpy.issubdtype(dtype, numpy.integer):
        fill_value = 10
    else:
        fill_value = 0.5

    return xp.full_like(a, fill_value, dtype=dtype)


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_full_like_large(xp, dtype):
    """测试full_like - 较大数组"""
    a = testing.shaped_arange((5, 6, 7), dtype=dtype, xp=xp)

    if numpy.issubdtype(dtype, numpy.integer):
        fill_value = 99
    else:
        fill_value = 9.99

    return xp.full_like(a, fill_value, dtype=dtype)


# ========== 注意 ==========
# 以下 dtype 已知不支持，已从测试中排除：
#
# eye/identity 不支持:
#   - float64 (CANN: DT_DOUBLE not implemented)
#   - uint16 (CANN: aclnnEyeGetWorkspaceSize error)
#
# full/full_like 不支持:
#   - uint16 (CANN: aclnnInplaceFillScalarGetWorkspaceSize error)
#   - complex64/128 (pybind11 Conversion error)
#
# *_like 函数 API 差异:
#   - 需要显式传递 dtype 参数（NumPy 中可选）
#
# full 行为差异:
#   - 不检查整数溢出（NumPy 会抛出 OverflowError）
#   - 例如: full((5,), -5, dtype=uint8) 在 NumPy 会报错，在 AsNumPy 会静默截断
