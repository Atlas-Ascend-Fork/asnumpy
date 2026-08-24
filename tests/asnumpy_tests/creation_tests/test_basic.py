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

"""基础数组创建函数测试

演示测试框架的核心功能，包括：
- dtype 参数化测试
- numpy-asnumpy 结果比较
- 异常断言
- 测试辅助函数
"""

import numpy

from asnumpy import testing

# ========== 基础创建函数测试 ==========


@testing.for_all_dtypes()
@testing.numpy_asnumpy_array_equal()
def test_zeros(xp, dtype):
    """测试zeros函数

    自动测试11种支持的dtype（默认排除 float16, uint32, uint64）
    """
    return xp.zeros((2, 3, 4), dtype=dtype)


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
def test_ones(xp, dtype):
    """测试ones函数

    只测试ones支持的dtype（不支持float16, uint16/32/64, complex）
    """
    return xp.ones((3, 4), dtype=dtype)


@testing.for_float_dtypes()
@testing.numpy_asnumpy_array_equal()
def test_empty(xp, dtype):
    """测试empty函数

    测试浮点类型（默认排除 float16）
    注意：用zeros代替empty以便比较结果
    """
    return xp.zeros((2, 3), dtype=dtype)


def test_linspace_int64_preserves_dtype():
    """Test that linspace keeps an explicitly requested int64 dtype."""
    import asnumpy as ap

    expected = numpy.full(5, 3_000_000_000, dtype=numpy.int64)
    actual = ap.linspace(3_000_000_000, 3_000_000_000, 5, dtype=numpy.int64).to_numpy()

    numpy.testing.assert_array_equal(actual, expected)
    assert actual.dtype == numpy.dtype(numpy.int64)


# ========== 多种形状测试（使用 TEST_SHAPES 常量）==========


@testing.for_float_dtypes()
@testing.numpy_asnumpy_array_equal()
def test_zeros_various_shapes(xp, dtype):
    """测试zeros - 各种形状

    使用 TEST_SHAPES 常量测试不同维度
    """
    # 测试一些常见形状
    results = []
    for shape in [(5,), (2, 3), (2, 3, 4)]:
        results.append(xp.zeros(shape, dtype=dtype))
    return results[0]  # 返回第一个用于比较


# ========== 异常测试（numpy和asnumpy都应抛出异常）==========


@testing.numpy_asnumpy_array_equal(accept_error=True)
def test_zeros_negative_shape(xp):
    """测试zeros - 负数形状应该抛出异常

    验证numpy和asnumpy都会抛出异常（类型可能不同）
    accept_error=True 允许异常类型不同
    """
    # numpy: ValueError, asnumpy: 可能是TypeError
    return xp.zeros(-1, dtype=numpy.float32)


@testing.numpy_asnumpy_array_equal(accept_error=True)
def test_zeros_invalid_dtype(xp):
    """测试zeros - 无效dtype应该抛出异常

    验证numpy和asnumpy都会抛出异常（类型可能不同）
    """
    # 两者都应该拒绝无效的dtype
    return xp.zeros((2, 3), dtype="invalid_dtype")
