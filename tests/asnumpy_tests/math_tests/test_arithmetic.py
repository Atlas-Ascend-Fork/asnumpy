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

"""基础算术运算测试

包含：
1. 二元运算: add, subtract, multiply, divide, true_divide, floor_divide, remainder, mod, hypot, cross
2. 一元运算: negative, absolute, reciprocal
"""

import numpy
import pytest

from asnumpy import testing


def _create_array(xp, data, dtype):
    np_arr = numpy.array(data, dtype=dtype)
    if xp is numpy:
        return np_arr
    return xp.ndarray.from_numpy(np_arr)


# ========== 1. 基础四则运算 (Add, Sub, Mul, Div) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_add_basic(xp, dtype):
    a = _create_array(xp, [1, 2, 3], dtype)
    b = _create_array(xp, [4, 5, 6], dtype)
    return xp.add(a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_subtract_basic(xp, dtype):
    a = _create_array(xp, [10, 20, 30], dtype)
    b = _create_array(xp, [1, 2, 3], dtype)
    return xp.subtract(a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_multiply_basic(xp, dtype):
    a = _create_array(xp, [2, 3, 4], dtype)
    b = _create_array(xp, [5, 6, 7], dtype)
    return xp.multiply(a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_divide_basic(xp, dtype):
    a = _create_array(xp, [10, 20, 30], dtype)
    b = _create_array(xp, [2, 4, 5], dtype)
    return xp.divide(a, b)


# ========== 2. 整除与取余 (Floor_divide, Remainder) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_floor_divide_basic(xp, dtype):
    a = _create_array(xp, [10, 7, 2], dtype)
    b = _create_array(xp, [3, 2, 3], dtype)
    return xp.floor_divide(a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_remainder_basic(xp, dtype):
    a = _create_array(xp, [10, 7, 2], dtype)
    b = _create_array(xp, [3, 2, 3], dtype)
    return xp.remainder(a, b)


# ========== 3. 一元运算 (Negative, Absolute, Reciprocal) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_negative(xp, dtype):
    a = _create_array(xp, [-1, 0, 1], dtype)
    return xp.negative(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_absolute(xp, dtype):
    a = _create_array(xp, [-1.5, 0, 2.5], dtype)
    return xp.absolute(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-6, rtol=1e-6)
def test_reciprocal(xp, dtype):
    """测试 reciprocal(x) - 逐元素倒数"""
    a = _create_array(xp, [1.0, 2.0, 4.0, -2.0], dtype)
    return xp.reciprocal(a)


@pytest.mark.xfail(
    reason="[FIXABLE] aclnnReciprocal unsupported dtype (int32/int64/bool)", strict=True
)
@testing.for_dtypes([numpy.int32])
def test_reciprocal_int_xfail(xp, dtype):
    a = _create_array(xp, [1, 2], dtype)
    return xp.reciprocal(a)


# ========== 4. 广播与特殊 Dtype 限制 ==========


# float16 has an 11-bit mantissa, so its ULP near 1.0 is ~9.8e-4. The default rtol=1e-7 is four
# orders of magnitude tighter than the type can represent and would fail on a single-ULP
# disagreement between NumPy's float16 add and aclnnAdd.
@testing.for_dtypes([numpy.float16])
@testing.numpy_asnumpy_allclose(rtol=1e-3, atol=1e-3)
def test_arithmetic_float16(xp, dtype):
    a = _create_array(xp, [1.0], dtype)
    b = _create_array(xp, [2.0], dtype)
    return xp.add(a, b)


@testing.for_dtypes([numpy.int32])
@testing.numpy_asnumpy_allclose()
def test_arithmetic_int_mismatch_xfail(xp, dtype):
    """测试整数输入的精度提升不一致问题"""
    a = _create_array(xp, [1, 2], dtype)
    b = _create_array(xp, [3, 4], dtype)
    return xp.add(a, b)


@testing.for_dtypes([numpy.bool_])
@testing.numpy_asnumpy_array_equal()
def test_remainder_bool(xp, dtype):
    a = _create_array(xp, [True, False], dtype)
    b = _create_array(xp, [True, True], dtype)
    return xp.remainder(a, b)


@testing.for_dtypes([numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_remainder_int32(xp, dtype):
    a = _create_array(xp, [10, 7, -5], dtype)
    b = _create_array(xp, [3, 2, 3], dtype)
    return xp.remainder(a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_arithmetic_broadcasting(xp, dtype):
    """测试二元运算的广播机制 (例如: (2,3) + (3,))"""
    a = _create_array(xp, [[1, 2, 3], [4, 5, 6]], dtype)
    b = _create_array(xp, [1, 0, 1], dtype)
    return xp.add(a, b)


# ========== 5. True_divide 与 Mod ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_true_divide_basic(xp, dtype):
    """true_divide 为 divide 的独立入口，行为应一致"""
    a = _create_array(xp, [10.0, 7.0, -5.0], dtype)
    b = _create_array(xp, [3.0, 2.0, 3.0], dtype)
    return xp.true_divide(a, b)


@testing.for_dtypes([numpy.int32, numpy.int64])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_true_divide_int_promotion(xp, dtype):
    """整数相除两侧均提升为 float64，不做截断"""
    a = _create_array(xp, [1, 2, 3], dtype)
    b = _create_array(xp, [2, 2, 2], dtype)
    return xp.true_divide(a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_mod_basic(xp, dtype):
    """mod 为 remainder 的独立入口，行为应一致"""
    a = _create_array(xp, [10.0, 7.0, -5.0], dtype)
    b = _create_array(xp, [3.0, 2.0, 3.0], dtype)
    return xp.mod(a, b)


@testing.for_dtypes([numpy.int32, numpy.int64])
@testing.numpy_asnumpy_array_equal()
def test_mod_int(xp, dtype):
    """整数取余结果精确相等，且保持整数 dtype"""
    a = _create_array(xp, [10, 7, -5], dtype)
    b = _create_array(xp, [3, 2, 3], dtype)
    return xp.mod(a, b)


# ========== 6. 直角三角形斜边 (Hypot) ==========


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_hypot_basic(xp, dtype):
    """勾股数 (3,4,5)、(5,12,13)、(8,15,17) 验证"""
    a = _create_array(xp, [3.0, 5.0, 8.0], dtype)
    b = _create_array(xp, [4.0, 12.0, 15.0], dtype)
    return xp.hypot(a, b)


@pytest.mark.xfail(reason="[FIXABLE] Hypot float16 结果错误（返回 x 自身）", strict=True)
@testing.for_dtypes([numpy.float16])
@testing.numpy_asnumpy_allclose(rtol=1e-3, atol=1e-3)
def test_hypot_float16_xfail(xp, dtype):
    a = _create_array(xp, [3.0], dtype)
    b = _create_array(xp, [4.0], dtype)
    return xp.hypot(a, b)


@pytest.mark.xfail(reason="[FIXABLE] Hypot 整数输入转换后结果错误（溢出量级）", strict=True)
@testing.for_dtypes([numpy.int32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_hypot_int_xfail(xp, dtype):
    a = _create_array(xp, [3, 5, 8], dtype)
    b = _create_array(xp, [4, 12, 15], dtype)
    return xp.hypot(a, b)


# ========== 7. 叉积 (Cross) ==========


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_cross_basic(xp, dtype):
    """二维批量向量的叉积（axis=-1 为 3 维向量所在的轴）"""
    a = _create_array(xp, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype)
    b = _create_array(xp, [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype)
    if xp is numpy:
        return xp.cross(a, b, axis=-1)
    # 适配当前 AsNumpy 的 axis 必须显式传入
    return xp.cross(a, b, -1)


@testing.for_dtypes([numpy.int32, numpy.int64])
@testing.numpy_asnumpy_array_equal()
def test_cross_int(xp, dtype):
    """一维整数向量的叉积，结果保持整数 dtype"""
    a = _create_array(xp, [1, 2, 3], dtype)
    b = _create_array(xp, [4, 5, 6], dtype)
    if xp is numpy:
        return xp.cross(a, b)
    return xp.cross(a, b, -1)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_cross_axis0(xp, dtype):
    """3 维向量在 axis=0 的叉积：形状 (3, 2) 逐列计算"""
    a = _create_array(xp, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype)
    b = _create_array(xp, [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype)
    if xp is numpy:
        return xp.cross(a, b, axis=0)
    return xp.cross(a, b, 0)
