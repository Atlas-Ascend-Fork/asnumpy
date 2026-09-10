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

"""指数和对数函数测试

包含：
1. 指数: exp, exp2, expm1
2. 对数: log, log2, log10, log1p
3. 幂运算: power, float_power
"""

import numpy
import pytest

from asnumpy import testing


def _create_array(xp, data, dtype):
    np_arr = numpy.array(data, dtype=dtype)
    if xp is numpy:
        return np_arr
    return xp.ndarray.from_numpy(np_arr)


# ========== 1. 指数运算 (Exp, Expm1) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_exp_basic(xp, dtype):
    data = [-1.0, 0.0, 1.0, 2.0]
    a = _create_array(xp, data, dtype)
    return xp.exp(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_expm1_basic(xp, dtype):
    """测试 exp(x) - 1，常用于极小值场景"""
    data = [1e-5, 0.0, -1e-5]
    a = _create_array(xp, data, dtype)
    return xp.expm1(a)


# ========== 2. 对数运算 (Log, Log10, Log1p) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_log_basic(xp, dtype):
    data = [0.1, 1.0, numpy.e, 10.0]
    a = _create_array(xp, data, dtype)
    return xp.log(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_log10_basic(xp, dtype):
    data = [0.1, 1.0, 10.0, 100.0]
    a = _create_array(xp, data, dtype)
    return xp.log10(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_log1p_basic(xp, dtype):
    """测试 log(1 + x)"""
    data = [1e-5, 0.0, -1e-5]
    a = _create_array(xp, data, dtype)
    return xp.log1p(a)


# ========== 3. 限制性测试 (XFAIL) ==========


# float16's ULP near e is ~2e-3, so the default rtol=1e-7 is far tighter than the type can hold.
@testing.for_dtypes([numpy.float16])
@testing.numpy_asnumpy_allclose(rtol=1e-3, atol=1e-3)
def test_exp_float16(xp, dtype):
    a = _create_array(xp, [1.0], dtype)
    return xp.exp(a)


@testing.for_dtypes([numpy.int32])
@testing.numpy_asnumpy_allclose()
def test_exp_int_dtype_promotion(xp, dtype):
    data = [1, 2]
    a = _create_array(xp, data, dtype)
    return xp.exp(a)


@testing.for_dtypes([numpy.int32])
@testing.numpy_asnumpy_allclose()
def test_log_int_dtype_promotion(xp, dtype):
    data = [1, 2, 4]
    a = _create_array(xp, data, dtype)
    return xp.log(a)


@testing.for_dtypes([numpy.int32])
@testing.numpy_asnumpy_allclose()
def test_log2_int_dtype_promotion(xp, dtype):
    data = [1, 2, 4]
    a = _create_array(xp, data, dtype)
    return xp.log2(a)


@testing.for_dtypes([numpy.int32])
@testing.numpy_asnumpy_allclose()
def test_log10_int_dtype_promotion(xp, dtype):
    data = [1, 2, 4]
    a = _create_array(xp, data, dtype)
    return xp.log10(a)


@testing.for_dtypes([numpy.int32, numpy.float64])
@testing.numpy_asnumpy_allclose()
def test_logaddexp_dtype_promotion(xp, dtype):
    a = _create_array(xp, [1.0, 2.0], dtype)
    b = _create_array(xp, [0.5, 1.5], dtype)
    return xp.logaddexp(a, b)


@testing.for_dtypes([numpy.int32, numpy.float64])
@testing.numpy_asnumpy_allclose()
def test_logaddexp2_dtype_promotion(xp, dtype):
    a = _create_array(xp, [1.0, 2.0], dtype)
    b = _create_array(xp, [0.5, 1.5], dtype)
    return xp.logaddexp2(a, b)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_log_domain_error(xp, dtype):
    """测试对数定义域（非正数）。Numpy 返回 NaN/Inf。"""
    data = [0.0, -1.0]
    a = _create_array(xp, data, dtype)
    return xp.log(a)


# ========== 4. 幂运算 (Exp2, Power, Float_power) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_exp2_basic(xp, dtype):
    """测试 2**x"""
    data = [0.0, 1.0, 2.0, 3.5, -1.5]
    a = _create_array(xp, data, dtype)
    return xp.exp2(a)


@testing.for_dtypes([numpy.float64, numpy.int32, numpy.int64])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_exp2_dtype_promotion(xp, dtype):
    """整数输入两侧均提升为 float64"""
    data = [0, 1, 2, -1]
    a = _create_array(xp, data, dtype)
    return xp.exp2(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_power_basic(xp, dtype):
    """测试 x1**x2（含负指数）"""
    a = _create_array(xp, [2.0, 3.0, 2.0], dtype)
    b = _create_array(xp, [3.0, 2.0, -1.0], dtype)
    return xp.power(a, b)


@testing.for_dtypes([numpy.int32, numpy.int64])
@testing.numpy_asnumpy_array_equal()
def test_power_int(xp, dtype):
    """整数非负指数的幂，结果保持整数 dtype"""
    a = _create_array(xp, [2, 3, 2], dtype)
    b = _create_array(xp, [3, 2, 2], dtype)
    return xp.power(a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_float_power_explicit_dtype(xp, dtype):
    """显式指定 dtype=float64 时与 NumPy 的 float64 输出一致"""
    a = _create_array(xp, [2.0, 3.0, 2.0], dtype)
    b = _create_array(xp, [3.0, 2.0, -1.0], dtype)
    return xp.float_power(a, b, dtype=numpy.float64)


@pytest.mark.xfail(
    reason="[FIXABLE] float_power 未显式指定 dtype 时返回 float32，NumPy 恒为 float64",
    strict=True,
)
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_float_power_dtype_xfail(xp, dtype):
    a = _create_array(xp, [2.0, 3.0], dtype)
    b = _create_array(xp, [3.0, 2.0], dtype)
    return xp.float_power(a, b)
