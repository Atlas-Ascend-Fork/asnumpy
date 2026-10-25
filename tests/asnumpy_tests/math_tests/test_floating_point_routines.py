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

"""浮点数处理算子测试

针对已记录的针对 signbit(-0.0) 问题以及硬件限制的标注。
"""

import numpy
import pytest

from asnumpy import testing

# ========== 辅助函数 ==========


def _create_array(xp, data, dtype):
    """辅助函数：创建数组"""
    np_arr = numpy.array(data, dtype=dtype)
    if xp is numpy:
        return np_arr
    return xp.ndarray.from_numpy(np_arr)


# ========== 1. 状态检查 (isinf, isfinite) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_isinf(xp, dtype):
    """测试无穷大检查"""
    data = [numpy.inf, -numpy.inf, 1.0, 0.0]
    a = _create_array(xp, data, dtype)
    return xp.isinf(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_isfinite(xp, dtype):
    """测试有限值检查"""
    data = [numpy.nan, numpy.inf, 1.0, 0.0]
    a = _create_array(xp, data, dtype)
    return xp.isfinite(a)


@testing.for_dtypes([numpy.float16])
@testing.numpy_asnumpy_array_equal()
def test_isinf_float16(xp, dtype):
    """测试 float16 无穷大检查"""
    data = [numpy.inf, -numpy.inf, 1.0, 0.0]
    a = _create_array(xp, data, dtype)
    return xp.isinf(a)


# ========== 2. 符号处理与负零 Bug (signbit, copysign) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_signbit_basic(xp, dtype):
    """测试基础符号位提取（排除负零）"""
    data = [-1.0, 1.0, -10.5, 10.5]
    a = _create_array(xp, data, dtype)
    return xp.signbit(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_signbit_negative_zero_xfail(xp, dtype):
    """
    记录：AsNumpy 在处理 -0.0 时将其视作正数。
    预期: True, 实际: False
    """
    data = [-0.0]
    a = _create_array(xp, data, dtype)
    return xp.signbit(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_copysign_basic(xp, dtype):
    """将第二个参数的符号复制给第一个参数"""
    x1 = _create_array(xp, [1, 2, -3], dtype)
    x2 = _create_array(xp, [-1, 1, -1], dtype)
    return xp.copysign(x1, x2)


# ========== 3. 其他浮点 Routines (ldexp, fmod) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_ldexp_basic(xp, dtype):
    """x1 * 2**x2"""
    x1 = _create_array(xp, [1, 2], dtype)
    x2 = _create_array(xp, [1, 2], numpy.int32)
    return xp.ldexp(x1, x2)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_fmod_basic(xp, dtype):
    """测试浮点余数"""
    x1 = _create_array(xp, [5.5, -5.5], dtype)
    x2 = _create_array(xp, [3, 3], dtype)
    return xp.fmod(x1, x2)


# ========== 4. Dtype 与硬件限制 (XFAIL) ==========


@pytest.mark.xfail(
    reason="[FIXABLE] isnan/isinf on int dtypes may be unsupported or return wrong types",
    strict=True,
)
@testing.for_dtypes([numpy.int32])
def test_float_checks_int_xfail(xp, dtype):
    a = _create_array(xp, [1, 2], dtype)
    return xp.isinf(a)
