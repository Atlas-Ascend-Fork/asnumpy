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

"""其他特殊数学函数测试

针对已记录的算子限制进行标注：
1. sinc: 基础功能支持 float16, float32, float64。
2. int32: 触发 RuntimeError 161002，底层算子输出类型不支持。
"""

import numpy

from asnumpy import testing

# ========== 辅助函数 ==========


def _create_array(xp, data, dtype):
    """辅助函数：创建数组"""
    np_arr = numpy.array(data, dtype=dtype)
    if xp is numpy:
        return np_arr
    return xp.ndarray.from_numpy(np_arr)


# ========== 1. Sinc 浮点正常链路 ==========


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_sinc_basic(xp, dtype):
    """测试 sinc 基础功能（已知支持的浮点类型）"""
    data = [-3.0, -1.5, 0.5, 2.0, 3.5]
    a = _create_array(xp, data, dtype)
    return xp.sinc(a)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_allclose()
def test_sinc_zero(xp, dtype):
    """测试 sinc(0) = 1"""
    data = [0.0]
    a = _create_array(xp, data, dtype)
    return xp.sinc(a)


# float16's ULP near sinc(0.5)~0.64 is ~4.9e-4, and sinc(-3.0) underflows to 0, so the default
# rtol=1e-7/atol=0 is far tighter than the type can hold.
@testing.for_dtypes([numpy.float16])
@testing.numpy_asnumpy_allclose(rtol=1e-3, atol=1e-3)
def test_sinc_float16(xp, dtype):
    data = [-3.0, -1.5, 0.5, 2.0, 3.5]
    a = _create_array(xp, data, dtype)
    return xp.sinc(a)


# ========== 2. Sinc 整数 dtype 链路 (Int32) ==========


@testing.for_dtypes([numpy.int32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_sinc_int(xp, dtype):
    data = [1, 2]
    a = _create_array(xp, data, dtype)
    return xp.sinc(a)
