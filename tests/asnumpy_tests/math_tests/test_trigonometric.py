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

# ========== 辅助函数 ==========


def _create_array(xp, data, dtype):
    np_arr = numpy.array(data, dtype=dtype)
    if xp is numpy:
        return np_arr
    return xp.ndarray.from_numpy(np_arr)


# ========== 1. 基础三角函数测试 (Sin, Cos, Tan) ==========


# float16's ULP near sin(pi/4)~0.707 is ~4.9e-4, so atol/rtol=1e-5 was ~30x tighter than a single
# ULP -- any disagreement between NumPy's float16 sin and aclnnSin would hard-fail the test.
@testing.for_dtypes([numpy.float16])
@testing.numpy_asnumpy_allclose(atol=1e-3, rtol=1e-3)
def test_trig_float16(xp, dtype):
    data = [0.0, numpy.pi / 4]
    a = _create_array(xp, data, dtype)
    return xp.sin(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_sin_basic(xp, dtype):
    data = [0.0, numpy.pi / 2, numpy.pi]
    a = _create_array(xp, data, dtype)
    return xp.sin(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_cos_basic(xp, dtype):
    data = [0.0, numpy.pi / 2, numpy.pi]
    a = _create_array(xp, data, dtype)
    return xp.cos(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_tan_basic(xp, dtype):
    """测试 tan (使用 allclose 处理 NPU 硬件计算位差)"""
    data = [0.0, numpy.pi / 4, numpy.pi / 3]
    a = _create_array(xp, data, dtype)
    return xp.tan(a)


# --- 针对非浮点类型的 xfail 标注 (遵循 Logic 风格) ---


@testing.for_dtypes([numpy.int32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_sin_int(xp, dtype):
    data = [1, 2, 3]
    a = _create_array(xp, data, dtype)
    return xp.sin(a)


@testing.for_dtypes([numpy.bool_])
@testing.numpy_asnumpy_allclose(atol=1e-3, rtol=1e-3)
def test_cos_bool(xp, dtype):
    """NumPy returns float16 for bool; asnumpy promotes to float32 (float16 mapping gap).

    Compare against float32 reference so float16 rounding on the NumPy side does not dominate.
    """
    data = [True, False]
    if xp is numpy:
        return numpy.cos(numpy.array(data, dtype=numpy.float32))
    a = _create_array(xp, data, dtype)
    return xp.cos(a)


# ========== 2. 反三角函数测试 (Arcsin, Arccos, Arctan) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_arcsin_basic(xp, dtype):
    data = [-1.0, 0.0, 1.0]
    a = _create_array(xp, data, dtype)
    return xp.arcsin(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_arccos_basic(xp, dtype):
    data = [-1.0, 0.0, 1.0]
    a = _create_array(xp, data, dtype)
    return xp.arccos(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_arctan_basic(xp, dtype):
    data = [-10.0, 0.0, 10.0]
    a = _create_array(xp, data, dtype)
    return xp.arctan(a)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_arccos_out_of_domain(xp, dtype):
    """测试反余弦越界行为"""
    data = [2.0]
    a = _create_array(xp, data, dtype)
    return xp.arccos(a)


# ========== 3. 双输入反正切 (Arctan2) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_arctan2_basic(xp, dtype):
    y = [1.0, 0.0, -1.0]
    x = [1.0, 1.0, 1.0]
    t_y = _create_array(xp, y, dtype)
    t_x = _create_array(xp, x, dtype)
    return xp.arctan2(t_y, t_x)


@testing.for_dtypes([numpy.int32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_arctan2_int(xp, dtype):
    y = [1, 0]
    x = [1, 1]
    t_y = _create_array(xp, y, dtype)
    t_x = _create_array(xp, x, dtype)
    return xp.arctan2(t_y, t_x)


# ========== 4. 角度弧度转换 (Deg2rad, Radians, Degrees, Rad2deg) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_deg2rad_basic(xp, dtype):
    data = [0.0, 90.0, 180.0, 45.5, -270.0]
    a = _create_array(xp, data, dtype)
    return xp.deg2rad(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_radians_basic(xp, dtype):
    """radians 与 deg2rad 为同一实现的不同入口"""
    data = [0.0, 90.0, 180.0, 45.5, -270.0]
    a = _create_array(xp, data, dtype)
    return xp.radians(a)


# float16 的 ULP 在 90 附近约 0.0625，默认 rtol=1e-7 远小于单精度位差（同 test_trig_float16）。
@testing.for_dtypes([numpy.float16])
@testing.numpy_asnumpy_allclose(rtol=1e-3, atol=1e-3)
def test_deg2rad_float16(xp, dtype):
    data = [0.0, 90.0, 180.0]
    a = _create_array(xp, data, dtype)
    return xp.deg2rad(a)


@pytest.mark.xfail(
    reason="[FIXABLE] aclnnForeachMulScalar unsupported dtype (float64)", strict=True
)
@testing.for_dtypes([numpy.float64])
def test_deg2rad_float64_xfail(xp, dtype):
    data = [0.0, 90.0, 180.0]
    a = _create_array(xp, data, dtype)
    return xp.deg2rad(a)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-12)
def test_degrees_basic(xp, dtype):
    data = [0.0, numpy.pi / 2, numpy.pi, 1.5, -numpy.pi]
    a = _create_array(xp, data, dtype)
    return xp.degrees(a)


@testing.for_dtypes([numpy.int32, numpy.int64])
@testing.numpy_asnumpy_allclose(rtol=1e-12)
def test_degrees_int(xp, dtype):
    """整数输入两侧均提升为 float64"""
    data = [0, 1, -1, 2]
    a = _create_array(xp, data, dtype)
    return xp.degrees(a)


@pytest.mark.xfail(reason="[FIXABLE] Degrees float32/float16 输出量级错误（约 1e-22）", strict=True)
@testing.for_dtypes([numpy.float32, numpy.float16])
@testing.numpy_asnumpy_allclose(atol=1e-3, rtol=1e-3)
def test_degrees_float32_xfail(xp, dtype):
    data = [0.0, numpy.pi / 2, numpy.pi]
    a = _create_array(xp, data, dtype)
    return xp.degrees(a)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-12)
def test_rad2deg_basic(xp, dtype):
    """rad2deg 与 degrees 为同一实现的不同入口"""
    data = [0.0, numpy.pi / 2, numpy.pi, 1.5, -numpy.pi]
    a = _create_array(xp, data, dtype)
    return xp.rad2deg(a)


@pytest.mark.xfail(reason="[FIXABLE] Rad2deg float32/float16 输出量级错误（约 1e-22）", strict=True)
@testing.for_dtypes([numpy.float32, numpy.float16])
@testing.numpy_asnumpy_allclose(atol=1e-3, rtol=1e-3)
def test_rad2deg_float32_xfail(xp, dtype):
    data = [0.0, numpy.pi / 2, numpy.pi]
    a = _create_array(xp, data, dtype)
    return xp.rad2deg(a)
