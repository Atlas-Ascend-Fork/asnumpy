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

"""激活函数测试

包含：
1. Softmax 激活函数
2. Gelu 激活函数（tanh 近似）
3. Relu 激活函数

优化维度：
- 数值稳定性（极端值输入）
- 多维输入
- axis 参数
- 多 dtype 支持
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
    # asnumpy 环境
    return xp.ndarray.from_numpy(np_arr)


def _numpy_softmax(x, axis=-1):
    """NumPy 参考实现的 softmax（数值稳定版）"""
    x = numpy.asarray(x)
    x_max = numpy.max(x, axis=axis, keepdims=True)
    exp_x = numpy.exp(x - x_max)
    return exp_x / numpy.sum(exp_x, axis=axis, keepdims=True)


# 将 softmax 注册到 numpy 模块上，以便 numpy_asnumpy_allclose 装饰器可以使用
numpy.softmax = _numpy_softmax


def _numpy_gelu(x):
    """NumPy 参考实现的 GELU（tanh 近似，与 CANN aclnnGelu 实现一致）"""
    x = numpy.asarray(x)
    # sqrt(2/pi) 用 Python float 常数，避免 0-d float64 数组把结果提升为 float64
    return 0.5 * x * (1.0 + numpy.tanh(0.7978845608028654 * (x + 0.044715 * numpy.power(x, 3))))


def _numpy_relu(x):
    """NumPy 参考实现的 ReLU"""
    return numpy.maximum(numpy.asarray(x), 0)


# 将 gelu/relu 注册到 numpy 模块上，以便 numpy_asnumpy_allclose 装饰器可以使用
numpy.gelu = _numpy_gelu
numpy.relu = _numpy_relu


# ==========================================================================
# 1. 基础功能测试 (Basic)
# ==========================================================================


# ---------- 1.1 一维数组 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_1d(xp, dtype):
    """测试一维数组的 softmax"""
    data = [1.0, 2.0, 3.0]
    a = _create_array(xp, data, dtype)
    return xp.softmax(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_1d_negative_values(xp, dtype):
    """测试含负值的一维数组 softmax"""
    data = [-1.0, 0.0, 1.0]
    a = _create_array(xp, data, dtype)
    return xp.softmax(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_1d_default_axis(xp, dtype):
    """测试 softmax 默认 axis（应为 -1）"""
    data = [0.5, 1.5, 2.5, 3.5]
    a = _create_array(xp, data, dtype)
    return xp.softmax(a)


# ---------- 1.2 单元素 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_single_element(xp, dtype):
    """测试单元素数组 softmax（结果应为 1.0）"""
    data = [5.0]
    a = _create_array(xp, data, dtype)
    return xp.softmax(a)


# ==========================================================================
# 2. axis 参数测试 (Axis Parameter)
# ==========================================================================


# ---------- 2.1 二维数组 axis ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_2d_axis0(xp, dtype):
    """测试二维数组沿 axis=0 的 softmax"""
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    a = _create_array(xp, data, dtype)
    return xp.softmax(a, axis=0)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_2d_axis1(xp, dtype):
    """测试二维数组沿 axis=1 的 softmax"""
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    a = _create_array(xp, data, dtype)
    return xp.softmax(a, axis=1)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_2d_negative_axis(xp, dtype):
    """测试二维数组负数 axis（axis=-1 等价于 axis=1）"""
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    a = _create_array(xp, data, dtype)
    return xp.softmax(a, axis=-1)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_2d_negative_axis0(xp, dtype):
    """测试二维数组负数 axis（axis=-2 等价于 axis=0）"""
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    a = _create_array(xp, data, dtype)
    return xp.softmax(a, axis=-2)


# ---------- 2.2 三维数组 axis ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_3d_axis0(xp, dtype):
    """测试三维数组沿 axis=0 的 softmax"""
    numpy.random.seed(42)
    np_data = numpy.random.uniform(-5, 5, (2, 3, 4)).astype(dtype)
    a = _create_array(xp, np_data, dtype)
    return xp.softmax(a, axis=0)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_3d_axis1(xp, dtype):
    """测试三维数组沿 axis=1 的 softmax"""
    numpy.random.seed(42)
    np_data = numpy.random.uniform(-5, 5, (2, 3, 4)).astype(dtype)
    a = _create_array(xp, np_data, dtype)
    return xp.softmax(a, axis=1)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_3d_axis2(xp, dtype):
    """测试三维数组沿 axis=2 的 softmax"""
    numpy.random.seed(42)
    np_data = numpy.random.uniform(-5, 5, (2, 3, 4)).astype(dtype)
    a = _create_array(xp, np_data, dtype)
    return xp.softmax(a, axis=2)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_3d_negative_axis(xp, dtype):
    """测试三维数组负数 axis（axis=-1 等价于 axis=2）"""
    numpy.random.seed(42)
    np_data = numpy.random.uniform(-5, 5, (2, 3, 4)).astype(dtype)
    a = _create_array(xp, np_data, dtype)
    return xp.softmax(a, axis=-1)


# ==========================================================================
# 3. 多维输入测试 (Multi-dimensional Input)
# ==========================================================================


# ---------- 3.1 二维不同形状 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_2d_wide(xp, dtype):
    """测试二维宽矩阵 softmax (2x10)"""
    data = [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    ]
    a = _create_array(xp, data, dtype)
    return xp.softmax(a, axis=1)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_2d_tall(xp, dtype):
    """测试二维高矩阵 softmax (10x2)"""
    numpy.random.seed(42)
    np_data = numpy.random.uniform(-3, 3, (10, 2)).astype(dtype)
    a = _create_array(xp, np_data, dtype)
    return xp.softmax(a, axis=1)


# ---------- 3.2 三维输入 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_3d_input(xp, dtype):
    """测试三维数组 softmax"""
    numpy.random.seed(42)
    np_data = numpy.random.randn(2, 3, 4).astype(dtype)
    a = _create_array(xp, np_data, dtype)
    return xp.softmax(a, axis=2)


# ---------- 3.3 四维输入（模拟深度学习批处理场景）----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_4d_batch(xp, dtype):
    """测试四维数组 softmax（批处理场景: batch=2, channels=3, height=4, width=5）"""
    numpy.random.seed(42)
    np_data = numpy.random.uniform(-2, 2, (2, 3, 4, 5)).astype(dtype)
    a = _create_array(xp, np_data, dtype)
    return xp.softmax(a, axis=1)


# ---------- 3.4 批量分类场景 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_batch_classification(xp, dtype):
    """测试批量分类场景 softmax (batch=8, classes=10)"""
    numpy.random.seed(42)
    np_data = numpy.random.randn(8, 10).astype(dtype)
    a = _create_array(xp, np_data, dtype)
    return xp.softmax(a, axis=1)


# ==========================================================================
# 4. 数值稳定性测试 (Numerical Stability)
# ==========================================================================


# ---------- 4.1 大正值 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_large_positive(xp, dtype):
    """测试大正值输入的数值稳定性（不应产生 NaN 或 Inf）"""
    data = [100.0, 200.0, 300.0]
    a = _create_array(xp, data, dtype)
    return xp.softmax(a)


# ---------- 4.2 大负值 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_large_negative(xp, dtype):
    """测试大负值输入的数值稳定性"""
    data = [-100.0, -200.0, -300.0]
    a = _create_array(xp, data, dtype)
    return xp.softmax(a)


# ---------- 4.3 混合极端值 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_mixed_extreme(xp, dtype):
    """测试混合极端值（大正、零、大负）的数值稳定性"""
    data = [-100.0, 0.0, 100.0]
    a = _create_array(xp, data, dtype)
    return xp.softmax(a)


# ---------- 4.4 极大值差异 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_huge_range(xp, dtype):
    """测试极大值差异的数值稳定性"""
    data = [-1e10, 0.0, 1e10]
    a = _create_array(xp, data, dtype)
    return xp.softmax(a)


# ---------- 4.5 非常接近的值 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_near_zero(xp, dtype):
    """测试接近零的小值"""
    data = [1e-7, 2e-7, 3e-7]
    a = _create_array(xp, data, dtype)
    return xp.softmax(a)


# ---------- 4.6 所有值相同 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_uniform_values(xp, dtype):
    """测试所有值相同（结果应为均匀分布 1/n）"""
    data = [5.0, 5.0, 5.0, 5.0]
    a = _create_array(xp, data, dtype)
    return xp.softmax(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_uniform_negative(xp, dtype):
    """测试所有负值相同"""
    data = [-3.0, -3.0, -3.0]
    a = _create_array(xp, data, dtype)
    return xp.softmax(a)


# ---------- 4.7 二维极端值 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_2d_large_values(xp, dtype):
    """测试二维数组大数值的数值稳定性"""
    data = [[10.0, 20.0, 30.0], [30.0, 20.0, 10.0]]
    a = _create_array(xp, data, dtype)
    return xp.softmax(a, axis=1)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_2d_extreme_mixed(xp, dtype):
    """测试二维数组混合极端值"""
    data = [[-1000.0, 0.0, 1000.0], [1000.0, 0.0, -1000.0]]
    a = _create_array(xp, data, dtype)
    return xp.softmax(a, axis=1)


# ==========================================================================
# 5. 多 dtype 测试 (Dtype Support)
# ==========================================================================


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_dtypes(xp, dtype):
    """测试 softmax 对多种浮点 dtype 的支持"""
    data = [1.0, 2.0, 3.0, 4.0]
    a = _create_array(xp, data, dtype)
    return xp.softmax(a)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_2d_dtypes(xp, dtype):
    """测试二维数组 softmax 多 dtype 支持"""
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    a = _create_array(xp, data, dtype)
    return xp.softmax(a, axis=1)


# ==========================================================================
# 6. 空数组 / 边界情况测试 (Edge Cases)
# ==========================================================================


@pytest.mark.xfail(
    reason="[UPSTREAM] CANN 8.0.RC1: operator does not support empty arrays", strict=True
)
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_softmax_empty(xp):
    """测试 softmax: 空数组"""
    a = _create_array(xp, [], numpy.float32)
    return xp.softmax(a)


# ==========================================================================
# 7. Gelu 激活函数 (Gelu)
# ==========================================================================


# ---------- 7.1 基础功能 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-6)
def test_gelu_basic(xp, dtype):
    """测试 gelu 基本功能（tanh 近似参考实现）"""
    data = [-2.0, -0.5, 0.0, 0.5, 2.0]
    a = _create_array(xp, data, dtype)
    return xp.gelu(a)


# ---------- 7.2 多维输入 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-6)
def test_gelu_2d(xp, dtype):
    """测试二维数组 gelu"""
    data = [[-1.0, 0.0, 1.0], [-3.0, 2.0, 0.5]]
    a = _create_array(xp, data, dtype)
    return xp.gelu(a)


# ---------- 7.3 限制性测试 ----------
@pytest.mark.xfail(reason="[FIXABLE] aclnnGelu unsupported dtypes (float64/int/bool)", strict=True)
@testing.for_dtypes([numpy.float64, numpy.int32])
def test_gelu_unsupported_xfail(xp, dtype):
    a = _create_array(xp, [-1.0, 1.0], dtype)
    return xp.gelu(a)


# ==========================================================================
# 8. Relu 激活函数 (Relu)
# ==========================================================================


# ---------- 8.1 基础功能 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_relu_basic(xp, dtype):
    """测试 relu 基本功能：负值置零、正值保留"""
    data = [-2.0, -0.5, 0.0, 0.5, 2.0]
    a = _create_array(xp, data, dtype)
    return xp.relu(a)


@testing.for_dtypes([numpy.int32, numpy.int64])
@testing.numpy_asnumpy_array_equal()
def test_relu_int(xp, dtype):
    """测试整数 relu，结果保持整数 dtype"""
    data = [-2, 0, 3]
    a = _create_array(xp, data, dtype)
    return xp.relu(a)


# ---------- 8.2 多维输入 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_relu_2d(xp, dtype):
    """测试二维数组 relu"""
    data = [[-1.0, 0.0, 1.0], [-3.0, 2.0, 0.5]]
    a = _create_array(xp, data, dtype)
    return xp.relu(a)


# ---------- 8.3 限制性测试 ----------
@pytest.mark.xfail(reason="[FIXABLE] aclnnRelu unsupported dtypes (float64/bool)", strict=True)
@testing.for_dtypes([numpy.float64])
def test_relu_float64_xfail(xp, dtype):
    a = _create_array(xp, [-1.0, 1.0], dtype)
    return xp.relu(a)
