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

"""矩阵乘积函数测试

包含：
1. 点积: dot
2. 内积: inner
3. 外积: outer
4. 向量点积: vdot
5. 矩阵乘法: matmul
6. 矩阵幂: matrix_power
7. 爱因斯坦求和约定: einsum

优化维度：
- FP32/FP64 精度验证
- 奇异矩阵边界
- 非方阵输入
- 广播行为
- 空矩阵输入
"""

import numpy
import pytest

import asnumpy
from asnumpy import testing


# ========== 辅助函数 ==========
def _create_array(xp, data, dtype):
    """辅助函数：创建数组"""
    np_arr = numpy.array(data, dtype=dtype)
    if xp is numpy:
        return np_arr
    # asnumpy 环境
    return xp.ndarray.from_numpy(np_arr)


# ==========================================================================
# 1. dot 点积测试
# ==========================================================================


# ---------- 1.1 基础功能: 1D 向量点积 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_dot_1d(xp, dtype):
    """1D · 1D → 标量"""
    a = _create_array(xp, [1.0, 2.0, 3.0], dtype)
    b = _create_array(xp, [4.0, 5.0, 6.0], dtype)
    return xp.dot(a, b)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_dot_1d_negative(xp, dtype):
    """1D · 1D 含负数"""
    a = _create_array(xp, [1.0, -2.0, 3.0], dtype)
    b = _create_array(xp, [-4.0, 5.0, -6.0], dtype)
    return xp.dot(a, b)


# ---------- 1.2 基础功能: 2D 矩阵乘法 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_dot_2d(xp, dtype):
    """2D × 2D → 矩阵乘法"""
    a = _create_array(xp, [[1.0, 2.0], [3.0, 4.0]], dtype)
    b = _create_array(xp, [[5.0, 6.0], [7.0, 8.0]], dtype)
    return xp.dot(a, b)


# ---------- 1.3 FP32/FP64 精度验证 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-4)
def test_dot_fp32_precision(xp, dtype):
    """FP32 精度: 随机向量点积"""
    numpy.random.seed(42)
    a_data = numpy.random.uniform(-10, 10, (5,)).astype(dtype)
    b_data = numpy.random.uniform(-10, 10, (5,)).astype(dtype)
    a = _create_array(xp, a_data, dtype)
    b = _create_array(xp, b_data, dtype)
    return xp.dot(a, b)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-10, atol=1e-10)
def test_dot_fp64_precision(xp, dtype):
    """FP64 精度: 随机向量点积"""
    numpy.random.seed(42)
    a_data = numpy.random.uniform(-10, 10, (5,)).astype(dtype)
    b_data = numpy.random.uniform(-10, 10, (5,)).astype(dtype)
    a = _create_array(xp, a_data, dtype)
    b = _create_array(xp, b_data, dtype)
    return xp.dot(a, b)


# ---------- 1.4 非方阵输入 (2D) ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_dot_nonsquare(xp, dtype):
    """非方阵: (2,3) × (3,4) → (2,4)"""
    a = _create_array(xp, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype)
    b = _create_array(
        xp, [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], dtype
    )
    return xp.dot(a, b)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_dot_nonsquare_3x2_2x1(xp, dtype):
    """非方阵: (3,2) × (2,1) → (3,1)"""
    a = _create_array(xp, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype)
    b = _create_array(xp, [[1.0], [2.0]], dtype)
    return xp.dot(a, b)


# ---------- 1.5 空矩阵输入 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_dot_empty(xp, dtype):
    """空矩阵: 空向量点积"""
    a = _create_array(xp, [], dtype)
    b = _create_array(xp, [], dtype)
    return xp.dot(a, b)


# ==========================================================================
# 2. inner 内积测试
# ==========================================================================


# ---------- 2.1 基础功能 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_inner_1d(xp, dtype):
    """1D × 1D 内积"""
    a = _create_array(xp, [1.0, 2.0, 3.0], dtype)
    b = _create_array(xp, [4.0, 5.0, 6.0], dtype)
    return xp.inner(a, b)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_inner_2d(xp, dtype):
    """2D × 2D 内积"""
    a = _create_array(xp, [[1.0, 2.0], [3.0, 4.0]], dtype)
    b = _create_array(xp, [[5.0, 6.0], [7.0, 8.0]], dtype)
    return xp.inner(a, b)


# ---------- 2.2 FP32/FP64 精度验证 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-4)
def test_inner_fp32_precision(xp, dtype):
    """FP32 精度: 内积"""
    numpy.random.seed(55)
    a_data = numpy.random.uniform(-5, 5, (4,)).astype(dtype)
    b_data = numpy.random.uniform(-5, 5, (4,)).astype(dtype)
    a = _create_array(xp, a_data, dtype)
    b = _create_array(xp, b_data, dtype)
    return xp.inner(a, b)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-10, atol=1e-10)
def test_inner_fp64_precision(xp, dtype):
    """FP64 精度: 内积"""
    numpy.random.seed(55)
    a_data = numpy.random.uniform(-5, 5, (4,)).astype(dtype)
    b_data = numpy.random.uniform(-5, 5, (4,)).astype(dtype)
    a = _create_array(xp, a_data, dtype)
    b = _create_array(xp, b_data, dtype)
    return xp.inner(a, b)


# ---------- 2.3 空矩阵输入 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_inner_empty(xp, dtype):
    """空矩阵: 空向量内积"""
    a = _create_array(xp, [], dtype)
    b = _create_array(xp, [], dtype)
    return xp.inner(a, b)


# ==========================================================================
# 3. outer 外积测试
# ==========================================================================


# ---------- 3.1 基础功能 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_outer_1d(xp, dtype):
    """1D × 1D 外积"""
    a = _create_array(xp, [1.0, 2.0, 3.0], dtype)
    b = _create_array(xp, [4.0, 5.0], dtype)
    return xp.outer(a, b)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_outer_2d_flatten(xp, dtype):
    """2D 输入自动展平后外积"""
    a = _create_array(xp, [[1.0, 2.0], [3.0, 4.0]], dtype)
    b = _create_array(xp, [5.0, 6.0], dtype)
    return xp.outer(a, b)


# ---------- 3.2 FP32/FP64 精度验证 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-4)
def test_outer_fp32_precision(xp, dtype):
    """FP32 精度: 外积"""
    numpy.random.seed(33)
    a_data = numpy.random.uniform(-3, 3, (4,)).astype(dtype)
    b_data = numpy.random.uniform(-3, 3, (5,)).astype(dtype)
    a = _create_array(xp, a_data, dtype)
    b = _create_array(xp, b_data, dtype)
    return xp.outer(a, b)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-10, atol=1e-10)
def test_outer_fp64_precision(xp, dtype):
    """FP64 精度: 外积"""
    numpy.random.seed(33)
    a_data = numpy.random.uniform(-3, 3, (4,)).astype(dtype)
    b_data = numpy.random.uniform(-3, 3, (5,)).astype(dtype)
    a = _create_array(xp, a_data, dtype)
    b = _create_array(xp, b_data, dtype)
    return xp.outer(a, b)


# ---------- 3.3 空矩阵输入 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_outer_empty(xp, dtype):
    """空矩阵: 空向量外积"""
    a = _create_array(xp, [], dtype)
    b = _create_array(xp, [], dtype)
    return xp.outer(a, b)


# ==========================================================================
# 4. vdot 向量点积测试
# ==========================================================================


# ---------- 4.1 基础功能 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_vdot_1d(xp, dtype):
    """1D vdot"""
    a = _create_array(xp, [1.0, 2.0, 3.0], dtype)
    b = _create_array(xp, [4.0, 5.0, 6.0], dtype)
    return xp.vdot(a, b)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_vdot_2d(xp, dtype):
    """2D vdot (自动展平)"""
    a = _create_array(xp, [[1.0, 2.0], [3.0, 4.0]], dtype)
    b = _create_array(xp, [[5.0, 6.0], [7.0, 8.0]], dtype)
    return xp.vdot(a, b)


# ---------- 4.2 FP32/FP64 精度验证 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-4)
def test_vdot_fp32_precision(xp, dtype):
    """FP32 精度: vdot"""
    numpy.random.seed(44)
    a_data = numpy.random.uniform(-5, 5, (6,)).astype(dtype)
    b_data = numpy.random.uniform(-5, 5, (6,)).astype(dtype)
    a = _create_array(xp, a_data, dtype)
    b = _create_array(xp, b_data, dtype)
    return xp.vdot(a, b)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-10, atol=1e-10)
def test_vdot_fp64_precision(xp, dtype):
    """FP64 精度: vdot"""
    numpy.random.seed(44)
    a_data = numpy.random.uniform(-5, 5, (6,)).astype(dtype)
    b_data = numpy.random.uniform(-5, 5, (6,)).astype(dtype)
    a = _create_array(xp, a_data, dtype)
    b = _create_array(xp, b_data, dtype)
    return xp.vdot(a, b)


# ---------- 4.3 空矩阵输入 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_vdot_empty(xp, dtype):
    """空矩阵: 空向量 vdot"""
    a = _create_array(xp, [], dtype)
    b = _create_array(xp, [], dtype)
    return xp.vdot(a, b)


# ==========================================================================
# 5. matmul 矩阵乘法测试
# ==========================================================================


# ---------- 5.1 基础功能 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_matmul_2d(xp, dtype):
    """2D × 2D 矩阵乘法"""
    a = _create_array(xp, [[1.0, 2.0], [3.0, 4.0]], dtype)
    b = _create_array(xp, [[5.0, 6.0], [7.0, 8.0]], dtype)
    return xp.matmul(a, b)


# ---------- 5.2 FP32/FP64 精度验证 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-4)
def test_matmul_fp32_precision(xp, dtype):
    """FP32 精度: 矩阵乘法"""
    numpy.random.seed(66)
    a_data = numpy.random.uniform(-5, 5, (4, 3)).astype(dtype)
    b_data = numpy.random.uniform(-5, 5, (3, 4)).astype(dtype)
    a = _create_array(xp, a_data, dtype)
    b = _create_array(xp, b_data, dtype)
    return xp.matmul(a, b)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-10, atol=1e-10)
def test_matmul_fp64_precision(xp, dtype):
    """FP64 精度: 矩阵乘法"""
    numpy.random.seed(66)
    a_data = numpy.random.uniform(-5, 5, (4, 3)).astype(dtype)
    b_data = numpy.random.uniform(-5, 5, (3, 4)).astype(dtype)
    a = _create_array(xp, a_data, dtype)
    b = _create_array(xp, b_data, dtype)
    return xp.matmul(a, b)


# ---------- 5.3 非方阵输入 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_matmul_nonsquare(xp, dtype):
    """非方阵: (2,3) × (3,2) → (2,2)"""
    a = _create_array(xp, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype)
    b = _create_array(xp, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype)
    return xp.matmul(a, b)


# ---------- 5.4 广播行为 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_matmul_broadcast(xp, dtype):
    """广播: 批量矩阵乘法 (2,2,3) × (2,3,2)"""
    data_a = numpy.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    ).astype(dtype)
    data_b = numpy.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ]
    ).astype(dtype)
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.matmul(a, b)


# ---------- 5.5 空矩阵输入 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_matmul_empty(xp, dtype):
    """空矩阵: 空输入"""
    a = _create_array(xp, numpy.zeros((0, 3)), dtype)
    b = _create_array(xp, numpy.zeros((3, 0)), dtype)
    return xp.matmul(a, b)


# ==========================================================================
# 6. matrix_power 矩阵幂测试
# ==========================================================================


# ---------- 6.1 基础功能: 正幂 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-4)
def test_matrix_power_positive(xp, dtype):
    """正幂: n=2"""
    data = [[1.0, 2.0], [3.0, 4.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.matrix_power(a, 2)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-4)
def test_matrix_power_n3(xp, dtype):
    """正幂: n=3"""
    data = [[1.0, 0.0], [0.0, 2.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.matrix_power(a, 3)


# ---------- 6.2 基础功能: 零幂 (返回单位阵) ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_matrix_power_zero(xp, dtype):
    """零幂: n=0 → 单位阵"""
    data = [[3.0, 1.0], [2.0, 4.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.matrix_power(a, 0)


# ---------- 6.3 负幂 (逆矩阵的幂) ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-3, atol=1e-3)
def test_matrix_power_negative(xp, dtype):
    """负幂: n=-1 → 逆矩阵"""
    data = [[1.0, 2.0], [3.0, 4.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.matrix_power(a, -1)


# ---------- 6.4 FP32/FP64 精度验证 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-3, atol=1e-3)
def test_matrix_power_fp32_precision(xp, dtype):
    """FP32 精度: 矩阵幂"""
    numpy.random.seed(77)
    data = numpy.random.uniform(-2, 2, (3, 3)).astype(dtype)
    a = _create_array(xp, data, dtype)
    return xp.linalg.matrix_power(a, 2)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-8, atol=1e-8)
def test_matrix_power_fp64_precision(xp, dtype):
    """FP64 精度: 矩阵幂"""
    numpy.random.seed(77)
    data = numpy.random.uniform(-2, 2, (3, 3)).astype(dtype)
    a = _create_array(xp, data, dtype)
    return xp.linalg.matrix_power(a, 2)


# ---------- 6.5 奇异矩阵边界 ----------
def test_matrix_power_singular_negative():
    """奇异矩阵边界: 负幂应对奇异矩阵抛出异常"""
    data = [[1.0, 2.0], [2.0, 4.0]]
    a = asnumpy.ndarray.from_numpy(numpy.array(data, dtype=numpy.float32))
    with pytest.raises((RuntimeError, ValueError, Exception)):
        asnumpy.linalg.matrix_power(a, -1)


# ---------- 6.6 空矩阵输入 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_matrix_power_empty(xp, dtype):
    """空矩阵: 空输入"""
    a = _create_array(xp, numpy.zeros((0, 0)), dtype)
    return xp.linalg.matrix_power(a, 0)


# ==========================================================================
# 7. einsum 爱因斯坦求和约定测试
# ==========================================================================


# ---------- 7.1 基础功能: 外积模式 "a,b->ab" ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_einsum_outer_basic(xp, dtype):
    """外积模式: (3,) × (2,) → (3,2)"""
    a = _create_array(xp, [1.0, 2.0, 3.0], dtype)
    b = _create_array(xp, [4.0, 5.0], dtype)
    return xp.einsum("a,b->ab", a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_einsum_outer_asymmetric(xp, dtype):
    """外积模式: 非对称长度 (3,) × (5,) → (3,5)"""
    a = _create_array(xp, [1.0, -2.0, 3.0], dtype)
    b = _create_array(xp, [4.0, 5.0, -6.0, 7.0, 8.0], dtype)
    return xp.einsum("a,b->ab", a, b)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_einsum_outer_zero_and_negative(xp, dtype):
    """外积模式: 含零与负数"""
    a = _create_array(xp, [0.0, -1.5, 2.5], dtype)
    b = _create_array(xp, [-3.0, 0.0, 4.0], dtype)
    return xp.einsum("a,b->ab", a, b)


# ---------- 7.2 FP32 精度验证 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-4)
def test_einsum_outer_fp32_precision(xp, dtype):
    """FP32 精度: 随机向量外积"""
    numpy.random.seed(88)
    a_data = numpy.random.uniform(-3, 3, (4,)).astype(dtype)
    b_data = numpy.random.uniform(-3, 3, (5,)).astype(dtype)
    a = _create_array(xp, a_data, dtype)
    b = _create_array(xp, b_data, dtype)
    return xp.einsum("a,b->ab", a, b)


# ---------- 7.3 整数 dtype ----------
@testing.for_dtypes([numpy.int32, numpy.int64])
@testing.numpy_asnumpy_array_equal()
def test_einsum_outer_int(xp, dtype):
    """整数 dtype: 外积精确匹配"""
    a = _create_array(xp, [1, -2, 3, 4], dtype)
    b = _create_array(xp, [5, -6, 7], dtype)
    return xp.einsum("a,b->ab", a, b)


# ---------- 7.4 空矩阵输入 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_einsum_outer_empty(xp, dtype):
    """空矩阵: 空向量外积 → (0,0)"""
    a = _create_array(xp, [], dtype)
    b = _create_array(xp, [], dtype)
    return xp.einsum("a,b->ab", a, b)


# ---------- 7.5 注意力模式 "abcd,abced->abce" (要求 e == d) ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-4)
def test_einsum_attention_basic(xp, dtype):
    """注意力模式: (2,3,4,5) × (2,3,4,5,5) → (2,3,4,5)"""
    numpy.random.seed(99)
    x1_data = numpy.random.uniform(-1, 1, (2, 3, 4, 5)).astype(dtype)
    x2_data = numpy.random.uniform(-1, 1, (2, 3, 4, 5, 5)).astype(dtype)
    x1 = _create_array(xp, x1_data, dtype)
    x2 = _create_array(xp, x2_data, dtype)
    return xp.einsum("abcd,abced->abce", x1, x2)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-4)
def test_einsum_attention_fp32_precision(xp, dtype):
    """注意力模式 FP32 精度: 随机数据批量缩并"""
    numpy.random.seed(100)
    x1_data = numpy.random.uniform(-2, 2, (3, 2, 4, 6)).astype(dtype)
    x2_data = numpy.random.uniform(-2, 2, (3, 2, 4, 6, 6)).astype(dtype)
    x1 = _create_array(xp, x1_data, dtype)
    x2 = _create_array(xp, x2_data, dtype)
    return xp.einsum("abcd,abced->abce", x1, x2)
