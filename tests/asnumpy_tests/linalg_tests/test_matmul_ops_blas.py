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

"""asnumpy.matmul 的 ops-blas 后端设备测试（issue #28 第三阶段）

对照 numpy 2.5.3 与原实现，覆盖非方阵/非对齐尺寸、K 尾块、零维、非法参数、后端选择的
可观察性、输入不被修改与重复调用。ops-blas 只覆盖 2-D float32，其余输入在显式选择该后端时
必须显式报错（不静默回退）。
"""

import os

import numpy
import pytest

import asnumpy
from asnumpy.linalg import direct

RTOL = 1e-4
ATOL = 1e-4

SHAPES = [
    (1, 1, 1),
    (2, 3, 4),
    (7, 5, 3),
    (64, 128, 32),
    (129, 257, 33),
    (256, 256, 256),
    (1, 64, 128),
    (128, 64, 1),
]


def _pair(m, k, n, seed=17):
    rng = numpy.random.default_rng(seed + m * 131 + k * 17 + n)
    a = rng.standard_normal((m, k)).astype(numpy.float32)
    b = rng.standard_normal((k, n)).astype(numpy.float32)
    return a, b


def _ops_blas_matmul(a, b):
    """Call matmul through the ops-blas backend and prove that it really was used."""
    before = direct._OPS_BLAS_CALLS
    os.environ["ASNUMPY_MATMUL_BACKEND"] = "ops-blas"
    try:
        result = asnumpy.matmul(a, b)
    finally:
        os.environ.pop("ASNUMPY_MATMUL_BACKEND", None)
    assert direct._OPS_BLAS_CALLS == before + 1, "the ops-blas backend was not executed"
    return result


@pytest.mark.parametrize(("m", "k", "n"), SHAPES)
def test_matches_numpy_and_original_backend(m, k, n):
    a_host, b_host = _pair(m, k, n)
    a = asnumpy.ndarray.from_numpy(a_host)
    b = asnumpy.ndarray.from_numpy(b_host)
    a_before, b_before = a.to_numpy().copy(), b.to_numpy().copy()

    expected = numpy.matmul(a_host, b_host)
    calls_before = direct._OPS_BLAS_CALLS
    original = asnumpy.matmul(a, b)  # default backend
    assert direct._OPS_BLAS_CALLS == calls_before, "the default backend must not touch ops-blas"
    got = _ops_blas_matmul(a, b)

    assert isinstance(got, asnumpy.ndarray)
    assert got.shape == (m, n)
    assert got.dtype == numpy.float32
    numpy.testing.assert_allclose(got.to_numpy(), expected, rtol=RTOL, atol=ATOL)
    numpy.testing.assert_allclose(original.to_numpy(), expected, rtol=RTOL, atol=ATOL)
    numpy.testing.assert_array_equal(a.to_numpy(), a_before)
    numpy.testing.assert_array_equal(b.to_numpy(), b_before)


def test_k_zero_yields_zeros():
    a_host = numpy.zeros((4, 0), dtype=numpy.float32)
    b_host = numpy.zeros((0, 3), dtype=numpy.float32)
    got = _ops_blas_matmul(asnumpy.ndarray.from_numpy(a_host), asnumpy.ndarray.from_numpy(b_host))
    assert got.shape == (4, 3)
    numpy.testing.assert_array_equal(got.to_numpy(), numpy.zeros((4, 3), dtype=numpy.float32))


@pytest.mark.parametrize(("m", "k", "n"), [(0, 5, 3), (4, 5, 0)])
def test_zero_dimension_returns_empty(m, k, n):
    a_host, b_host = _pair(m, k, n)
    got = _ops_blas_matmul(asnumpy.ndarray.from_numpy(a_host), asnumpy.ndarray.from_numpy(b_host))
    expected = numpy.matmul(a_host, b_host)
    assert got.shape == expected.shape
    assert got.to_numpy().size == 0


def test_identity_and_zero_matrices():
    eye = numpy.eye(16, dtype=numpy.float32)
    rng = numpy.random.default_rng(3)
    x_host = rng.standard_normal((16, 16)).astype(numpy.float32)
    got = _ops_blas_matmul(asnumpy.ndarray.from_numpy(eye), asnumpy.ndarray.from_numpy(x_host))
    numpy.testing.assert_allclose(got.to_numpy(), x_host, rtol=RTOL, atol=ATOL)

    zeros = numpy.zeros((16, 16), dtype=numpy.float32)
    got_zero = _ops_blas_matmul(
        asnumpy.ndarray.from_numpy(zeros), asnumpy.ndarray.from_numpy(x_host)
    )
    numpy.testing.assert_array_equal(got_zero.to_numpy(), zeros)


def test_large_dynamic_range_and_cancellation():
    values = numpy.array([1e-6, 1e6, -1e6, 1.0, -1.0, 3.25e5], dtype=numpy.float32)
    k = values.size
    a_host = numpy.tile(values, (8, 1)).astype(numpy.float32)
    b_host = numpy.tile(values[::-1], (k, 4)).astype(numpy.float32)
    got = _ops_blas_matmul(asnumpy.ndarray.from_numpy(a_host), asnumpy.ndarray.from_numpy(b_host))
    expected = numpy.matmul(a_host.astype(numpy.float64), b_host.astype(numpy.float64))
    scale = numpy.abs(expected).max()
    assert numpy.abs(got.to_numpy() - expected).max() / scale < 1e-5


def test_inner_dimension_mismatch_raises():
    a = asnumpy.ndarray.from_numpy(numpy.ones((3, 4), dtype=numpy.float32))
    b = asnumpy.ndarray.from_numpy(numpy.ones((5, 6), dtype=numpy.float32))
    calls_before = direct._OPS_BLAS_CALLS
    os.environ["ASNUMPY_MATMUL_BACKEND"] = "ops-blas"
    try:
        with pytest.raises(ValueError, match="not aligned"):
            asnumpy.matmul(a, b)
    finally:
        os.environ.pop("ASNUMPY_MATMUL_BACKEND", None)
    assert direct._OPS_BLAS_CALLS == calls_before, (
        "a rejected call must not count as an ops-blas dispatch"
    )


@pytest.mark.parametrize(
    ("shape", "exc"),
    [
        ((4,), NotImplementedError),
        ((2, 3, 4), NotImplementedError),
    ],
)
def test_non_2d_operands_raise(shape, exc):
    a = asnumpy.ndarray.from_numpy(numpy.ones(shape, dtype=numpy.float32))
    b = asnumpy.ndarray.from_numpy(numpy.ones((3, 3), dtype=numpy.float32))
    os.environ["ASNUMPY_MATMUL_BACKEND"] = "ops-blas"
    try:
        with pytest.raises(exc):
            asnumpy.matmul(a, b)
    finally:
        os.environ.pop("ASNUMPY_MATMUL_BACKEND", None)


@pytest.mark.parametrize("dtype", [numpy.float64, numpy.float16, numpy.complex64, numpy.int32])
def test_non_float32_operands_raise(dtype):
    a = asnumpy.ndarray.from_numpy(numpy.ones((4, 4), dtype=dtype))
    b = asnumpy.ndarray.from_numpy(numpy.ones((4, 4), dtype=dtype))
    os.environ["ASNUMPY_MATMUL_BACKEND"] = "ops-blas"
    try:
        with pytest.raises(TypeError, match="float32"):
            asnumpy.matmul(a, b)
    finally:
        os.environ.pop("ASNUMPY_MATMUL_BACKEND", None)


def test_host_numpy_operand_raises_with_ops_blas():
    a = asnumpy.ndarray.from_numpy(numpy.ones((4, 4), dtype=numpy.float32))
    os.environ["ASNUMPY_MATMUL_BACKEND"] = "ops-blas"
    try:
        with pytest.raises(TypeError, match="asnumpy.ndarray"):
            asnumpy.matmul(a, numpy.ones((4, 4), dtype=numpy.float32))
    finally:
        os.environ.pop("ASNUMPY_MATMUL_BACKEND", None)


def test_invalid_backend_value_raises():
    a = asnumpy.ndarray.from_numpy(numpy.ones((4, 4), dtype=numpy.float32))
    os.environ["ASNUMPY_MATMUL_BACKEND"] = "nope"
    try:
        with pytest.raises(ValueError, match="ASNUMPY_MATMUL_BACKEND"):
            asnumpy.matmul(a, a)
    finally:
        os.environ.pop("ASNUMPY_MATMUL_BACKEND", None)


def test_default_backend_keeps_full_capability():
    """默认后端（原实现）不受可选后端影响：多维、广播、float64 仍可用"""
    rng = numpy.random.default_rng(5)
    calls_before = direct._OPS_BLAS_CALLS
    a3 = rng.standard_normal((2, 3, 4)).astype(numpy.float32)
    b3 = rng.standard_normal((2, 4, 5)).astype(numpy.float32)
    got = asnumpy.matmul(asnumpy.ndarray.from_numpy(a3), asnumpy.ndarray.from_numpy(b3))
    numpy.testing.assert_allclose(got.to_numpy(), numpy.matmul(a3, b3), rtol=RTOL, atol=ATOL)

    a64 = rng.standard_normal((4, 4))
    b64 = rng.standard_normal((4, 4))
    got64 = asnumpy.matmul(asnumpy.ndarray.from_numpy(a64), asnumpy.ndarray.from_numpy(b64))
    numpy.testing.assert_allclose(got64.to_numpy(), numpy.matmul(a64, b64), rtol=1e-10, atol=1e-12)
    assert direct._OPS_BLAS_CALLS == calls_before


def test_repeated_calls_are_stable():
    a_host, b_host = _pair(64, 64, 64, seed=9)
    a = asnumpy.ndarray.from_numpy(a_host)
    b = asnumpy.ndarray.from_numpy(b_host)
    first = _ops_blas_matmul(a, b).to_numpy()
    before = direct._OPS_BLAS_CALLS
    for _ in range(20):
        got = _ops_blas_matmul(a, b).to_numpy()
        numpy.testing.assert_array_equal(got, first)
    assert direct._OPS_BLAS_CALLS == before + 20


def test_missing_backend_message_is_actionable():
    """未启用 ops-blas 的构建应给出可操作的报错；启用时应能正常调用

    ``matmul_ops_blas`` 与 ``matmul`` 一样绑定在顶层 ``_core`` 上（bind_linalg_no_submodule）。
    """
    from asnumpy._core import matmul_ops_blas

    a = asnumpy.ndarray.from_numpy(numpy.ones((4, 4), dtype=numpy.float32))
    try:
        matmul_ops_blas(a, a)
    except RuntimeError as exc:
        assert "ops-blas" in str(exc)
        assert "ASNUMPY_ENABLE_OPS_BLAS" in str(exc)
    else:
        numpy.testing.assert_allclose(
            matmul_ops_blas(a, a).to_numpy(),
            numpy.full((4, 4), 4.0, dtype=numpy.float32),
            rtol=RTOL,
            atol=ATOL,
        )


def test_failure_path_does_not_break_later_calls():
    """有界重复失败（内维不匹配）后，正常调用仍必须正确"""
    good_a = asnumpy.ndarray.from_numpy(numpy.ones((4, 4), dtype=numpy.float32))
    bad_b = asnumpy.ndarray.from_numpy(numpy.ones((7, 4), dtype=numpy.float32))
    os.environ["ASNUMPY_MATMUL_BACKEND"] = "ops-blas"
    try:
        for _ in range(10):
            with pytest.raises(ValueError):
                asnumpy.matmul(good_a, bad_b)
        got = asnumpy.matmul(good_a, good_a).to_numpy()
    finally:
        os.environ.pop("ASNUMPY_MATMUL_BACKEND", None)
    numpy.testing.assert_allclose(
        got, numpy.full((4, 4), 4.0, dtype=numpy.float32), rtol=RTOL, atol=ATOL
    )
