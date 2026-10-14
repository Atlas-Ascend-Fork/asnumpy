# *****************************************************************************
# Copyright (c) 2025 ISE Group at Harbin Institute of Technology. All Rights Reserved.
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

"""SVD 与伪逆测试 (issue #26)

包含：
1. linalg.svd (full/reduced/compute_uv=False/批量)
2. linalg.pinv

SVD 的左右奇异向量在符号/简并子空间上不唯一，因此不做逐元素比较，而是验证
NumPy 与 AsNumpy 各自都满足的不变量：奇异值一致、可重构、U/V 正交、伪逆满足
Moore-Penrose 条件。
"""

import numpy
import pytest

import asnumpy as ap

DTYPES = [numpy.float32, numpy.float64]
RTOL = {numpy.float32: 1e-4, numpy.float64: 1e-10}


def _arr(data, dtype):
    return ap.ndarray.from_numpy(numpy.asarray(data, dtype=dtype))


def _random(shape, dtype, seed=42):
    rng = numpy.random.default_rng(seed)
    return rng.uniform(-5, 5, shape).astype(dtype)


def _as_tuple(result):
    return tuple(x.to_numpy() for x in result)


# ==========================================================================
# 1. svd
# ==========================================================================


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(4, 3), (3, 4), (5, 5)])
@pytest.mark.parametrize("full_matrices", [True, False])
def test_svd_shapes_match_numpy(dtype, shape, full_matrices):
    data = _random(shape, dtype)
    expected = numpy.linalg.svd(data, full_matrices=full_matrices)
    u, s, vh = _as_tuple(ap.linalg.svd(_arr(data, dtype), full_matrices=full_matrices))
    for got, want in zip((u, s, vh), expected, strict=True):
        assert got.shape == want.shape


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(4, 3), (3, 4), (6, 6)])
def test_svd_singular_values_match_numpy(dtype, shape):
    data = _random(shape, dtype)
    expected = numpy.linalg.svd(data, compute_uv=False)
    _, s, _ = _as_tuple(ap.linalg.svd(_arr(data, dtype)))
    numpy.testing.assert_allclose(
        numpy.sort(s), numpy.sort(expected), rtol=RTOL[dtype], atol=RTOL[dtype]
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(4, 3), (3, 4), (5, 5)])
@pytest.mark.parametrize("full_matrices", [True, False])
def test_svd_reconstructs_input(dtype, shape, full_matrices):
    data = _random(shape, dtype, seed=7)
    u, s, vh = _as_tuple(ap.linalg.svd(_arr(data, dtype), full_matrices=full_matrices))
    # u @ diag(s) @ vh: for the reduced SVD the middle factor is (K, K), for the full SVD (M, N).
    # Deriving both dimensions from u/vh covers the two cases without special-casing.
    sigma = numpy.zeros((u.shape[-1], vh.shape[-2]), dtype=dtype)
    numpy.fill_diagonal(sigma, s)
    reconstructed = u @ sigma @ vh
    numpy.testing.assert_allclose(reconstructed, data, rtol=RTOL[dtype] * 10, atol=RTOL[dtype] * 10)


@pytest.mark.parametrize("dtype", DTYPES)
def test_svd_factors_are_orthonormal(dtype):
    data = _random((5, 3), dtype, seed=11)
    u, _, vh = _as_tuple(ap.linalg.svd(_arr(data, dtype), full_matrices=False))
    k = min(data.shape)
    numpy.testing.assert_allclose(
        u.T @ u, numpy.eye(k, dtype=dtype), rtol=RTOL[dtype], atol=RTOL[dtype]
    )
    numpy.testing.assert_allclose(
        vh @ vh.T, numpy.eye(k, dtype=dtype), rtol=RTOL[dtype], atol=RTOL[dtype]
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_svd_compute_uv_false_returns_singular_values_only(dtype):
    data = _random((4, 2), dtype, seed=3)
    result = ap.linalg.svd(_arr(data, dtype), compute_uv=False)
    assert isinstance(result, ap.ndarray)
    numpy.testing.assert_allclose(
        result.to_numpy(),
        numpy.linalg.svd(data, compute_uv=False),
        rtol=RTOL[dtype],
        atol=RTOL[dtype],
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_svd_batched(dtype):
    data = numpy.stack([_random((3, 3), dtype, seed=s) for s in (1, 2)])
    u, s, vh = _as_tuple(ap.linalg.svd(_arr(data, dtype), full_matrices=False))
    assert u.shape == (2, 3, 3)
    assert s.shape == (2, 3)
    assert vh.shape == (2, 3, 3)
    numpy.testing.assert_allclose(
        (u * s[..., None, :]) @ vh, data, rtol=RTOL[dtype] * 10, atol=RTOL[dtype] * 10
    )


def test_svd_one_dimensional_input_raises():
    with pytest.raises(numpy.linalg.LinAlgError):
        ap.linalg.svd(_arr(numpy.arange(4.0, dtype=numpy.float32), numpy.float32))


def test_svd_rejects_integer_input_with_clear_error():
    # aclnnSvd only accepts float/double; fail loudly instead of surfacing an ACL error.
    with pytest.raises(TypeError):
        ap.linalg.svd(_arr(numpy.arange(4, dtype=numpy.int32).reshape(2, 2), numpy.int32))


# ==========================================================================
# 2. pinv
# ==========================================================================


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(3, 3), (4, 3), (3, 4)])
def test_pinv_matches_numpy(dtype, shape):
    data = _random(shape, dtype, seed=5)
    got = ap.linalg.pinv(_arr(data, dtype)).to_numpy()
    expected = numpy.linalg.pinv(data)
    numpy.testing.assert_allclose(got, expected, rtol=RTOL[dtype] * 100, atol=RTOL[dtype] * 100)


@pytest.mark.parametrize("dtype", DTYPES)
def test_pinv_satisfies_moore_penrose(dtype):
    data = _random((5, 3), dtype, seed=9)
    pinv = ap.linalg.pinv(_arr(data, dtype)).to_numpy()
    numpy.testing.assert_allclose(
        data @ pinv @ data, data, rtol=RTOL[dtype] * 100, atol=RTOL[dtype] * 100
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_pinv_rcond_zero_keeps_small_singular_values(dtype):
    data = _random((4, 4), dtype, seed=13)
    got = ap.linalg.pinv(_arr(data, dtype), rcond=0.0).to_numpy()
    expected = numpy.linalg.pinv(data, rcond=0.0)
    numpy.testing.assert_allclose(got, expected, rtol=RTOL[dtype] * 100, atol=RTOL[dtype] * 100)


def test_pinv_one_dimensional_input_raises():
    with pytest.raises(numpy.linalg.LinAlgError):
        ap.linalg.pinv(_arr(numpy.arange(4.0, dtype=numpy.float32), numpy.float32))


def test_pinv_batched_matches_numpy():
    data = _random((2, 4, 3), numpy.float32, seed=17)
    got = ap.linalg.pinv(_arr(data, numpy.float32)).to_numpy()
    assert got.shape == (2, 3, 4)
    numpy.testing.assert_allclose(got, numpy.linalg.pinv(data), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_pinv_rank_deficient_matches_numpy(dtype):
    # Second row is 2x the first: rank 1, so the cutoff branch of pinv is exercised.
    data = numpy.array([[1.0, 2.0], [2.0, 4.0]], dtype=dtype)
    got = ap.linalg.pinv(_arr(data, dtype)).to_numpy()
    numpy.testing.assert_allclose(
        got, numpy.linalg.pinv(data), rtol=RTOL[dtype] * 100, atol=RTOL[dtype] * 100
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_hermitian_keyword_is_accepted(dtype):
    # aclnnSvd has no Hermitian fast path; the general decomposition must still be correct.
    data = numpy.array([[2.0, 1.0], [1.0, 3.0]], dtype=dtype)
    u, s, vh = _as_tuple(ap.linalg.svd(_arr(data, dtype), hermitian=True))
    sigma = numpy.zeros((u.shape[-1], vh.shape[-2]), dtype=dtype)
    numpy.fill_diagonal(sigma, s)
    numpy.testing.assert_allclose(
        u @ sigma @ vh, data, rtol=RTOL[dtype] * 10, atol=RTOL[dtype] * 10
    )
    numpy.testing.assert_allclose(
        ap.linalg.pinv(_arr(data, dtype), hermitian=True).to_numpy(),
        numpy.linalg.pinv(data),
        rtol=RTOL[dtype] * 100,
        atol=RTOL[dtype] * 100,
    )
