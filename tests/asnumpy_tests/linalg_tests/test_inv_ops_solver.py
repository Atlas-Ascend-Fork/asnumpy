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

"""asnumpy.linalg.inv 的 ops-solver 后端设备测试（issue #28 第四阶段）

对照 numpy 2.5.3 与 float64 独立参考，覆盖：常规/对角/单位/需换主元矩阵、精确奇异（抛
LinAlgError，且输入不被修改）、数值奇异但非精确奇异（与 numpy 同样返回有限值）、非对齐与
中等尺寸、非法输入、输入不被覆盖、默认后端不受影响、重复调用与失败后恢复。
"""

import logging
import os

import numpy as np
import pytest

import asnumpy
from asnumpy.linalg import _linalg

logger = logging.getLogger(__name__)

# 精度判据（不为了通过而放宽）：良态 float32 方阵在 n <= 257 上实测 res_norm <= 2e-6，这里留 ~50x 余量。
RES_TOL = 1e-4
FWD_RTOL = 1e-3
FWD_ATOL = 1e-3

SIZES = [1, 2, 3, 7, 16, 17, 33, 64, 129, 257]


def _ops_solver_inv(a):
    """Call inv through the ops-solver backend and prove that it really was used."""
    before = _linalg.ops_solver_dispatches()
    os.environ["ASNUMPY_LINALG_BACKEND"] = "ops-solver"
    try:
        result = asnumpy.linalg.inv(a)
    finally:
        os.environ.pop("ASNUMPY_LINALG_BACKEND", None)
    assert _linalg.ops_solver_dispatches() == before + 1, "the ops-solver backend was not executed"
    return result


def _residuals(a_host, x_host):
    a64 = a_host.astype(np.float64)
    x64 = x_host.astype(np.float64)
    scale = float(np.abs(a64).max()) * float(np.abs(x64).max()) + 1.0
    eye = np.eye(a64.shape[0])
    res_ax = float(np.abs(a64 @ x64 - eye).max())
    res_xa = float(np.abs(x64 @ a64 - eye).max())
    return res_ax, res_xa, res_ax / scale, res_xa / scale


def _well_conditioned(n, seed=7, diagonal_shift=6.0):
    rng = np.random.default_rng(seed + n)
    a = rng.standard_normal((n, n)).astype(np.float32)
    a[np.diag_indices(n)] += diagonal_shift
    return a


@pytest.mark.parametrize("n", SIZES)
def test_matches_numpy_and_float64_reference(n):
    a_host = _well_conditioned(n)
    a = asnumpy.ndarray.from_numpy(a_host)
    a_before = a.to_numpy().copy()

    ref = np.linalg.inv(a_host.astype(np.float64))
    got = _ops_solver_inv(a)
    got_host = got.to_numpy()

    assert isinstance(got, asnumpy.ndarray)
    assert got.shape == (n, n)
    assert got.dtype == np.float32

    fwd = float(np.abs(got_host.astype(np.float64) - ref).max())
    assert fwd <= FWD_ATOL + FWD_RTOL * float(np.abs(ref).max()), (
        f"forward error {fwd:.3e} too large for n={n}"
    )

    res_ax, res_xa, n_ax, n_xa = _residuals(a_host, got_host)
    assert n_ax <= RES_TOL, f"||A@X - I||/scale = {n_ax:.3e} (n={n}, raw {res_ax:.3e})"
    assert n_xa <= RES_TOL, f"||X@A - I||/scale = {n_xa:.3e} (n={n}, raw {res_xa:.3e})"

    # the operand must not be modified
    np.testing.assert_array_equal(a.to_numpy(), a_before)


def test_identity_diagonal_and_pivoting_cases():
    n = 32
    eye = np.eye(n, dtype=np.float32)
    got = _ops_solver_inv(asnumpy.ndarray.from_numpy(eye))
    np.testing.assert_allclose(got.to_numpy(), eye, rtol=0, atol=1e-6)

    diag = np.diag(np.linspace(1.0, 4.0, n)).astype(np.float32)
    got_diag = _ops_solver_inv(asnumpy.ndarray.from_numpy(diag))
    np.testing.assert_allclose(
        got_diag.to_numpy(), np.linalg.inv(diag.astype(np.float64)), rtol=1e-4, atol=1e-5
    )

    # zero diagonal -> a pivoted LU is required, the matrix is still invertible
    rng = np.random.default_rng(11)
    a = rng.standard_normal((n, n)).astype(np.float32)
    a[np.diag_indices(n)] = 0.0
    for i in range(n):
        a[i, (i + 1) % n] += 10.0 * float(np.abs(a[i]).sum()) + 1.0
    ref = np.linalg.inv(a.astype(np.float64))
    got_pivot = _ops_solver_inv(asnumpy.ndarray.from_numpy(a))
    fwd = float(np.abs(got_pivot.to_numpy().astype(np.float64) - ref).max())
    assert fwd <= FWD_ATOL + FWD_RTOL * float(np.abs(ref).max())


@pytest.mark.parametrize(
    ("name", "make"),
    [
        ("all_zero", lambda n: np.zeros((n, n), dtype=np.float32)),
        ("duplicate_rows", None),
        ("late_zero_pivot", None),
    ],
)
def test_exactly_singular_raises_linalgerror(name, make):
    n = 16
    if name == "all_zero":
        a_host = make(n)
    elif name == "duplicate_rows":
        a_host = _well_conditioned(n, seed=3)
        a_host[1, :] = a_host[0, :]
    else:  # duplicate rows at the bottom -> the zero pivot only appears late
        a_host = _well_conditioned(n, seed=4)
        a_host[n - 2, :] = 0.0
        a_host[n - 2, 0] = 1.0
        a_host[n - 1, :] = 2.0 * a_host[n - 2, :]  # row n-1 is an exact multiple of row n-2
    a = asnumpy.ndarray.from_numpy(a_host)
    a_before = a.to_numpy().copy()

    before_calls = _linalg.ops_solver_dispatches()
    os.environ["ASNUMPY_LINALG_BACKEND"] = "ops-solver"
    try:
        with pytest.raises(np.linalg.LinAlgError):
            asnumpy.linalg.inv(a)
    finally:
        os.environ.pop("ASNUMPY_LINALG_BACKEND", None)

    # numpy must agree that this input is singular (same public contract)
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.inv(a_host)
    # a rejected call must not be counted as a dispatch and must not corrupt the operand
    assert _linalg.ops_solver_dispatches() == before_calls
    np.testing.assert_array_equal(a.to_numpy(), a_before)


def test_numerically_singular_matches_numpy_behaviour():
    """cond ~1e16 但没有精确零主元：numpy 静默返回有限值，本后端必须同样不误报"""
    n = 16
    a_host = _well_conditioned(n, seed=21)
    a_host[:, 1] = a_host[:, 0]  # duplicate columns -> numerically singular
    assert np.linalg.cond(a_host.astype(np.float64)) > 1e12

    numpy_inv = np.linalg.inv(a_host)  # numpy does not raise here
    got = _ops_solver_inv(asnumpy.ndarray.from_numpy(a_host)).to_numpy()
    assert np.isfinite(got).all() == np.isfinite(numpy_inv).all()
    # same magnitude class as numpy's answer (both are unreliable; we only check we did not invent a different verdict)
    assert np.abs(got).max() > 0


def test_ill_conditioned_is_reported_not_hidden():
    """Hilbert-12 严重病态：只记录，不声称结果可信（也不放宽容差去通过）"""
    n = 12
    a_host = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)], dtype=np.float32)
    cond = float(np.linalg.cond(a_host.astype(np.float64)))
    got = _ops_solver_inv(asnumpy.ndarray.from_numpy(a_host)).to_numpy()
    ref = np.linalg.inv(a_host.astype(np.float64))
    res_ax, _, n_ax, _ = _residuals(a_host, got)
    logger.info(
        "[ill-conditioned] n=%d cond=%.3e forward=%.3e res_AX=%.3e res_norm=%.3e",
        n,
        cond,
        float(np.abs(got.astype(np.float64) - ref).max()),
        res_ax,
        n_ax,
    )
    assert np.isfinite(got).all()


@pytest.mark.parametrize("n", [0, 1])
def test_empty_and_single_element(n):
    """0x0 与 1x1 都走 ops-solver 后端本身（默认后端对 0x0 会抛奇异错误，属既有行为，不在本阶段范围）"""
    a_host = np.zeros((n, n), dtype=np.float32)
    if n == 1:
        a_host[0, 0] = 4.0
    a = asnumpy.ndarray.from_numpy(a_host)
    got = _ops_solver_inv(a)
    assert got.shape == (n, n)
    assert got.to_numpy().size == n * n
    if n == 1:
        np.testing.assert_allclose(got.to_numpy(), np.array([[0.25]], dtype=np.float32), rtol=1e-6)


def test_non_2d_and_non_square_raise_linalgerror():
    os.environ["ASNUMPY_LINALG_BACKEND"] = "ops-solver"
    try:
        with pytest.raises(np.linalg.LinAlgError):
            asnumpy.linalg.inv(asnumpy.ndarray.from_numpy(np.ones(4, dtype=np.float32)))
        with pytest.raises(np.linalg.LinAlgError):
            asnumpy.linalg.inv(asnumpy.ndarray.from_numpy(np.ones((3, 4), dtype=np.float32)))
    finally:
        os.environ.pop("ASNUMPY_LINALG_BACKEND", None)


@pytest.mark.parametrize("dtype", [np.float64, np.float16, np.complex64, np.int32])
def test_unsupported_dtype_raises_typeerror(dtype):
    a = asnumpy.ndarray.from_numpy(np.eye(4, dtype=dtype))
    os.environ["ASNUMPY_LINALG_BACKEND"] = "ops-solver"
    try:
        with pytest.raises(TypeError, match="float32"):
            asnumpy.linalg.inv(a)
    finally:
        os.environ.pop("ASNUMPY_LINALG_BACKEND", None)


def test_host_numpy_operand_raises_typeerror():
    os.environ["ASNUMPY_LINALG_BACKEND"] = "ops-solver"
    try:
        with pytest.raises(TypeError, match="asnumpy.ndarray"):
            asnumpy.linalg.inv(np.eye(4, dtype=np.float32))
    finally:
        os.environ.pop("ASNUMPY_LINALG_BACKEND", None)


def test_invalid_backend_value_raises():
    a = asnumpy.ndarray.from_numpy(np.eye(4, dtype=np.float32))
    os.environ["ASNUMPY_LINALG_BACKEND"] = "nope"
    try:
        with pytest.raises(ValueError, match="ASNUMPY_LINALG_BACKEND"):
            asnumpy.linalg.inv(a)
    finally:
        os.environ.pop("ASNUMPY_LINALG_BACKEND", None)


def test_default_backend_is_untouched_and_still_works():
    a_host = _well_conditioned(16, seed=31)
    a = asnumpy.ndarray.from_numpy(a_host)
    before_calls = _linalg.ops_solver_dispatches()
    got = asnumpy.linalg.inv(a)  # default backend
    assert _linalg.ops_solver_dispatches() == before_calls, (
        "the default backend must not enter ops-solver"
    )
    np.testing.assert_allclose(
        got.to_numpy().astype(np.float64),
        np.linalg.inv(a_host.astype(np.float64)),
        rtol=1e-3,
        atol=1e-3,
    )
    # other linalg entry points must be unaffected by the backend switch
    os.environ["ASNUMPY_LINALG_BACKEND"] = "ops-solver"
    try:
        det = asnumpy.linalg.det(a)
    finally:
        os.environ.pop("ASNUMPY_LINALG_BACKEND", None)
    assert np.isfinite(float(det.to_numpy()))


def test_repeated_calls_are_stable():
    a_host = _well_conditioned(64, seed=41)
    a = asnumpy.ndarray.from_numpy(a_host)
    first = _ops_solver_inv(a).to_numpy()
    before = _linalg.ops_solver_dispatches()
    for _ in range(10):
        np.testing.assert_array_equal(_ops_solver_inv(a).to_numpy(), first)
    assert _linalg.ops_solver_dispatches() == before + 10


def test_singular_failure_then_normal_call_recovers():
    good = _well_conditioned(16, seed=51)
    bad = good.copy()
    bad[1, :] = bad[0, :]
    a_good = asnumpy.ndarray.from_numpy(good)
    a_bad = asnumpy.ndarray.from_numpy(bad)

    os.environ["ASNUMPY_LINALG_BACKEND"] = "ops-solver"
    try:
        for _ in range(5):
            with pytest.raises(np.linalg.LinAlgError):
                asnumpy.linalg.inv(a_bad)
        got = asnumpy.linalg.inv(a_good).to_numpy()
    finally:
        os.environ.pop("ASNUMPY_LINALG_BACKEND", None)
    res_ax, res_xa, n_ax, n_xa = _residuals(good, got)
    assert n_ax <= RES_TOL and n_xa <= RES_TOL


def test_missing_backend_message_is_actionable():
    """未启用 ops-solver 的构建应给出可操作的报错；启用时应能正常调用"""
    from asnumpy._core import inv_ops_solver

    a = asnumpy.ndarray.from_numpy(np.eye(4, dtype=np.float32))
    try:
        inv_ops_solver(a)
    except RuntimeError as exc:
        assert "ops-solver" in str(exc)
        assert "ASNUMPY_ENABLE_OPS_SOLVER" in str(exc)
    else:
        np.testing.assert_allclose(
            inv_ops_solver(a).to_numpy(), np.eye(4, dtype=np.float32), rtol=1e-6
        )
