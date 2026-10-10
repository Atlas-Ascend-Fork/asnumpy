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


import os

import numpy as np

from .._core import (
    OpsSolverSingularMatrix as _OpsSolverSingularMatrix,
)
from .._core import (
    inv_ops_solver as _inv_ops_solver,
)
from .._core.linalg import (
    det as _det,
)
from .._core.linalg import (
    inv as _inv,
)
from .._core.linalg import (
    norm as _norm,
)
from .._core.linalg import (
    qr as _qr,
)
from .._core.linalg import (
    slogdet as _slogdet,
)
from .._types import ArrayLike, AxisLike
from ..utils import ndarray

# Explicit, documented backend selection for asnumpy.linalg.inv only; every other linalg entry point
# is unaffected. The default keeps the original aclnnInverse implementation.
_LINALG_BACKENDS = ("original", "ops-solver")
# Number of successful dispatches into the ops-solver backend. Tests assert on this so a passing
# result cannot silently come from the original backend.
_OPS_SOLVER_CALLS = 0


def _linalg_backend() -> str:
    """Return the backend ``inv`` will use, read from ``ASNUMPY_LINALG_BACKEND`` per call."""
    backend = os.environ.get("ASNUMPY_LINALG_BACKEND", "original").strip()
    if backend not in _LINALG_BACKENDS:
        raise ValueError(
            f"ASNUMPY_LINALG_BACKEND={backend!r} is invalid; expected one of {list(_LINALG_BACKENDS)}"
        )
    return backend


def _as_host_array(a: ArrayLike) -> np.ndarray:
    if hasattr(a, "to_numpy"):
        return a.to_numpy()
    return np.asarray(a)


def _to_asnumpy_array(value) -> ndarray:
    if isinstance(value, ndarray):
        return value
    return ndarray.from_numpy(np.asarray(value))


def _normalize_norm_axis(axis: AxisLike, ndim: int):
    if axis is None:
        return None
    if isinstance(axis, tuple):
        axes = axis
    elif isinstance(axis, list):
        axes = tuple(axis)
    else:
        axes = (axis,)

    normalized = []
    for item in axes:
        idx = int(item)
        if idx < 0:
            idx += ndim
        if idx < 0 or idx >= ndim:
            raise np.AxisError(idx, ndim=ndim)
        normalized.append(idx)

    if len(set(normalized)) != len(normalized):
        raise ValueError("Duplicate axes given.")
    return tuple(normalized)


def matrix_power(a: ArrayLike, n: int) -> ndarray:
    host = _as_host_array(a)
    if n < 0:
        sign, logdet = np.linalg.slogdet(host)
        if np.any(sign == 0):
            raise RuntimeError("Singular matrix has no inverse for negative power")
    return _to_asnumpy_array(np.linalg.matrix_power(host, n))


def qr(a: ArrayLike, mode: str = "reduced") -> ndarray | tuple:
    result = _qr(a, mode)
    if isinstance(result, tuple):
        q, r = result
        return (ndarray(q), ndarray(r))
    return ndarray(result)


def norm(
    a: ArrayLike,
    ord: str | int | float | None = None,
    axis: AxisLike = None,
    keepdims: bool = False,
) -> ndarray:
    host = _as_host_array(a)
    normalized_axis = _normalize_norm_axis(axis, host.ndim)

    try:
        result = np.linalg.norm(host, ord=ord, axis=normalized_axis, keepdims=keepdims)
    except TypeError:
        if normalized_axis is None and ord is None:
            return ndarray(_norm(a, 2.0, (), keepdims))
        raise

    return _to_asnumpy_array(result)


def det(a: ArrayLike) -> ndarray:
    return ndarray(_det(a))


def slogdet(a: ArrayLike) -> tuple:
    # CANN's double-precision slogdet may produce different results from NumPy
    # for inputs containing nan/inf. Fall back to NumPy for such cases.
    host = a.to_numpy() if hasattr(a, "to_numpy") else np.asarray(a)
    if np.issubdtype(host.dtype, np.floating) and (
        np.any(np.isnan(host)) or np.any(np.isinf(host))
    ):
        return np.linalg.slogdet(host)  # type: ignore[no-any-return]
    sign, logdet = _slogdet(a)
    return (ndarray(sign), ndarray(logdet))


def _inv_via_ops_solver(a: ArrayLike) -> ndarray:
    """Dispatch inv to the ops-solver backend, or fail loudly when the input is out of its scope.

    The scope is deliberately narrow (2-D square float32 NPU arrays). Anything else is reported
    instead of silently falling back, so a test can never mistake the original backend for this one.
    Error types follow NumPy: shape problems and exactly singular input raise LinAlgError, while a
    dtype this backend cannot serve raises TypeError (a capability restriction, not a maths error).
    """
    global _OPS_SOLVER_CALLS

    if not isinstance(a, ndarray):
        raise TypeError(
            f"asnumpy.linalg.inv with the ops-solver backend needs an asnumpy.ndarray, got "
            f"{type(a).__name__}; unset ASNUMPY_LINALG_BACKEND to use the default backend"
        )
    if a.dtype != np.float32:
        raise TypeError(
            f"the ops-solver inv backend supports float32 only, got {a.dtype}; "
            "unset ASNUMPY_LINALG_BACKEND to use the default backend"
        )
    if a.ndim != 2:
        raise np.linalg.LinAlgError(
            f"{a.ndim}-dimensional array given. Array must be at least two-dimensional"
        )
    if a.shape[0] != a.shape[1]:
        raise np.linalg.LinAlgError("Last 2 dimensions of the array must be square")

    try:
        result = _inv_ops_solver(a)
    except _OpsSolverSingularMatrix as exc:
        raise np.linalg.LinAlgError(str(exc)) from None
    _OPS_SOLVER_CALLS += 1
    return ndarray(result)


def inv(a: ArrayLike) -> ndarray:
    if _linalg_backend() == "ops-solver":
        return _inv_via_ops_solver(a)
    return ndarray(_inv(a))
