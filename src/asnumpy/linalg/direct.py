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
    dot as _dot,
)
from .._core import (
    einsum as _einsum,
)
from .._core import (
    matmul_ops_blas as _matmul_ops_blas,
)
from .._core import (
    vdot as _vdot,
)
from .._types import ArrayLike
from ..utils import ndarray

# Explicit, documented backend selection for asnumpy.matmul. The default keeps the original
# behaviour; see docs/matmul_backend.md.
_MATMUL_BACKENDS = ("original", "ops-blas")
# Number of successful dispatches into the ops-blas backend. Tests assert on this so that a passing
# result cannot silently come from the original path.
_OPS_BLAS_CALLS = 0


def ops_blas_dispatches() -> int:
    """Return how many calls have been dispatched into the ops-blas backend since import.

    Read-only view of the internal counter, exposed so tests (and users diagnosing which backend ran)
    do not have to touch a private attribute.
    """
    return _OPS_BLAS_CALLS


def _matmul_backend() -> str:
    """Return the backend ``matmul`` will use, read from ``ASNUMPY_MATMUL_BACKEND`` per call."""
    backend = os.environ.get("ASNUMPY_MATMUL_BACKEND", "original").strip()
    if backend not in _MATMUL_BACKENDS:
        raise ValueError(
            f"ASNUMPY_MATMUL_BACKEND={backend!r} is invalid; expected one of {list(_MATMUL_BACKENDS)}"
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


def _requires_fp64_fallback(*arrays: ArrayLike) -> bool:
    return any(np.asarray(_as_host_array(arr)).dtype == np.float64 for arr in arrays)


def dot(a: ArrayLike, b: ArrayLike) -> ndarray:
    if _requires_fp64_fallback(a, b):
        return _to_asnumpy_array(np.dot(_as_host_array(a), _as_host_array(b)))
    return ndarray(_dot(a, b))


def inner(a: ArrayLike, b: ArrayLike) -> ndarray:
    na = _as_host_array(a)
    nb = _as_host_array(b)
    return ndarray.from_numpy(np.asarray(np.inner(na, nb)))


def outer(a: ArrayLike, b: ArrayLike) -> ndarray:
    na = _as_host_array(a)
    nb = _as_host_array(b)
    return ndarray.from_numpy(np.asarray(np.outer(na, nb)))


def vdot(a: ArrayLike, b: ArrayLike) -> ndarray:
    if _requires_fp64_fallback(a, b):
        return _to_asnumpy_array(np.vdot(_as_host_array(a), _as_host_array(b)))
    return ndarray(_vdot(a, b))


def _matmul_via_ops_blas(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    """Dispatch matmul to the ops-blas backend, or fail loudly when the input is out of its scope.

    The scope is deliberately narrow (2-D float32 NPU arrays); anything else is reported instead of
    silently falling back, so a test can never mistake the original path for the ops-blas one.
    """
    global _OPS_BLAS_CALLS

    for name, value in (("x1", x1), ("x2", x2)):
        if not isinstance(value, ndarray):
            raise TypeError(
                f"asnumpy.matmul with the ops-blas backend needs asnumpy.ndarray operands, got "
                f"{type(value).__name__} for {name}; unset ASNUMPY_MATMUL_BACKEND to use the default backend"
            )
        if value.ndim != 2:
            raise NotImplementedError(
                f"the ops-blas matmul backend supports 2-D operands only, got {value.ndim}-D for {name}; "
                "unset ASNUMPY_MATMUL_BACKEND to use the default backend"
            )
        if value.dtype != np.float32:
            raise TypeError(
                f"the ops-blas matmul backend supports float32 only, got {value.dtype} for {name}; "
                "unset ASNUMPY_MATMUL_BACKEND to use the default backend"
            )
    if x1.shape[1] != x2.shape[0]:
        raise ValueError(f"matmul: shapes {tuple(x1.shape)} and {tuple(x2.shape)} are not aligned")

    _OPS_BLAS_CALLS += 1
    return ndarray(_matmul_ops_blas(x1, x2))


def matmul(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    if _matmul_backend() == "ops-blas":
        return _matmul_via_ops_blas(x1, x2)
    return _to_asnumpy_array(np.matmul(_as_host_array(x1), _as_host_array(x2)))


def einsum(subscripts: str, *operands: ArrayLike) -> ndarray:
    return ndarray(_einsum(subscripts, *operands))


_direct_all_ = [
    "dot",
    "einsum",
    "inner",
    "matmul",
    "outer",
    "vdot",
]
