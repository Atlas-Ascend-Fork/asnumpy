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


from typing import Literal

import numpy as np

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
from ..utils import _ensure_ndarray, ndarray


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
            raise np.exceptions.AxisError(idx, ndim=ndim)
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
    result = _qr(_ensure_ndarray(a), mode)
    if isinstance(result, tuple):
        q, r = result
        return (ndarray(q), ndarray(r))
    return ndarray(result)


def norm(
    a: ArrayLike,
    ord: Literal["fro", "nuc"] | int | float | None = None,
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
    return ndarray(_det(_ensure_ndarray(a)))


def slogdet(a: ArrayLike) -> tuple:
    # CANN's double-precision slogdet may produce different results from NumPy
    # for inputs containing nan/inf. Fall back to NumPy for such cases.
    host = a.to_numpy() if hasattr(a, "to_numpy") else np.asarray(a)
    if np.issubdtype(host.dtype, np.floating) and (
        np.any(np.isnan(host)) or np.any(np.isinf(host))
    ):
        return np.linalg.slogdet(host)  # type: ignore[no-any-return]
    sign, logdet = _slogdet(_ensure_ndarray(a))
    return (ndarray(sign), ndarray(logdet))


def inv(a: ArrayLike) -> ndarray:
    return ndarray(_inv(_ensure_ndarray(a)))
