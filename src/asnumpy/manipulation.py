# *****************************************************************************
# Copyright (c) 2025 AISS and ISE Group at Harbin Institute of Technology. All Rights Reserved.
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

"""Array manipulation routines (issue #25).

Adds the NumPy-compatible array-assembly / shape-adjustment entry points that were missing from the
top-level namespace:

- :func:`concatenate`, :func:`stack`      -- join a sequence of arrays (aclnnCat / aclnnStack)
- :func:`where`                           -- element-wise selection (aclnnSWhere)
- :func:`expand_dims`, :func:`squeeze`    -- add / remove length-one axes

Signatures, keyword names and error types mirror NumPy, so ``import asnumpy as np`` stays a
drop-in replacement for the supported subset.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from loguru import logger

from ._core.array import (
    concatenate as _concatenate,
)
from ._core.array import (
    stack as _stack,
)
from ._core.array import (
    where as _where,
)
from ._types import ArrayLike, AxisLike, AxisOptional
from .utils import ndarray

__all__ = [
    "concatenate",
    "expand_dims",
    "squeeze",
    "stack",
    "where",
]


def _as_npu(value: ArrayLike, like: ndarray | None = None) -> ndarray:
    """Return *value* as an asnumpy array, uploading NumPy/scalar inputs to the device.

    A Python scalar is materialized with ``like``'s dtype when one is supplied, which reproduces
    NumPy's weak-scalar rule (``np.where(bool_array, float32_array, 0)`` stays float32 rather than
    jumping to float64).
    """
    if isinstance(value, ndarray):
        return value
    if like is not None and isinstance(value, (bool, int, float, complex)):
        return ndarray.from_numpy(np.asarray(value, dtype=like.dtype))
    return ndarray.from_numpy(np.asarray(value))


def _as_host(value: ArrayLike) -> np.ndarray:
    """Return *value* as a host ``numpy.ndarray`` without copying when already host-resident."""
    if isinstance(value, ndarray):
        return value.to_numpy()
    return np.asarray(value)


def _as_sequence(arrays: Sequence[ArrayLike]) -> list[ArrayLike]:
    """Normalize the ``arrays`` argument of concatenate/stack.

    NumPy accepts any sequence (list, tuple). An empty sequence is rejected here with NumPy's
    message so the failure is reported before any device work starts.
    """
    if isinstance(arrays, (str, bytes)) or not isinstance(arrays, Sequence):
        raise TypeError("arrays must be a sequence of arrays")
    items = list(arrays)
    if not items:
        raise ValueError("need at least one array to concatenate")
    return items


@logger.catch(reraise=True)
def concatenate(arrays: Sequence[ArrayLike], axis: AxisOptional = 0) -> ndarray:
    """Join a sequence of arrays along an existing axis.

    Mirrors :func:`numpy.concatenate`: all dimensions except *axis* must match, ``axis=None``
    flattens every input first, and the result dtype follows NumPy's promotion rules.
    """
    items = _as_sequence(arrays)
    logger.debug(f"concatenate: {len(items)} arrays, axis={axis}")
    if axis is None:
        # NumPy flattens every input before joining. Flattening is a pure host-side shape rewrite
        # and CANN exposes no aclnn reshape entry point, so that (rare) form is computed on the
        # host; the common axis form stays entirely on the NPU.
        return ndarray.from_numpy(np.concatenate([_as_host(a) for a in items], axis=None))
    operands = [_as_npu(a) for a in items]
    return ndarray(_concatenate(operands, axis))


@logger.catch(reraise=True)
def stack(arrays: Sequence[ArrayLike], axis: AxisOptional = 0) -> ndarray:
    """Join a sequence of arrays along a new axis.

    Mirrors :func:`numpy.stack`: every input must have the same shape, and *axis* positions the new
    dimension in the result.
    """
    items = _as_sequence(arrays)
    logger.debug(f"stack: {len(items)} arrays, axis={axis}")
    operands = [_as_npu(a) for a in items]
    return ndarray(_stack(operands, axis))


@logger.catch(reraise=True)
def where(condition: ArrayLike, x: ArrayLike | None = None, y: ArrayLike | None = None):
    """Return elements chosen from *x* or *y* depending on *condition*.

    With all three arguments this is ``numpy.where(condition, x, y)`` and the three operands are
    broadcast together. With only *condition* it degrades to the one-argument form and returns the
    indices of the non-zero entries as a host ``numpy.ndarray`` tuple, exactly like NumPy.
    """
    if x is None and y is None:
        logger.debug("where: one-argument form (nonzero indices)")
        return np.where(
            condition.to_numpy() if isinstance(condition, ndarray) else np.asarray(condition)
        )
    if x is None or y is None:
        raise ValueError("either both or neither of x and y should be given")

    logger.debug("where: three-argument form")
    cond = _as_npu(condition)
    if cond.dtype != np.dtype(bool):
        cond = cond.astype(np.bool_)
    # Coerce a scalar operand against the other operand so both land on one device dtype.
    left = _as_npu(x, like=y if isinstance(y, ndarray) else None)
    right = _as_npu(y, like=x if isinstance(x, ndarray) else None)
    return ndarray(_where(cond, left, right))


@logger.catch(reraise=True)
def expand_dims(a: ArrayLike, axis: AxisLike) -> ndarray:
    """Expand the shape of an array by inserting a new length-one axis at *axis*.

    ``numpy.expand_dims(a, axis)`` is exactly ``numpy.stack((a,), axis=axis)``; reusing the device
    stack keeps the result on the NPU without a host round trip.
    """
    logger.debug(f"expand_dims: axis={axis}")
    return stack((a,), axis=axis)


@logger.catch(reraise=True)
def squeeze(a: ArrayLike, axis: AxisOptional = None) -> ndarray:
    """Remove length-one axes from an array.

    ``axis=None`` removes every length-one axis; an explicit *axis* (or sequence of axes) is
    validated against the input shape and raises :class:`ValueError` when it is not length one.

    The shape-only rewrite is performed on the host because CANN exposes no aclnn reshape entry
    point in the supported operator set; the values themselves are unchanged.
    """
    logger.debug(f"squeeze: axis={axis}")
    host = a.to_numpy() if isinstance(a, ndarray) else np.asarray(a)
    return ndarray.from_numpy(np.squeeze(host, axis=axis))
