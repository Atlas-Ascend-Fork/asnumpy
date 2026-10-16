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

from ._core.logic import (
    all as _all,
)
from ._core.logic import (
    any as _any,
)
from ._core.logic import (
    equal as _equal,
)
from ._core.logic import (
    greater as _greater,
)
from ._core.logic import (
    greater_equal as _greater_equal,
)
from ._core.logic import (
    isfinite as _isfinite,
)
from ._core.logic import (
    isinf as _isinf,
)
from ._core.logic import (
    isneginf as _isneginf,
)
from ._core.logic import (
    isposinf as _isposinf,
)
from ._core.logic import (
    less as _less,
)
from ._core.logic import (
    less_equal as _less_equal,
)
from ._core.logic import (
    logical_and as _logical_and,
)
from ._core.logic import (
    logical_not as _logical_not,
)
from ._core.logic import (
    logical_or as _logical_or,
)
from ._core.logic import (
    logical_xor as _logical_xor,
)
from ._core.logic import (
    not_equal as _not_equal,
)
from ._types import ArrayLike, AxisLike, DTypeLike
from ._ufunc import create_ufunc as _create_ufunc
from .utils import _convert_dtype, _ensure_ndarray, ndarray


def _normalize_axis_arg(axis: AxisLike) -> list[int] | None:
    """Wrap a bare int axis into a list; the C++ layer owns axis semantics."""
    if axis is None:
        return None
    if isinstance(axis, int):
        return [axis]
    return list(axis)


def all(x: ArrayLike, axis: AxisLike = None, keepdims: bool = False) -> ndarray:
    x = _ensure_ndarray(x)
    axes = _normalize_axis_arg(axis)
    if axes is None:
        return ndarray(_all(x))
    return ndarray(_all(x, axes, keepdims))


def any(x: ArrayLike, axis: AxisLike = None, keepdims: bool = False) -> ndarray:
    x = _ensure_ndarray(x)
    axes = _normalize_axis_arg(axis)
    if axes is None:
        return ndarray(_any(x))
    return ndarray(_any(x, axes, keepdims))


# ---------- Unary is-checks ----------


def _is_check_fallback(name, func, x, dtype=None):
    if dtype is not None:
        raise TypeError(f"{name}() does not support the 'dtype' parameter")
    return ndarray(func(x))


# Prototype slice: unary is-check with bool output + fallback (representative #5)
isfinite = _create_ufunc(
    "isfinite",
    (("f->?", _isfinite, False), ("e->?", _isfinite, False), ("d->?", _isfinite, False)),
    fallback=lambda x, dtype=None: _is_check_fallback("isfinite", _isfinite, x, dtype),
    doc="Test element-wise for finiteness (not infinity and not Not a Number).",
)


def isinf(x: ArrayLike) -> ndarray:
    return ndarray(_isinf(x))


def isneginf(x: ArrayLike) -> ndarray:
    return ndarray(_isneginf(x))


def isposinf(x: ArrayLike) -> ndarray:
    return ndarray(_isposinf(x))


def logical_and(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    return ndarray(_logical_and(x1, x2))


def logical_or(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    return ndarray(_logical_or(x1, x2))


def logical_not(x: ArrayLike) -> ndarray:
    return ndarray(_logical_not(x))


def logical_xor(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    return ndarray(_logical_xor(x1, x2))


def greater(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_greater(x1, x2, _convert_dtype(dtype)))


def greater_equal(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_greater_equal(x1, x2, _convert_dtype(dtype)))


def less(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_less(x1, x2, _convert_dtype(dtype)))


def less_equal(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_less_equal(x1, x2, _convert_dtype(dtype)))


def _equal_fallback(x1, x2, dtype=None):
    """Fallback for equal with dtypes not in loop table (e.g. int64, float16)."""
    return ndarray(_equal(x1, x2, _convert_dtype(dtype)))


# Prototype slice: binary comparison with bool output + float+int loops (representative #4)
equal = _create_ufunc(
    "equal",
    (("ff->?", _equal), ("dd->?", _equal), ("ii->?", _equal)),
    doc="Returns (x1 == x2) element-wise.",
    fallback=_equal_fallback,
)


def not_equal(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_not_equal(x1, x2, _convert_dtype(dtype)))
