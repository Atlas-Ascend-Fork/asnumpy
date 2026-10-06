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

"""Discrete Fourier transforms on the NPU.

Scope of this first version:

* only one-dimensional transforms of ``complex64`` arrays are supported;
* ``n`` is either ``None`` or the input length (no padding/truncation);
* ``axis`` is ``-1`` or ``0``, which are the same axis for a one-dimensional array;
* ``norm`` is ``None``/``"backward"``: the forward transform is unscaled and the
  inverse transform is scaled by ``1/n``, exactly like NumPy.

Anything outside that scope raises ``NotImplementedError`` rather than silently
falling back to a host/NumPy implementation or ignoring an argument.
"""

__all__ = ["fft", "ifft"]

import operator

import numpy as np

from .._core import fft as _fft
from .._types import ArrayLike
from ..utils import ndarray

_SUPPORTED_NORMS = (None, "backward")


def _as_complex64_vector(a: ArrayLike, name: str) -> ndarray:
    """Return ``a`` after checking that the NPU backend can handle it."""
    if not isinstance(a, ndarray):
        raise TypeError(
            f"asnumpy.fft.{name} only accepts an asnumpy.ndarray on the NPU, got "
            f"{type(a).__name__}; convert host data with asnumpy.ndarray.from_numpy(...) first"
        )
    if a.ndim != 1:
        raise NotImplementedError(
            f"asnumpy.fft.{name} currently supports one-dimensional transforms only, got {a.ndim} dimensions"
        )
    if a.dtype != np.complex64:
        raise TypeError(f"asnumpy.fft.{name} only supports complex64 input, got {a.dtype}")
    return a


def _check_args(a: ArrayLike, n, axis, norm, name: str) -> None:
    if n is not None:
        length = operator.index(n)
        if length != a.shape[0]:
            raise NotImplementedError(
                f"asnumpy.fft.{name} does not support padding or truncation yet: n={length} "
                f"but the input length is {a.shape[0]}"
            )
    if isinstance(axis, bool):
        raise TypeError("an integer is required")
    index = operator.index(axis)
    if index not in (0, -1):
        raise np.exceptions.AxisError(axis, ndim=1)
    if norm not in _SUPPORTED_NORMS:
        raise NotImplementedError(
            f"asnumpy.fft.{name} only supports norm=None/'backward', got norm={norm!r}"
        )


def fft(a: ArrayLike, n=None, axis: int = -1, norm=None) -> ndarray:
    """Compute the one-dimensional discrete Fourier transform of an NPU array.

    This is the NPU-backed counterpart of :func:`numpy.fft.fft` for
    one-dimensional ``complex64`` input; the result is unscaled.
    """
    a = _as_complex64_vector(a, "fft")
    _check_args(a, n, axis, norm, "fft")
    return ndarray(_fft.fft(a))


def ifft(a: ArrayLike, n=None, axis: int = -1, norm=None) -> ndarray:
    """Compute the one-dimensional inverse discrete Fourier transform of an NPU array.

    This is the NPU-backed counterpart of :func:`numpy.fft.ifft` for
    one-dimensional ``complex64`` input; the result is scaled by ``1/n``.
    """
    a = _as_complex64_vector(a, "ifft")
    _check_args(a, n, axis, norm, "ifft")
    return ndarray(_fft.ifft(a))
