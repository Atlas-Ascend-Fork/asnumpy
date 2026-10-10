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

"""Discrete Fourier transforms on the NPU, backed by CANN ops-fft.

Implemented so far: ``fft``/``ifft`` for complex64 arrays along one axis (every other dimension is
treated as a batch), with NumPy's ``n``/``axis``/``norm`` semantics.

The numerical transform always runs on the NPU through ops-fft. ops-fft's public ``aclfftExec*``
functions take *host* pointers and perform the device transfers internally, so each call bridges the
array through the host once; the layout work (truncation, zero padding, batching) happens on the
host while the FFT itself stays on the device. See ``docs/fft.md`` for the support matrix and the
measured backend conventions.
"""

__all__ = ["fft", "fft2", "ifft", "ifft2", "irfft", "rfft"]

import math
import operator

import numpy as np

from .._core import fft as _fft
from .._types import ArrayLike
from ..utils import ndarray

_COMPLEX64 = np.dtype(np.complex64)
_REAL32 = np.dtype(np.float32)
_NORMS = (None, "backward", "forward", "ortho")


def _require_array(a: ArrayLike, name: str) -> ndarray:
    if not isinstance(a, ndarray):
        raise TypeError(
            f"asnumpy.fft.{name} only accepts an asnumpy.ndarray on the NPU, got {type(a).__name__}; "
            "convert host data with asnumpy.ndarray.from_numpy(...) first"
        )
    return a


def _require_dtype(a: ndarray, dtype: np.dtype, name: str) -> None:
    if a.dtype != dtype:
        raise TypeError(f"asnumpy.fft.{name} only supports {dtype.name} input, got {a.dtype}")


def _normalize_axis(axis: int, ndim: int) -> int:
    """Turn a (possibly negative) axis into a non-negative index, mirroring NumPy's checks."""
    if isinstance(axis, bool):
        raise TypeError("an integer is required")
    index = operator.index(axis)
    if index < 0:
        index += ndim
    if index < 0 or index >= ndim:
        raise np.exceptions.AxisError(axis, ndim=ndim)
    return index


def _resolve_length(n, default: int) -> int:
    """Resolve ``n`` to a positive transform length, rejecting the values NumPy rejects."""
    if n is None:
        length = default
    else:
        if isinstance(n, bool):
            raise TypeError("an integer is required")
        length = operator.index(n)
    if length <= 0:
        raise ValueError(f"Invalid number of FFT data points ({length}) specified.")
    return length


def _norm_factor(inverse: bool, norm, n: int) -> float:
    """Factor applied exactly once to the ops-fft result.

    Measured convention of ops-fft (commit 294cbc5, Ascend 910B, CANN 9.1.0): every direction
    returns the *unnormalised* transform, i.e. ``ACLFFT_BACKWARD == n * numpy.ifft`` and the C2R
    inverse is ``n * numpy.irfft``, while the forward C2C and R2C results already match NumPy's
    unscaled forward transforms. NumPy's normalisation is therefore applied here, once, and never
    combined with a second backend-side factor.
    """
    if norm not in _NORMS:
        raise ValueError(f'Invalid norm value {norm}; should be "backward", "ortho" or "forward".')
    if norm in (None, "backward"):
        return 1.0 / n if inverse else 1.0
    if norm == "forward":
        return 1.0 if inverse else 1.0 / n
    return 1.0 / math.sqrt(n)


def fft(a: ArrayLike, n=None, axis: int = -1, norm=None) -> ndarray:
    """Compute the discrete Fourier transform of an NPU array along one axis.

    Counterpart of :func:`numpy.fft.fft` for complex64 input: the transform runs along ``axis``
    (default: the last one) and every other dimension is a batch. ``n`` shorter than the axis
    truncates, longer zero pads; ``norm`` accepts ``None``/``"backward"``, ``"forward"`` and
    ``"ortho"`` with NumPy's scaling.
    """
    a = _require_array(a, "fft")
    _require_dtype(a, _COMPLEX64, "fft")
    ax = _normalize_axis(axis, a.ndim)
    length = _resolve_length(n, a.shape[ax])
    return ndarray(_fft.fft_c2c(a, ax, length, False, _norm_factor(False, norm, length)))


def ifft(a: ArrayLike, n=None, axis: int = -1, norm=None) -> ndarray:
    """Compute the inverse discrete Fourier transform of an NPU array along one axis.

    Counterpart of :func:`numpy.fft.ifft` for complex64 input, with the same ``n``/``axis``/``norm``
    semantics as :func:`asnumpy.fft.fft` (the inverse is scaled by ``1/n`` when ``norm`` is
    ``None``/``"backward"``).
    """
    a = _require_array(a, "ifft")
    _require_dtype(a, _COMPLEX64, "ifft")
    ax = _normalize_axis(axis, a.ndim)
    length = _resolve_length(n, a.shape[ax])
    return ndarray(_fft.fft_c2c(a, ax, length, True, _norm_factor(True, norm, length)))


def rfft(a: ArrayLike, n=None, axis: int = -1, norm=None) -> ndarray:
    """Compute the one-dimensional FFT of a real NPU array.

    Counterpart of :func:`numpy.fft.rfft` for float32 input: the transformed axis of the result holds
    the non-negative frequency terms and has length ``n // 2 + 1``. ``n`` truncates or zero pads the
    real input; ``norm`` follows NumPy (``None``/``"backward"`` leaves the forward transform
    unscaled).
    """
    a = _require_array(a, "rfft")
    _require_dtype(a, _REAL32, "rfft")
    ax = _normalize_axis(axis, a.ndim)
    length = _resolve_length(n, a.shape[ax])
    return ndarray(_fft.rfft(a, ax, length, _norm_factor(False, norm, length)))


def irfft(a: ArrayLike, n=None, axis: int = -1, norm=None) -> ndarray:
    """Compute the inverse of :func:`asnumpy.fft.rfft` for an NPU half-spectrum.

    Counterpart of :func:`numpy.fft.irfft` for complex64 input. With ``n=None`` the output length is
    derived from the input length ``m`` as ``2 * (m - 1)``, which cannot recover an odd original
    length: an odd-length round trip must pass the original length explicitly, i.e.
    ``irfft(rfft(x), n=len(x))``. When ``n`` is given the input is truncated or zero padded to
    ``n // 2 + 1`` bins first, exactly as NumPy does.
    """
    a = _require_array(a, "irfft")
    _require_dtype(a, _COMPLEX64, "irfft")
    ax = _normalize_axis(axis, a.ndim)
    if n is None:
        length = 2 * (a.shape[ax] - 1)
        if length <= 0:
            raise ValueError(f"Invalid number of FFT data points ({length}) specified.")
    else:
        length = _resolve_length(n, a.shape[ax])
    return ndarray(_fft.irfft(a, ax, length, _norm_factor(True, norm, length)))


def _normalize_axes(axes, ndim: int, name: str) -> tuple[int, int]:
    if isinstance(axes, (bool, int, np.integer)) or not hasattr(axes, "__iter__"):
        raise TypeError(
            f"asnumpy.fft.{name} expects a sequence of two axes, got {type(axes).__name__}"
        )
    items = tuple(axes)
    if len(items) != 2:
        raise NotImplementedError(
            f"asnumpy.fft.{name} implements exactly two axes, got {len(items)}; transforming a "
            "different number of axes (fftn/ifftn) is out of scope for this version"
        )
    return (_normalize_axis(items[0], ndim), _normalize_axis(items[1], ndim))


def _resolve_sizes(s, axes: tuple[int, int], shape, name: str) -> tuple[int, int]:
    if s is None:
        return (shape[axes[0]], shape[axes[1]])
    if isinstance(s, (bool, int, np.integer)) or not hasattr(s, "__iter__"):
        raise TypeError(
            f"asnumpy.fft.{name} expects s to be a sequence of two lengths, got {type(s).__name__}"
        )
    items = tuple(s)
    if len(items) != len(axes):
        raise ValueError("Shape and axes have different lengths.")
    return (_resolve_length(items[0], 0), _resolve_length(items[1], 0))


def fft2(a: ArrayLike, s=None, axes=(-2, -1), norm=None) -> ndarray:
    """Compute the two-dimensional discrete Fourier transform of an NPU array.

    Counterpart of :func:`numpy.fft.fft2` for complex64 input: the transform runs over ``axes``
    (default: the last two dimensions) and every other dimension is a batch. ``s`` gives the
    transform length for each of the two axes (shorter input is truncated, longer zero padded).

    Execution path: two one-dimensional ops-fft transforms, one per axis, both on the NPU — the
    native two-dimensional ops-fft kernel only covers 32/64/128 per axis on Ascend 910B.
    """
    a = _require_array(a, "fft2")
    _require_dtype(a, _COMPLEX64, "fft2")
    ax0, ax1 = _normalize_axes(axes, a.ndim, "fft2")
    n0, n1 = _resolve_sizes(s, (ax0, ax1), a.shape, "fft2")
    return ndarray(
        _fft.fft2_c2c(
            a, ax0, ax1, n0, n1, False, _norm_factor(False, norm, n0), _norm_factor(False, norm, n1)
        )
    )


def ifft2(a: ArrayLike, s=None, axes=(-2, -1), norm=None) -> ndarray:
    """Compute the two-dimensional inverse discrete Fourier transform of an NPU array.

    Counterpart of :func:`numpy.fft.ifft2` for complex64 input, with the same ``s``/``axes``/``norm``
    semantics as :func:`asnumpy.fft.fft2` (``norm=None``/``"backward"`` scales by ``1/(n0*n1)``).
    """
    a = _require_array(a, "ifft2")
    _require_dtype(a, _COMPLEX64, "ifft2")
    ax0, ax1 = _normalize_axes(axes, a.ndim, "ifft2")
    n0, n1 = _resolve_sizes(s, (ax0, ax1), a.shape, "ifft2")
    return ndarray(
        _fft.fft2_c2c(
            a, ax0, ax1, n0, n1, True, _norm_factor(True, norm, n0), _norm_factor(True, norm, n1)
        )
    )
