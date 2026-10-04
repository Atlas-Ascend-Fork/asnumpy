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

"""Array comparison functions.

Provides array comparison assertion functions for use in tests.
"""

__all__ = ["assert_array_equal", "assert_allclose"]

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _ComparisonOptions:
    err_msg: str = ""
    verbose: bool = True
    strides_check: bool = False
    type_check: bool = True
    rtol: float = 1e-7
    atol: float = 0


def _to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "to_numpy"):
        return value.to_numpy()
    return np.asarray(value)


def _raise_mismatch(msg, err_msg):
    raise AssertionError(f"{err_msg}\n{msg}" if err_msg else msg)


def _check_metadata(x, y, options):
    if x.shape != y.shape:
        _raise_mismatch(f"Shape mismatch: x.shape={x.shape}, y.shape={y.shape}", options.err_msg)
    if options.type_check and x.dtype != y.dtype:
        _raise_mismatch(f"Dtype mismatch: x.dtype={x.dtype}, y.dtype={y.dtype}", options.err_msg)
    if options.strides_check and x.strides != y.strides:
        _raise_mismatch(
            f"Strides mismatch: x.strides={x.strides}, y.strides={y.strides}", options.err_msg
        )


def _equal_difference(x, y):
    if x.dtype.kind not in "biu f":
        return ""
    try:
        if x.dtype.kind == "b":
            return f"\nNumber of differing elements: {np.sum(x ^ y)}"
        return f"\nMax absolute difference: {np.abs(x - y).max()}"
    except Exception:
        # Difference diagnostics must not replace the comparison assertion.
        return ""


def _close_difference(x, y):
    if x.dtype.kind not in "uif":
        return ""
    try:
        return f"\nMax absolute difference: {np.abs(x - y).max()}"
    except TypeError as exc:
        return f"\nWarning: Cannot calculate difference - {str(exc)}"
    except Exception as exc:
        return f"\nWarning: Difference calculation failed - {type(exc).__name__}: {str(exc)}"


def _assert_array_equal(x, y, options):
    x, y = _to_numpy(x), _to_numpy(y)
    _check_metadata(x, y, options)
    if np.array_equal(x, y):
        return
    msg = "Arrays are not equal."
    if options.verbose:
        msg += _equal_difference(x, y)
        if x.size > 0:
            msg += f"\nIndices where elements differ: {np.where(x != y)}"
        msg += f"\nNumPy:\n{x}\nAsNumPy:\n{y}"
    _raise_mismatch(msg, options.err_msg)


def _assert_allclose(x, y, options):
    x, y = _to_numpy(x), _to_numpy(y)
    _check_metadata(x, y, options)
    if np.allclose(x, y, rtol=options.rtol, atol=options.atol, equal_nan=True):
        return
    msg = "Arrays are not almost equal."
    if options.verbose:
        msg += _close_difference(x, y)
        if x.size > 0:
            mask = ~np.isclose(x, y, rtol=options.rtol, atol=options.atol, equal_nan=True)
            indices = np.atleast_1d(mask).nonzero()
            msg += f"\nIndices where elements differ: {indices}"
        msg += f"\nNumPy:\n{x}\nAsNumPy:\n{y}"
    _raise_mismatch(msg, options.err_msg)


def assert_array_equal(x, y, err_msg="", verbose=True, strides_check=False):
    """Assert that two arrays are exactly equal.

    Args:
        x: First array.
        y: Second array.
        err_msg: Custom error message.
        verbose: Whether to show detailed error information.
        strides_check: Whether to check strides.

    Raises:
        AssertionError: If the arrays are not equal.
    """
    options = _ComparisonOptions(err_msg=err_msg, verbose=verbose, strides_check=strides_check)
    _assert_array_equal(x, y, options)


def assert_allclose(x, y, rtol=1e-7, atol=0, err_msg="", verbose=True, strides_check=False):
    """Assert equality within a tolerance, including shape and dtype."""
    options = _ComparisonOptions(
        err_msg=err_msg, verbose=verbose, strides_check=strides_check, rtol=rtol, atol=atol
    )
    _assert_allclose(x, y, options)
