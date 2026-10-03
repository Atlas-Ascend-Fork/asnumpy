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

import numpy as np


def assert_array_equal(x, y, err_msg="", verbose=True, strides_check=False, type_check=True):
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
    # Convert to numpy arrays
    if not isinstance(x, np.ndarray):
        if hasattr(x, "to_numpy"):
            x = x.to_numpy()
        else:
            x = np.asarray(x)

    if not isinstance(y, np.ndarray):
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()
        else:
            y = np.asarray(y)

    # Check shape
    if x.shape != y.shape:
        msg = f"Shape mismatch: x.shape={x.shape}, y.shape={y.shape}"
        raise AssertionError(f"{err_msg}\n{msg}" if err_msg else msg)

    # Check dtype
    if type_check and x.dtype != y.dtype:
        msg = f"Dtype mismatch: x.dtype={x.dtype}, y.dtype={y.dtype}"
        raise AssertionError(f"{err_msg}\n{msg}" if err_msg else msg)

    # Check strides
    if strides_check and x.strides != y.strides:
        msg = f"Strides mismatch: x.strides={x.strides}, y.strides={y.strides}"
        raise AssertionError(f"{err_msg}\n{msg}" if err_msg else msg)

    # Check values
    if not np.array_equal(x, y):
        if verbose:
            msg = "Arrays are not equal."
            try:
                # Compute difference only for numeric types
                if x.dtype.kind in "biu f":  # bool, int, uint, float
                    if x.dtype.kind == "b":
                        # For bool, use XOR to compute difference (subtraction raises an error)
                        diff = x ^ y
                        msg += f"\nNumber of differing elements: {np.sum(diff)}"
                    else:
                        diff = x - y
                        msg += f"\nMax absolute difference: {np.abs(diff).max()}"
            except Exception:
                pass  # Ignore if difference computation fails

            if x.size > 0:
                msg += f"\nIndices where elements differ: {np.where(x != y)}"
            msg += f"\nNumPy:\n{x}\nAsNumPy:\n{y}"
        else:
            msg = "Arrays are not equal."
        raise AssertionError(f"{err_msg}\n{msg}" if err_msg else msg)


def assert_allclose(
    x, y, rtol=1e-7, atol=0, err_msg="", verbose=True, strides_check=False, type_check=True
):
    """Assert that two arrays are equal within a tolerance (for floating-point comparison)."""
    # Convert to numpy arrays
    if not isinstance(x, np.ndarray):
        if hasattr(x, "to_numpy"):
            x = x.to_numpy()
        else:
            x = np.asarray(x)

    if not isinstance(y, np.ndarray):
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()
        else:
            y = np.asarray(y)

    # Check shape
    if x.shape != y.shape:
        msg = f"Shape mismatch: x.shape={x.shape}, y.shape={y.shape}"
        raise AssertionError(f"{err_msg}\n{msg}" if err_msg else msg)

    # Check dtype
    if type_check and x.dtype != y.dtype:
        msg = f"Dtype mismatch: x.dtype={x.dtype}, y.dtype={y.dtype}"
        raise AssertionError(f"{err_msg}\n{msg}" if err_msg else msg)

    # Check strides
    if strides_check and x.strides != y.strides:
        msg = f"Strides mismatch: x.strides={x.strides}, y.strides={y.strides}"
        raise AssertionError(f"{err_msg}\n{msg}" if err_msg else msg)

    # Check values within tolerance
    if not np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True):
        if verbose:
            msg = "Arrays are not almost equal."
            try:
                if x.dtype.kind in "uif":
                    diff = x - y
                    msg += f"\nMax absolute difference: {np.abs(diff).max()}"
            except TypeError as e:
                # Type error: cannot perform arithmetic - log but continue
                msg += f"\nWarning: Cannot calculate difference - {str(e)}"
            except Exception as e:
                # Other unexpected errors - log details
                msg += f"\nWarning: Difference calculation failed - {type(e).__name__}: {str(e)}"

            if x.size > 0:
                mask = ~np.isclose(x, y, rtol=rtol, atol=atol, equal_nan=True)
                # Use atleast_1d to avoid DeprecationWarning on 0d arrays
                indices = np.atleast_1d(mask).nonzero()
                msg += f"\nIndices where elements differ: {indices}"
            msg += f"\nNumPy:\n{x}\nAsNumPy:\n{y}"
        else:
            msg = "Arrays are not almost equal."
        raise AssertionError(f"{err_msg}\n{msg}" if err_msg else msg)
