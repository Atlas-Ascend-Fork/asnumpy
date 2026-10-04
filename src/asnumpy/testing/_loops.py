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

"""Test loop decorators.

Provides decorators for parameterized tests across dtype, order, and other dimensions.
"""

__all__ = [
    "for_dtypes",
    "for_all_dtypes",
    "for_float_dtypes",
    "for_int_dtypes",
    "for_signed_dtypes",
    "for_unsigned_dtypes",
    "for_complex_dtypes",
    "for_orders",
    "for_cf_orders",
    "numpy_asnumpy_array_equal",
    "numpy_asnumpy_allclose",
]

import functools
import inspect

import numpy
from loguru import logger

from . import _array

# dtype constants
_float_dtypes = (numpy.float16, numpy.float32, numpy.float64)
_complex_dtypes = (numpy.complex64, numpy.complex128)
_signed_dtypes = (numpy.int8, numpy.int16, numpy.int32, numpy.int64)
_unsigned_dtypes = (numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64)
_int_dtypes = _signed_dtypes + _unsigned_dtypes


def _make_all_dtypes(
    no_float16=True, no_bool=False, no_complex=False, no_uint32=True, no_uint64=True
):
    """Build the list of all supported dtypes.

    Excludes types unsupported by asnumpy by default: float16, uint32, uint64.

    Args:
        no_float16: Exclude float16 (default True - not supported).
        no_bool: Exclude bool.
        no_complex: Exclude complex types.
        no_uint32: Exclude uint32 (default True - NPU op not supported).
        no_uint64: Exclude uint64 (default True - NPU op not supported).

    Returns:
        Tuple of included dtypes.
    """
    # Float types
    dtypes = list(_float_dtypes)
    if no_float16:
        dtypes.remove(numpy.float16)

    # Complex types
    if not no_complex:
        dtypes.extend(_complex_dtypes)

    # Integer types (signed and unsigned)
    int_types = list(_int_dtypes)
    if no_uint32 and numpy.uint32 in int_types:
        int_types.remove(numpy.uint32)
    if no_uint64 and numpy.uint64 in int_types:
        int_types.remove(numpy.uint64)
    dtypes.extend(int_types)

    # Bool type
    if not no_bool:
        dtypes.append(numpy.bool_)

    return tuple(dtypes)


def _wraps_partial(impl, name):
    """Function wrapper helper."""

    def decorator(wrapper):
        return functools.wraps(impl)(wrapper)

    return decorator


# ========== dtype decorators ==========


def for_dtypes(dtypes, name="dtype"):
    """Parameterize a test over multiple dtypes.

    Recommended usage:
        @for_dtypes([numpy.float32, numpy.float64])
        def test_func(xp, dtype):
            return xp.some_function(...)

    Args:
        dtypes: List of dtypes.
        name: Parameter name (default 'dtype').

    Returns:
        Decorator function.
    """

    def decorator(impl):
        # Wrap as a no-argument function so pytest does not treat params as fixtures
        @functools.wraps(impl)
        def test_func():
            for dtype in dtypes:
                try:
                    impl(**{name: dtype})
                except Exception:
                    logger.debug(f"{name} is {dtype}")
                    raise

        # Clear signature so pytest sees a no-argument function
        test_func.__signature__ = inspect.Signature()
        return test_func

    return decorator


def for_all_dtypes(
    name="dtype",
    no_float16=True,
    no_bool=False,
    no_complex=False,
    no_uint32=True,
    no_uint64=True,
    exclude=None,
):
    """Parameterize a test over all supported dtypes.

    Excludes types unsupported by asnumpy by default: float16, uint32, uint64.

    Args:
        name: Parameter name.
        no_float16: Exclude float16 (default True - not supported).
        no_bool: Exclude bool.
        no_complex: Exclude complex types.
        no_uint32: Exclude uint32 (default True - NPU op not supported).
        no_uint64: Exclude uint64 (default True - NPU op not supported).
        exclude: Additional dtypes to exclude, e.g. [numpy.float64, numpy.uint8].

    Returns:
        Decorator function.
    """
    dtypes = list(_make_all_dtypes(no_float16, no_bool, no_complex, no_uint32, no_uint64))
    if exclude:
        dtypes = [dt for dt in dtypes if dt not in exclude]
    return for_dtypes(dtypes, name=name)


def for_float_dtypes(name="dtype", no_float16=True, exclude=None):
    """Parameterize a test over floating-point dtypes.

    Excludes float16 by default (not supported by asnumpy).

    Args:
        name: Parameter name.
        no_float16: Exclude float16 (default True - not supported).
        exclude: Additional dtypes to exclude, e.g. [numpy.float64].

    Returns:
        Decorator function.
    """
    dtypes = list(_float_dtypes)
    if no_float16:
        dtypes.remove(numpy.float16)
    if exclude:
        dtypes = [dt for dt in dtypes if dt not in exclude]
    return for_dtypes(tuple(dtypes), name=name)


def for_int_dtypes(name="dtype", exclude=None):
    """Parameterize a test over all integer dtypes.

    Args:
        name: Parameter name.
        exclude: Dtypes to exclude, e.g. [numpy.uint8].

    Returns:
        Decorator function.
    """
    dtypes = list(_int_dtypes)
    if exclude:
        dtypes = [dt for dt in dtypes if dt not in exclude]
    return for_dtypes(dtypes, name=name)


def for_signed_dtypes(name="dtype"):
    """Parameterize a test over signed integer dtypes."""
    return for_dtypes(_signed_dtypes, name=name)


def for_unsigned_dtypes(name="dtype", no_uint32=True, no_uint64=True):
    """Parameterize a test over unsigned integer dtypes.

    Excludes uint32 and uint64 by default (NPU ops not supported).

    Args:
        name: Parameter name.
        no_uint32: Exclude uint32 (default True - NPU op not supported).
        no_uint64: Exclude uint64 (default True - NPU op not supported).

    Returns:
        Decorator function.
    """
    dtypes = list(_unsigned_dtypes)
    if no_uint32 and numpy.uint32 in dtypes:
        dtypes.remove(numpy.uint32)
    if no_uint64 and numpy.uint64 in dtypes:
        dtypes.remove(numpy.uint64)
    return for_dtypes(tuple(dtypes), name=name)


def for_complex_dtypes(name="dtype", exclude=None):
    """Parameterize a test over complex dtypes.

    Args:
        name: Parameter name.
        exclude: Dtypes to exclude, e.g. [numpy.complex64].

    Returns:
        Decorator function.
    """
    dtypes = list(_complex_dtypes)
    if exclude:
        dtypes = [dt for dt in dtypes if dt not in exclude]
    return for_dtypes(tuple(dtypes), name=name)


# ========== order decorators ==========


def for_orders(orders, name="order"):
    """Parameterize a test over multiple memory orders.

    Test function should accept parameters via **kw:
        @for_orders(['C', 'F'])
        def test_func(**kw):
            xp = kw['xp']
            order = kw['order']
            return xp.some_function(...)

    Args:
        orders: List of memory orders.
        name: Parameter name (default 'order').

    Returns:
        Decorator function.
    """

    def decorator(impl):
        @_wraps_partial(impl, name)
        def test_func(*args, **kw):
            for order in orders:
                try:
                    kw[name] = order
                    impl(*args, **kw)
                except Exception:
                    logger.debug(f"{name} is {order}")
                    raise

        return test_func

    return decorator


def for_cf_orders(name="order"):
    """Parameterize a test over C and F memory orders."""
    return for_orders([None, "C", "F", "c", "f"], name)


# ========== numpy-asnumpy comparison decorators ==========


def _make_decorator(check_func, name, type_check, accept_error, sp_name=None, scipy_name=None):
    """Core factory for numpy-asnumpy comparison decorators.

    Recommended usage:
        @numpy_asnumpy_array_equal()
        def test_func(xp, dtype):
            return xp.some_function(...)

    Args:
        check_func: Function used to compare results.
        name: Name of the xp parameter.
        type_check: Whether to perform type checking.
        accept_error: Whether to accept errors.
        sp_name: scipy parameter name (reserved).
        scipy_name: scipy module name (reserved).

    Returns:
        Decorator function.
    """

    def decorator(impl):
        # Inspect the original function's parameters
        sig = inspect.signature(impl)
        params = list(sig.parameters.keys())

        # Check whether there are parameters other than xp
        other_params = [p for p in params if p != name]
        needs_external_params = len(other_params) > 0

        if needs_external_params:
            # Other parameters exist; an outer decorator will supply them
            @functools.wraps(impl)
            def test_func(**kw):
                # Run numpy version
                kw_numpy = kw.copy()
                kw_numpy[name] = numpy
                try:
                    numpy_result = impl(**kw_numpy)
                    numpy_error = None
                except Exception as e:
                    numpy_result = None
                    numpy_error = e

                # Run asnumpy version
                import asnumpy as ap

                kw_asnumpy = kw.copy()
                kw_asnumpy[name] = ap

                # Convert dtype parameter
                if "dtype" in kw_asnumpy and kw_asnumpy["dtype"] is not None:
                    kw_asnumpy["dtype"] = numpy.dtype(kw_asnumpy["dtype"])

                # Remove parameters unsupported by asnumpy (e.g. order)
                if "order" in kw_asnumpy:
                    kw_asnumpy.pop("order")

                try:
                    asnumpy_result = impl(**kw_asnumpy)
                    asnumpy_error = None
                except Exception as e:
                    asnumpy_result = None
                    asnumpy_error = e

                # Compare results
                if numpy_error is not None:
                    if asnumpy_error is None:
                        raise AssertionError(
                            f"NumPy raised {type(numpy_error).__name__} "
                            f"but AsNumPy did not raise an exception\n"
                            f"NumPy error: {numpy_error}"
                        )
                    elif not isinstance(asnumpy_error, type(numpy_error)):
                        if not accept_error:
                            raise AssertionError(
                                f"Exception types differ:\n"
                                f"  NumPy: {type(numpy_error).__name__}\n"
                                f"  AsNumPy: {type(asnumpy_error).__name__}"
                            )
                    # Same exception type - test passes
                    return
                elif asnumpy_error is not None:
                    raise AssertionError(
                        f"AsNumPy raised {type(asnumpy_error).__name__} "
                        f"but NumPy did not raise an exception\n"
                        f"AsNumPy error: {asnumpy_error}"
                    )

                # No exceptions - compare values
                check_func(numpy_result, asnumpy_result)
        else:
            # No other parameters; only xp - return a no-argument function
            @functools.wraps(impl)
            def test_func():
                # Run numpy version
                try:
                    numpy_result = impl(**{name: numpy})
                    numpy_error = None
                except Exception as e:
                    numpy_result = None
                    numpy_error = e

                # Run asnumpy version
                import asnumpy as ap

                try:
                    asnumpy_result = impl(**{name: ap})
                    asnumpy_error = None
                except Exception as e:
                    asnumpy_result = None
                    asnumpy_error = e

                # Compare results
                if numpy_error is not None:
                    if asnumpy_error is None:
                        raise AssertionError(
                            f"NumPy raised {type(numpy_error).__name__} "
                            f"but AsNumPy did not raise an exception\n"
                            f"NumPy error: {numpy_error}"
                        )
                    elif not isinstance(asnumpy_error, type(numpy_error)):
                        if not accept_error:
                            raise AssertionError(
                                f"Exception types differ:\n"
                                f"  NumPy: {type(numpy_error).__name__}\n"
                                f"  AsNumPy: {type(asnumpy_error).__name__}"
                            )
                    # Same exception type - test passes
                    return
                elif asnumpy_error is not None:
                    raise AssertionError(
                        f"AsNumPy raised {type(asnumpy_error).__name__} "
                        f"but NumPy did not raise an exception\n"
                        f"AsNumPy error: {asnumpy_error}"
                    )

                # No exceptions - compare values
                check_func(numpy_result, asnumpy_result)

            # Clear signature so pytest does not treat it as a fixture
            test_func.__signature__ = inspect.Signature()

        return test_func

    return decorator


def numpy_asnumpy_array_equal(
    err_msg="",
    verbose=True,
    name="xp",
    type_check=True,
    accept_error=False,
    sp_name=None,
    scipy_name=None,
    strides_check=False,
):
    """Decorator: assert that NumPy and AsNumPy results are exactly equal.

    Args:
        err_msg: Error message.
        verbose: Whether to show detailed information.
        name: Name of the xp parameter.
        type_check: Whether to perform type checking.
        accept_error: Whether to accept errors.
        sp_name: scipy parameter name (reserved).
        scipy_name: scipy module name (reserved).
        strides_check: Whether to check strides.

    Returns:
        Decorator function.
    """

    options = _array.ComparisonOptions(
        err_msg=err_msg, verbose=verbose, strides_check=strides_check, type_check=type_check
    )

    def check_func(x, y):
        _array.assert_array_equal_with_options(x, y, options)

    return _make_decorator(check_func, name, type_check, accept_error, sp_name, scipy_name)


def numpy_asnumpy_allclose(
    rtol=1e-7,
    atol=0,
    err_msg="",
    verbose=True,
    name="xp",
    type_check=True,
    accept_error=False,
    sp_name=None,
    scipy_name=None,
    strides_check=False,
):
    """Decorator: assert that NumPy and AsNumPy float results are within tolerance.

    Args:
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        err_msg: Error message.
        verbose: Whether to show detailed information.
        name: Name of the xp parameter.
        type_check: Whether to perform type checking.
        accept_error: Whether to accept errors.
        sp_name: scipy parameter name (reserved).
        scipy_name: scipy module name (reserved).
        strides_check: Whether to check strides.

    Returns:
        Decorator function.
    """

    options = _array.ComparisonOptions(
        err_msg=err_msg,
        verbose=verbose,
        strides_check=strides_check,
        type_check=type_check,
        rtol=rtol,
        atol=atol,
    )

    def check_func(x, y):
        _array.assert_allclose_with_options(x, y, options)

    return _make_decorator(check_func, name, type_check, accept_error, sp_name, scipy_name)
