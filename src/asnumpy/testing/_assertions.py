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

"""Exception and warning assertions.

Provides assertion functions for testing exceptions and warnings.
"""

__all__ = [
    "assert_raises",
    "assert_raises_regex",
    "assert_warns",
    "assert_no_warnings",
    "assert_equal",
    "assert_string_equal",
    "assert_warns_message",
]

import re
import warnings
from contextlib import contextmanager


def assert_raises(exception_class, func=None, *args, **kwargs):
    """Assert that a function raises the specified exception type.

    Can be used as a context manager or called directly.

    Args:
        exception_class: Expected exception type.
        func: Function under test (optional).
        *args: Positional arguments passed to func.
        **kwargs: Keyword arguments passed to func.

    Returns:
        Context manager object when used as a context manager.

    Raises:
        AssertionError: If no exception is raised or the type does not match.

    Examples:
        # As a context manager
        with assert_raises(ValueError):
            numpy.zeros(-1)

        # As a function call
        assert_raises(ValueError, numpy.zeros, -1)

        # As a decorator
        @assert_raises(TypeError)
        def test_invalid_type():
            numpy.array('invalid')
    """
    if func is None:
        return _AssertRaisesContext(exception_class)

    try:
        func(*args, **kwargs)
        raise AssertionError(
            f"Expected {exception_class.__name__} to be raised, but no exception was raised"
        )
    except exception_class:
        return None
    except Exception as e:
        raise AssertionError(
            f"Expected {exception_class.__name__}, but got {type(e).__name__}: {e}"
        ) from e


class _AssertRaisesContext:
    """Context manager implementation for assert_raises."""

    def __init__(self, exception_class):
        self.exception_class = exception_class
        self.exception = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(
                f"Expected {self.exception_class.__name__} to be raised, "
                f"but no exception was raised"
            )

        if not issubclass(exc_type, self.exception_class):
            # Wrong exception type - let it propagate
            return False

        # Correct exception captured
        self.exception = exc_val
        return True


def assert_raises_regex(exception_class, regex_pattern, func=None, *args, **kwargs):
    """Assert that a function raises the specified exception with a matching message.

    Args:
        exception_class: Expected exception type.
        regex_pattern: Regex pattern the exception message must match.
        func: Function under test (optional).
        *args: Positional arguments passed to func.
        **kwargs: Keyword arguments passed to func.

    Returns:
        Context manager object when used as a context manager.

    Examples:
        # As a context manager
        with assert_raises_regex(ValueError, "invalid.*shape"):
            numpy.zeros(-1)

        # As a function call
        assert_raises_regex(ValueError, "negative", numpy.zeros, -1)
    """
    if func is None:
        return _AssertRaisesRegexContext(exception_class, regex_pattern)

    try:
        func(*args, **kwargs)
        raise AssertionError(
            f"Expected {exception_class.__name__} to be raised, but no exception was raised"
        )
    except exception_class as e:
        if not re.search(regex_pattern, str(e)):
            raise AssertionError(
                f"Exception message '{e}' does not match pattern '{regex_pattern}'"
            ) from e
        return None
    except Exception as e:
        raise AssertionError(
            f"Expected {exception_class.__name__}, but got {type(e).__name__}: {e}"
        ) from e


class _AssertRaisesRegexContext:
    """Context manager implementation for assert_raises_regex."""

    def __init__(self, exception_class, regex_pattern):
        self.exception_class = exception_class
        self.regex_pattern = regex_pattern
        self.exception = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(
                f"Expected {self.exception_class.__name__} to be raised, "
                f"but no exception was raised"
            )

        if not issubclass(exc_type, self.exception_class):
            # Wrong exception type - let it propagate
            return False

        if not re.search(self.regex_pattern, str(exc_val)):
            raise AssertionError(
                f"Exception message '{exc_val}' does not match pattern '{self.regex_pattern}'"
            )

        self.exception = exc_val
        return True


def assert_warns(warning_class, func=None, *args, **kwargs):
    """Assert that a function emits the specified warning type.

    Args:
        warning_class: Expected warning type.
        func: Function under test (optional).
        *args: Positional arguments passed to func.
        **kwargs: Keyword arguments passed to func.

    Returns:
        Context manager object when used as a context manager.

    Examples:
        # As a context manager
        with assert_warns(DeprecationWarning):
            old_function()

        # As a function call
        assert_warns(UserWarning, some_function, arg1, arg2)
    """
    if func is None:
        return _AssertWarnsContext(warning_class)

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        func(*args, **kwargs)

        for w in warning_list:
            if issubclass(w.category, warning_class):
                return None

        raise AssertionError(
            f"Expected {warning_class.__name__} to be raised, but no such warning was issued"
        )


class _AssertWarnsContext:
    """Context manager implementation for assert_warns."""

    def __init__(self, warning_class):
        self.warning_class = warning_class
        self.warnings = []
        self._warnings_manager = None

    def __enter__(self):
        self._warnings_manager = warnings.catch_warnings(record=True)
        self.warnings = self._warnings_manager.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._warnings_manager.__exit__(exc_type, exc_val, exc_tb)

        for w in self.warnings:
            if issubclass(w.category, self.warning_class):
                return False

        raise AssertionError(
            f"Expected {self.warning_class.__name__} to be raised, but no such warning was issued"
        )


def assert_no_warnings(func=None, *args, **kwargs):
    """Assert that a function does not emit any warnings.

    Args:
        func: Function under test.
        *args: Positional arguments passed to func.
        **kwargs: Keyword arguments passed to func.

    Raises:
        AssertionError: If any warning is emitted.

    Examples:
        assert_no_warnings(some_function, arg1, arg2)

        # Or as a context manager
        with assert_no_warnings():
            some_function()
    """
    if callable(func):
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            result = func(*args, **kwargs)

            if warning_list:
                warning_msgs = [f"{w.category.__name__}: {w.message}" for w in warning_list]
                raise AssertionError(
                    f"Expected no warnings, but got {len(warning_list)} warning(s):\n"
                    + "\n".join(warning_msgs)
                )

            return result
    else:
        return _AssertNoWarningsContext()


class _AssertNoWarningsContext:
    """Context manager implementation for assert_no_warnings."""

    def __init__(self):
        self._warnings_manager = None
        self.warnings = []

    def __enter__(self):
        self._warnings_manager = warnings.catch_warnings(record=True)
        self.warnings = self._warnings_manager.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._warnings_manager.__exit__(exc_type, exc_val, exc_tb)

        if self.warnings:
            warning_msgs = [f"{w.category.__name__}: {w.message}" for w in self.warnings]
            raise AssertionError(
                f"Expected no warnings, but got {len(self.warnings)} warning(s):\n"
                + "\n".join(warning_msgs)
            )

        return False


def assert_equal(actual, desired, err_msg=""):
    """Assert that two objects are equal.

    General-purpose equality check for any Python objects.

    Args:
        actual: Actual value.
        desired: Expected value.
        err_msg: Custom error message.

    Raises:
        AssertionError: If the two objects are not equal.

    Examples:
        assert_equal(result, expected)
        assert_equal(a.shape, (3, 4))
        assert_equal(type(arr), numpy.ndarray)
    """
    if actual != desired:
        msg = f"\nActual: {actual}\nDesired: {desired}"
        if err_msg:
            msg = f"{err_msg}\n{msg}"
        raise AssertionError(msg)


def assert_string_equal(actual, desired):
    """Assert that two strings are equal with a friendly diff message.

    Args:
        actual: Actual string.
        desired: Expected string.

    Raises:
        AssertionError: If the strings are not equal.
    """
    if actual != desired:
        for i, (a, d) in enumerate(zip(actual, desired, strict=False)):
            if a != d:
                raise AssertionError(
                    f"Strings differ at position {i}:\n"
                    f"Actual  : {repr(actual)}\n"
                    f"Desired : {repr(desired)}\n"
                    f"First difference: {repr(a)} != {repr(d)}"
                )

        raise AssertionError(
            f"Strings have different lengths:\n"
            f"Actual  : {repr(actual)} (length {len(actual)})\n"
            f"Desired : {repr(desired)} (length {len(desired)})"
        )


@contextmanager
def assert_warns_message(warning_class, message_pattern):
    """Assert that a specific warning with a matching message is emitted.

    Combines assert_warns with message matching.

    Args:
        warning_class: Expected warning type.
        message_pattern: Regex pattern the warning message must match.

    Examples:
        with assert_warns_message(DeprecationWarning, "deprecated.*use.*instead"):
            old_function()
    """
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        yield

        for w in warning_list:
            if issubclass(w.category, warning_class) and re.search(message_pattern, str(w.message)):
                return

        raise AssertionError(
            f"Expected {warning_class.__name__} with message matching '{message_pattern}', "
            f"but no such warning was issued"
        )
