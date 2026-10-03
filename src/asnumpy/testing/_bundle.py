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

"""Test class generation utilities.

Responsible for dynamically generating test classes:
- make_decorator() - create decorators
- _generate_case() - generate concrete test classes
"""

__all__ = [
    "make_decorator",
    "generate_test_classes",
    "TestBundle",
]

import functools
import unittest


def make_decorator(decorator_func):
    """General-purpose utility for creating decorators.

    Converts a plain function into a decorator.

    Args:
        decorator_func: The decorator function.

    Returns:
        Decorator.
    """

    @functools.wraps(decorator_func)
    def wrapper(*args, **kwargs):
        # Used directly as @decorator
        if len(args) == 1 and callable(args[0]) and not kwargs:
            func = args[0]
            return decorator_func(func)
        # Used as @decorator(...)
        else:

            def actual_decorator(func):
                return decorator_func(func, *args, **kwargs)

            return actual_decorator

    return wrapper


def _generate_case(test_class, params):
    """Generate a concrete test case class.

    Creates a concrete test case class for a given combination of parameters.

    Args:
        test_class: Base test class.
        params: Parameter dict.

    Returns:
        Generated test class.
    """
    # Build class name
    class_name = test_class.__name__
    for key, value in params.items():
        value_str = str(value)
        if hasattr(value, "__name__"):
            value_str = value.__name__
        class_name += f"_{key}_{value_str}"

    # Build attribute dict for the new class
    class_dict = {}

    # Copy all methods from the test class
    for attr_name in dir(test_class):
        if attr_name.startswith("_"):
            continue
        attr = getattr(test_class, attr_name)
        if not callable(attr):
            continue

        # Bind parameters into each method
        def make_method(original_method, test_params):
            @functools.wraps(original_method)
            def method(self, *args, **kwargs):
                kwargs.update(test_params)
                return original_method(self, *args, **kwargs)

            return method

        class_dict[attr_name] = make_method(attr, params)

    # Create the new class
    new_class = type(class_name, (test_class,), class_dict)
    return new_class


def generate_test_classes(base_class, param_combinations):
    """Generate multiple parameterized versions of a test class.

    Args:
        base_class: Base test class.
        param_combinations: List of parameter combination dicts.

    Returns:
        List of generated test classes.
    """
    test_classes = []

    for params in param_combinations:
        test_class = _generate_case(base_class, params)
        test_classes.append(test_class)

    return test_classes


class TestBundle:
    """Wrapper class for a collection of test classes.

    Wraps multiple test classes for convenient batch management.
    """

    def __init__(self, test_classes):
        """Initialize the test bundle.

        Args:
            test_classes: List of test classes.
        """
        self.test_classes = test_classes

    def run(self, runner):
        """Run all test classes.

        Args:
            runner: Test runner.

        Returns:
            List of test results.
        """
        results = []
        for test_class in self.test_classes:
            loader = getattr(
                runner, "loadTestsFromTestCase", unittest.defaultTestLoader.loadTestsFromTestCase
            )
            suite = loader(test_class)
            result = runner.run(suite)
            results.append(result)
        return results

    def get_test_count(self):
        """Return the total number of test methods.

        Returns:
            Total count of test methods across all classes.
        """
        count = 0
        for test_class in self.test_classes:
            for attr_name in dir(test_class):
                if attr_name.startswith("test_"):
                    count += 1
        return count
