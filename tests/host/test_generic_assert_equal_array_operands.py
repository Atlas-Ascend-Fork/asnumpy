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

"""Host regressions for the generic assertion's NumPy operands."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def assertions():
    root = Path(__file__).resolve().parents[2]
    path = root / "src/asnumpy/testing/_assertions.py"
    spec = importlib.util.spec_from_file_location("assertions_host_leaf", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_equal_finite_numpy_arrays_accept_all_elements(assertions):
    array = np.arange(6, dtype=np.float32).reshape(2, 3)

    assertions.assert_equal(array, array.copy())


def test_numpy_array_difference_raises_assertion_with_message(assertions):
    actual = np.array([1, 2, 3])
    expected = np.array([1, 9, 3])

    with pytest.raises(AssertionError, match="finite array mismatch"):
        assertions.assert_equal(actual, expected, err_msg="finite array mismatch")


@pytest.mark.parametrize("container", [list, lambda value: {"values": value}], ids=["list", "dict"])
def test_equal_containers_with_numpy_arrays_accept_elements(assertions, container):
    actual = container([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
    expected = container([np.array([1.0, 2.0]), np.array([3.0, 4.0])])

    assertions.assert_equal(actual, expected)


@pytest.mark.parametrize("value", [7, "label", (2, 3)], ids=["integer", "string", "shape"])
def test_equal_scalar_string_and_shape_controls(assertions, value):
    assertions.assert_equal(value, value)


def test_scalar_difference_still_raises(assertions):
    with pytest.raises(AssertionError, match="scalar mismatch"):
        assertions.assert_equal(1, 2, err_msg="scalar mismatch")
