# Copyright (c) 2026 Ethan Zou.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Host tests of actual testing leaf modules, without importing the NPU extension."""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def helpers(monkeypatch):
    package_name = "array_assertion_host"
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path(__file__).resolve().parents[3] / "src/asnumpy/testing")]
    monkeypatch.setitem(sys.modules, package_name, package)
    array = importlib.import_module(package_name + "._array")
    helper = importlib.import_module(package_name + "._helper")
    yield array, helper
    for name in list(sys.modules):
        if name.startswith(package_name + "."):
            del sys.modules[name]


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("scalar", [True, False])
def test_matching_nan_agrees_with_numpy_testing(helpers, dtype, scalar):
    array, _ = helpers
    x = np.array(np.nan if scalar else [0, np.nan, np.inf, -np.inf], dtype=dtype)
    y = x.copy()
    np.testing.assert_array_equal(x, y, strict=True)
    array.assert_array_equal(x, y)


def test_noncontiguous_matching_nans(helpers):
    array, _ = helpers
    x = np.array([[np.nan, 0, 1, 2], [3, 4, np.nan, 5]], dtype=np.float32)[:, ::2]
    y = x.copy()
    assert not x.flags.c_contiguous
    np.testing.assert_array_equal(x, y, strict=True)
    array.assert_array_equal(x, y)


def test_complex_nan_semantics_match_numpy_testing(helpers):
    array, _ = helpers
    x = np.array([complex(np.nan, 1), complex(2, np.nan)])
    y = np.array([complex(3, np.nan), complex(np.nan, 4)])
    np.testing.assert_array_equal(x, y, strict=True)
    array.assert_array_equal(x, y)


@pytest.mark.parametrize(
    "x,y",
    [([np.nan, 1], [1, np.nan]), ([np.nan, 1], [np.nan, 2]), ([np.inf, np.nan], [-np.inf, np.nan])],
)
def test_true_value_mismatches_still_fail(helpers, x, y):
    array, _ = helpers
    x, y = np.asarray(x), np.asarray(y)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(x, y, strict=True)
    with pytest.raises(AssertionError):
        array.assert_array_equal(x, y, verbose=False)


@pytest.mark.parametrize("dtype", [np.bool_, np.int64, np.uint64, "U2", "S2", object])
def test_non_nan_dtypes_keep_their_comparison(helpers, dtype):
    array, _ = helpers
    x = np.array([0, 1], dtype=dtype)
    array.assert_array_equal(x, x.copy())


@pytest.mark.parametrize("guard", ["shape", "dtype", "strides"])
def test_existing_strict_guards_remain(helpers, guard):
    array, _ = helpers
    x = np.array([[np.nan, 1], [2, 3]], dtype=np.float32)
    y = (
        x.reshape(-1)
        if guard == "shape"
        else x.astype(np.float64)
        if guard == "dtype"
        else x.copy(order="F")
    )
    with pytest.raises(AssertionError, match=guard.capitalize() + " mismatch"):
        array.assert_array_equal(x, y, strides_check=guard == "strides")


def test_actual_array_list_helper_uses_nan_comparison(helpers):
    _, helper = helpers
    arrays = [np.array([np.nan, 1]), np.array(np.nan)]
    helper.assert_array_list_equal(arrays, [x.copy() for x in arrays])
