# Copyright (c) 2026 ISE Group at Harbin Institute of Technology. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-only regression tests for the actual exported testing helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SOURCE = Path(__file__).resolve().parents[2] / "src/asnumpy/testing/_helper.py"
_SPEC = importlib.util.spec_from_file_location("asnumpy_testing_helper_host", _SOURCE)
_HELPER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPER)


@pytest.mark.parametrize("shape", [(), (0,), (1,), (2,), (5,), (2, 3), (2, 3, 4), (2, 0, 3)])
@pytest.mark.parametrize("dtype", [np.bool_, np.dtype("bool"), "bool"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_boolean_sequential_helpers(shape, dtype, order):
    size = int(np.prod(shape))
    expected = np.array([bool(i) for i in range(size)], dtype=np.bool_).reshape(shape, order=order)
    reverse = np.array([bool(i) for i in range(size - 1, -1, -1)], dtype=np.bool_).reshape(
        shape, order=order
    )

    actual = _HELPER.shaped_arange(shape, dtype=dtype, order=order, xp=np)
    actual_reverse = _HELPER.shaped_reverse_arange(shape, dtype=dtype, order=order, xp=np)

    for result, reference in [(actual, expected), (actual_reverse, reverse)]:
        assert result.dtype == np.dtype("bool")
        assert result.shape == shape
        np.testing.assert_array_equal(result, reference)
        assert result.flags.c_contiguous == reference.flags.c_contiguous
        assert result.flags.f_contiguous == reference.flags.f_contiguous


@pytest.mark.parametrize("start", [-3, 0, 2, 2**65])
@pytest.mark.parametrize("shape", [0, 1, 2, 5])
def test_boolean_shaped_arange_start_and_integer_shape(start, shape):
    expected = np.array([bool(start + i) for i in range(shape)], dtype=np.bool_)
    result = _HELPER.shaped_arange(shape, dtype=np.bool_, start=start, xp=np)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (shape,)
    assert result.dtype == expected.dtype


@pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.uint8,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ],
)
@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
@pytest.mark.parametrize("order", ["C", "F"])
def test_nonboolean_sequential_helpers_preserve_numpy_behavior(dtype, shape, order):
    size = int(np.prod(shape))
    for result, expected in [
        (
            _HELPER.shaped_arange(shape, dtype=dtype, order=order, xp=np),
            np.arange(size, dtype=dtype).reshape(shape, order=order),
        ),
        (
            _HELPER.shaped_reverse_arange(shape, dtype=dtype, order=order, xp=np),
            np.arange(size - 1, -1, -1, dtype=dtype).reshape(shape, order=order),
        ),
    ]:
        assert result.dtype == expected.dtype
        assert result.shape == expected.shape
        assert result.tobytes(order=order) == expected.tobytes(order=order)
