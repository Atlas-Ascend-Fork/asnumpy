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

"""Regression tests for moving private _core results into public ndarrays."""

import gc

import numpy
import pytest

import asnumpy
from asnumpy import _core


def _from_numpy(values) -> asnumpy.ndarray:
    return asnumpy.ndarray.from_numpy(numpy.asarray(values, dtype=numpy.float32))


def test_representative_operator_results_are_public_ndarrays():
    x = _from_numpy([[0.0, 0.5, 1.0], [1.5, 2.0, 2.5]])
    y = _from_numpy([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    zeros_result = asnumpy.zeros((2, 3), dtype=numpy.float32)
    sin_result = asnumpy.sin(x)
    add_result = asnumpy.add(x, y)
    sum_result = asnumpy.sum(y, axis=0)

    for result in (zeros_result, sin_result, add_result, sum_result):
        assert isinstance(result, asnumpy.ndarray)

    numpy.testing.assert_array_equal(
        zeros_result.to_numpy(), numpy.zeros((2, 3), dtype=numpy.float32)
    )
    numpy.testing.assert_allclose(sin_result.to_numpy(), numpy.sin(x.to_numpy()), rtol=1e-5)
    numpy.testing.assert_allclose(add_result.to_numpy(), x.to_numpy() + y.to_numpy(), rtol=1e-5)
    numpy.testing.assert_allclose(sum_result.to_numpy(), numpy.sum(y.to_numpy(), axis=0), rtol=1e-5)


def test_multi_output_results_are_wrapped_once():
    # Use non-negative inputs so the fractional/integral split is convention
    # independent (asnumpy's modf floors, numpy truncates toward zero; they only
    # differ for negative values, which is a pre-existing kernel behavior).
    host = numpy.array([0.25, 0.5, 1.75, 2.25], dtype=numpy.float32)
    x = asnumpy.ndarray.from_numpy(host)

    fraction, integral = asnumpy.modf(x)
    expected_fraction, expected_integral = numpy.modf(host)

    assert isinstance(fraction, asnumpy.ndarray)
    assert isinstance(integral, asnumpy.ndarray)
    numpy.testing.assert_allclose(fraction.to_numpy(), expected_fraction, rtol=1e-5)
    numpy.testing.assert_allclose(integral.to_numpy(), expected_integral, rtol=1e-5)
    # The two outputs must reconstruct the input, proving both were moved out intact.
    numpy.testing.assert_allclose(fraction.to_numpy() + integral.to_numpy(), host, rtol=1e-5)


def test_from_numpy_preserves_metadata_and_values():
    host = numpy.arange(12, dtype=numpy.float32).reshape(3, 4)

    result = asnumpy.ndarray.from_numpy(host)

    assert isinstance(result, asnumpy.ndarray)
    assert result.shape == host.shape
    assert result.dtype == host.dtype
    assert result.strides == host.strides
    numpy.testing.assert_array_equal(result.to_numpy(), host)


def test_from_numpy_strided_slice_roundtrip():
    # issue #41: non-contiguous inputs are normalized before the H2D copy
    host = numpy.arange(10, dtype=numpy.float32)[::2]

    result = asnumpy.ndarray.from_numpy(host)

    assert result.shape == host.shape
    numpy.testing.assert_array_equal(result.to_numpy(), host)


def test_from_numpy_transpose_roundtrip():
    host = numpy.arange(6, dtype=numpy.float32).reshape(2, 3).T

    result = asnumpy.ndarray.from_numpy(host)

    assert result.shape == host.shape
    numpy.testing.assert_array_equal(result.to_numpy(), host)


def test_from_numpy_negative_stride_roundtrip():
    # a negative stride points info.ptr at the buffer end; a raw memcpy would
    # read host memory out of bounds
    host = numpy.arange(5, dtype=numpy.float32)[::-1]

    result = asnumpy.ndarray.from_numpy(host)

    assert result.shape == host.shape
    numpy.testing.assert_array_equal(result.to_numpy(), host)


def test_public_ndarray_constructor_keeps_source_valid():
    expected = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
    source = asnumpy.ndarray.from_numpy(expected)

    copied = asnumpy.ndarray(source)

    numpy.testing.assert_array_equal(source.to_numpy(), expected)
    numpy.testing.assert_array_equal(copied.to_numpy(), expected)


def test_internal_move_constructor_rejects_false_marker():
    expected = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
    raw = _core.ndarray.from_numpy(expected)

    with pytest.raises(ValueError, match="_move must be true"):
        _core.ndarray(raw, _move=False)

    # Rejection happens before ownership is transferred.
    numpy.testing.assert_array_equal(raw.to_numpy(), expected)


def test_repeated_operator_result_lifetime():
    x = asnumpy.ones((1024,), dtype=numpy.float32)

    for _ in range(100):
        result = asnumpy.sin(x)
        assert result.shape == x.shape

    del result
    gc.collect()

    assert x.to_numpy().shape == (1024,)
