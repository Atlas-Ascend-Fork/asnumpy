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

"""Tests for the NumPy array protocol implemented by asnumpy arrays.

``__array__`` / ``__iter__`` / ``__len__`` are what let ``numpy.asarray`` and every library built on
it -- pandas in particular, see ``test_pandas_interop.py`` -- consume an asnumpy array without an
explicit ``to_numpy()`` call. The protocol lives on the core ``NPUArray`` type, so both the public
``asnumpy.ndarray`` and the raw ``asnumpy._core.ndarray`` (e.g. what ``asnumpy.load()`` returns)
support it.
"""

import numpy as np
import pytest
from asnumpy._core import ndarray as core_ndarray

import asnumpy as ap


def _arr(host):
    return ap.ndarray.from_numpy(np.asarray(host))


# ========== __array__ ==========


@pytest.mark.parametrize(
    "host",
    [
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        np.array(7.0, dtype=np.float32),
    ],
    ids=["1d", "2d", "0d"],
)
def test_asarray_preserves_shape_and_dtype(host):
    got = np.asarray(_arr(host))
    assert isinstance(got, np.ndarray)
    assert got.shape == host.shape
    assert got.dtype == host.dtype
    np.testing.assert_array_equal(got, host)


def test_asarray_honours_requested_dtype():
    host = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    got = np.asarray(_arr(host), dtype=np.float64)
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, host.astype(np.float64))


def test_numpy_array_copy_keyword_is_accepted():
    """NumPy 2.x passes ``copy`` to ``__array__``; omitting the parameter raises a warning/error."""
    host = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert np.array(_arr(host), copy=True).shape == host.shape
    assert np.array(_arr(host), copy=False).shape == host.shape


# ========== __iter__ ==========


def test_iteration_yields_rows_of_a_2d_array():
    host = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rows = list(_arr(host))
    assert len(rows) == 2
    for got, expected in zip(rows, host, strict=True):
        assert isinstance(got, np.ndarray)
        np.testing.assert_array_equal(got, expected)


def test_iteration_yields_scalars_of_a_1d_array():
    host = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(list(_arr(host))), host)


def test_iteration_of_a_0d_array_raises():
    with pytest.raises(TypeError):
        iter(_arr(np.float32(1.0)))


# ========== __len__ ==========


def test_len_is_the_first_dimension():
    assert len(_arr(np.zeros((4, 3), dtype=np.float32))) == 4
    assert len(_arr(np.zeros((7,), dtype=np.float32))) == 7


def test_len_of_a_0d_array_raises_type_error():
    """NumPy raises TypeError here, not 0 -- the protocol must match."""
    with pytest.raises(TypeError):
        len(_arr(np.float32(1.0)))


# ========== core type coverage ==========


@pytest.mark.parametrize("name", ["__array__", "__iter__", "__len__"])
def test_protocol_is_defined_on_the_core_type(name):
    """The raw ``_core.ndarray`` must support the protocol too, not just the public subclass."""
    assert name in dir(core_ndarray)
    assert name in dir(ap.ndarray)
