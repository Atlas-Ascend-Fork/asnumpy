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

"""Tests for passing asnumpy arrays directly to pandas.

Regression tests for issue #24: pandas reported asnumpy arrays as an unrecognized type because the
core ``NDArray`` implemented neither ``__array__`` nor the iteration protocol. pandas uses
``__iter__``/``__len__`` to recognize a value as array-like and ``__array__`` to materialize it, so
all three are required for ``pd.Series``/``pd.DataFrame`` to accept an asnumpy array directly.
"""

import numpy as np
import pytest

import asnumpy as ap

pd = pytest.importorskip("pandas")


def _arr(host):
    return ap.ndarray.from_numpy(np.asarray(host))


def test_series_from_1d_array():
    host = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    series = pd.Series(_arr(host))
    assert series.dtype == np.float32
    np.testing.assert_array_equal(series.to_numpy(), host)


def test_dataframe_from_2d_array():
    host = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    frame = pd.DataFrame(_arr(host))
    assert frame.shape == host.shape
    np.testing.assert_array_equal(frame.to_numpy(), host)


def test_dataframe_from_dict_of_arrays():
    host = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    frame = pd.DataFrame({"a": _arr(host), "b": _arr(host * 2.0)})
    assert list(frame.columns) == ["a", "b"]
    np.testing.assert_array_equal(frame["a"].to_numpy(), host)
    np.testing.assert_array_equal(frame["b"].to_numpy(), host * 2.0)


def test_series_rejects_2d_array_like_numpy():
    with pytest.raises(ValueError):
        pd.Series(_arr(np.zeros((2, 3), dtype=np.float32)))


def test_pandas_reduction_uses_numpy_semantics():
    host = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert pd.Series(_arr(host)).sum() == pytest.approx(6.0)
