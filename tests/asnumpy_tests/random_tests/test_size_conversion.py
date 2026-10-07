# Copyright (c) 2026 Ethan_Zou
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shape conversion used by every random distribution wrapper."""

import numpy as np
import pytest

from asnumpy.utils import _convert_size


@pytest.mark.parametrize(
    "dtype", [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]
)
@pytest.mark.parametrize("size", [0, 1, 5])
def test_numpy_integer_size(dtype, size):
    scalar = dtype(size)
    shape = _convert_size(scalar)
    assert shape == (size,)
    assert isinstance(shape, tuple)
    assert type(shape[0]) is int
    # NumPy's real random API accepts the same scalar as a vector size.
    assert np.random.default_rng(0).standard_normal(size=scalar).shape == shape


@pytest.mark.parametrize("size", [0, 1, 5, False, True])
def test_python_integer_size(size):
    assert _convert_size(size) == (int(size),)


@pytest.mark.parametrize("size", [(), (2, 3), (0, 3), [], [2, 3], [np.int64(2), np.int64(3)]])
def test_sequence_size_is_preserved(size):
    assert _convert_size(size) is size


@pytest.mark.parametrize("size", [None, 3.5, "5"])
def test_other_inputs_are_preserved_for_binding_validation(size):
    assert _convert_size(size) is size
