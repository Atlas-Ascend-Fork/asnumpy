# Copyright (c) 2026 Ethan_Zou
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

import numpy as np
import pytest

from asnumpy.testing._loops import numpy_asnumpy_allclose, numpy_asnumpy_array_equal


@pytest.mark.parametrize("factory", [numpy_asnumpy_array_equal, numpy_asnumpy_allclose])
def test_disabled_type_check_compares_actual_values(factory):
    def matching(xp):
        dtype = np.float32 if xp is np else np.float64
        return np.array([1, 2], dtype=dtype)

    factory(type_check=False)(matching)()
    with pytest.raises(AssertionError, match="Dtype mismatch"):
        factory(type_check=True)(matching)()


@pytest.mark.parametrize("factory", [numpy_asnumpy_array_equal, numpy_asnumpy_allclose])
def test_disabled_type_check_keeps_value_and_shape_checks(factory):
    def different(xp):
        return np.array([1, 2] if xp is np else [1, 3], dtype=np.float64)

    with pytest.raises(AssertionError):
        factory(type_check=False)(different)()


@pytest.mark.parametrize("factory", [numpy_asnumpy_array_equal, numpy_asnumpy_allclose])
def test_disabled_type_check_keeps_shape_and_strides(factory):
    def different_shape(xp):
        return np.ones((2,) if xp is np else (1, 2), dtype=np.float64)

    with pytest.raises(AssertionError, match="Shape mismatch"):
        factory(type_check=False)(different_shape)()

    def different_strides(xp):
        return np.ones(4)[::2] if xp is np else np.ones(2)

    with pytest.raises(AssertionError, match="Strides mismatch"):
        factory(type_check=False, strides_check=True)(different_strides)()


def test_public_assertions_keep_positional_api_and_diagnostics():
    from asnumpy.testing._array import assert_allclose, assert_array_equal

    with pytest.raises(AssertionError, match="context\nArrays are not equal"):
        assert_array_equal([True, False], [False, False], "context", True, False)
    with pytest.raises(AssertionError, match="context\nArrays are not almost equal"):
        assert_allclose([1.0], [2.0], 1e-7, 0, "context", False, False)
    assert_allclose([np.nan], [np.nan])
