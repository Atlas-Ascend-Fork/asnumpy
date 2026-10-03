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
