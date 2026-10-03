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

from asnumpy.testing._array import assert_array_equal


def test_scalar_difference_raises_documented_assertion():
    with pytest.raises(AssertionError, match="scalar fixture"):
        assert_array_equal(np.array(1.0), np.array(2.0), err_msg="scalar fixture")


def test_equal_scalar_and_vector_difference():
    assert_array_equal(np.array(1.0), np.array(1.0))
    with pytest.raises(AssertionError, match="Indices where"):
        assert_array_equal(np.array([1, 2]), np.array([1, 3]))
