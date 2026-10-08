# Copyright (c) 2026 ISE Group at Harbin Institute of Technology. All Rights Reserved.
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

from asnumpy.testing._helper import shaped_random


def test_explicit_seed_preserves_callers_global_sequence():
    state = np.random.get_state()
    try:
        np.random.seed(123)
        expected = np.random.random(5)
        np.random.seed(123)
        generated = shaped_random((2, 3), seed=7)
        np.testing.assert_array_equal(generated, np.random.RandomState(7).random((2, 3)))
        np.testing.assert_array_equal(np.random.random(5), expected)
    finally:
        np.random.set_state(state)


def test_unseeded_calls_still_consume_global_generator():
    state = np.random.get_state()
    try:
        np.random.seed(123)
        expected = np.random.random(5)
        np.random.seed(123)
        np.testing.assert_array_equal(shaped_random((5,)), expected)
        assert not np.array_equal(shaped_random((5,)), expected)
    finally:
        np.random.set_state(state)
