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

import warnings

import pytest

from asnumpy.testing._assertions import assert_no_warnings


def test_documented_context_accepts_no_argument():
    with assert_no_warnings():
        value = 3 + 4
    assert value == 7


def test_context_rejects_an_actual_warning():
    with pytest.raises(AssertionError, match="Expected no warnings"):
        with assert_no_warnings():
            warnings.warn("fixture", UserWarning, stacklevel=1)


def test_callable_path_retains_return_value():
    assert assert_no_warnings(lambda value: value + 1, 4) == 5
