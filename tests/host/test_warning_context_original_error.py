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

from asnumpy.testing._assertions import assert_no_warnings, assert_warns


@pytest.mark.parametrize("kind", ["required", "forbidden"])
def test_body_exception_keeps_identity_and_restores_filters(kind):
    before = list(warnings.filters)
    error = ValueError("original failure")
    context = assert_warns(UserWarning) if kind == "required" else assert_no_warnings(None)
    with pytest.raises(ValueError) as caught:
        with context:
            if kind == "forbidden":
                warnings.warn("fixture", UserWarning, stacklevel=1)
            raise error
    assert caught.value is error
    assert warnings.filters == before


def test_normal_completion_still_checks_warnings():
    with pytest.raises(AssertionError, match="no such warning"):
        with assert_warns(UserWarning):
            pass
    with pytest.raises(AssertionError, match="Expected no warnings"):
        with assert_no_warnings(None):
            warnings.warn("fixture", UserWarning, stacklevel=1)
