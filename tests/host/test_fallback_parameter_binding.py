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

import pytest

from asnumpy.testing import _pytest_impl as module


def test_fallback_plain_function_and_single_value(monkeypatch):
    monkeypatch.setattr(module, "is_available", lambda: False)
    seen = []

    @module.parameterize("value", [2, 3])
    def run(value):
        seen.append(value)

    run()
    assert seen == [2, 3]


@pytest.mark.parametrize("names", ["left, right", ["left", "right"]])
def test_fallback_method_binds_multiple_names(monkeypatch, names):
    monkeypatch.setattr(module, "is_available", lambda: False)
    seen = []

    class Subject:
        @module.parameterize(names, [(2, 3), (4, 5)])
        def run(self, left, right):
            seen.append(left + right)

    Subject().run()
    assert seen == [5, 9]


def test_fallback_rejects_inconsistent_row(monkeypatch):
    monkeypatch.setattr(module, "is_available", lambda: False)

    @module.parameterize("left,right", [(2,)])
    def run(left, right):
        pass

    with pytest.raises(ValueError, match="different lengths"):
        run()
