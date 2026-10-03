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

import inspect

import numpy as np

from asnumpy.testing._loops import for_dtypes, for_orders


def test_dtype_decorator_accepts_bound_method_and_keyword():
    seen = []

    class Subject:
        @for_dtypes([np.float32, np.float64])
        def run(self, value, dtype):
            seen.append((value, dtype))

    Subject().run(value=4)
    assert seen == [(4, np.float32), (4, np.float64)]


def test_external_signature_keeps_fixture_parameters():
    @for_dtypes([np.float32])
    def run(external_value, dtype):
        return external_value

    assert list(inspect.signature(run).parameters) == ["external_value"]


def test_order_loop_can_supply_dtype_loop():
    seen = []

    @for_orders(["C", "F"])
    @for_dtypes([np.float32, np.float64])
    def run(order, dtype):
        seen.append((order, dtype))

    run()
    assert seen == [(order, dtype) for order in ["C", "F"] for dtype in [np.float32, np.float64]]
