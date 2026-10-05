# *****************************************************************************
# Copyright (c) 2025 ISE Group at Harbin Institute of Technology. All Rights Reserved.
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

"""Host regressions for NumPy-compatible ufunc output destinations."""

import importlib.util
from pathlib import Path

import numpy
import pytest

path = Path(__file__).resolve().parents[2] / "src/asnumpy/_ufunc.py"
spec = importlib.util.spec_from_file_location("asnumpy._ufunc", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ufunc = module.ufunc


class TestUfuncOut:
    @staticmethod
    @pytest.mark.parametrize("as_tuple", [False, True])
    def test_single_output_returns_destination(as_tuple):
        values = numpy.array([1.0, 2.0], dtype=numpy.float32)
        expected = numpy.empty_like(values)
        target = numpy.empty_like(values)
        expected_out = (expected,) if as_tuple else expected
        out = (target,) if as_tuple else target
        oracle = numpy.add(values, 0, out=expected_out)
        result = ufunc._write_out(out, values)
        assert oracle is expected
        assert result is target
        numpy.testing.assert_array_equal(target, expected)

    @staticmethod
    def test_none_slot_returns_allocated_result():
        values = numpy.array([1.0, 2.0], dtype=numpy.float32)
        oracle = numpy.add(values, 0, out=(None,))
        result = ufunc._write_out((None,), values)
        assert result is values
        numpy.testing.assert_array_equal(result, oracle)

    @staticmethod
    @pytest.mark.parametrize("extra", [False, True])
    def test_invalid_output_count_does_not_write(extra):
        values = numpy.array([1.0, 2.0], dtype=numpy.float32)
        target = numpy.full_like(values, -1)
        before = target.copy()
        out = (target, target) if extra else ()
        with pytest.raises(ValueError):
            ufunc._write_out(out, values)
        numpy.testing.assert_array_equal(target, before)
