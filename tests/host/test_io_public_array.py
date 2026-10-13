# *****************************************************************************
# Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
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

"""Host regressions for choosing the public array wrapper after file loading."""

import importlib.util
import io
import sys
import types
from pathlib import Path

import numpy
import pytest


@pytest.fixture
def io_module(monkeypatch):
    class CoreArray:
        def __init__(self, values):
            self.values = values

        @classmethod
        def from_numpy(cls, values):
            return cls(values)

        def to_numpy(self):
            return self.values

    class PublicArray(CoreArray):
        pass

    package = types.ModuleType("asnumpy")
    package.__path__ = []
    core = types.ModuleType("asnumpy._core")
    core.ndarray = CoreArray
    utils = types.ModuleType("asnumpy.utils")
    utils.ndarray = PublicArray
    for name, module in (("asnumpy", package), ("asnumpy._core", core), ("asnumpy.utils", utils)):
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[2] / "src/asnumpy/io.py"
    spec = importlib.util.spec_from_file_location("asnumpy.io", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, CoreArray, PublicArray


@pytest.mark.parametrize("archive", [False, True])
def test_loaded_arrays_use_public_wrapper(io_module, archive):
    module, _, public = io_module
    values = numpy.arange(6, dtype=numpy.float32).reshape(2, 3)
    buffer = io.BytesIO()
    if archive:
        numpy.savez(buffer, values=values)
    else:
        numpy.save(buffer, values)
    buffer.seek(0)
    loaded = module.load(buffer)
    if archive:
        with loaded:
            result = loaded["values"]
            assert isinstance(result, public)
            assert loaded["values"] is result
    else:
        result = loaded
        assert isinstance(result, public)
    numpy.testing.assert_array_equal(result.to_numpy(), values)


def test_save_still_accepts_native_core_arrays(io_module):
    module, core, _ = io_module
    values = numpy.arange(3, dtype=numpy.float32)
    buffer = io.BytesIO()
    module.save(buffer, core.from_numpy(values))
    buffer.seek(0)
    numpy.testing.assert_array_equal(numpy.load(buffer), values)
