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

"""Host NumPy oracle for the real Python logical reduction wrapper."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy
import pytest


class HostUtils(types.ModuleType):
    """Minimal utility implementation owned by this host backend."""

    ndarray = staticmethod(numpy.asarray)
    _ensure_ndarray = staticmethod(numpy.asarray)

    @staticmethod
    def _convert_dtype(dtype):
        return numpy.dtype(dtype)


def reduce_all(values, axis=None, keepdims=False):
    native_axes = tuple(axis) if isinstance(axis, list) else axis
    return numpy.all(values, axis=native_axes, keepdims=keepdims)


def reduce_any(values, axis=None, keepdims=False):
    native_axes = tuple(axis) if isinstance(axis, list) else axis
    return numpy.any(values, axis=native_axes, keepdims=keepdims)


@pytest.fixture
def logic(monkeypatch):
    """Replace native kernels, retaining the public wrapper's argument dispatch."""
    package = "logic_keepdims_host"
    module_names = (
        package,
        f"{package}._core",
        f"{package}._core.logic",
        f"{package}._types",
        f"{package}._ufunc",
    )
    modules = {}
    for name in module_names:
        modules[name] = types.ModuleType(name)
    modules[f"{package}.utils"] = HostUtils(f"{package}.utils")
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    native = modules.get(f"{package}._core.logic")
    assert native is not None
    for name in (
        "all",
        "any",
        "equal",
        "greater",
        "greater_equal",
        "isfinite",
        "isinf",
        "isneginf",
        "isposinf",
        "less",
        "less_equal",
        "logical_and",
        "logical_not",
        "logical_or",
        "logical_xor",
        "not_equal",
    ):
        setattr(native, name, getattr(numpy, name))

    native.all = reduce_all
    native.any = reduce_any
    type_module = modules.get(f"{package}._types")
    assert type_module is not None
    type_module.ArrayLike = type_module.AxisLike = type_module.DTypeLike = object

    def unused_ufunc(*args, **kwargs):
        return None

    ufunc_module = modules.get(f"{package}._ufunc")
    assert ufunc_module is not None
    ufunc_module.create_ufunc = unused_ufunc
    path = Path(__file__).resolve().parents[2] / "src/asnumpy/logic.py"
    spec = importlib.util.spec_from_file_location(f"{package}.logic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("operation", ["all", "any"])
@pytest.mark.parametrize("shape", [(), (6,), (2, 3), (2, 1, 3)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_global_reduction_matches_numpy(logic, operation, shape, keepdims):
    values = numpy.array(1, dtype=numpy.int32) if not shape else numpy.arange(6).reshape(shape)
    expected = getattr(numpy, operation)(values, keepdims=keepdims)
    result = getattr(logic, operation)(values, keepdims=keepdims)
    assert result.shape == expected.shape
    numpy.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("operation", ["all", "any"])
@pytest.mark.parametrize("axis", [(0,), (0, 1)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_explicit_axes_match_numpy(logic, operation, axis, keepdims):
    values = numpy.arange(6).reshape(2, 3)
    expected = getattr(numpy, operation)(values, axis=axis, keepdims=keepdims)
    result = getattr(logic, operation)(values, axis=axis, keepdims=keepdims)
    assert result.shape == expected.shape
    numpy.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("operation", ["all", "any"])
@pytest.mark.parametrize("keepdims", [False, True])
def test_empty_global_reduction_matches_numpy(logic, operation, keepdims):
    values = numpy.empty((0, 3), dtype=numpy.bool_)
    expected = getattr(numpy, operation)(values, keepdims=keepdims)
    result = getattr(logic, operation)(values, keepdims=keepdims)
    assert result.shape == expected.shape
    numpy.testing.assert_array_equal(result, expected)
