# *****************************************************************************
# Copyright (c) 2026 Ethan_Zou. All Rights Reserved.
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

"""Host dtype/dispatch regressions, with only NPU conversion substituted."""

import importlib.util
import itertools
import sys
import types
import unittest
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[2] / "src" / "asnumpy" / "_ufunc.py"
SHAPES = ((), (0,), (3,), (2, 3))
SCALARS = (1.25 - 2.5j, -0.5 + 3j, complex(float("nan"), 1))


class HostArray:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.dtype = self.value.dtype


def host_add(a, b, dtype=None):
    return np.add(a, b, dtype=dtype)


def load_dispatcher():
    name = "cann_weak_complex_host"
    package = types.ModuleType(name)
    package.__path__ = []
    utils = types.ModuleType(f"{name}.utils")
    utils.ndarray = HostArray
    spec = importlib.util.spec_from_file_location(f"{name}._ufunc", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    sys.modules[utils.__name__] = utils
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        sys.modules.pop(utils.__name__, None)
        raise
    return module, name


class TestWeakComplexScalarHost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module, cls.package_name = load_dispatcher()
        codes = ("e", "f", "d", "F", "D", "?")
        table = tuple(
            (f"{a}{b}->{np.result_type(np.dtype(a), np.dtype(b)).char}", host_add)
            for a, b in itertools.product(codes, repeat=2)
        )
        cls.add = cls.module.create_ufunc("host_fixture_add", table)

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop(cls.package_name, None)
        sys.modules.pop(f"{cls.package_name}.utils", None)

    def check_dispatch(self, dtype, shape, scalar, reverse):
        x = np.resize(np.array([0.25, 1.5, -2.0]), int(np.prod(shape)))
        x = x.astype(dtype).reshape(shape)
        before = x.tobytes()
        args = (scalar, x) if reverse else (x, scalar)
        expected = np.add(*args)
        actual = self.add(*args).value
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(x.tobytes(), before)

    def test_direct_complex_coercion_matches_numpy(self):
        for dtype, scalar in itertools.product((np.float16, np.float32, np.float64), SCALARS):
            with self.subTest(dtype=dtype, scalar=scalar):
                expected_dtype = np.add(np.array(1, dtype=dtype), scalar).dtype
                result = self.module._coerce_weak_scalar(scalar, np.dtype(dtype))
                self.assertEqual(result.dtype, expected_dtype)
                np.testing.assert_array_equal(result, np.array(scalar, dtype=expected_dtype))

    def test_real_arrays_and_python_complex_use_registered_dispatch(self):
        cases = itertools.product(
            (np.float16, np.float32, np.float64), SHAPES, SCALARS, (False, True)
        )
        for dtype, shape, scalar, reverse in cases:
            with self.subTest(dtype=dtype, shape=shape, scalar=scalar, reverse=reverse):
                self.check_dispatch(dtype, shape, scalar, reverse)

    def test_numpy_complex_scalars_keep_strong_precision(self):
        cases = itertools.product(
            (np.float16, np.float32, np.float64),
            SHAPES,
            (np.complex64(1 + 2j), np.complex128(1 + 2j)),
            (False, True),
        )
        for dtype, shape, scalar, reverse in cases:
            with self.subTest(dtype=dtype, shape=shape, scalar_dtype=scalar.dtype, reverse=reverse):
                self.check_dispatch(dtype, shape, scalar, reverse)

    def test_existing_python_float_integer_boolean_controls(self):
        cases = itertools.product(
            (np.float16, np.float32, np.float64), SHAPES, (2, 1.5, True), (False, True)
        )
        for dtype, shape, scalar, reverse in cases:
            with self.subTest(dtype=dtype, shape=shape, scalar=scalar, reverse=reverse):
                self.check_dispatch(dtype, shape, scalar, reverse)


if __name__ == "__main__":
    unittest.main()
