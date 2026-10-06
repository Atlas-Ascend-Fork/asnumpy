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


import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np


class HostArray:
    def __init__(self, value):
        self.value = np.asarray(value)

    @classmethod
    def from_numpy(cls, value):
        return cls(value)

    def to_numpy(self):
        return self.value


def load_linalg():
    prefix = "norm_axis_host_pkg"
    modules = {}
    for suffix in ("", ".linalg", "._core", "._core.linalg", "._types", ".utils"):
        module = types.ModuleType(prefix + suffix)
        module.__path__ = []
        modules[module.__name__] = module
    backend = modules[prefix + "._core.linalg"]
    for name in ("det", "inv", "norm", "qr", "slogdet"):
        setattr(backend, name, Mock(side_effect=RuntimeError("unexpected native call")))
    modules[prefix + "._types"].ArrayLike = object
    modules[prefix + "._types"].AxisLike = object
    modules[prefix + ".utils"].ndarray = HostArray
    source = Path(__file__).resolve().parents[2] / "src/asnumpy/linalg/_linalg.py"
    spec = importlib.util.spec_from_file_location(prefix + ".linalg._linalg", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module, backend


class TestNormAxisErrors(unittest.TestCase):
    def setUp(self):
        self.linalg, self.backend = load_linalg()

    def tearDown(self):
        for name in ("det", "inv", "norm", "qr", "slogdet"):
            getattr(self.backend, name).assert_not_called()

    def assert_invalid_axes(self, shape, cases):
        array = np.ones(shape)
        for axis, invalid in cases:
            with self.subTest(shape=shape, axis=axis):
                with self.assertRaises(np.exceptions.AxisError) as caught:
                    self.linalg.norm(array, axis=axis)
                self.assertEqual(caught.exception.axis, invalid)
                self.assertEqual(caught.exception.ndim, array.ndim)

    def test_matrix_out_of_range_axes(self):
        self.assert_invalid_axes((2, 3), [(2, 2), (3, 3), (-3, -3), (-4, -4)])

    def test_sequence_out_of_range_axes(self):
        self.assert_invalid_axes((2, 3), [((0, 2), 2), ([-1, -3], -3)])

    def test_higher_rank_out_of_range_axes(self):
        self.assert_invalid_axes((2, 3, 4), [(3, 3), ((1, 3), 3), ([0, -4], -4)])

    def test_scalar_out_of_range_axes(self):
        self.assert_invalid_axes((), [(0, 0), (-1, -1)])

    def test_valid_axes_match_numpy(self):
        array = np.arange(1, 7, dtype=np.float64).reshape(2, 3)
        for axis in (None, 0, -1, [1], (0, 1), np.int64(1)):
            with self.subTest(axis=axis):
                expected_axis = tuple(axis) if isinstance(axis, list) else axis
                expected = np.linalg.norm(array, axis=expected_axis, keepdims=True)
                actual = self.linalg.norm(array, axis=axis, keepdims=True)
                np.testing.assert_array_equal(actual.to_numpy(), expected)

    def test_duplicate_axes_remain_value_error(self):
        with self.assertRaisesRegex(ValueError, "Duplicate axes"):
            self.linalg.norm(np.ones((2, 3)), axis=(0, -2))


if __name__ == "__main__":
    unittest.main()
