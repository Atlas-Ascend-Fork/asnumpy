# Copyright (c) 2026 Ethan Zou.
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
"""Check random test data preparation with real NumPy, without the NPU extension."""

import runpy
import unittest
from itertools import product
from pathlib import Path

import numpy as np


class HostArray:
    @staticmethod
    def from_numpy(array):
        """Inspect the CPU array passed to the backend conversion boundary."""
        return array


class HostBackend:
    ndarray = HostArray


class TestShapedRandomDtype(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helper = Path(__file__).resolve().parents[2] / "src/asnumpy/testing/_helper.py"
        cls.shaped_random = staticmethod(runpy.run_path(str(helper))["shaped_random"])

    def setUp(self):
        self.random_state = np.random.get_state()

    def tearDown(self):
        np.random.set_state(self.random_state)

    def test_integer_dtype_is_retained_after_default_scaling(self):
        for dtype, backend in product((np.int32, np.int64, np.uint8), (np, HostBackend)):
            with self.subTest(dtype=dtype, backend=backend):
                array = self.shaped_random((2, 3), dtype=dtype, seed=42, xp=backend)
                self.assertEqual(array.dtype, np.dtype(dtype))
                self.assertEqual(array.shape, (2, 3))

    def test_scaled_integer_data_has_multiple_values(self):
        for dtype, backend in product((np.int32, np.int64, np.uint8), (np, HostBackend)):
            with self.subTest(dtype=dtype, backend=backend):
                array = self.shaped_random((32,), dtype=dtype, scale=10.0, seed=42, xp=backend)
                self.assertGreater(np.unique(array).size, 1)
                self.assertEqual(array.dtype, np.dtype(dtype))
                self.assertTrue(np.all((array >= 0) & (array < 10)))

    def test_float_data_keeps_dtype_shape_and_range(self):
        for dtype, backend in product((np.float32, np.float64), (np, HostBackend)):
            with self.subTest(dtype=dtype, backend=backend):
                array = self.shaped_random((4, 5), dtype=dtype, scale=3.5, seed=42, xp=backend)
                self.assertEqual(array.dtype, np.dtype(dtype))
                self.assertEqual(array.shape, (4, 5))
                self.assertTrue(np.all((array >= 0) & (array < 3.5)))
                self.assertGreater(np.unique(array).size, 1)

    def test_float_seed_reproducibility_and_empty_arrays(self):
        for backend in (np, HostBackend):
            with self.subTest(backend=backend):
                first = self.shaped_random((8,), dtype=np.float64, scale=3.5, seed=42, xp=backend)
                second = self.shaped_random((8,), dtype=np.float64, scale=3.5, seed=42, xp=backend)
                np.testing.assert_array_equal(first, second)
                empty = self.shaped_random((0,), dtype=np.float64, scale=3.5, seed=42, xp=backend)
                self.assertEqual(empty.size, 0)
                self.assertEqual(empty.dtype, np.dtype(np.float64))


if __name__ == "__main__":
    unittest.main()
