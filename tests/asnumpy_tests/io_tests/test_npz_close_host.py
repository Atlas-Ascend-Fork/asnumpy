# Copyright (c) 2026 Ethan Zou.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real NPZ I/O and Python reference ownership; the NPU conversion boundary is substituted."""

import gc
import importlib.util
import io
import itertools
import sys
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np


class HostArray:
    conversions = 0

    def __init__(self, data):
        self.data = data.copy()

    @classmethod
    def from_numpy(cls, data):
        cls.conversions += 1
        return cls(data)


def load_io_source():
    package_name = "_asnp_npz_close_fixture"
    package = types.ModuleType(package_name)
    package.__path__ = []
    core = types.ModuleType(package_name + "._core")
    core.ndarray = HostArray
    source = Path(__file__).resolve().parents[3] / "src/asnumpy/io.py"
    spec = importlib.util.spec_from_file_location(package_name + ".io", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {package_name: package, core.__name__: core}):
        spec.loader.exec_module(module)
    return module


class TestNpzCloseOwnership(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_io_source()

    def open_archive(self, compressed):
        buffer = io.BytesIO()
        save = np.savez_compressed if compressed else np.savez
        save(buffer, left=np.array([1, 2], dtype=np.int32), right=np.array([3, 4]))
        buffer.seek(0)
        loaded = self.module.load(buffer)
        self.addCleanup(loaded.close)
        return loaded

    def test_cache_only_arrays_released(self):
        for compressed, exit_kind in itertools.product(
            (False, True), ("close", "context", "exception")
        ):
            with self.subTest(compressed=compressed, exit=exit_kind):
                loaded = self.open_archive(compressed)
                references = []
                for name, values in (("left", [1, 2]), ("right", [3, 4])):
                    array = loaded[name]
                    np.testing.assert_array_equal(array.data, values)
                    references.append(weakref.ref(array))
                del array
                self.assertTrue(all(reference() is not None for reference in references))
                self.close_archive(loaded, exit_kind)
                gc.collect()
                self.assertTrue(all(reference() is None for reference in references))
                loaded.close()

    def close_archive(self, loaded, exit_kind):
        if exit_kind == "close":
            loaded.close()
        elif exit_kind == "context":
            with loaded:
                pass
        else:
            with self.assertRaisesRegex(RuntimeError, "consumer failed"), loaded:
                raise RuntimeError("consumer failed")

    def test_caller_owned_array_survives_close(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                loaded = self.open_archive(compressed)
                kept = loaded["left"]
                self.assertIs(loaded["left"], kept)
                reference = weakref.ref(kept)
                loaded.close()
                np.testing.assert_array_equal(kept.data, [1, 2])
                with self.assertRaises((AttributeError, ValueError)):
                    loaded["left"]
                del kept
                gc.collect()
                self.assertIsNone(reference())

    def test_unread_archive_stays_lazy(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                before = HostArray.conversions
                loaded = self.open_archive(compressed)
                self.assertEqual(loaded.files, ["left", "right"])
                with self.assertRaises(KeyError):
                    loaded["missing"]
                loaded.close()
                self.assertEqual(HostArray.conversions, before)


if __name__ == "__main__":
    unittest.main()
