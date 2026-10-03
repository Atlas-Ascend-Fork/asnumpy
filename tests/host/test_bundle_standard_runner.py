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

import io
import unittest

from asnumpy.testing._bundle import TestBundle as Bundle


def test_bundle_runs_with_standard_text_runner():
    class Sample(unittest.TestCase):
        def test_value(self):
            self.assertEqual(1 + 1, 2)

    output = io.StringIO()
    results = Bundle([Sample]).run(unittest.TextTestRunner(stream=output))
    assert len(results) == 1
    assert results[0].wasSuccessful()
    assert results[0].testsRun == 1


def test_custom_combined_loader_runner_remains_supported():
    class Sample(unittest.TestCase):
        def test_value(self):
            pass

    class Runner(unittest.TestLoader):
        @staticmethod
        def run(suite):
            result = unittest.TestResult()
            suite.run(result)
            return result

    assert Bundle([Sample]).run(Runner())[0].testsRun == 1
