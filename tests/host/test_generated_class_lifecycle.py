# Copyright (c) 2026 ISE Group at Harbin Institute of Technology. All Rights Reserved.
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

import unittest

from asnumpy.testing._bundle import generate_test_classes


def test_generated_class_preserves_standard_lifecycle():
    events = []

    class Base(unittest.TestCase):
        def setUp(self):
            events.append("setup")

        def tearDown(self):
            events.append("teardown")

        def helper(self):
            return 7

        def test_value(self, value):
            self.assertEqual(value, self.helper())
            events.append("test")

    generated = generate_test_classes(Base, [{"value": 7}])[0]
    result = unittest.TestResult()
    unittest.defaultTestLoader.loadTestsFromTestCase(generated).run(result)
    assert result.wasSuccessful(), result.errors
    assert result.testsRun == 1
    assert events == ["setup", "test", "teardown"]
