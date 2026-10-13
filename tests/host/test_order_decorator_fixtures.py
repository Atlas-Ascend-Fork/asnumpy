# *****************************************************************************
# Copyright (c) 2026 AISS Group at Harbin Institute of Technology. All Rights Reserved.
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

import os
import subprocess
import sys
from pathlib import Path

import pytest

from asnumpy.testing._loops import for_orders


@pytest.mark.parametrize("kind", ["function", "method", "fixture"])
def test_pytest_does_not_require_internally_supplied_order(tmp_path, kind):
    source_root = Path(__file__).resolve().parents[2] / "src"
    prefix = "import sys, types\nfrom pathlib import Path\n"
    prefix += "root = Path(" + repr(str(source_root)) + ")\n"
    prefix += "for name in ('asnumpy', 'asnumpy.testing'):\n"
    prefix += "    package = types.ModuleType(name)\n"
    prefix += "    package.__path__ = [str(root.joinpath(*name.split('.')))]\n"
    prefix += "    sys.modules[name] = package\n"
    prefix += "from asnumpy.testing._loops import for_orders\nimport pytest\n"
    bodies = {
        "function": "@for_orders(['C', 'F'])\ndef test_order(order):\n    assert order in ('C', 'F')\n",
        "method": "class TestOrders:\n    @for_orders(['C', 'F'])\n    def test_order(self, order):\n        assert order in ('C', 'F')\n",
        "fixture": "@pytest.fixture\ndef value():\n    return 42\n@for_orders(['C', 'F'], name='layout')\ndef test_order(layout, value):\n    assert layout in ('C', 'F') and value == 42\n",
    }
    test_file = tmp_path / "test_decorated_orders.py"
    test_file.write_text(prefix + bodies[kind])
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(test_file),
            "--confcutdir=" + str(tmp_path),
            "-o",
            "addopts=",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_direct_call_still_visits_each_order():
    seen = []

    @for_orders(["C", "F"])
    def check(order):
        seen.append(order)

    check()
    assert seen == ["C", "F"]
