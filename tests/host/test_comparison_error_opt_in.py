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

import pytest

from asnumpy.testing._loops import numpy_asnumpy_allclose, numpy_asnumpy_array_equal


@pytest.mark.parametrize("factory", [numpy_asnumpy_array_equal, numpy_asnumpy_allclose])
@pytest.mark.parametrize("external", [False, True])
def test_same_callback_errors_require_opt_in(factory, external):
    if external:

        def failing(xp, value):
            raise TypeError("fixture failure")

        kwargs = {"value": 3}
    else:

        def failing(xp):
            raise TypeError("fixture failure")

        kwargs = {}
    with pytest.raises(AssertionError):
        factory(accept_error=False)(failing)(**kwargs)
    factory(accept_error=True)(failing)(**kwargs)
