# *****************************************************************************
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import numpy as np
import pytest

import asnumpy as ap


@pytest.mark.parametrize("operation", ["set_device", "reset_device", "reset_device_force"])
@pytest.mark.parametrize("device_id", [-1, 2147483647])
def test_device_operations_report_invalid_ids(operation, device_id):
    with pytest.raises(RuntimeError, match=rf"{operation}.*error"):
        getattr(ap, operation)(device_id)


@pytest.mark.parametrize("operation", ["set_device", "reset_device", "reset_device_force"])
@pytest.mark.parametrize("device_id", [None, "invalid"])
def test_device_operations_preserve_binding_type_errors(operation, device_id):
    with pytest.raises(TypeError):
        getattr(ap, operation)(device_id)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.int32])
def test_valid_device_still_supports_array_transfers(dtype):
    assert ap.set_device(0) is None
    host = np.array([1, 2, 3], dtype=dtype)
    device = ap.ndarray.from_numpy(host)
    np.testing.assert_array_equal(device.to_numpy(), host)
