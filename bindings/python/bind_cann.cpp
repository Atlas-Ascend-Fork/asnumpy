/******************************************************************************
 * Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include <asnumpy/cann/driver.hpp>
#include <acl/acl.h>
#include <algorithm>
#include <pybind11/pybind11.h>

void bind_cann(pybind11::module_& cann) {
    cann.doc() = "cann module of asnumpy";
    cann.def("set_device", &aclrtSetDevice, pybind11::arg("device_id"));
    cann.def("reset_device", &aclrtResetDevice, pybind11::arg("device_id"));
    cann.def("reset_device_force", &aclrtResetDeviceForce, pybind11::arg("device_id"));
    cann.def("get_device_count", &asnumpy::cann::get_device_count);
    cann.def("init", &asnumpy::cann::init);
    cann.def("finalize", &asnumpy::cann::finalize);
}
