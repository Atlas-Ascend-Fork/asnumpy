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
 *****************************************************************************/

#pragma once

#include <acl/acl.h>
#include "../utils/npu_array.hpp"
#include "../utils/npu_scalar.hpp"

namespace asnumpy {
NPUArray Zeros(const std::vector<int64_t>& shape, py::dtype dtype);

NPUArray Zeros_like(const NPUArray& other, py::dtype dtype);

NPUArray Full(const std::vector<int64_t>& shape, const py::object& value, py::dtype dtype);

NPUArray Full_like(const NPUArray& other, const py::object& value, py::dtype dtype);

NPUArray Empty(const std::vector<int64_t>& shape, py::dtype dtype);

NPUArray EmptyLike(const NPUArray& prototype, py::dtype dtype = py::none());

NPUArray Eye(int64_t n, py::dtype dtype);

NPUArray Ones(const std::vector<int64_t>& shape, py::dtype dtype);

NPUArray Identity(int64_t n, py::dtype dtype);

NPUArray ones_like(const NPUArray& other, py::dtype dtype);

NPUArray Linspace(const py::object& start, const py::object& end, const py::object& steps, const py::object& dtype,
                  bool endpoint = true);

} // namespace asnumpy
