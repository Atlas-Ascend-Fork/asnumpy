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

#include <asnumpy/array/basic.hpp>
#include <asnumpy/array/manipulation.hpp>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace asnumpy;

namespace {

// Collect the operands of a variadic op without copying them: asnumpy.ndarray instances hold their
// NPUArray by value, so casting to a reference reaches the device tensor directly. Copying the
// NPUArray (as a by-value std::vector<NPUArray> binding would) deep-copies device memory.
std::vector<const NPUArray*> ResolveOperands(const py::sequence& arrays, const char* op) {
    std::vector<const NPUArray*> operands;
    operands.reserve(py::len(arrays));
    for (auto item : arrays) {
        operands.push_back(&item.cast<const NPUArray&>());
    }
    if (operands.empty()) {
        throw std::invalid_argument(std::string("[bind_array.cpp](bind_array) ") + op + " needs at least one array");
    }
    return operands;
}

} // namespace

void bind_array(pybind11::module_& array) {
    array.doc() = "array module of asnumpy";
    array.def("zeros", &Zeros, py::arg("shape"), py::arg("dtype"));
    array.def("zeros_like", &Zeros_like, py::arg("other"), py::arg("dtype"));
    array.def("full", &Full, py::arg("shape"), py::arg("value"), py::arg("dtype"));
    array.def("full_like", &Full_like, py::arg("other"), py::arg("value"), py::arg("dtype"));
    array.def("empty", &Empty, py::arg("shape"), py::arg("dtype"));
    array.def("empty_like", &EmptyLike, py::arg("prototype"), py::arg("dtype") = py::none());
    array.def("eye", &Eye, py::arg("n"), py::arg("dtype"));
    array.def("ones", &Ones, py::arg("shape"), py::arg("dtype"));
    array.def("ones_like", &ones_like, py::arg("other"), py::arg("dtype"));
    array.def("identity", &Identity, py::arg("n"), py::arg("dtype"));
    array.def("linspace", &Linspace, py::arg("start"), py::arg("end"), py::arg("steps") = 50,
              py::arg("dtype") = py::none());
    // Array manipulation routines (issue #25).
    array.def(
        "concatenate",
        [](const py::sequence& arrays, int64_t axis) {
            return Concatenate(ResolveOperands(arrays, "concatenate"), axis);
        },
        py::arg("arrays"), py::arg("axis") = 0, "Join a sequence of arrays along an existing axis.");
    array.def(
        "stack", [](const py::sequence& arrays, int64_t axis) { return Stack(ResolveOperands(arrays, "stack"), axis); },
        py::arg("arrays"), py::arg("axis") = 0, "Join a sequence of arrays along a new axis.");
    array.def("where", &Where, py::arg("condition"), py::arg("self"), py::arg("other"),
              "Select elements from two arrays according to a boolean condition.");
}
