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

#include <asnumpy/logic/logic.hpp>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace asnumpy;

void bind_logic(py::module_& logic) {
    logic.doc() = "logic module of asnumpy";

    // Reduction-like operations
    logic.def("all", py::overload_cast<const NPUArray&>(&All), py::arg("x"));
    logic.def("all", py::overload_cast<const NPUArray&, const std::vector<int64_t>&, bool>(&All), py::arg("x"),
              py::arg("axis"), py::arg("keepdims") = false);
    logic.def("any", py::overload_cast<const NPUArray&>(&Any), py::arg("x"));
    logic.def("any", py::overload_cast<const NPUArray&, const std::vector<int64_t>&, bool>(&Any), py::arg("x"),
              py::arg("axis"), py::arg("keepdims") = false);

    // Finite / infinite checks
    logic.def("isfinite", &IsFinite, py::arg("x"));
    logic.def("isinf", &IsInf, py::arg("x"));
    logic.def("isneginf", &IsNegInf, py::arg("x"));
    logic.def("isposinf", &IsPosInf, py::arg("x"));

    // Logical operators
    logic.def("logical_and", &LogicalAnd, py::arg("x1"), py::arg("x2"));
    logic.def("logical_or", &LogicalOr, py::arg("x1"), py::arg("x2"));
    logic.def("logical_not", &LogicalNot, py::arg("x"));
    logic.def("logical_xor", &LogicalXor, py::arg("x1"), py::arg("x2"));

    // Comparisons (array-array / array-scalar), dtype optional
    logic.def("greater", py::overload_cast<const NPUArray&, const NPUArray&, std::optional<py::dtype>>(&greater),
              py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());

    logic.def("greater", py::overload_cast<const NPUArray&, const py::object&, std::optional<py::dtype>>(&greater),
              py::arg("x1"), py::arg("scalar"), py::arg("dtype") = py::none());

    logic.def("greater_equal",
              py::overload_cast<const NPUArray&, const NPUArray&, std::optional<py::dtype>>(&greater_equal),
              py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());

    logic.def("greater_equal",
              py::overload_cast<const NPUArray&, const py::object&, std::optional<py::dtype>>(&greater_equal),
              py::arg("x1"), py::arg("scalar"), py::arg("dtype") = py::none());

    // Less comparisons
    logic.def("less", py::overload_cast<const NPUArray&, const NPUArray&, std::optional<py::dtype>>(&less),
              py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());

    logic.def("less", py::overload_cast<const NPUArray&, const py::object&, std::optional<py::dtype>>(&less),
              py::arg("x1"), py::arg("scalar"), py::arg("dtype") = py::none());

    logic.def("less_equal", py::overload_cast<const NPUArray&, const NPUArray&, std::optional<py::dtype>>(&less_equal),
              py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());

    logic.def("less_equal",
              py::overload_cast<const NPUArray&, const py::object&, std::optional<py::dtype>>(&less_equal),
              py::arg("x1"), py::arg("scalar"), py::arg("dtype") = py::none());

    // Equal / Not equal comparisons
    logic.def("equal", py::overload_cast<const NPUArray&, const NPUArray&, std::optional<py::dtype>>(&equal),
              py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());

    logic.def("equal", py::overload_cast<const NPUArray&, const py::object&, std::optional<py::dtype>>(&equal),
              py::arg("x1"), py::arg("scalar"), py::arg("dtype") = py::none());

    logic.def("not_equal", py::overload_cast<const NPUArray&, const NPUArray&, std::optional<py::dtype>>(&not_equal),
              py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());

    logic.def("not_equal", py::overload_cast<const NPUArray&, const py::object&, std::optional<py::dtype>>(&not_equal),
              py::arg("x1"), py::arg("scalar"), py::arg("dtype") = py::none());

    logic.def("where", &where, py::arg("condition"), py::arg("x1"), py::arg("x2"));
}
