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

#include <asnumpy/statistics/averages_and_variances.hpp>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace asnumpy {

void bind_statistics(py::module_& statistics) {
    statistics.doc() = "statistics module of asnumpy";
    statistics.def("mean", py::overload_cast<const NPUArray&, int64_t, bool, std::optional<py::dtype>>(&Mean),
                   py::arg("a"), py::arg("axis"), py::arg("keepdims"), py::arg("dtype") = py::none());
    statistics.def("mean", py::overload_cast<const NPUArray&, std::optional<py::dtype>>(&Mean), py::arg("a"),
                   py::arg("dtype") = py::none());

    statistics.def("nanmean", py::overload_cast<const NPUArray&, int64_t, bool, std::optional<py::dtype>>(&Nanmean),
                   py::arg("a"), py::arg("axis"), py::arg("keepdims"), py::arg("dtype") = py::none());
    statistics.def("nanmean", py::overload_cast<const NPUArray&, std::optional<py::dtype>>(&Nanmean), py::arg("a"),
                   py::arg("dtype") = py::none());
}

} // namespace asnumpy
