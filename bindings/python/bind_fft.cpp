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

#include <asnumpy/fft/fft.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_fft(py::module_& fft) {
    fft.doc() = "fft module of asnumpy";

    fft.def(
        "fft", [](const NPUArray& x) { return asnumpy::fft::Fft1DC2C(x, /*inverse=*/false); }, py::arg("x"),
        "One-dimensional complex-to-complex discrete Fourier transform of an NPU complex64 array.");

    fft.def(
        "ifft", [](const NPUArray& x) { return asnumpy::fft::Fft1DC2C(x, /*inverse=*/true); }, py::arg("x"),
        "One-dimensional complex-to-complex inverse discrete Fourier transform, scaled by 1/n.");
}
