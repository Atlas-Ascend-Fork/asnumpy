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

    fft.def("fft_c2c", &asnumpy::fft::FftC2C, py::arg("x"), py::arg("axis"), py::arg("n"), py::arg("inverse"),
            py::arg("scale"),
            "Complex-to-complex FFT along one axis, batched over the remaining dimensions. "
            "The caller passes an already-normalised non-negative axis, the transform length and the "
            "single scale factor applied to the backend result.");

    fft.def("rfft", &asnumpy::fft::Rfft, py::arg("x"), py::arg("axis"), py::arg("n"), py::arg("scale"),
            "Real-to-complex FFT along one axis; the transformed axis becomes n/2+1 complex bins.");

    fft.def("irfft", &asnumpy::fft::Irfft, py::arg("x"), py::arg("axis"), py::arg("n"), py::arg("scale"),
            "Complex-to-real inverse FFT along one axis; the spectrum axis must be Hermitian and the "
            "output axis has length n.");

    fft.def("fft2_c2c", &asnumpy::fft::Fft2C2C, py::arg("x"), py::arg("axis0"), py::arg("axis1"), py::arg("n0"),
            py::arg("n1"), py::arg("inverse"), py::arg("scale0"), py::arg("scale1"),
            "Two-dimensional complex-to-complex FFT over two axes, implemented as two one-dimensional "
            "ops-fft passes (the native fft2_d kernel is limited to 32/64/128 on Ascend 910B).");
}
