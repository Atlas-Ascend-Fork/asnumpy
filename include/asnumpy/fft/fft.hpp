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

#include <asnumpy/utils/npu_array.hpp>

namespace asnumpy {
namespace fft {

/**
 * @brief One-dimensional complex-to-complex FFT, backed by CANN ops-fft.
 *
 * Equivalent to numpy.fft.fft(x) / numpy.fft.ifft(x) for a one-dimensional
 * complex64 array. Only the transform itself runs on the NPU: ops-fft's public
 * aclfftExecC2C contract takes host pointers and performs the Device transfers
 * and the stream synchronisation internally, so this wrapper copies the input
 * to the host, executes the plan and copies the result back to a new NPU array.
 *
 * The inverse transform is normalised by 1/N here: ops-fft returns the
 * unnormalised result (ACLFFT_BACKWARD == N * ifft).
 *
 * @param x One-dimensional complex64 array on the NPU.
 * @param inverse false for the forward transform, true for the inverse transform.
 * @return NPUArray One-dimensional complex64 array on the NPU, same length as x.
 * @throws std::invalid_argument for unsupported rank, dtype or length.
 * @throws std::runtime_error when the backend is unavailable or the plan fails.
 */
NPUArray Fft1DC2C(const NPUArray& x, bool inverse);

} // namespace fft
} // namespace asnumpy
