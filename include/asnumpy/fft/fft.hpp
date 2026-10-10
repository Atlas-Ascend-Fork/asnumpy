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
 * @brief Complex-to-complex FFT along one axis, batched over the remaining dimensions.
 *
 * Equivalent to numpy.fft.fft(x, n=n, axis=axis) / numpy.fft.ifft(x, n=n, axis=axis) for a
 * complex64 array: the transform runs along `axis` (already a normalised non-negative index) and
 * every other dimension is a batch.
 *
 * The array is gathered on the host into (batch, n) row-major buffers, transformed by one ops-fft
 * plan created with that batch count, and copied back into a new NPU array. Only the transform
 * runs on the NPU: ops-fft's aclfftExecC2C takes host pointers and performs the device transfers
 * and the synchronisation internally, so exactly one host<->device round trip happens here.
 *
 * @param x complex64 input array on the NPU.
 * @param axis non-negative index of the transformed axis.
 * @param n transform length along `axis`; shorter input is truncated, longer input is zero padded.
 * @param inverse false for the forward transform, true for the inverse transform.
 * @param scale factor applied once to the backend result (all NumPy normalisation lives here).
 * @return NPUArray complex64 array whose `axis` dimension has length `n`.
 * @throws std::invalid_argument for an unsupported dtype, axis, or length.
 * @throws std::runtime_error when the backend is unavailable or a plan call fails.
 */
NPUArray FftC2C(const NPUArray& x, int64_t axis, int64_t n, bool inverse, double scale);

/**
 * @brief Real-to-complex FFT along one axis, batched over the remaining dimensions.
 *
 * Equivalent to numpy.fft.rfft(x, n=n, axis=axis) for a float32 array: the transformed axis
 * becomes ``n / 2 + 1`` complex bins (NumPy's ``floor(n/2) + 1``) and every other dimension is a
 * batch. The packing, batching and single host round trip are the same as :func:`FftC2C`.
 *
 * @param x float32 input array on the NPU.
 * @param axis non-negative index of the transformed axis.
 * @param n transform length along `axis`; shorter input is truncated, longer input is zero padded.
 * @param scale factor applied once to the backend result.
 * @return NPUArray complex64 array whose `axis` dimension has length ``n/2 + 1``.
 */
NPUArray Rfft(const NPUArray& x, int64_t axis, int64_t n, double scale);

/**
 * @brief Complex-to-real inverse FFT along one axis, batched over the remaining dimensions.
 *
 * Equivalent to numpy.fft.irfft(x, n=n, axis=axis): the input holds ``n/2 + 1`` Hermitian bins
 * along `axis` (a longer spectrum is truncated, a shorter one zero padded, as NumPy does) and the
 * output axis has length ``n``.
 *
 * @param x complex64 half-spectrum input array on the NPU.
 * @param axis non-negative index of the transformed axis.
 * @param n length of the *output* axis, matching NumPy's ``irfft`` parameter.
 * @param scale factor applied once to the backend result.
 * @return NPUArray float32 array whose `axis` dimension has length ``n``.
 */
NPUArray Irfft(const NPUArray& x, int64_t axis, int64_t n, double scale);

} // namespace fft
} // namespace asnumpy
