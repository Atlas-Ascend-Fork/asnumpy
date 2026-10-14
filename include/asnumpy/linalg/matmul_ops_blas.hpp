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

/**
 * @brief Two-dimensional float32 matrix multiplication through the optional CANN ops-blas backend.
 *
 * Computes the same result as numpy.matmul for two 2-D float32 arrays, using
 * aclblasSgemmStridedBatched with batchCount = 1 (the plain aclblasSgemm / GemmEx / SgemmBatched
 * entries have no Ascend 910B implementation, see docs/matmul_backend.md).
 *
 * The operands stay on the device: asnumpy passes NPUArray device pointers straight to ops-blas, so
 * there is no host round trip (unlike the FFT bridge). Row-major asnumpy buffers are consumed as
 * column-major BLAS matrices by swapping the operands; the derivation and the lda/ldb/ldc values
 * are documented in docs/matmul_backend.md and covered by non-square device tests.
 *
 * @param x1 Left operand, 2-D float32 NPUArray with shape (M, K).
 * @param x2 Right operand, 2-D float32 NPUArray with shape (K, N).
 * @return NPUArray Result of shape (M, N) on the NPU.
 * @throws std::invalid_argument for unsupported rank/dtype or mismatched inner dimensions.
 * @throws std::runtime_error when the backend is unavailable or a call fails.
 */
NPUArray MatmulOpsBlas(const NPUArray& x1, const NPUArray& x2);
