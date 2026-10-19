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

#include <stdexcept>
#include <string>

#include <asnumpy/utils/npu_array.hpp>

/**
 * @brief Raised when the optional ops-solver backend reports an exactly singular matrix.
 *
 * ops-solver signals this through `info = k > 0` (LAPACK convention, `U(k,k)` exactly zero) together
 * with a non-zero aclError. The Python wrapper converts it into `numpy.linalg.LinAlgError`, which is
 * what NumPy raises for the same inputs.
 */
class OpsSolverSingularMatrix : public std::runtime_error {
  public:
    explicit OpsSolverSingularMatrix(const std::string& message) : std::runtime_error(message) {}
};

/**
 * @brief Square float32 matrix inverse through the optional CANN ops-solver backend.
 *
 * Contract notes (established by reading the ops-solver sources and by device measurements, see
 * docs/inv_backend.md):
 *  - `aclsolverSgetri` takes the ORIGINAL matrix and performs its own LU factorisation; passing the
 *    factors from `aclsolverSgetrf` silently produces a wrong result.
 *  - its `A` argument is a HOST pointer: the library copies it to the device and copies the inverse
 *    back, so this backend stages the operand through pinned host memory (two transfers) and never
 *    hands the caller's device buffer to the library.
 *  - the library validates `lda == n` and `1 <= n <= 8192`, and synchronises its stream before
 *    returning, so the result is complete when the call returns.
 *
 * @param a Square float32 NPUArray of shape (n, n); left untouched.
 * @return NPUArray The inverse, shape (n, n), on the NPU.
 * @throws std::invalid_argument for unsupported rank/dtype/shape or sizes outside the library range.
 * @throws OpsSolverSingularMatrix when the library reports an exactly singular matrix.
 * @throws std::runtime_error when the backend is unavailable or the library reports another failure.
 */
NPUArray InvOpsSolver(const NPUArray& a);
