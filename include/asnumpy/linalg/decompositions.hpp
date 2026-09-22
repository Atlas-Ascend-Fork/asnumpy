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

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

#include <pybind11/pybind11.h>
#include <utility>

py::object Linalg_Qr(const NPUArray& a, const std::string& mode);

/**
 * @brief Singular value decomposition of a matrix (or a batch of matrices).
 *
 * Delegates to aclnnSvd. For an input of shape (..., M, N) with K = min(M, N):
 *   - full_matrices=true:  u (..., M, M), sigma (..., K), v (..., N, N)
 *   - full_matrices=false: u (..., M, K), sigma (..., K), v (..., N, K)
 * The returned `v` is V, not V^H; the Python wrapper transposes it to NumPy's `vh`.
 *
 * @param a Input array with at least two dimensions.
 * @param full_matrices Whether to return the full u/v or the economy-size ones.
 * @param compute_uv When false only the singular values are computed and returned.
 * @return py::object Either (u, sigma, v) or sigma alone.
 * @throws std::invalid_argument If @p a has fewer than two dimensions.
 */
py::object Linalg_Svd(const NPUArray& a, bool full_matrices, bool compute_uv);
