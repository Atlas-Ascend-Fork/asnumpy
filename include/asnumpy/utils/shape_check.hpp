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

#include <algorithm>
#include <cstdint>
#include <fmt/core.h>
#include <stdexcept>
#include <vector>

namespace asnumpy {
namespace detail {

/**
 * @brief Check that an array has at least `rank` dimensions.
 *
 * Matrix operators index shape[ndim - 2]; with fewer dimensions that is an
 * out-of-bounds access, so reject such inputs up front like NumPy does.
 *
 * @param a Input array.
 * @param rank Minimum number of dimensions required.
 * @param func Caller name for the error message (pass __func__).
 * @throw std::invalid_argument Mapped to ValueError by the binding layer.
 */
inline void CheckNdimsAtLeast(const NPUArray& a, size_t rank, const char* func) {
    if (a.shape.size() < rank) {
        throw std::invalid_argument(fmt::format("{}() {}-dimensional array given. Array must be at least {} "
                                                "dimensional",
                                                func, a.shape.size(), rank));
    }
}

/**
 * @brief Apply NumPy axis semantics: negative axes count from the last
 *        dimension; out-of-range and duplicate axes raise.
 *
 * @param axes Axes as given by the user (may be negative / unsorted).
 * @param ndim Number of dimensions of the array being reduced.
 * @param func Caller name for the error message (pass __func__).
 * @return std::vector<int64_t> Non-negative axes sorted ascending.
 */
inline std::vector<int64_t> NormalizeAxes(const std::vector<int64_t>& axes, size_t ndim, const char* func) {
    std::vector<int64_t> normalized;
    normalized.reserve(axes.size());
    for (auto axis : axes) {
        auto ax = axis < 0 ? static_cast<int64_t>(ndim) + axis : axis;
        if (ax < 0 || ax >= static_cast<int64_t>(ndim)) {
            throw std::invalid_argument(fmt::format("axis {} is out of bounds for array of dimension {}", axis, ndim));
        }
        if (std::find(normalized.begin(), normalized.end(), ax) != normalized.end()) {
            throw std::invalid_argument(fmt::format("{}(): duplicate value in 'axis'", func));
        }
        normalized.push_back(ax);
    }
    std::sort(normalized.begin(), normalized.end());
    return normalized;
}

} // namespace detail
} // namespace asnumpy
