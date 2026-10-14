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
#include <cstdint>
#include <vector>

namespace asnumpy {

/**
 * @brief Join a sequence of arrays along an existing axis.
 *
 * Mirrors numpy.concatenate. Every input must have the same rank and the same dtype; all
 * dimensions except @p axis must match exactly.
 *
 * @param arrays Input arrays, at least one.
 * @param axis Axis along which the arrays are joined. Negative values count from the end.
 * @return NPUArray The concatenated array.
 * @throws std::invalid_argument If the sequence is empty, ranks/dtypes differ, a non-axis
 *         dimension differs, or @p axis is out of bounds.
 */
NPUArray Concatenate(const std::vector<const NPUArray*>& arrays, int64_t axis);

/**
 * @brief Join a sequence of arrays along a new axis.
 *
 * Mirrors numpy.stack. Every input must have an identical shape and dtype. A single-element
 * sequence inserts a length-one axis, which is exactly numpy.expand_dims.
 *
 * @param arrays Input arrays, at least one.
 * @param axis Position of the new axis in the result. Negative values count from the end.
 * @return NPUArray The stacked array.
 * @throws std::invalid_argument If the sequence is empty, shapes/dtypes differ, or @p axis is out
 *         of bounds.
 */
NPUArray Stack(const std::vector<const NPUArray*>& arrays, int64_t axis);

/**
 * @brief Select elements from two arrays according to a boolean condition.
 *
 * Mirrors the three-argument numpy.where: condition, self and other are broadcast together and
 * the result has the broadcast shape and the dtype of @p self (which must equal @p other's).
 *
 * @param condition Boolean (ACL_BOOL) condition array.
 * @param self Values taken where the condition is true.
 * @param other Values taken where the condition is false.
 * @return NPUArray The selected values.
 * @throws std::invalid_argument If the condition is not boolean, the operand dtypes differ, or the
 *         shapes are not broadcastable.
 */
NPUArray Where(const NPUArray& condition, const NPUArray& self, const NPUArray& other);

} // namespace asnumpy
