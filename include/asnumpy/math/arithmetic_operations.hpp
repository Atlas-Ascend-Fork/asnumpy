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

#include <utility>

namespace asnumpy {

/**
 * @brief Element-wise addition of two arrays with broadcasting.
 *
 * Computes x1 + x2 element-wise on NPU using aclnnAdd.
 * Supports broadcasting of input shapes according to array broadcasting rules.
 *
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype (optional) Target dtype for the output array.
 * @return NPUArray Output array with element-wise sums.
 * @throws std::runtime_error If shapes are not broadcastable, dtype unsupported, or ACL op fails.
 */
NPUArray Add(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Compute the reciprocal (1/x) of each element in the input array.
 *
 * Performs element-wise reciprocal operation using aclnnReciprocal.
 *
 * @param x Input array.
 * @param dtype (optional) Target dtype for the output array.
 * @return NPUArray Array with element-wise reciprocals.
 * @throws std::runtime_error If dtype unsupported or ACL operation fails.
 */
NPUArray Reciprocal(const NPUArray& x, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Unary positive operator: return a copy or cast of the input array.
 *
 * If the target dtype matches the input, this returns a deep copy.
 * Otherwise, performs dtype casting on NPU using aclnnCast.
 *
 * @param x Input array.
 * @param dtype (optional) Target dtype for the output array.
 * @return NPUArray Copy of x or casted array.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray Positive(const NPUArray& x, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Unary negative operator: element-wise negation (-x).
 *
 * Computes the element-wise negation of the input array using aclnnNeg.
 *
 * @param x Input array.
 * @param dtype (optional) Target dtype for the output array.
 * @return NPUArray Array with element-wise negated values.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray Negative(const NPUArray& x, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Element-wise multiplication of two arrays with broadcasting.
 *
 * Computes x1 * x2 element-wise on NPU using aclnnMul.
 * Supports broadcasting of input shapes according to array broadcasting rules.
 *
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype (optional) Target dtype for the output array.
 * @return NPUArray Array with element-wise products.
 * @throws std::runtime_error If shapes are not broadcastable, dtype unsupported, or ACL op fails.
 */
NPUArray Multiply(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Element-wise division of two arrays with broadcasting.
 *
 * Computes x1 / x2 element-wise on NPU using aclnnDiv.
 * Supports broadcasting of input shapes according to array broadcasting rules.
 *
 * @param x1 Dividend array.
 * @param x2 Divisor array.
 * @param dtype (optional) Target dtype for the output array.
 * @return NPUArray Array with element-wise quotients.
 * @throws std::runtime_error If shapes are not broadcastable, dtype unsupported, or ACL op fails.
 */
NPUArray Divide(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Element-wise true division of two arrays.
 *
 * Delegates to Divide() for computation on NPU.
 *
 * @param x1 Dividend array.
 * @param x2 Divisor array.
 * @param dtype (optional) Target dtype for the output array.
 * @return NPUArray Array with element-wise quotients.
 * @throws std::runtime_error If Divide fails.
 */
NPUArray TrueDivide(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Element-wise subtraction of two arrays with broadcasting.
 *
 * Computes x1 - x2 element-wise on NPU using aclnnSub.
 * Supports broadcasting of input shapes according to array broadcasting rules.
 *
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype (optional) Target dtype for the output array.
 * @return NPUArray Array with element-wise differences.
 * @throws std::runtime_error If shapes are not broadcastable, dtype unsupported, or ACL op fails.
 */
NPUArray Subtract(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Element-wise floor division of two arrays with broadcasting.
 *
 * Computes floor(x1 / x2) element-wise on NPU using aclnnFloorDivide.
 * Supports broadcasting of input shapes according to array broadcasting rules.
 *
 * @param x1 Dividend array.
 * @param x2 Divisor array.
 * @param dtype (optional) Target dtype for the output array.
 * @return NPUArray Array with element-wise floored quotients.
 * @throws std::runtime_error If shapes are not broadcastable, dtype unsupported, or ACL op fails.
 */
NPUArray FloorDivide(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Element-wise power of two arrays with broadcasting.
 *
 * Computes x1 ** x2 element-wise on NPU using aclnnPowTensorTensor.
 *
 * @param x1 Base array.
 * @param x2 Exponent array.
 * @param dtype (optional) Target dtype for the output array.
 * @return NPUArray Array with element-wise powers.
 * @throws std::runtime_error If shapes are not broadcastable, dtype unsupported, or ACL op fails.
 */
NPUArray Power(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Scalar ** Tensor power.
 *
 * Computes scalar ** x2 element-wise on NPU using aclnnPowScalarTensor.
 *
 * @param x1 Scalar base (Python object convertible to number).
 * @param x2 Exponent array.
 * @param dtype (optional) Target dtype for the output array.
 * @return NPUArray Array with element-wise powers.
 * @throws std::runtime_error If conversion fails or ACL op fails.
 */
NPUArray Power(const py::object& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Tensor ** Scalar power.
 *
 * Computes x1 ** scalar element-wise on NPU using aclnnPowTensorScalar.
 *
 * @param x1 Base array.
 * @param x2 Scalar exponent (Python object convertible to number).
 * @param dtype (optional) Target dtype for the output array.
 * @return NPUArray Array with element-wise powers.
 * @throws std::runtime_error If conversion fails or ACL op fails.
 */
NPUArray Power(const NPUArray& x1, const py::object& x2, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Compute the element-wise floating-point power x1 ** x2.
 *
 * Similar to NumPy's float_power, always returns floating-point results
 * even if inputs are integers. Uses NPU operator aclnnPowTensorTensor.
 * Supports broadcasting of input shapes according to standard array broadcasting rules.
 *
 * @param x1 Base array.
 * @param x2 Exponent array.
 * @param dtype (optional) Target numpy dtype for the output array (must be float or double).
 * @return NPUArray Array of element-wise powers.
 * @throws std::runtime_error If inputs are not broadcastable, dtype is unsupported, or ACL operation fails.
 */
NPUArray FloatPower(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Compute the element-wise floating-point remainder (fmod) of two arrays.
 *
 * The result of fmod(x1, x2) is defined as:
 *     fmod(x1, x2) = x1 - trunc(x1/x2) * x2,
 * and its sign follows the dividend (x1).
 *
 * Supports broadcasting of input shapes according to standard array broadcasting rules.
 * Uses NPU operator aclnnFmodTensor.
 *
 * @param x1 Dividend array.
 * @param x2 Divisor array.
 * @param dtype (optional) Target numpy dtype for the output array (must be float or double).
 * @return NPUArray Array with element-wise fmod results.
 * @throws std::runtime_error If shapes are not broadcastable, dtype is unsupported, or ACL operation fails.
 */
NPUArray Fmod(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Compute the element-wise remainder (mod) of two arrays.
 *
 * The result follows NumPy's remainder semantics:
 *     remainder(x1, x2) = x1 - floor(x1 / x2) * x2,
 * so its non-zero result has the same sign as the divisor (x2), while fmod's
 * sign follows the dividend (x1).
 *
 * Supports broadcasting of input shapes according to standard array broadcasting rules.
 * Uses NPU operator aclnnRemainderTensorTensor.
 *
 * @param x1 Dividend array.
 * @param x2 Divisor array.
 * @param dtype (optional) Target numpy dtype for the output array (must be float or double).
 * @return NPUArray Array with element-wise remainder results.
 * @throws std::runtime_error If shapes are not broadcastable, dtype is unsupported, or ACL operation fails.
 */
NPUArray Mod(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Split input into fractional and integral parts element-wise.
 *
 * For each element x:
 *     frac = x - floor(x)
 *     int  = floor(x)
 *
 * The fractional part has the same sign as the input.
 * Returns a pair of arrays (frac, int), consistent with NumPy's modf.
 *
 * Uses NPU operators aclnnFloor and aclnnSub.
 *
 * @param x Input array (must be float or double).
 * @return std::pair<NPUArray, NPUArray> Pair of (fractional part, integral part).
 * @throws std::runtime_error If dtype is unsupported or ACL operation fails.
 */
std::pair<NPUArray, NPUArray> Modf(const NPUArray& x);

/**
 * @brief Compute the element-wise remainder of division.
 *
 * The result follows NumPy's remainder semantics:
 *     remainder(x1, x2) = x1 - floor(x1 / x2) * x2
 *
 * Unlike fmod, a non-zero remainder has the same sign as the divisor (x2);
 * fmod's sign follows the dividend (x1).
 *
 * Supports broadcasting of input shapes according to standard array broadcasting rules.
 * Internally reuses Mod() which calls aclnnRemainderTensorTensor.
 *
 * @param x1 Dividend array.
 * @param x2 Divisor array.
 * @param dtype (optional) Target numpy dtype for the output array (must be float or double).
 * @return NPUArray Array with element-wise remainder results.
 * @throws std::runtime_error If shapes are not broadcastable, dtype is unsupported, or ACL operation fails.
 */
NPUArray Remainder(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Compute element-wise quotient and remainder of division.
 *
 * For each pair (x1, x2):
 *     quotient  = floor(x1 / x2)
 *     remainder = x1 - quotient * x2
 *
 * Returns a pair of arrays (quotient, remainder).
 * Supports broadcasting of input shapes according to standard array broadcasting rules.
 * Uses aclnnDivMod with mode=2 (floor) for quotient, and computes remainder manually.
 *
 * @param x1 Dividend array.
 * @param x2 Divisor array.
 * @param dtype (optional) Target dtype for the output arrays (supports int32, float, double).
 * @return std::pair<NPUArray, NPUArray> Pair (quotient, remainder).
 * @throws std::runtime_error If inputs are not broadcastable, dtype unsupported, or ACL op fails.
 */
std::pair<NPUArray, NPUArray> Divmod(const NPUArray& x1, const NPUArray& x2,
                                     std::optional<py::dtype> dtype = std::nullopt);

} // namespace asnumpy
