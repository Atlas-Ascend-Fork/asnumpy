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

#include <asnumpy/logic/logic.hpp>
#include <asnumpy/dtypes/promote.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/dtype_promotion.hpp>
#include <asnumpy/utils/status_handler.hpp>
#include <fmt/core.h>
#include <stdexcept>

#include <aclnnop/aclnn_all.h>
#include <aclnnop/aclnn_any.h>
#include <aclnnop/aclnn_eq_scalar.h>
#include <aclnnop/aclnn_eq_tensor.h>
#include <aclnnop/aclnn_ge_scalar.h>
#include <aclnnop/aclnn_ge_tensor.h>
#include <aclnnop/aclnn_gt_scalar.h>
#include <aclnnop/aclnn_gt_tensor.h>
#include <aclnnop/aclnn_is_inf.h>
#include <aclnnop/aclnn_isfinite.h>
#include <aclnnop/aclnn_isneginf.h>
#include <aclnnop/aclnn_isposinf.h>
#include <aclnnop/aclnn_le_scalar.h>
#include <aclnnop/aclnn_le_tensor.h>
#include <aclnnop/aclnn_logical_and.h>
#include <aclnnop/aclnn_logical_not.h>
#include <aclnnop/aclnn_logical_or.h>
#include <aclnnop/aclnn_logical_xor.h>
#include <aclnnop/aclnn_lt_scalar.h>
#include <aclnnop/aclnn_lt_tensor.h>
#include <aclnnop/aclnn_ne_scalar.h>
#include <aclnnop/aclnn_ne_tensor.h>
#include <aclnnop/aclnn_where.h>

#include <algorithm>
#include <vector>

namespace asnumpy {

/**
 * @brief Create aclScalar object for scalar comparison operations
 *
 * Determine the target scalar dtype from the input array dtype and an optional
 * output dtype, then create an aclScalar using CreateScalar.
 *
 * @param x1 Input array
 * @param scalar Python scalar object
 * @param dtype Optional output data type
 * @return aclScalar* Created scalar object
 */
static aclScalar* CreateScalarForComparison(const NPUArray& x1, const py::object& scalar,
                                            std::optional<py::dtype> dtype) {

    // determine scalar dtype: use specified dtype if provided, otherwise use input array dtype
    aclDataType scalar_dtype;
    if (dtype.has_value()) {
        scalar_dtype = NPUArray::GetACLDataType(*dtype);
    } else {
        scalar_dtype = x1.aclDtype;
    }

    return CreateScalar(scalar, scalar_dtype);
}

/// Reduce array by logical AND operation over all elements.
NPUArray All(const NPUArray& x) {
    LOG_DEBUG("aclnnAll start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(x.shape), x.tensorSize,
              AclDtypeName(x.aclDtype));
    auto result = NPUArray({}, ACL_BOOL);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;

    // dim=[] (global reduction)
    aclIntArray* aclDim = aclCreateIntArray(nullptr, 0);
    if (!aclDim) {
        throw std::runtime_error(fmt::format("[logic.cpp]({}) failed to create empty aclIntArray", __func__));
    }

    auto error = aclnnAllGetWorkspaceSize(x.tensorPtr, aclDim,
                                          false, // keepdims = false
                                          result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnAllGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnAll(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnAll");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    aclDestroyIntArray(aclDim);

    LOG_INFO("aclnnAll completed");
    return result;
}

/// Reduce array by logical AND operation over specified dimensions.
NPUArray All(const NPUArray& x, const std::vector<int64_t>& dim, bool keepdims) {
    LOG_DEBUG("aclnnAll start: input_shape={}, tensorSize={}, aclDtype={}, dim={}, keepdims={}",
              detail::FormatShape(x.shape), x.tensorSize, AclDtypeName(x.aclDtype), detail::FormatShape(dim), keepdims);
    std::vector<int64_t> shape = x.shape;
    if (keepdims) {
        for (int i = 0; i < dim.size(); i++) {
            shape[dim[i]] = 1;
        }
    } else {
        std::vector<int64_t> dim_remove = dim;
        std::sort(dim_remove.begin(), dim_remove.end(), std::greater<int64_t>());
        for (int i = 0; i < dim.size(); i++) {
            shape.erase(shape.begin() + dim_remove[i]);
        }
    }
    auto result = NPUArray(shape, ACL_BOOL);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;

    // build dim array
    aclIntArray* aclDim = aclCreateIntArray(dim.data(), dim.size());
    if (!aclDim) {
        throw std::runtime_error(fmt::format("[logic.cpp]({}) failed to create aclIntArray", __func__));
    }

    auto error = aclnnAllGetWorkspaceSize(x.tensorPtr, aclDim, keepdims, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnAllGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnAll(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnAll");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    aclDestroyIntArray(aclDim);

    LOG_INFO("aclnnAll completed");
    return result;
}

/// Reduce array by logical OR operation over all elements.
NPUArray Any(const NPUArray& x) {
    LOG_DEBUG("aclnnAny start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(x.shape), x.tensorSize,
              AclDtypeName(x.aclDtype));
    auto result = NPUArray({}, ACL_BOOL);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;

    // dim=[] (global reduction)
    aclIntArray* aclDim = aclCreateIntArray(nullptr, 0);
    if (!aclDim) {
        throw std::runtime_error("[logic.cpp](Any) failed to create empty aclIntArray");
    }

    auto error = aclnnAnyGetWorkspaceSize(x.tensorPtr, aclDim,
                                          false, // keepdims = false
                                          result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnAnyGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnAny(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnAny");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    aclDestroyIntArray(aclDim);

    LOG_INFO("aclnnAny completed");
    return result;
}

/// Reduce array by logical OR operation over specified dimensions.
NPUArray Any(const NPUArray& x, const std::vector<int64_t>& dim, bool keepdims) {
    LOG_DEBUG("aclnnAny start: input_shape={}, tensorSize={}, aclDtype={}, dim={}, keepdims={}",
              detail::FormatShape(x.shape), x.tensorSize, AclDtypeName(x.aclDtype), detail::FormatShape(dim), keepdims);
    std::vector<int64_t> shape = x.shape;
    if (keepdims) {
        for (int i = 0; i < dim.size(); i++) {
            shape[dim[i]] = 1;
        }
    } else {
        std::vector<int64_t> dim_remove = dim;
        std::sort(dim_remove.begin(), dim_remove.end(), std::greater<int64_t>());
        for (int i = 0; i < dim.size(); i++) {
            shape.erase(shape.begin() + dim_remove[i]);
        }
    }
    auto result = NPUArray(shape, ACL_BOOL);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;

    // build dim array
    aclIntArray* aclDim = aclCreateIntArray(dim.data(), dim.size());
    if (!aclDim) {
        throw std::runtime_error(fmt::format("[logic.cpp]({}) failed to create aclIntArray", __func__));
    }

    auto error = aclnnAnyGetWorkspaceSize(x.tensorPtr, aclDim, keepdims, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnAnyGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnAny(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnAny");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    aclDestroyIntArray(aclDim);

    LOG_INFO("aclnnAny completed");
    return result;
}

/// Check element-wise finiteness of the input array.
NPUArray IsFinite(const NPUArray& x) {
    // output boolean array with same shape as input
    py::dtype dtype = NPUArray::GetPyDtype(ACL_BOOL);
    return EXECUTE_UNARY_OP(
        x, dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnIsFiniteGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnIsFinite(workspace, workspaceSize, executor, nullptr);
        },
        "IsFinite", "aclnnIsFinite");
}

/// Check element-wise infinity of the input array.
NPUArray IsInf(const NPUArray& x) {
    py::dtype dtype = NPUArray::GetPyDtype(ACL_BOOL);
    return EXECUTE_UNARY_OP(
        x, dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnIsInfGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnIsInf(workspace, workspaceSize, executor, nullptr);
        },
        "IsInf", "aclnnIsInf");
}

/// Test element-wise for negative infinity (-inf).
NPUArray IsNegInf(const NPUArray& x) {
    py::dtype dtype = NPUArray::GetPyDtype(ACL_BOOL);
    return EXECUTE_UNARY_OP(
        x, dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnIsNegInfGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnIsNegInf(workspace, workspaceSize, executor, nullptr);
        },
        "IsNegInf", "aclnnIsNegInf");
}

/// Test element-wise for positive infinity (+inf).
NPUArray IsPosInf(const NPUArray& x) {
    py::dtype dtype = NPUArray::GetPyDtype(ACL_BOOL);
    return EXECUTE_UNARY_OP(
        x, dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnIsPosInfGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnIsPosInf(workspace, workspaceSize, executor, nullptr);
        },
        "IsPosInf", "aclnnIsPosInf");
}

/// Perform element-wise logical AND between two boolean arrays.
NPUArray LogicalAnd(const NPUArray& x, const NPUArray& y) {
    py::dtype dtype = NPUArray::GetPyDtype(ACL_BOOL);
    return EXECUTE_BINARY_OP(
        x, y, dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnLogicalAndGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnLogicalAnd(workspace, workspaceSize, executor, nullptr);
        },
        "LogicalAnd", "aclnnLogicalAnd");
}

/// Perform element-wise logical OR between two boolean arrays.
NPUArray LogicalOr(const NPUArray& x, const NPUArray& y) {
    py::dtype dtype = NPUArray::GetPyDtype(ACL_BOOL);
    return EXECUTE_BINARY_OP(
        x, y, dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnLogicalOrGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnLogicalOr(workspace, workspaceSize, executor, nullptr);
        },
        "LogicalOr", "aclnnLogicalOr");
}

/// Perform element-wise logical NOT on a boolean array.
NPUArray LogicalNot(const NPUArray& x) {
    py::dtype dtype = NPUArray::GetPyDtype(ACL_BOOL);
    return EXECUTE_UNARY_OP(
        x, dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnLogicalNotGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnLogicalNot(workspace, workspaceSize, executor, nullptr);
        },
        "LogicalNot", "aclnnLogicalNot");
}

/// Perform element-wise logical XOR between two boolean arrays.
NPUArray LogicalXor(const NPUArray& x, const NPUArray& y) {
    py::dtype dtype = NPUArray::GetPyDtype(ACL_BOOL);
    return EXECUTE_BINARY_OP(
        x, y, dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnLogicalXorGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnLogicalXor(workspace, workspaceSize, executor, nullptr);
        },
        "LogicalXor", "aclnnLogicalXor");
}

/// Element-wise greater-than comparison between two arrays.
NPUArray greater(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    py::dtype out_dtype = dtype.value_or(py::dtype::of<bool>());
    return EXECUTE_BINARY_OP(
        x1, x2, out_dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnGtTensorGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnGtTensor(workspace, workspaceSize, executor, nullptr);
        },
        "greater", "aclnnGtTensor");
}

/// Element-wise greater-than comparison between an array and a scalar.
NPUArray greater(const NPUArray& x1, const py::object& scalar, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnGtScalar start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(x1.shape),
              x1.tensorSize, AclDtypeName(x1.aclDtype));
    auto out = NPUArray(x1.shape, dtype.value_or(py::dtype::of<bool>()));
    aclScalar* acl_scalar = CreateScalarForComparison(x1, scalar, dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnGtScalarGetWorkspaceSize(x1.tensorPtr, acl_scalar, out.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnGtScalarGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnGtScalar(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnGtScalar");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    aclDestroyScalar(acl_scalar);
    LOG_INFO("aclnnGtScalar completed");
    return out;
}

/// Element-wise greater-than-or-equal comparison between two arrays.
NPUArray greater_equal(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    py::dtype out_dtype = dtype.value_or(py::dtype::of<bool>());
    return EXECUTE_BINARY_OP(
        x1, x2, out_dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnGeTensorGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnGeTensor(workspace, workspaceSize, executor, nullptr);
        },
        "greater_equal", "aclnnGeTensor");
}

/// Element-wise greater-than-or-equal comparison between an array and a scalar.
NPUArray greater_equal(const NPUArray& x1, const py::object& scalar, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnGeScalar start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(x1.shape),
              x1.tensorSize, AclDtypeName(x1.aclDtype));
    auto out = NPUArray(x1.shape, dtype.value_or(py::dtype::of<bool>()));
    aclScalar* acl_scalar = CreateScalarForComparison(x1, scalar, dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnGeScalarGetWorkspaceSize(x1.tensorPtr, acl_scalar, out.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnGeScalarGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnGeScalar(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnGeScalar");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    aclDestroyScalar(acl_scalar);
    LOG_INFO("aclnnGeScalar completed");
    return out;
}

/// Element-wise less-than comparison between two arrays.
NPUArray less(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    py::dtype out_dtype = dtype.value_or(py::dtype::of<bool>());
    return EXECUTE_BINARY_OP(
        x1, x2, out_dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnLtTensorGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnLtTensor(workspace, workspaceSize, executor, nullptr);
        },
        "less", "aclnnLtTensor");
}

/// Element-wise less-than comparison between an array and a scalar.
NPUArray less(const NPUArray& x1, const py::object& scalar, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnLtScalar start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(x1.shape),
              x1.tensorSize, AclDtypeName(x1.aclDtype));
    auto out = NPUArray(x1.shape, dtype.value_or(py::dtype::of<bool>()));
    aclScalar* acl_scalar = CreateScalarForComparison(x1, scalar, dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnLtScalarGetWorkspaceSize(x1.tensorPtr, acl_scalar, out.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnLtScalarGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnLtScalar(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnLtScalar");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    aclDestroyScalar(acl_scalar);
    LOG_INFO("aclnnLtScalar completed");
    return out;
}

/// Element-wise less-than-or-equal comparison between two arrays.
NPUArray less_equal(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    py::dtype out_dtype = dtype.value_or(py::dtype::of<bool>());
    return EXECUTE_BINARY_OP(
        x1, x2, out_dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnLeTensorGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnLeTensor(workspace, workspaceSize, executor, nullptr);
        },
        "less_equal", "aclnnLeTensor");
}

/// Element-wise less-than-or-equal comparison between an array and a scalar.
NPUArray less_equal(const NPUArray& x1, const py::object& scalar, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnLeScalar start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(x1.shape),
              x1.tensorSize, AclDtypeName(x1.aclDtype));
    auto out = NPUArray(x1.shape, dtype.value_or(py::dtype::of<bool>()));
    aclScalar* acl_scalar = CreateScalarForComparison(x1, scalar, dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnLeScalarGetWorkspaceSize(x1.tensorPtr, acl_scalar, out.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnLeScalarGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnLeScalar(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnLeScalar");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    aclDestroyScalar(acl_scalar);
    LOG_INFO("aclnnLeScalar completed");
    return out;
}

/// Element-wise equality comparison between two arrays.
NPUArray equal(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    py::dtype out_dtype = dtype.value_or(py::dtype::of<bool>());
    return EXECUTE_BINARY_OP(
        x1, x2, out_dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnEqTensorGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnEqTensor(workspace, workspaceSize, executor, nullptr);
        },
        "equal", "aclnnEqTensor");
}

/// Element-wise equal comparison between an array and a scalar.
NPUArray equal(const NPUArray& x1, const py::object& scalar, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnEqScalar start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(x1.shape),
              x1.tensorSize, AclDtypeName(x1.aclDtype));
    auto out = NPUArray(x1.shape, dtype.value_or(py::dtype::of<bool>()));
    aclScalar* acl_scalar = CreateScalarForComparison(x1, scalar, dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnEqScalarGetWorkspaceSize(x1.tensorPtr, acl_scalar, out.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnEqScalarGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnEqScalar(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnEqScalar");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    aclDestroyScalar(acl_scalar);
    LOG_INFO("aclnnEqScalar completed");
    return out;
}

/// Element-wise not-equal comparison between two arrays.
NPUArray not_equal(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    py::dtype out_dtype = dtype.value_or(py::dtype::of<bool>());
    return EXECUTE_BINARY_OP(
        x1, x2, out_dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnNeTensorGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnNeTensor(workspace, workspaceSize, executor, nullptr);
        },
        "not_equal", "aclnnNeTensor");
}

/// Element-wise not-equal comparison between an array and a scalar.
NPUArray not_equal(const NPUArray& x1, const py::object& scalar, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnNeScalar start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(x1.shape),
              x1.tensorSize, AclDtypeName(x1.aclDtype));
    auto out = NPUArray(x1.shape, dtype.value_or(py::dtype::of<bool>()));
    aclScalar* acl_scalar = CreateScalarForComparison(x1, scalar, dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnNeScalarGetWorkspaceSize(x1.tensorPtr, acl_scalar, out.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnNeScalarGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnNeScalar(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnNeScalar");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    aclDestroyScalar(acl_scalar);
    LOG_INFO("aclnnNeScalar completed");
    return out;
}

namespace {

// GetBroadcastShape 只支持两个 NPUArray,where 需要三元广播,这里补一个 shape 级别的二元广播
std::vector<int64_t> BroadcastTwoShapes(const std::vector<int64_t>& lhs, const std::vector<int64_t>& rhs) {
    const size_t n = std::max(lhs.size(), rhs.size());
    std::vector<int64_t> out(n, 0);
    for (size_t i = 0; i < n; ++i) {
        const int64_t l = (i >= n - lhs.size()) ? lhs[i - (n - lhs.size())] : 1;
        const int64_t r = (i >= n - rhs.size()) ? rhs[i - (n - rhs.size())] : 1;
        if (l != r && l != 1 && r != 1) {
            throw std::invalid_argument(
                fmt::format("[logic.cpp]({}) shapes are not broadcastable", __func__));
        }
        out[i] = std::max(l, r);
    }
    return out;
}

} // anonymous namespace

NPUArray where(const NPUArray& condition, const NPUArray& x1, const NPUArray& x2) {
    LOG_DEBUG("aclnnWhere start: cond_shape={}, x1_shape={}, x2_shape={}, condDtype={}, x1Dtype={}, x2Dtype={}",
              detail::FormatShape(condition.shape), detail::FormatShape(x1.shape), detail::FormatShape(x2.shape),
              AclDtypeName(condition.aclDtype), AclDtypeName(x1.aclDtype), AclDtypeName(x2.aclDtype));
    if (condition.aclDtype != ACL_BOOL) {
        throw std::invalid_argument(fmt::format(
            "[logic.cpp]({}) condition must be ACL_BOOL, got {}", __func__, AclDtypeName(condition.aclDtype)));
    }
    aclDataType common = ResultType(x1.aclDtype, x2.aclDtype);
    NPUArray in_x1 = EnsureAclDtype(x1, common);
    NPUArray in_x2 = EnsureAclDtype(x2, common);
    std::vector<int64_t> out_shape = BroadcastTwoShapes(GetBroadcastShape(condition, in_x1), in_x2.shape);

    auto result = NPUArray(out_shape, common);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnWhereGetWorkspaceSize(condition.tensorPtr, in_x1.tensorPtr, in_x2.tensorPtr, result.tensorPtr,
                                            &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnWhereGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnWhere(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnWhere");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnWhere completed");
    return result;
}

} // namespace asnumpy
