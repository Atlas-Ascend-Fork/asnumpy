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

#include <asnumpy/math/trigonometric_functions.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/dtype_promotion.hpp>
#include <asnumpy/utils/npu_array.hpp>

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_acos.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_asin.h>
#include <aclnnop/aclnn_atan.h>
#include <aclnnop/aclnn_atan2.h>
#include <aclnnop/aclnn_cos.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_sin.h>
#include <aclnnop/aclnn_sqrt.h>
#include <aclnnop/aclnn_tan.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fmt/core.h>
#include <fmt/format.h>
#include <stdexcept>

namespace asnumpy {

namespace {

uint16_t FloatToFp16Bits(float value) {
    uint32_t float_bits;
    std::memcpy(&float_bits, &value, sizeof(float_bits));
    return static_cast<uint16_t>(((float_bits >> 16) & 0x8000U) | (((float_bits >> 13) - 0x1C000U) & 0x7C00U) |
                                 ((float_bits >> 13) & 0x03FFU));
}

} // namespace

NPUArray Sin(const NPUArray& x) {
    return UnaryFloatingPromoteOp(
        x, /*supports_float64=*/true,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnSinGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnSin(workspace, workspaceSize, executor, nullptr);
        },
        "Sin", "aclnnSin");
}

NPUArray Cos(const NPUArray& x) {
    return UnaryFloatingPromoteOp(
        x, /*supports_float64=*/true,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnCosGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnCos(workspace, workspaceSize, executor, nullptr);
        },
        "Cos", "aclnnCos");
}

NPUArray Tan(const NPUArray& x) {
    return UnaryFloatingPromoteOp(
        x, /*supports_float64=*/true,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnTanGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnTan(workspace, workspaceSize, executor, nullptr);
        },
        "Tan", "aclnnTan");
}

NPUArray Arcsin(const NPUArray& x) {
    aclDataType aclType = PromoteUnaryFloating(x.aclDtype);
    ACL_DTYPE_WARN(x.aclDtype, aclType, __func__);
    NPUArray input = EnsureAclDtype(x, aclType);
    py::dtype dtype = NPUArray::GetPyDtype(aclType);
    return EXECUTE_UNARY_OP(
        input, dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnAsinGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnAsin(workspace, workspaceSize, executor, nullptr);
        },
        "Arcsin", "aclnnAsin");
}

NPUArray Arccos(const NPUArray& x) {
    aclDataType aclType = PromoteUnaryFloating(x.aclDtype);
    ACL_DTYPE_WARN(x.aclDtype, aclType, __func__);
    NPUArray input = EnsureAclDtype(x, aclType);
    py::dtype dtype = NPUArray::GetPyDtype(aclType);
    return EXECUTE_UNARY_OP(
        input, dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnAcosGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnAcos(workspace, workspaceSize, executor, nullptr);
        },
        "Arccos", "aclnnAcos");
}

NPUArray Arctan(const NPUArray& x) {
    aclDataType aclType = PromoteUnaryFloating(x.aclDtype);
    ACL_DTYPE_WARN(x.aclDtype, aclType, __func__);
    NPUArray input = EnsureAclDtype(x, aclType);
    py::dtype dtype = NPUArray::GetPyDtype(aclType);
    return EXECUTE_UNARY_OP(
        input, dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnAtanGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnAtan(workspace, workspaceSize, executor, nullptr);
        },
        "Arctan", "aclnnAtan");
}

NPUArray Hypot(const NPUArray& a, const NPUArray& b) {
    LOG_DEBUG("Hypot start: a_shape={}, b_shape={}, aclDtype={}", detail::FormatShape(a.shape),
              detail::FormatShape(b.shape), AclDtypeName(a.aclDtype));

    aclDataType dtype = PromoteBinaryFloating(a.aclDtype, b.aclDtype);
    ACL_DTYPE_WARN(a.aclDtype, dtype, __func__);
    ACL_DTYPE_WARN(b.aclDtype, dtype, __func__);
    NPUArray in_a = EnsureAclDtype(a, dtype);
    NPUArray in_b = EnsureAclDtype(b, dtype);

    auto broadcast = GetBroadcastShape(in_a, in_b);

    // step 1: compute a squared (a²)
    NPUArray a_squared(in_a.shape, in_a.dtype);
    uint64_t a_sq_workspace_size = 0;
    aclOpExecutor* a_sq_executor = nullptr;
    auto error = aclnnMulGetWorkspaceSize(in_a.tensorPtr, in_a.tensorPtr, a_squared.tensorPtr, &a_sq_workspace_size,
                                          &a_sq_executor);
    ACLNN_CHECK(error, "aclnnMulGetWorkspaceSize");

    AclWorkspace a_sq_workspace(a_sq_workspace_size);

    error = aclnnMul(a_sq_workspace.get(), a_sq_workspace_size, a_sq_executor, nullptr);
    ACLNN_CHECK(error, "aclnnMul");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnMul completed");

    // step 2: compute b squared (b²)
    NPUArray b_squared(in_b.shape, in_b.dtype);
    uint64_t b_sq_workspace_size = 0;
    aclOpExecutor* b_sq_executor = nullptr;
    error = aclnnMulGetWorkspaceSize(in_b.tensorPtr, in_b.tensorPtr, b_squared.tensorPtr, &b_sq_workspace_size,
                                     &b_sq_executor);
    ACLNN_CHECK(error, "aclnnMulGetWorkspaceSize");

    AclWorkspace b_sq_workspace(b_sq_workspace_size);

    error = aclnnMul(b_sq_workspace.get(), b_sq_workspace_size, b_sq_executor, nullptr);
    ACLNN_CHECK(error, "aclnnMul");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnMul completed");

    // step 3: compute sum of squares (a² + b²)
    LOG_DEBUG("aclnnAdd start: a_squared_shape={}, b_squared_shape={}, aclDtype={}",
              detail::FormatShape(a_squared.shape), detail::FormatShape(b_squared.shape), AclDtypeName(dtype));
    NPUArray sum_squares(broadcast, dtype);
    uint64_t add_workspace_size = 0;
    aclOpExecutor* add_executor = nullptr;
    aclScalar* alpha_scalar;
    if (dtype == ACL_DOUBLE) {
        double alpha = 1.0;
        alpha_scalar = aclCreateScalar(&alpha, dtype);
    } else if (dtype == ACL_FLOAT16) {
        uint16_t alpha = FloatToFp16Bits(1.0f);
        alpha_scalar = aclCreateScalar(&alpha, dtype);
    } else {
        float alpha = 1.0f;
        alpha_scalar = aclCreateScalar(&alpha, dtype);
    }

    error = aclnnAddGetWorkspaceSize(a_squared.tensorPtr, b_squared.tensorPtr, alpha_scalar, sum_squares.tensorPtr,
                                     &add_workspace_size, &add_executor);
    ACLNN_CHECK(error, "aclnnAddGetWorkspaceSize");

    AclWorkspace add_workspace(add_workspace_size);

    error = aclnnAdd(add_workspace.get(), add_workspace_size, add_executor, nullptr);
    ACLNN_CHECK(error, "aclnnAdd");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnAdd completed");

    // step 4: compute square root (√(a² + b²))
    LOG_DEBUG("aclnnSqrt start: input_shape={}, aclDtype={}", detail::FormatShape(sum_squares.shape),
              AclDtypeName(sum_squares.aclDtype));
    NPUArray result(broadcast, dtype);
    uint64_t sqrt_workspace_size = 0;
    aclOpExecutor* sqrt_executor = nullptr;
    error = aclnnSqrtGetWorkspaceSize(sum_squares.tensorPtr, result.tensorPtr, &sqrt_workspace_size, &sqrt_executor);
    ACLNN_CHECK(error, "aclnnSqrtGetWorkspaceSize");

    AclWorkspace sqrt_workspace(sqrt_workspace_size);

    error = aclnnSqrt(sqrt_workspace.get(), sqrt_workspace_size, sqrt_executor, nullptr);
    ACLNN_CHECK(error, "aclnnSqrt");

    // synchronize device and release resources
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    aclDestroyScalar(alpha_scalar);

    LOG_INFO("aclnnSqrt completed");

    return result;
}

NPUArray Arctan2(const NPUArray& y, const NPUArray& x) {
    aclDataType desired = PromoteBinaryFloating(y.aclDtype, x.aclDtype);
    ACL_DTYPE_WARN(y.aclDtype, desired, __func__);
    ACL_DTYPE_WARN(x.aclDtype, desired, __func__);
    // aclnnAtan2 supports float/double; cast integer inputs first.
    aclDataType compute = AclComputeFloatingDtype(desired, /*supports_float64=*/true);
    NPUArray in_y = EnsureAclDtype(y, compute);
    NPUArray in_x = EnsureAclDtype(x, compute);
    py::dtype dtype = NPUArray::GetPyDtype(compute);
    NPUArray out = EXECUTE_BINARY_OP(
        in_y, in_x, dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnAtan2GetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnAtan2(workspace, workspaceSize, executor, nullptr);
        },
        "Arctan2", "aclnnAtan2");
    if (desired != compute) {
        return CastToDtype(out, desired);
    }
    return out;
}

NPUArray Radians(const NPUArray& x) {
    LOG_DEBUG("aclnnMuls start: input_shape={}, aclDtype={}", detail::FormatShape(x.shape), AclDtypeName(x.aclDtype));

    // validate input parameters
    if (x.tensorSize == 0) {
        throw std::invalid_argument(
            fmt::format("[trigonometric_functions.cpp]({}) input tensor has no elements", __func__));
    }
    if (x.aclDtype != ACL_FLOAT && x.aclDtype != ACL_DOUBLE && x.aclDtype != ACL_FLOAT16) {
        throw std::invalid_argument(
            fmt::format("[trigonometric_functions.cpp]({}) input must be float, double or float16 type, got {}",
                        __func__, AclDtypeName(x.aclDtype)));
    }

    // initialize output tensor (same shape and dtype as input)
    NPUArray result(x.shape, x.aclDtype);

    // degrees-to-radians factor: π/180
    const double rad_factor = M_PI / 180.0;
    aclScalar* factor_scalar;
    if (x.aclDtype == ACL_DOUBLE) {
        double factor = rad_factor;
        factor_scalar = aclCreateScalar(&factor, x.aclDtype);
    } else if (x.aclDtype == ACL_FLOAT16) {
        uint16_t fp16_bits = FloatToFp16Bits(static_cast<float>(rad_factor));
        factor_scalar = aclCreateScalar(&fp16_bits, x.aclDtype);
    } else {
        float factor = static_cast<float>(rad_factor);
        factor_scalar = aclCreateScalar(&factor, x.aclDtype);
    }

    // get workspace size
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnMulsGetWorkspaceSize(x.tensorPtr, factor_scalar, result.tensorPtr, &workspace_size, &executor);
    ACLNN_CHECK(error, "aclnnMulsGetWorkspaceSize");

    // allocate workspace
    AclWorkspace workspace(workspace_size);

    // execute scalar multiplication
    error = aclnnMuls(workspace.get(), workspace_size, executor, nullptr);
    ACLNN_CHECK(error, "aclnnMuls");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

    aclDestroyScalar(factor_scalar);

    LOG_INFO("aclnnMuls completed");

    return result;
}

NPUArray Degrees(const NPUArray& x) {
    LOG_DEBUG("aclnnMul start: input_shape={}, aclDtype={}", detail::FormatShape(x.shape), AclDtypeName(x.aclDtype));

    aclDataType aclType = ACL_DOUBLE;
    if (x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_FLOAT16 || x.aclDtype == ACL_DOUBLE) {
        aclType = x.aclDtype;
    }
    auto out = NPUArray(x.shape, aclType);
    const double factor = 180.0 / M_PI;
    auto factorArr = NPUArray({1}, aclType);
    void* factorPtr = nullptr;
    auto error = aclGetRawTensorAddr(factorArr.tensorPtr, &factorPtr);
    ACL_RT_CHECK(error, "aclGetRawTensorAddr");
    double hostValue = factor;
    error = aclrtMemcpy(factorPtr, sizeof(double), &hostValue, sizeof(double), ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_RT_CHECK(error, "Write const factor");
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    error = aclnnMulGetWorkspaceSize(x.tensorPtr, factorArr.tensorPtr, out.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnMulGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnMul(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnMul");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

    LOG_INFO("aclnnMul completed");

    return out;
}
} // namespace asnumpy
