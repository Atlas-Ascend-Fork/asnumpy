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

#include <asnumpy/math/rounding.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/dtype_promotion.hpp>
#include <asnumpy/utils/npu_array.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_ceil.h>
#include <aclnnop/aclnn_floor.h>
#include <aclnnop/aclnn_round.h>
#include <aclnnop/aclnn_trunc.h>

#include <fmt/core.h>
#include <fmt/format.h>
#include <stdexcept>

namespace asnumpy {

namespace {

template <typename GetWs, typename Exec>
NPUArray RoundingUnaryOp(const NPUArray& x, std::optional<py::dtype> dtype, bool supports_float64,
                         bool promote_int_to_float, GetWs&& get_ws, Exec&& exec, const char* op_name,
                         const char* api_name) {
    aclDataType desired = x.aclDtype;
    if (dtype != std::nullopt) {
        desired = NPUArray::GetACLDataType(*dtype);
    } else if (promote_int_to_float && !IsFloatingAclDtype(x.aclDtype)) {
        desired = PromoteUnaryFloating(x.aclDtype);
    }
    ACL_DTYPE_WARN(x.aclDtype, desired, op_name);

    aclDataType compute = desired;
    if (!IsFloatingAclDtype(desired)) {
        // ceil/trunc/fix keep integer dtype in NumPy but ACL needs floating compute.
        compute = ACL_FLOAT;
    } else {
        compute = AclComputeFloatingDtype(desired, supports_float64);
    }

    NPUArray input = EnsureAclDtype(x, compute);
    NPUArray out = EXECUTE_UNARY_OP(input, NPUArray::GetPyDtype(compute), std::forward<GetWs>(get_ws),
                                    std::forward<Exec>(exec), op_name, api_name);
    if (desired != compute) {
        return CastToDtype(out, desired);
    }
    return out;
}

} // namespace

NPUArray Around(const NPUArray& x, int decimals, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnRoundDecimals start: input_shape={}, tensorSize={}, aclDtype={}, decimals={}",
              detail::FormatShape(x.shape), x.tensorSize, AclDtypeName(x.aclDtype), decimals);
    auto shape = x.shape;
    py::dtype py_dtype = x.dtype;
    aclDataType in_dtype = NPUArray::GetACLDataType(py_dtype);
    aclDataType out_dtype = in_dtype;
    // convert out_dtype back to py::dtype for NPUArray constructor
    py::dtype out_py_dtype = NPUArray::GetPyDtype(out_dtype);
    if (dtype != std::nullopt) {
        out_py_dtype = *dtype;
        out_dtype = NPUArray::GetACLDataType(out_py_dtype);
    }
    NPUArray out(shape, out_py_dtype);

    if (out.tensorPtr == nullptr) {
        throw std::runtime_error("[rounding.cpp](around) out.tensorPtr is null, failed to allocate output tensor");
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnRoundDecimalsGetWorkspaceSize(x.tensorPtr, decimals, out.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnRoundDecimalsGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnRoundDecimals(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnRoundDecimals");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnRoundDecimals completed");
    return out;
}

NPUArray Round_(const NPUArray& x, int decimals, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnRoundDecimals start: input_shape={}, tensorSize={}, aclDtype={}, decimals={}",
              detail::FormatShape(x.shape), x.tensorSize, AclDtypeName(x.aclDtype), decimals);
    auto result = Around(x, decimals, dtype);
    LOG_INFO("aclnnRoundDecimals completed");
    return result;
}

NPUArray Rint(const NPUArray& x, std::optional<py::dtype> dtype) {
    // NumPy rint promotes integers to floating.
    return RoundingUnaryOp(
        x, dtype, /*supports_float64=*/true, /*promote_int_to_float=*/true,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnRoundGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnRound(workspace, workspaceSize, executor, nullptr);
        },
        "Rint", "aclnnRound");
}

NPUArray Fix(const NPUArray& x, std::optional<py::dtype> dtype) {
    // aclnnTrunc often float32-only; keep NumPy dtype via cast-back.
    return RoundingUnaryOp(
        x, dtype, /*supports_float64=*/false, /*promote_int_to_float=*/false,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnTruncGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnTrunc(workspace, workspaceSize, executor, nullptr);
        },
        "Fix", "aclnnTrunc");
}

NPUArray Floor(const NPUArray& x, std::optional<py::dtype> dtype) {
    return RoundingUnaryOp(
        x, dtype, /*supports_float64=*/true, /*promote_int_to_float=*/false,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnFloorGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnFloor(workspace, workspaceSize, executor, nullptr);
        },
        "Floor", "aclnnFloor");
}

NPUArray Ceil(const NPUArray& x, std::optional<py::dtype> dtype) {
    return RoundingUnaryOp(
        x, dtype, /*supports_float64=*/true, /*promote_int_to_float=*/false,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnCeilGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnCeil(workspace, workspaceSize, executor, nullptr);
        },
        "Ceil", "aclnnCeil");
}

NPUArray Trunc(const NPUArray& x, std::optional<py::dtype> dtype) {
    return RoundingUnaryOp(
        x, dtype, /*supports_float64=*/false, /*promote_int_to_float=*/false,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnTruncGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnTrunc(workspace, workspaceSize, executor, nullptr);
        },
        "Trunc", "aclnnTrunc");
}

} // namespace asnumpy
