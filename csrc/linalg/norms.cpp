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

#include <asnumpy/linalg/norms.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/cast.hpp>
#include <asnumpy/utils/shape_check.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_exp.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_norm.h>
#include <aclnnop/aclnn_slogdet.h>

#include <fmt/core.h>
#include <fmt/format.h>
#include <stdexcept>

using namespace asnumpy;

namespace {

// Cast to a target dtype. Thin alias kept so the call sites below read unchanged; the primitive
// itself now lives in asnumpy/utils/cast.hpp and is shared with the promotion layer and astype.
NPUArray CastToDtype(const NPUArray& input, aclDataType targetDtype) { return asnumpy::CastTo(input, targetDtype); }

} // anonymous namespace

NPUArray Linalg_Norm(const NPUArray& a, double ord, const std::vector<int64_t>& axis, bool keepdims) {
    auto axes = detail::NormalizeAxes(axis, a.shape.size(), "norm");
    LOG_DEBUG("aclnnNorm start: input_shape={}, aclDtype={}, ord={}, axis={}, keepdims={}",
              detail::FormatShape(a.shape), AclDtypeName(a.aclDtype), ord, detail::FormatShape(axis), keepdims);
    auto shape = a.shape;
    if (keepdims) {
        for (auto ax : axes) {
            shape[ax] = 1;
        }
    } else {
        // sorted ascending; erase back-to-front
        for (auto it = axes.rbegin(); it != axes.rend(); ++it) {
            shape.erase(shape.begin() + *it);
        }
    }
    auto ord_scalar = aclCreateScalar(&ord, ACL_DOUBLE);
    aclIntArray* axis_array = aclCreateIntArray(axes.data(), axes.size());
    auto result = NPUArray(shape, ACL_FLOAT);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnNormGetWorkspaceSize(a.tensorPtr, ord_scalar, axis_array, keepdims, result.tensorPtr,
                                           &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnNormGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnNorm(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnNorm");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    aclDestroyScalar(ord_scalar);
    aclDestroyIntArray(axis_array);
    LOG_INFO("aclnnNorm completed");
    return result;
}

NPUArray Linalg_Det(const NPUArray& a) {
    detail::CheckNdimsAtLeast(a, 2, "det");
    LOG_DEBUG("aclnnSlogdet start: input_shape={}, aclDtype={}", detail::FormatShape(a.shape),
              AclDtypeName(a.aclDtype));
    std::vector<int64_t> shape = a.shape;
    shape.erase(shape.end() - 2, shape.end());

    // Cast input to double for numerical stability in LU decomposition
    NPUArray aDouble = (a.aclDtype != ACL_DOUBLE) ? CastToDtype(a, ACL_DOUBLE) : a;

    auto sign = NPUArray(shape, ACL_DOUBLE);
    auto logdet = NPUArray(shape, ACL_DOUBLE);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error =
        aclnnSlogdetGetWorkspaceSize(aDouble.tensorPtr, sign.tensorPtr, logdet.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnSlogdetGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnSlogdet(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnSlogdet");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

    auto absDet = EXECUTE_UNARY_OP(
        logdet, NPUArray::GetPyDtype(ACL_DOUBLE),
        [](aclTensor* in, aclTensor* out, uint64_t* ws, aclOpExecutor** exec) {
            return aclnnExpGetWorkspaceSize(in, out, ws, exec);
        },
        [](void* ws, uint64_t wsSize, aclOpExecutor* exec, void* stream) {
            return aclnnExp(ws, wsSize, exec, nullptr);
        },
        "Linalg_Det_Exp", "aclnnExp");

    auto detDouble = EXECUTE_BINARY_OP(
        sign, absDet, NPUArray::GetPyDtype(ACL_DOUBLE),
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* ws, aclOpExecutor** exec) {
            return aclnnMulGetWorkspaceSize(in1, in2, out, ws, exec);
        },
        [](void* ws, uint64_t wsSize, aclOpExecutor* exec, void* stream) {
            return aclnnMul(ws, wsSize, exec, nullptr);
        },
        "Linalg_Det_Mul", "aclnnMul");

    // Cast result back to input dtype
    LOG_INFO("aclnnSlogdet completed");
    return (a.aclDtype != ACL_DOUBLE) ? CastToDtype(detDouble, a.aclDtype) : detDouble;
}

std::pair<NPUArray, NPUArray> Linalg_Slogdet(const NPUArray& a) {
    detail::CheckNdimsAtLeast(a, 2, "slogdet");
    LOG_DEBUG("aclnnSlogdet start: input_shape={}, aclDtype={}", detail::FormatShape(a.shape),
              AclDtypeName(a.aclDtype));
    auto shape = a.shape;
    shape.erase(shape.end() - 2, shape.end());

    // Cast input to double for numerical stability in LU decomposition
    NPUArray aDouble = (a.aclDtype != ACL_DOUBLE) ? CastToDtype(a, ACL_DOUBLE) : a;

    auto signout = NPUArray(shape, ACL_DOUBLE);
    auto logout = NPUArray(shape, ACL_DOUBLE);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error =
        aclnnSlogdetGetWorkspaceSize(aDouble.tensorPtr, signout.tensorPtr, logout.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnSlogdetGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnSlogdet(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnSlogdet");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

    // Cast results back to input dtype
    if (a.aclDtype != ACL_DOUBLE) {
        LOG_INFO("aclnnSlogdet completed");
        return {CastToDtype(signout, a.aclDtype), CastToDtype(logout, a.aclDtype)};
    }
    LOG_INFO("aclnnSlogdet completed");
    return {signout, logout};
}
