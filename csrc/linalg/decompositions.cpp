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

#include <asnumpy/linalg/decompositions.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_linalg_qr.h>

#include <fmt/core.h>
#include <fmt/format.h>
#include <stdexcept>

using namespace asnumpy;

py::object Linalg_Qr(const NPUArray& a, const std::string& mode) {
    LOG_DEBUG("aclnnLinalgQr start: input_shape={}, aclDtype={}, mode={}", detail::FormatShape(a.shape),
              AclDtypeName(a.aclDtype), mode);
    int size = a.shape.size();
    // Guard before indexing shape[size - 2]: a scalar or 1-D input makes the
    // index negative and underflows once converted to an unsigned subscript.
    if (size < 2) {
        throw std::runtime_error(fmt::format(
            "[decompositions.cpp](qr) requires an input with at least 2 dimensions, but got {}", size));
    }
    int64_t m = a.shape[size - 2];
    int64_t n = a.shape.back();
    int64_t k = m < n ? m : n;
    int64_t num = 0;
    std::vector<int64_t> shapeR;

    if (mode == "complete") {
        num = 1;
        shapeR = a.shape;
    } else if (mode == "r") {
        num = 2;
        shapeR = a.shape;
        shapeR[size - 2] = k;
    } else {
        // reduced (default)
        num = 0;
        shapeR = a.shape;
        shapeR[size - 2] = k;
    }

    auto resultR = NPUArray(shapeR, a.aclDtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    aclError error;

    if (mode == "r") {
        // r mode: only return R, create empty Q tensor directly (shape [0])
        int64_t emptyShape = 0;
        int64_t emptyStride = 1;
        aclTensor* emptyQ =
            aclCreateTensor(&emptyShape, 1, a.aclDtype, &emptyStride, 0, ACL_FORMAT_ND, &emptyShape, 1, nullptr);

        error = aclnnLinalgQrGetWorkspaceSize(a.tensorPtr, num, emptyQ, resultR.tensorPtr, &workspaceSize, &executor);
        ACLNN_CHECK(error, "aclnnLinalgQrGetWorkspaceSize");

        AclWorkspace workspace(workspaceSize);
        error = aclnnLinalgQr(workspace.get(), workspaceSize, executor, nullptr);
        ACLNN_CHECK(error, "aclnnLinalgQr");
        error = aclrtSynchronizeDevice();
        ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

        aclDestroyTensor(emptyQ);
        LOG_INFO("aclnnLinalgQr completed");
        return py::cast(std::move(resultR));
    } else {
        // complete / reduced: return (Q, R)
        std::vector<int64_t> shapeQ = a.shape;
        if (mode == "complete") {
            shapeQ.back() = m;
        } else {
            shapeQ.back() = k;
        }

        auto resultQ = NPUArray(shapeQ, a.aclDtype);

        error = aclnnLinalgQrGetWorkspaceSize(a.tensorPtr, num, resultQ.tensorPtr, resultR.tensorPtr, &workspaceSize,
                                              &executor);
        ACLNN_CHECK(error, "aclnnLinalgQrGetWorkspaceSize");

        AclWorkspace workspace(workspaceSize);
        error = aclnnLinalgQr(workspace.get(), workspaceSize, executor, nullptr);
        ACLNN_CHECK(error, "aclnnLinalgQr");
        error = aclrtSynchronizeDevice();
        ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

        LOG_INFO("aclnnLinalgQr completed");
        return py::make_tuple(py::cast(std::move(resultQ)), py::cast(std::move(resultR)));
    }
}
