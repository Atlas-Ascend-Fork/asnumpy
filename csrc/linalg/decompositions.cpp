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
#include <asnumpy/utils/shape_check.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_linalg_qr.h>
#include <aclnnop/aclnn_svd.h>

#include <algorithm>
#include <fmt/core.h>
#include <fmt/format.h>
#include <stdexcept>

using namespace asnumpy;

py::object Linalg_Qr(const NPUArray& a, const std::string& mode) {
    detail::CheckNdimsAtLeast(a, 2, "qr");
    LOG_DEBUG("aclnnLinalgQr start: input_shape={}, aclDtype={}, mode={}", detail::FormatShape(a.shape),
              AclDtypeName(a.aclDtype), mode);
    int size = a.shape.size();
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

namespace {

/// Run aclnnSvd once the output tensors have been prepared.
void RunSvd(const NPUArray& input, bool full_matrices, bool compute_uv, NPUArray& sigma, aclTensor* u, aclTensor* v) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnSvdGetWorkspaceSize(input.tensorPtr, full_matrices, compute_uv, sigma.tensorPtr, u, v,
                                          &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnSvdGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);
    error = aclnnSvd(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnSvd");

    auto sync = aclrtSynchronizeDevice();
    ACL_RT_CHECK(sync, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnSvd completed");
}

/// aclnnSvd demands non-null u/v handles even when computeUV is false; own two empty placeholders.
class EmptyUvPlaceholders {
  public:
    explicit EmptyUvPlaceholders(aclDataType dtype) {
        u_ = aclCreateTensor(shape_, 1, dtype, stride_, 0, ACL_FORMAT_ND, shape_, 1, nullptr);
        v_ = aclCreateTensor(shape_, 1, dtype, stride_, 0, ACL_FORMAT_ND, shape_, 1, nullptr);
    }

    ~EmptyUvPlaceholders() {
        aclDestroyTensor(u_);
        aclDestroyTensor(v_);
    }

    EmptyUvPlaceholders(const EmptyUvPlaceholders&) = delete;
    EmptyUvPlaceholders& operator=(const EmptyUvPlaceholders&) = delete;

    aclTensor* u() const { return u_; }
    aclTensor* v() const { return v_; }

  private:
    int64_t shape_[1] = {0};
    int64_t stride_[1] = {1};
    aclTensor* u_ = nullptr;
    aclTensor* v_ = nullptr;
};

} // namespace

py::object Linalg_Svd(const NPUArray& a, bool full_matrices, bool compute_uv) {
    const size_t rank = a.shape.size();
    if (rank < 2) {
        throw std::invalid_argument("[decompositions.cpp](svd) input must have at least two dimensions");
    }
    LOG_DEBUG("aclnnSvd start: input_shape={}, aclDtype={}, full_matrices={}, compute_uv={}",
              detail::FormatShape(a.shape), AclDtypeName(a.aclDtype), full_matrices, compute_uv);

    const int64_t m = a.shape[rank - 2];
    const int64_t n = a.shape[rank - 1];
    const int64_t k = std::min(m, n);

    std::vector<int64_t> batch(a.shape.begin(), a.shape.end() - 2);
    std::vector<int64_t> sigma_shape = batch;
    sigma_shape.push_back(k);
    std::vector<int64_t> u_shape = batch;
    u_shape.push_back(m);
    u_shape.push_back(full_matrices ? m : k);
    // aclnnSvd returns V (not V^H): reduced is (..., N, K), full is (..., N, N).
    std::vector<int64_t> v_shape = batch;
    v_shape.push_back(n);
    v_shape.push_back(full_matrices ? n : k);

    auto sigma = NPUArray(sigma_shape, a.aclDtype);
    if (!compute_uv) {
        EmptyUvPlaceholders placeholders(a.aclDtype);
        RunSvd(a, full_matrices, false, sigma, placeholders.u(), placeholders.v());
        return py::cast(std::move(sigma));
    }

    auto resultU = NPUArray(u_shape, a.aclDtype);
    auto resultV = NPUArray(v_shape, a.aclDtype);
    RunSvd(a, full_matrices, true, sigma, resultU.tensorPtr, resultV.tensorPtr);
    // NumPy returns (u, sigma, vh); vh is derived from V by the Python wrapper.
    return py::make_tuple(py::cast(std::move(resultU)), py::cast(std::move(sigma)), py::cast(std::move(resultV)));
}
