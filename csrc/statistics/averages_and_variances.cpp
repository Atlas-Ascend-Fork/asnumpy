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

#include <asnumpy/statistics/averages_and_variances.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_flatten.h>
#include <aclnnop/aclnn_mean.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fmt/core.h>
#include <fmt/format.h>
#include <limits>
#include <optional>
#include <pybind11/numpy.h>
#include <stdexcept>

namespace asnumpy {
namespace {
int64_t CalculateTotalElements(const std::vector<int64_t>& shape) {
    int64_t total = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        total *= shape[i];
    }
    return total;
}

NPUArray FlattenArray(const NPUArray& a) {
    LOG_DEBUG("aclnnFlatten start: input_shape={}, aclDtype={}", detail::FormatShape(a.shape),
              AclDtypeName(a.aclDtype));
    int64_t totalElements = CalculateTotalElements(a.shape);
    std::vector<int64_t> flatShape = {1, totalElements};
    auto temp = NPUArray(flatShape, a.aclDtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnFlattenGetWorkspaceSize(a.tensorPtr, 0, temp.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnFlattenGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnFlatten(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnFlatten");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnFlatten completed");

    return temp;
}
} // namespace

NPUArray Mean(const NPUArray& a, int64_t axis, bool keepdims, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnMean start: input_shape={}, tensorSize={}, aclDtype={}, axis={}, keepdims={}",
              detail::FormatShape(a.shape), a.tensorSize, AclDtypeName(a.aclDtype), axis, keepdims);
    py::dtype outDtype = dtype.has_value() ? dtype.value() : a.dtype;
    auto shape = a.shape;
    int64_t ax = axis;
    if (axis < 0) {
        ax = shape.size() + axis;
    }
    if (ax < 0 || ax >= static_cast<int64_t>(shape.size())) {
        throw std::out_of_range("axis out of range");
    }
    if (keepdims) {
        shape[ax] = 1;
    } else {
        shape.erase(shape.begin() + ax);
    }
    std::vector<int64_t> tmp{axis};
    aclIntArray* axis_array = aclCreateIntArray(tmp.data(), tmp.size());
    auto result = NPUArray(shape, outDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnMeanGetWorkspaceSize(a.tensorPtr, axis_array, keepdims, result.aclDtype, result.tensorPtr,
                                           &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnMeanGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnMean(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnMean");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnMean completed");
    return result;
}

py::object Mean(const NPUArray& a, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnMean start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(a.shape), a.tensorSize,
              AclDtypeName(a.aclDtype));
    auto temp = FlattenArray(a);

    std::vector<int64_t> tmp{1};
    aclIntArray* axis_array = aclCreateIntArray(tmp.data(), tmp.size());
    auto outDtype = dtype.has_value() ? dtype.value() : a.dtype;
    auto result = NPUArray({1}, outDtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnMeanGetWorkspaceSize(temp.tensorPtr, axis_array, false, result.aclDtype, result.tensorPtr,
                                           &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnMeanGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnMean(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnMean");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnMean completed");
    return result.ToNumpy()[py::int_(0)];
}
} // namespace asnumpy
