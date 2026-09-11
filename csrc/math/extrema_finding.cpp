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

#include <asnumpy/math/extrema_finding.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/npu_ops_macros.hpp>

#include <aclnnop/aclnn_amax.h>
#include <aclnnop/aclnn_amin.h>
#include <aclnnop/aclnn_max.h>
#include <aclnnop/aclnn_maximum.h>
#include <aclnnop/aclnn_min.h>
#include <aclnnop/aclnn_minimum.h>
#include <aclnnop/aclnn_nan_to_num.h>

#include <cstdint>
#include <fmt/core.h>
#include <fmt/format.h>
#include <limits>
#include <stdexcept>

namespace asnumpy {

/**
 * @brief Element-wise maximum of two arrays.
 *
 * Creates an output array on NPU and computes element-wise max(x1, x2)
 * using the aclnnMaximum operator.
 *
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise maxima.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray Maximum(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    return EXECUTE_BINARY_OP(
        x1, x2, dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnMaximumGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnMaximum(workspace, workspaceSize, executor, nullptr);
        },
        "Maximum", "aclnnMaximum");
}

/**
 * @brief Element-wise minimum of two arrays.


/**
 * @brief Element-wise minimum of two arrays.
 *
 * Creates an output array on NPU and computes element-wise min(x1, x2)
 * using the aclnnMinimum operator.
 *
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise minima.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray Minimum(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    return EXECUTE_BINARY_OP(
        x1, x2, dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnMinimumGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnMinimum(workspace, workspaceSize, executor, nullptr);
        },
        "Minimum", "aclnnMinimum");
}

NPUArray Fmax(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    return EXECUTE_BINARY_OP(
        x1, x2, dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnMaximumGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnMaximum(workspace, workspaceSize, executor, nullptr);
        },
        "Fmax", "aclnnMaximum");
}

NPUArray Fmin(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    return EXECUTE_BINARY_OP(
        x1, x2, dtype,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnMinimumGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnMinimum(workspace, workspaceSize, executor, nullptr);
        },
        "Fmin", "aclnnMinimum");
}

NPUArray Max(const NPUArray& a, int64_t axis, bool keepdims) {
    LOG_DEBUG("aclnnAmax start: input_shape={}, tensorSize={}, aclDtype={}, axis={}, keepdims={}",
              detail::FormatShape(a.shape), a.tensorSize, AclDtypeName(a.aclDtype), axis, keepdims);
    auto shape = a.shape;
    int64_t ax = axis;
    if (axis < 0) {
        ax = shape.size() + axis;
    }
    if (keepdims) {
        shape[ax] = 1;
    } else {
        shape.erase(shape.begin() + ax);
    }
    std::vector<int64_t> data = {ax};
    auto axis_array = aclCreateIntArray(data.data(), data.size());
    auto result = NPUArray(shape, a.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error =
        aclnnAmaxGetWorkspaceSize(a.tensorPtr, axis_array, keepdims, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnAmaxGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnAmax(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnAmax");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnAmax completed");
    return result;
}

double Max(const NPUArray& a) {
    LOG_DEBUG("aclnnMax start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(a.shape), a.tensorSize,
              AclDtypeName(a.aclDtype));
    std::vector<int64_t> shape = {1};
    auto result = NPUArray(shape, a.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnMaxGetWorkspaceSize(a.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnMaxGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnMax(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnMax");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

    py::array x = result.ToNumpy();
    py::dtype dt = x.dtype();
    py::buffer_info buf = x.request();
    if (dt.is(py::dtype::of<int>())) {
        int* results = static_cast<int*>(buf.ptr);
        LOG_INFO("aclnnMax completed");
        return results[0];
    } else if (dt.is(py::dtype::of<double>())) {
        double* results = static_cast<double*>(buf.ptr);
        LOG_INFO("aclnnMax completed");
        return results[0];
    } else if (dt.is(py::dtype::of<float>())) {
        float* results = static_cast<float*>(buf.ptr);
        LOG_INFO("aclnnMax completed");
        return results[0];
    } else {
        throw std::runtime_error(fmt::format("[extrema_finding.cpp]({}) unsupported dtype", __func__));
    }
    return 0;
}

NPUArray Nanmax(const NPUArray& a, int64_t axis, bool keepdims) {
    LOG_DEBUG("aclnnNanToNum start: input_shape={}, tensorSize={}, aclDtype={}, axis={}, keepdims={}",
              detail::FormatShape(a.shape), a.tensorSize, AclDtypeName(a.aclDtype), axis, keepdims);
    auto shape = a.shape;
    auto temp = NPUArray(a.shape, a.aclDtype);
    int64_t ax = axis;
    if (axis < 0) {
        ax = shape.size() + axis;
    }
    if (keepdims) {
        shape[ax] = 1;
    } else {
        shape.erase(shape.begin() + ax);
    }
    std::vector<int64_t> data = {ax};
    auto axis_array = aclCreateIntArray(data.data(), data.size());
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnNanToNumGetWorkspaceSize(
        a.tensorPtr, -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), temp.tensorPtr, &workspaceSize1, &executor1);
    ACLNN_CHECK(error1, "aclnnNanToNumGetWorkspaceSize");

    AclWorkspace workspace1(workspaceSize1);

    error1 = aclnnNanToNum(workspace1.get(), workspaceSize1, executor1, nullptr);
    ACLNN_CHECK(error1, "aclnnNanToNum");
    error1 = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error1, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnNanToNum completed");

    LOG_DEBUG("aclnnAmax start: input_shape={}, aclDtype={}, axis={}, keepdims={}", detail::FormatShape(temp.shape),
              AclDtypeName(temp.aclDtype), axis, keepdims);
    auto result = NPUArray(shape, a.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error =
        aclnnAmaxGetWorkspaceSize(temp.tensorPtr, axis_array, keepdims, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnAmaxGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnAmax(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnAmax");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnAmax completed");
    return result;
}

double Nanmax(const NPUArray& a) {
    LOG_DEBUG("aclnnNanToNum start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(a.shape),
              a.tensorSize, AclDtypeName(a.aclDtype));
    auto temp = NPUArray(a.shape, a.aclDtype);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnNanToNumGetWorkspaceSize(
        a.tensorPtr, -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), temp.tensorPtr, &workspaceSize1, &executor1);
    ACLNN_CHECK(error1, "aclnnNanToNumGetWorkspaceSize");

    AclWorkspace workspace1(workspaceSize1);

    error1 = aclnnNanToNum(workspace1.get(), workspaceSize1, executor1, nullptr);
    ACLNN_CHECK(error1, "aclnnNanToNum");
    error1 = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error1, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnNanToNum completed");

    std::vector<int64_t> shape = {1};
    auto result = NPUArray(shape, a.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    LOG_DEBUG("aclnnMax start: input_shape={}, aclDtype={}", detail::FormatShape(temp.shape),
              AclDtypeName(temp.aclDtype));
    auto error = aclnnMaxGetWorkspaceSize(temp.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnMaxGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnMax(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnMax");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

    py::array x = result.ToNumpy();
    py::dtype dt = x.dtype();
    py::buffer_info buf = x.request();
    if (dt.is(py::dtype::of<int>())) {
        int* results = static_cast<int*>(buf.ptr);
        LOG_INFO("aclnnMax completed");
        return results[0];
    } else if (dt.is(py::dtype::of<double>())) {
        double* results = static_cast<double*>(buf.ptr);
        LOG_INFO("aclnnMax completed");
        return results[0];
    } else if (dt.is(py::dtype::of<float>())) {
        float* results = static_cast<float*>(buf.ptr);
        LOG_INFO("aclnnMax completed");
        return results[0];
    } else {
        throw std::runtime_error(fmt::format("[extrema_finding.cpp]({}) unsupported dtype", __func__));
    }
    return 0;
}

NPUArray Min(const NPUArray& a, int64_t axis, bool keepdims) {
    LOG_DEBUG("aclnnAmin start: input_shape={}, tensorSize={}, aclDtype={}, axis={}, keepdims={}",
              detail::FormatShape(a.shape), a.tensorSize, AclDtypeName(a.aclDtype), axis, keepdims);
    auto shape = a.shape;
    int64_t ax = axis;
    if (axis < 0) {
        ax = shape.size() + axis;
    }
    if (keepdims) {
        shape[ax] = 1;
    } else {
        shape.erase(shape.begin() + ax);
    }
    std::vector<int64_t> data = {ax};
    auto axis_array = aclCreateIntArray(data.data(), data.size());
    auto result = NPUArray(shape, a.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error =
        aclnnAminGetWorkspaceSize(a.tensorPtr, axis_array, keepdims, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnAminGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnAmin(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnAmin");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnAmin completed");
    return result;
}

double Min(const NPUArray& a) {
    LOG_DEBUG("aclnnMin start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(a.shape), a.tensorSize,
              AclDtypeName(a.aclDtype));
    std::vector<int64_t> shape = {1};
    auto result = NPUArray(shape, a.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnMinGetWorkspaceSize(a.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnMinGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnMin(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnMin");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

    py::array x = result.ToNumpy();
    py::dtype dt = x.dtype();
    py::buffer_info buf = x.request();
    if (dt.is(py::dtype::of<int>())) {
        int* results = static_cast<int*>(buf.ptr);
        LOG_INFO("aclnnMin completed");
        return results[0];
    } else if (dt.is(py::dtype::of<double>())) {
        double* results = static_cast<double*>(buf.ptr);
        LOG_INFO("aclnnMin completed");
        return results[0];
    } else if (dt.is(py::dtype::of<float>())) {
        float* results = static_cast<float*>(buf.ptr);
        LOG_INFO("aclnnMin completed");
        return results[0];
    } else {
        throw std::runtime_error(fmt::format("[extrema_finding.cpp]({}) unsupported dtype", __func__));
    }
    return 0;
}

NPUArray Nanmin(const NPUArray& a, int64_t axis, bool keepdims) {
    LOG_DEBUG("aclnnNanToNum start: input_shape={}, tensorSize={}, aclDtype={}, axis={}, keepdims={}",
              detail::FormatShape(a.shape), a.tensorSize, AclDtypeName(a.aclDtype), axis, keepdims);
    auto shape = a.shape;
    auto temp = NPUArray(a.shape, a.aclDtype);
    int64_t ax = axis;
    if (axis < 0) {
        ax = shape.size() + axis;
    }
    if (keepdims) {
        shape[ax] = 1;
    } else {
        shape.erase(shape.begin() + ax);
    }
    std::vector<int64_t> data = {ax};
    auto axis_array = aclCreateIntArray(data.data(), data.size());
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnNanToNumGetWorkspaceSize(
        a.tensorPtr, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), temp.tensorPtr, &workspaceSize1, &executor1);
    ACLNN_CHECK(error1, "aclnnNanToNumGetWorkspaceSize");

    AclWorkspace workspace1(workspaceSize1);

    error1 = aclnnNanToNum(workspace1.get(), workspaceSize1, executor1, nullptr);
    ACLNN_CHECK(error1, "aclnnNanToNum");
    error1 = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error1, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnNanToNum completed");

    LOG_DEBUG("aclnnAmin start: input_shape={}, aclDtype={}, axis={}, keepdims={}", detail::FormatShape(temp.shape),
              AclDtypeName(temp.aclDtype), axis, keepdims);
    auto result = NPUArray(shape, a.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error =
        aclnnAminGetWorkspaceSize(temp.tensorPtr, axis_array, keepdims, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnAminGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnAmin(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnAmin");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnAmin completed");
    return result;
}

double Nanmin(const NPUArray& a) {
    LOG_DEBUG("aclnnNanToNum start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(a.shape),
              a.tensorSize, AclDtypeName(a.aclDtype));
    auto temp = NPUArray(a.shape, a.aclDtype);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnNanToNumGetWorkspaceSize(
        a.tensorPtr, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), temp.tensorPtr, &workspaceSize1, &executor1);
    ACLNN_CHECK(error1, "aclnnNanToNumGetWorkspaceSize");

    AclWorkspace workspace1(workspaceSize1);

    error1 = aclnnNanToNum(workspace1.get(), workspaceSize1, executor1, nullptr);
    ACLNN_CHECK(error1, "aclnnNanToNum");
    error1 = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error1, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnNanToNum completed");

    std::vector<int64_t> shape = {1};
    auto result = NPUArray(shape, a.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    LOG_DEBUG("aclnnMin start: input_shape={}, aclDtype={}", detail::FormatShape(temp.shape),
              AclDtypeName(temp.aclDtype));
    auto error = aclnnMinGetWorkspaceSize(temp.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnMinGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnMin(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnMin");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

    py::array x = result.ToNumpy();
    py::dtype dt = x.dtype();
    py::buffer_info buf = x.request();
    if (dt.is(py::dtype::of<int>())) {
        int* results = static_cast<int*>(buf.ptr);
        LOG_INFO("aclnnMin completed");
        return results[0];
    } else if (dt.is(py::dtype::of<double>())) {
        double* results = static_cast<double*>(buf.ptr);
        LOG_INFO("aclnnMin completed");
        return results[0];
    } else if (dt.is(py::dtype::of<float>())) {
        float* results = static_cast<float*>(buf.ptr);
        LOG_INFO("aclnnMin completed");
        return results[0];
    } else {
        throw std::runtime_error(fmt::format("[extrema_finding.cpp]({}) unsupported dtype", __func__));
    }
    return 0;
}

} // namespace asnumpy
