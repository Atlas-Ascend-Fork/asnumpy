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

#include <asnumpy/math/sums_products_differences.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_cumprod.h>
#include <aclnnop/aclnn_cumsum.h>
#include <aclnnop/aclnn_flatten.h>
#include <aclnnop/aclnn_linalg_cross.h>
#include <aclnnop/aclnn_nan_to_num.h>
#include <aclnnop/aclnn_prod.h>
#include <aclnnop/aclnn_reduce_nansum.h>
#include <aclnnop/aclnn_reduce_sum.h>

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
NPUArray Prod(const NPUArray& a, int64_t axis, bool keepdims, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnProdDim start: input_shape={}, tensorSize={}, aclDtype={}, axis={}, keepdims={}",
              detail::FormatShape(a.shape), a.tensorSize, AclDtypeName(a.aclDtype), axis, keepdims);
    py::dtype outDtype = dtype.has_value() ? dtype.value() : a.dtype;
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
    auto result = NPUArray(shape, outDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnProdDimGetWorkspaceSize(a.tensorPtr, axis, keepdims, result.aclDtype, result.tensorPtr,
                                              &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnProdDimGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnProdDim(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnProdDim");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnProdDim completed");
    return result;
}

double Prod(const NPUArray& a) {
    LOG_DEBUG("aclnnProd start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(a.shape), a.tensorSize,
              AclDtypeName(a.aclDtype));
    std::vector<int64_t> shape = {1};
    auto result = NPUArray(shape, a.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnProdGetWorkspaceSize(a.tensorPtr, result.aclDtype, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnProdGetWorkspaceSize");
    void* workspaceAddr = nullptr;
    AclWorkspace workspace(workspaceSize);
    error = aclnnProd(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnProd");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnProd completed");

    py::array temp = result.ToNumpy();
    py::dtype dt = temp.dtype();
    py::buffer_info buf = temp.request();
    if (dt.is(py::dtype::of<int>())) {
        int* results = static_cast<int*>(buf.ptr);
        return results[0];
    } else if (dt.is(py::dtype::of<double>())) {
        double* results = static_cast<double*>(buf.ptr);
        return results[0];
    } else if (dt.is(py::dtype::of<float>())) {
        float* results = static_cast<float*>(buf.ptr);
        return results[0];
    } else {
        throw std::runtime_error(fmt::format("[sums_products_differences.cpp]({}) unsupported dtype", __func__));
    }
    return 0;
}

NPUArray Sum(const NPUArray& a, int64_t axis, bool keepdims, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnReduceSum start: input_shape={}, tensorSize={}, aclDtype={}, axis={}, keepdims={}",
              detail::FormatShape(a.shape), a.tensorSize, AclDtypeName(a.aclDtype), axis, keepdims);
    py::dtype outDtype = dtype.has_value() ? dtype.value() : a.dtype;
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
    std::vector<int64_t> tmp{axis};
    aclIntArray* axis_array = aclCreateIntArray(tmp.data(), tmp.size());
    auto result = NPUArray(shape, outDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnReduceSumGetWorkspaceSize(a.tensorPtr, axis_array, keepdims, result.aclDtype, result.tensorPtr,
                                                &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnReduceSumGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnReduceSum(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnReduceSum");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    aclDestroyIntArray(axis_array);
    LOG_INFO("aclnnReduceSum completed");
    return result;
}

double Sum(const NPUArray& a) {
    LOG_DEBUG("aclnnFlatten start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(a.shape),
              a.tensorSize, AclDtypeName(a.aclDtype));
    std::vector<int64_t> shape = a.shape;
    int64_t pro = 1;
    for (int i = 0; i < shape.size(); i++) {
        pro = pro * shape[i];
    }
    shape = {1, pro};
    auto temp = NPUArray(shape, a.aclDtype);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnFlattenGetWorkspaceSize(a.tensorPtr, 0, temp.tensorPtr, &workspaceSize1, &executor1);
    ACLNN_CHECK(error1, "aclnnFlattenGetWorkspaceSize");
    AclWorkspace workspace1(workspaceSize1);
    error1 = aclnnFlatten(workspace1.get(), workspaceSize1, executor1, nullptr);
    ACLNN_CHECK(error1, "aclnnFlatten");
    error1 = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error1, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnFlatten completed");

    LOG_DEBUG("aclnnReduceSum start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(temp.shape),
              temp.tensorSize, AclDtypeName(temp.aclDtype));
    std::vector<int64_t> tmp{1};
    aclIntArray* axis_array = aclCreateIntArray(tmp.data(), tmp.size());
    auto result = NPUArray({1}, a.aclDtype);
    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    auto error2 = aclnnReduceSumGetWorkspaceSize(temp.tensorPtr, axis_array, false, result.aclDtype, result.tensorPtr,
                                                 &workspaceSize2, &executor2);
    ACLNN_CHECK(error2, "aclnnReduceSumGetWorkspaceSize");
    AclWorkspace workspace2(workspaceSize2);
    error2 = aclnnReduceSum(workspace2.get(), workspaceSize2, executor2, nullptr);
    ACLNN_CHECK(error2, "aclnnReduceSum");
    error2 = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error2, "aclrtSynchronizeDevice");
    aclDestroyIntArray(axis_array);
    LOG_INFO("aclnnReduceSum completed");

    py::array x = result.ToNumpy();
    py::dtype dt = x.dtype();
    py::buffer_info buf = x.request();
    if (dt.is(py::dtype::of<int>())) {
        int* results = static_cast<int*>(buf.ptr);
        return results[0];
    } else if (dt.is(py::dtype::of<double>())) {
        double* results = static_cast<double*>(buf.ptr);
        return results[0];
    } else if (dt.is(py::dtype::of<float>())) {
        float* results = static_cast<float*>(buf.ptr);
        return results[0];
    } else {
        throw std::runtime_error(fmt::format("[sums_products_differences.cpp]({}) unsupported dtype", __func__));
    }
    return 0;
}

NPUArray Nanprod(const NPUArray& a, int64_t axis, bool keepdims, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnNanToNum start: input_shape={}, tensorSize={}, aclDtype={}, axis={}, keepdims={}",
              detail::FormatShape(a.shape), a.tensorSize, AclDtypeName(a.aclDtype), axis, keepdims);
    py::dtype outDtype = dtype.has_value() ? dtype.value() : a.dtype;
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
    float scalar = 1.0;
    auto temp = NPUArray(a.shape, a.aclDtype);
    auto result = NPUArray(shape, outDtype);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnNanToNumGetWorkspaceSize(a.tensorPtr, scalar, std::numeric_limits<float>::infinity(),
                                                -std::numeric_limits<float>::infinity(), temp.tensorPtr,
                                                &workspaceSize1, &executor1);
    ACLNN_CHECK(error1, "aclnnNanToNumGetWorkspaceSize");
    AclWorkspace workspace1(workspaceSize1);
    error1 = aclnnNanToNum(workspace1.get(), workspaceSize1, executor1, nullptr);
    ACLNN_CHECK(error1, "aclnnNanToNum");
    error1 = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error1, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnNanToNum completed");

    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    LOG_DEBUG("aclnnProdDim start: input_shape={}, aclDtype={}, axis={}, keepdims={}", detail::FormatShape(temp.shape),
              AclDtypeName(temp.aclDtype), axis, keepdims);
    auto error2 = aclnnProdDimGetWorkspaceSize(temp.tensorPtr, axis, keepdims, result.aclDtype, result.tensorPtr,
                                               &workspaceSize2, &executor2);
    ACLNN_CHECK(error2, "aclnnProdDimGetWorkspaceSize");
    AclWorkspace workspace2(workspaceSize2);
    error2 = aclnnProdDim(workspace2.get(), workspaceSize2, executor2, nullptr);
    ACLNN_CHECK(error2, "aclnnProdDim");
    error2 = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error2, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnProdDim completed");
    return result;
}

double Nanprod(const NPUArray& a) {
    LOG_DEBUG("aclnnNanToNum start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(a.shape),
              a.tensorSize, AclDtypeName(a.aclDtype));
    std::vector<int64_t> shape = {1};
    float scalar = 1.0;
    auto temp = NPUArray(a.shape, a.aclDtype);
    auto result = NPUArray(shape, a.aclDtype);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnNanToNumGetWorkspaceSize(a.tensorPtr, scalar, std::numeric_limits<float>::infinity(),
                                                -std::numeric_limits<float>::infinity(), temp.tensorPtr,
                                                &workspaceSize1, &executor1);
    ACLNN_CHECK(error1, "aclnnNanToNumGetWorkspaceSize");
    AclWorkspace workspace1(workspaceSize1);
    error1 = aclnnNanToNum(workspace1.get(), workspaceSize1, executor1, nullptr);
    ACLNN_CHECK(error1, "aclnnNanToNum");
    error1 = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error1, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnNanToNum completed");

    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    LOG_DEBUG("aclnnProd start: input_shape={}, aclDtype={}", detail::FormatShape(temp.shape),
              AclDtypeName(temp.aclDtype));
    auto error2 =
        aclnnProdGetWorkspaceSize(temp.tensorPtr, result.aclDtype, result.tensorPtr, &workspaceSize2, &executor2);
    ACLNN_CHECK(error2, "aclnnProdGetWorkspaceSize");
    AclWorkspace workspace2(workspaceSize2);
    error2 = aclnnProd(workspace2.get(), workspaceSize2, executor2, nullptr);
    ACLNN_CHECK(error2, "aclnnProd");
    error2 = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error2, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnProd completed");

    py::array tmp = result.ToNumpy();
    py::dtype dt = tmp.dtype();
    py::buffer_info buf = tmp.request();
    if (dt.is(py::dtype::of<int>())) {
        int* results = static_cast<int*>(buf.ptr);
        return results[0];
    } else if (dt.is(py::dtype::of<double>())) {
        double* results = static_cast<double*>(buf.ptr);
        return results[0];
    } else if (dt.is(py::dtype::of<float>())) {
        float* results = static_cast<float*>(buf.ptr);
        return results[0];
    } else {
        throw std::runtime_error(fmt::format("[sums_products_differences.cpp]({}) unsupported dtype", __func__));
    }
    return 0;
}

NPUArray Nansum(const NPUArray& a, int64_t axis, bool keepdims, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnReduceNansum start: input_shape={}, tensorSize={}, aclDtype={}, axis={}, keepdims={}",
              detail::FormatShape(a.shape), a.tensorSize, AclDtypeName(a.aclDtype), axis, keepdims);
    py::dtype outDtype = dtype.has_value() ? dtype.value() : a.dtype;
    auto shape = a.shape;
    float scalar = 0.0;
    int64_t ax = axis;
    if (axis < 0) {
        ax = shape.size() + axis;
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
    auto error = aclnnReduceNansumGetWorkspaceSize(a.tensorPtr, axis_array, keepdims, result.aclDtype, result.tensorPtr,
                                                   &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnReduceNansumGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnReduceNansum(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnReduceNansum");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    aclDestroyIntArray(axis_array);
    LOG_INFO("aclnnReduceNansum completed");
    return result;
}

double Nansum(const NPUArray& a) {
    LOG_DEBUG("aclnnFlatten start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(a.shape),
              a.tensorSize, AclDtypeName(a.aclDtype));
    auto shape = a.shape;
    int64_t pro = 1;
    for (int i = 0; i < shape.size(); i++) {
        pro = pro * shape[i];
    }
    shape = {1, pro};
    auto temp = NPUArray(shape, a.aclDtype);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnFlattenGetWorkspaceSize(a.tensorPtr, 0, temp.tensorPtr, &workspaceSize1, &executor1);
    ACLNN_CHECK(error1, "aclnnFlattenGetWorkspaceSize");
    AclWorkspace workspace1(workspaceSize1);
    error1 = aclnnFlatten(workspace1.get(), workspaceSize1, executor1, nullptr);
    ACLNN_CHECK(error1, "aclnnFlatten");
    error1 = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error1, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnFlatten completed");

    LOG_DEBUG("aclnnReduceNansum start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(temp.shape),
              temp.tensorSize, AclDtypeName(temp.aclDtype));
    std::vector<int64_t> tmp{1};
    aclIntArray* axis_array = aclCreateIntArray(tmp.data(), tmp.size());
    auto result = NPUArray({1}, a.aclDtype);
    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    auto error2 = aclnnReduceNansumGetWorkspaceSize(temp.tensorPtr, axis_array, false, result.aclDtype,
                                                    result.tensorPtr, &workspaceSize2, &executor2);
    ACLNN_CHECK(error2, "aclnnReduceNansumGetWorkspaceSize");
    AclWorkspace workspace2(workspaceSize2);
    error2 = aclnnReduceNansum(workspace2.get(), workspaceSize2, executor2, nullptr);
    ACLNN_CHECK(error2, "aclnnReduceNansum");
    error2 = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error2, "aclrtSynchronizeDevice");
    aclDestroyIntArray(axis_array);
    LOG_INFO("aclnnReduceNansum completed");

    py::array x = result.ToNumpy();
    py::dtype dt = x.dtype();
    py::buffer_info buf = x.request();
    if (dt.is(py::dtype::of<int>())) {
        int* results = static_cast<int*>(buf.ptr);
        return results[0];
    } else if (dt.is(py::dtype::of<double>())) {
        double* results = static_cast<double*>(buf.ptr);
        return results[0];
    } else if (dt.is(py::dtype::of<float>())) {
        float* results = static_cast<float*>(buf.ptr);
        return results[0];
    } else {
        throw std::runtime_error(fmt::format("[sums_products_differences.cpp]({}) unsupported dtype", __func__));
    }
    return 0;
}

NPUArray Cumprod(const NPUArray& a, int64_t axis, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnCumprod start: input_shape={}, tensorSize={}, aclDtype={}, axis={}", detail::FormatShape(a.shape),
              a.tensorSize, AclDtypeName(a.aclDtype), axis);
    py::dtype outDtype = dtype.has_value() ? dtype.value() : a.dtype;
    auto shape = a.shape;
    auto axis_scalar = aclCreateScalar(&axis, ACL_INT64);
    auto result = NPUArray(shape, outDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnCumprodGetWorkspaceSize(a.tensorPtr, axis_scalar, result.aclDtype, result.tensorPtr,
                                              &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnCumprodGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnCumprod(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnCumprod");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    aclDestroyScalar(axis_scalar);
    LOG_INFO("aclnnCumprod completed");
    return result;
}

NPUArray Cumsum(const NPUArray& a, int64_t axis, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnCumsum start: input_shape={}, tensorSize={}, aclDtype={}, axis={}", detail::FormatShape(a.shape),
              a.tensorSize, AclDtypeName(a.aclDtype), axis);
    py::dtype outDtype = dtype.has_value() ? dtype.value() : a.dtype;
    auto shape = a.shape;
    auto result = NPUArray(shape, outDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error =
        aclnnCumsumGetWorkspaceSize(a.tensorPtr, axis, result.aclDtype, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnCumsumGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnCumsum(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnCumsum");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnCumsum completed");
    return result;
}

NPUArray Nancumprod(const NPUArray& a, int64_t axis, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnNanToNum start: input_shape={}, tensorSize={}, aclDtype={}, axis={}", detail::FormatShape(a.shape),
              a.tensorSize, AclDtypeName(a.aclDtype), axis);
    py::dtype outDtype = dtype.has_value() ? dtype.value() : a.dtype;
    auto shape = a.shape;
    auto axis_scalar = aclCreateScalar(&axis, ACL_INT64);
    float scalar = 1.0;
    auto temp = NPUArray(shape, a.aclDtype);
    auto result = NPUArray(shape, outDtype);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnNanToNumGetWorkspaceSize(a.tensorPtr, scalar, std::numeric_limits<float>::infinity(),
                                                -std::numeric_limits<float>::infinity(), temp.tensorPtr,
                                                &workspaceSize1, &executor1);
    ACLNN_CHECK(error1, "aclnnNanToNumGetWorkspaceSize");
    AclWorkspace workspace1(workspaceSize1);
    error1 = aclnnNanToNum(workspace1.get(), workspaceSize1, executor1, nullptr);
    ACLNN_CHECK(error1, "aclnnNanToNum");
    error1 = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error1, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnNanToNum completed");

    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    LOG_DEBUG("aclnnCumprod start: input_shape={}, aclDtype={}, axis={}", detail::FormatShape(temp.shape),
              AclDtypeName(temp.aclDtype), axis);
    auto error2 = aclnnCumprodGetWorkspaceSize(temp.tensorPtr, axis_scalar, result.aclDtype, result.tensorPtr,
                                               &workspaceSize2, &executor2);
    ACLNN_CHECK(error2, "aclnnCumprodGetWorkspaceSize");
    AclWorkspace workspace2(workspaceSize2);
    error2 = aclnnCumprod(workspace2.get(), workspaceSize2, executor2, nullptr);
    ACLNN_CHECK(error2, "aclnnCumprod");
    error2 = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error2, "aclrtSynchronizeDevice");
    aclDestroyScalar(axis_scalar);
    LOG_INFO("aclnnCumprod completed");
    return result;
}

NPUArray Nancumsum(const NPUArray& a, int64_t axis, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnNanToNum start: input_shape={}, tensorSize={}, aclDtype={}, axis={}", detail::FormatShape(a.shape),
              a.tensorSize, AclDtypeName(a.aclDtype), axis);
    py::dtype outDtype = dtype.has_value() ? dtype.value() : a.dtype;
    auto shape = a.shape;
    float scalar = 0.0;
    auto temp = NPUArray(shape, a.aclDtype);
    auto result = NPUArray(shape, outDtype);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnNanToNumGetWorkspaceSize(a.tensorPtr, scalar, std::numeric_limits<float>::infinity(),
                                                -std::numeric_limits<float>::infinity(), temp.tensorPtr,
                                                &workspaceSize1, &executor1);
    ACLNN_CHECK(error1, "aclnnNanToNumGetWorkspaceSize");
    AclWorkspace workspace1(workspaceSize1);
    error1 = aclnnNanToNum(workspace1.get(), workspaceSize1, executor1, nullptr);
    ACLNN_CHECK(error1, "aclnnNanToNum");
    error1 = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error1, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnNanToNum completed");

    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    LOG_DEBUG("aclnnCumsum start: input_shape={}, aclDtype={}, axis={}", detail::FormatShape(temp.shape),
              AclDtypeName(temp.aclDtype), axis);
    auto error2 = aclnnCumsumGetWorkspaceSize(temp.tensorPtr, axis, result.aclDtype, result.tensorPtr, &workspaceSize2,
                                              &executor2);
    ACLNN_CHECK(error2, "aclnnCumsumGetWorkspaceSize");
    AclWorkspace workspace2(workspaceSize2);
    error2 = aclnnCumsum(workspace2.get(), workspaceSize2, executor2, nullptr);
    ACLNN_CHECK(error2, "aclnnCumsum");
    error2 = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error2, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnCumsum completed");
    return result;
}

NPUArray Cross(const NPUArray& a, const NPUArray& b, int64_t axis) {
    LOG_DEBUG("aclnnLinalgCross start: a_shape={}, b_shape={}, axis={}, aclDtype={}", detail::FormatShape(a.shape),
              detail::FormatShape(b.shape), axis, AclDtypeName(a.aclDtype));
    auto broadcast = GetBroadcastShape(a, b);
    auto result = NPUArray(broadcast, a.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error =
        aclnnLinalgCrossGetWorkspaceSize(a.tensorPtr, b.tensorPtr, axis, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnLinalgCrossGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnLinalgCross(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnLinalgCross");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnLinalgCross completed");
    return result;
}
} // namespace asnumpy
