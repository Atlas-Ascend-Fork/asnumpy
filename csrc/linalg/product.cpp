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

#include <asnumpy/linalg/product.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnn/aclnn_base.h>
#include <cstring>
#include <fmt/core.h>
#include <fmt/format.h>
#include <stdexcept>
#include "aclnnop/aclnn_dot.h"
#include "aclnnop/aclnn_flatten.h"
#include "aclnnop/aclnn_mm.h"
#include "aclnnop/aclnn_mul.h"

using namespace asnumpy;

NPUArray Matmul(const NPUArray& x1, const NPUArray& x2) {
    LOG_DEBUG("aclnnMatmul start: x1_shape={}, x2_shape={}, aclDtype={}", detail::FormatShape(x1.shape),
              detail::FormatShape(x2.shape), AclDtypeName(x1.aclDtype));
    size_t x1_ndim = x1.shape.size();
    size_t x2_ndim = x2.shape.size();

    // 1-D × 1-D: inner product → scalar
    if (x1_ndim == 1 && x2_ndim == 1) {
        NPUArray out({}, x1.aclDtype);
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        auto err = aclnnDotGetWorkspaceSize(x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &ws, &exec);
        ACLNN_CHECK(err, "aclnnDotGetWorkspaceSize");
        AclWorkspace workspace(ws);
        err = aclnnDot(workspace.get(), ws, exec, nullptr);
        ACLNN_CHECK(err, "aclnnDot");
        err = aclrtSynchronizeDevice();
        ACL_RT_CHECK(err, "aclrtSynchronizeDevice");
        LOG_INFO("aclnnMatmul completed");
        return out;
    }

    // 2-D+ matrix multiplication via aclnnMatmul
    size_t x1_batch = x1_ndim - 2;
    size_t x2_batch = x2_ndim - 2;
    size_t out_batch = std::max(x1_batch, x2_batch);

    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < out_batch; i++) {
        ssize_t x1_pos = static_cast<ssize_t>(i) - static_cast<ssize_t>(out_batch - x1_batch);
        ssize_t x2_pos = static_cast<ssize_t>(i) - static_cast<ssize_t>(out_batch - x2_batch);
        int64_t d1 = (x1_pos >= 0 && x1_pos < static_cast<ssize_t>(x1_batch)) ? x1.shape[x1_pos] : 1;
        int64_t d2 = (x2_pos >= 0 && x2_pos < static_cast<ssize_t>(x2_batch)) ? x2.shape[x2_pos] : 1;
        out_shape.push_back(std::max(d1, d2));
    }
    out_shape.push_back(x1.shape[x1_ndim - 2]); // M
    out_shape.push_back(x2.shape[x2_ndim - 1]); // N

    auto result = NPUArray(out_shape, x1.aclDtype);
    int8_t use_fp16 = 2;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error =
        aclnnMatmulGetWorkspaceSize(x1.tensorPtr, x2.tensorPtr, result.tensorPtr, use_fp16, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnMatmulGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnMatmul(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnMatmul");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnMatmul completed");
    return result;
}

NPUArray Einsum(const char* subscripts, const std::vector<NPUArray>& operands) {
    // Guard before any operands[0] / operands[1] access. The Python binding collects
    // operands through py::args, so einsum("ab") or einsum("ab", a) reaches here with
    // fewer than two entries; the LOG_DEBUG arguments below are evaluated eagerly and
    // would read out of bounds.
    if (operands.size() < 2) {
        throw std::runtime_error(fmt::format(
            "[product.cpp](einsum) requires 2 operands, but {} were given", operands.size()));
    }
    LOG_DEBUG("aclnnEinsum start: subscripts={}, x1_shape={}, x2_shape={}, aclDtype={}", subscripts,
              detail::FormatShape(operands[0].shape), detail::FormatShape(operands[1].shape),
              AclDtypeName(operands[0].aclDtype));
    // aclnnEinsum currently supports only 'abcd,abced->abce' and 'a,b->ab'(outer); implement as two operands
    std::vector<aclTensor*> tmp{operands[0].tensorPtr, operands[1].tensorPtr};
    auto input = aclCreateTensorList(tmp.data(), tmp.size());
    // einsum output shape handling
    std::vector<int64_t> shape;
    if (!std::strcmp(subscripts, "a,b->ab")) {
        shape = {operands[0].shape[0], operands[1].shape[0]};
    } else {
        shape = {operands[1].shape[0], operands[1].shape[1], operands[1].shape[2], operands[1].shape[4]};
    }
    auto result = NPUArray(shape, operands[0].dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnEinsumGetWorkspaceSize(input, subscripts, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnEinsumGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnEinsum(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnEinsum");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnEinsum completed");
    return result;
}

NPUArray Matrix_power(const NPUArray& a, int64_t n) {
    LOG_DEBUG("aclnnMatmul start: input_shape={}, aclDtype={}, n={}", detail::FormatShape(a.shape),
              AclDtypeName(a.aclDtype), n);
    auto shape = a.shape;
    auto temp = NPUArray(shape, a.aclDtype);
    auto ax = NPUArray(shape, a.aclDtype);
    auto result = NPUArray(shape, a.aclDtype);
    int absn = 0;
    if (n == 0) {
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        auto error = aclnnEyeGetWorkspaceSize(shape[0], shape[1], result.tensorPtr, &workspaceSize, &executor);
        ACLNN_CHECK(error, "aclnnEyeGetWorkspaceSize");
        AclWorkspace workspace(workspaceSize);
        error = aclnnEye(workspace.get(), workspaceSize, executor, nullptr);
        ACLNN_CHECK(error, "aclnnEye");
        error = aclrtSynchronizeDevice();
        ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
        LOG_INFO("aclnnMatmul completed");
        return result;
    }

    else if (n < 0) {
        absn = -n;
        uint64_t workspaceSize1 = 0;
        aclOpExecutor* executor1;
        auto error1 = aclnnInverseGetWorkspaceSize(a.tensorPtr, temp.tensorPtr, &workspaceSize1, &executor1);
        ACLNN_CHECK(error1, "aclnnInverseGetWorkspaceSize");
        AclWorkspace workspace1(workspaceSize1);
        error1 = aclnnInverse(workspace1.get(), workspaceSize1, executor1, nullptr);
        ACLNN_CHECK(error1, "aclnnInverse");
        error1 = aclrtSynchronizeDevice();
        ACL_RT_CHECK(error1, "aclrtSynchronizeDevice");
        ax = NPUArray(temp);
    } else {
        absn = n;
        temp = NPUArray(a);
        ax = NPUArray(a);
    }

    if (absn == 1) {
        LOG_INFO("aclnnMatmul completed");
        return temp;
    }

    for (int i = 1; i < absn - 1; i++) {
        auto x = NPUArray(shape, a.aclDtype);
        int8_t use_fp16 = 2;
        uint64_t workspaceSize2 = 0;
        aclOpExecutor* executor2;
        auto error2 = aclnnMatmulGetWorkspaceSize(temp.tensorPtr, ax.tensorPtr, x.tensorPtr, use_fp16, &workspaceSize2,
                                                  &executor2);
        ACLNN_CHECK(error2, "aclnnMatmulGetWorkspaceSize");
        AclWorkspace workspace2(workspaceSize2);
        error2 = aclnnMatmul(workspace2.get(), workspaceSize2, executor2, nullptr);
        ACLNN_CHECK(error2, "aclnnMatmul");
        error2 = aclrtSynchronizeDevice();
        ACL_RT_CHECK(error2, "aclrtSynchronizeDevice");
        temp = std::move(x);
    }

    int8_t use_fp16 = 2;
    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    auto error2 = aclnnMatmulGetWorkspaceSize(temp.tensorPtr, ax.tensorPtr, result.tensorPtr, use_fp16, &workspaceSize2,
                                              &executor2);
    ACLNN_CHECK(error2, "aclnnMatmulGetWorkspaceSize");
    AclWorkspace workspace2(workspaceSize2);
    error2 = aclnnMatmul(workspace2.get(), workspaceSize2, executor2, nullptr);
    ACLNN_CHECK(error2, "aclnnMatmul");
    error2 = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error2, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnMatmul completed");
    return result;
}

/**
 * @brief Compute the dot product of two arrays.
 */
NPUArray dot(const NPUArray& a, const NPUArray& b) {
    LOG_DEBUG("aclnnDot start:={}, b_shape={}, aclDtype={}", detail::FormatShape(a.shape), detail::FormatShape(b.shape),
              AclDtypeName(a.aclDtype));
    // case 1: both are scalars
    if (a.shape.size() == 0 && b.shape.size() == 0) {
        // std::nullopt, not a.dtype: ExecuteBinaryOp promotes the operands, so pinning a's dtype
        // would give the kernel an out narrower than its inputs.
        return EXECUTE_BINARY_OP(
            a, b, std::nullopt,
            [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnDotGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
            },
            [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
                return aclnnDot(workspace, workspaceSize, executor, nullptr);
            },
            "dot", "aclnnDot");
    }

    // case 2: 1D · 1D → scalar
    if (a.shape.size() == 1 && b.shape.size() == 1) {
        auto out = NPUArray({}, a.dtype);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        auto error = aclnnDotGetWorkspaceSize(a.tensorPtr, b.tensorPtr, out.tensorPtr, &workspaceSize, &executor);
        ACLNN_CHECK(error, "aclnnDotGetWorkspaceSize");

        AclWorkspace workspace(workspaceSize);

        error = aclnnDot(workspace.get(), workspaceSize, executor, nullptr);
        ACLNN_CHECK(error, "aclnnDot");

        error = aclrtSynchronizeDevice();
        ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

        LOG_INFO("aclnnDot completed");
        return out;
    }

    // case 3: 2D × 2D → matrix multiply
    if (a.shape.size() == 2 && b.shape.size() == 2) {
        std::vector<int64_t> out_shape = {a.shape[0], b.shape[1]};
        auto out = NPUArray(out_shape, a.dtype);

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        int8_t cubeMathType = 0; // KEEP_DTYPE
        auto error =
            aclnnMmGetWorkspaceSize(a.tensorPtr, b.tensorPtr, out.tensorPtr, cubeMathType, &workspaceSize, &executor);
        ACLNN_CHECK(error, "aclnnMmGetWorkspaceSize");

        AclWorkspace workspace(workspaceSize);

        error = aclnnMm(workspace.get(), workspaceSize, executor, nullptr);
        ACLNN_CHECK(error, "aclnnMm");

        error = aclrtSynchronizeDevice();
        ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

        LOG_INFO("aclnnDot completed");
        return out;
    }

    // case 4: higher-dim fallback (throw exception for now; implementation deferred)
    throw std::runtime_error("[product.cpp](dot) Higher-dimensional dot is not yet implemented");
}

/**
 * @brief Compute the dot product of two arrays after flattening.
 *
 * TODO: Currently requires reshape support to convert (1, N) -> (N,).
 *       This project has not implemented reshape yet.
//  */
NPUArray vdot(const NPUArray& a, const NPUArray& b) {
    LOG_DEBUG("aclnnDot start: a_shape={}, b_shape={}, aclDtype={}", detail::FormatShape(a.shape),
              detail::FormatShape(b.shape), AclDtypeName(a.aclDtype));
    aclnnStatus ret; // For receiving API return status
    const int64_t num_elements = a.tensorSize;
    // ===================================================================================
    // Step 1: Flatten
    // ===================================================================================
    std::vector<int64_t> flat_shape = {1, num_elements};
    NPUArray a_flat(flat_shape, a.dtype);
    NPUArray b_flat(flat_shape, b.dtype);

    // Flatten 'a'
    {
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        ret = aclnnFlattenGetWorkspaceSize(a.tensorPtr, 0, a_flat.tensorPtr, &workspaceSize, &executor);
        ACLNN_CHECK(ret, "aclnnFlattenGetWorkspaceSize");

        AclWorkspace workspace(workspaceSize);

        // Pass nullptr for the stream, as in the dot function
        ret = aclnnFlatten(workspace.get(), workspaceSize, executor, nullptr);
        ACLNN_CHECK(ret, "aclnnFlatten");
    }

    // Flatten 'b'
    {
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        ret = aclnnFlattenGetWorkspaceSize(b.tensorPtr, 0, b_flat.tensorPtr, &workspaceSize, &executor);
        ACLNN_CHECK(ret, "aclnnFlattenGetWorkspaceSize");

        AclWorkspace workspace(workspaceSize);

        // Pass nullptr for the stream
        ret = aclnnFlatten(workspace.get(), workspaceSize, executor, nullptr);
        ACLNN_CHECK(ret, "aclnnFlatten");
    }

    // ===================================================================================
    // Step 2: Metadata Reshape
    // ===================================================================================
    std::vector<int64_t> vector_shape = {num_elements};
    std::vector<int64_t> vector_stride = {1};

    aclTensor* a_1d_view =
        aclCreateTensor(vector_shape.data(), vector_shape.size(), a.aclDtype, vector_stride.data(), 0, ACL_FORMAT_ND,
                        flat_shape.data(), flat_shape.size(), a_flat.device_address());
    if (!a_1d_view) {
        throw std::runtime_error("[product.cpp](vdot) aclCreateTensor for a_1d_view failed");
    }

    aclTensor* b_1d_view =
        aclCreateTensor(vector_shape.data(), vector_shape.size(), b.aclDtype, vector_stride.data(), 0, ACL_FORMAT_ND,
                        flat_shape.data(), flat_shape.size(), b_flat.device_address());
    if (!b_1d_view) {
        aclDestroyTensor(a_1d_view);
        throw std::runtime_error("[product.cpp](vdot) aclCreateTensor for b_1d_view failed");
    }

    // ===================================================================================
    // Step 3: Dot Product
    // ===================================================================================
    NPUArray out({}, a.dtype);

    {
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        ret = aclnnDotGetWorkspaceSize(a_1d_view, b_1d_view, out.tensorPtr, &workspaceSize, &executor);
        ACLNN_CHECK(ret, "aclnnDotGetWorkspaceSize");

        AclWorkspace workspace(workspaceSize);

        // Pass nullptr for the stream
        ret = aclnnDot(workspace.get(), workspaceSize, executor, nullptr);
        ACLNN_CHECK(ret, "aclnnDot");
    }

    // Manually destroy the created tensor views before returning
    aclDestroyTensor(a_1d_view);
    aclDestroyTensor(b_1d_view);

    // Synchronize the device to wait for computation to complete, as in the dot function
    ret = aclrtSynchronizeDevice();
    ACL_RT_CHECK(ret, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnDot completed");
    return out;
}

/**
 * @brief Compute the inner product of two arrays.
 */
NPUArray inner(const NPUArray& a, const NPUArray& b) {
    LOG_DEBUG("aclnnDot start: a_shape={}, b_shape={}, aclDtype={}", detail::FormatShape(a.shape),
              detail::FormatShape(b.shape), AclDtypeName(a.aclDtype));
    py::dtype dtype = a.dtype;
    // case 1: 1D × 1D
    if (a.shape.size() == 1 && b.shape.size() == 1) {
        NPUArray out({}, dtype);

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        auto error = aclnnDotGetWorkspaceSize(a.tensorPtr, b.tensorPtr, out.tensorPtr, &workspaceSize, &executor);
        ACLNN_CHECK(error, "aclnnDotGetWorkspaceSize");

        AclWorkspace workspace(workspaceSize);

        error = aclnnDot(workspace.get(), workspaceSize, executor, nullptr);
        ACLNN_CHECK(error, "aclnnDot");

        error = aclrtSynchronizeDevice();
        ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
        LOG_INFO("aclnnDot completed");
        return out;
    }

    // case 2: 2D × 2D
    if (a.shape.size() == 2 && b.shape.size() == 2) {
        std::vector<int64_t> out_shape = {a.shape[0], b.shape[1]};
        NPUArray out(out_shape, dtype);

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        int8_t cubeMathType = 0; // KEEP_DTYPE
        auto error =
            aclnnMmGetWorkspaceSize(a.tensorPtr, b.tensorPtr, out.tensorPtr, cubeMathType, &workspaceSize, &executor);
        ACLNN_CHECK(error, "aclnnMmGetWorkspaceSize");

        AclWorkspace workspace(workspaceSize);

        error = aclnnMm(workspace.get(), workspaceSize, executor, nullptr);
        ACLNN_CHECK(error, "aclnnMm");

        error = aclrtSynchronizeDevice();
        ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
        LOG_INFO("aclnnDot completed");
        return out;
    }

    // case 3: higher dimensions not supported yet
    throw std::runtime_error("[product.cpp](inner) Only 1D and 2D inputs are supported for now");
}

/**
 * @brief Compute the outer product of two arrays.
 */
NPUArray outer(const NPUArray& a, const NPUArray& b) {
    LOG_DEBUG("aclnnMul start: a_shape={}, b_shape={}, aclDtype={}", detail::FormatShape(a.shape),
              detail::FormatShape(b.shape), AclDtypeName(a.aclDtype));
    py::dtype dtype = a.dtype;
    // Step 1: flatten a -> (m, 1) - bug: aclnnFlatten cannot expand to (m, 1) format
    auto a_flat = NPUArray({static_cast<int64_t>(a.tensorSize), 1}, a.aclDtype);
    {
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        auto err = aclnnFlattenGetWorkspaceSize(a.tensorPtr, 1, a_flat.tensorPtr, &ws, &exec);
        ACLNN_CHECK(err, "aclnnFlattenGetWorkspaceSize");

        AclWorkspace workspace(ws);

        err = aclnnFlatten(workspace.get(), ws, exec, nullptr);
        ACLNN_CHECK(err, "aclnnFlatten");
    }

    // Step 2: flatten b -> (1, n)
    auto b_flat = NPUArray({1, static_cast<int64_t>(b.tensorSize)}, b.aclDtype);
    {
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        auto err = aclnnFlattenGetWorkspaceSize(b.tensorPtr, 0, b_flat.tensorPtr, &ws, &exec);
        ACLNN_CHECK(err, "aclnnFlattenGetWorkspaceSize");

        AclWorkspace workspace(ws);

        err = aclnnFlatten(workspace.get(), ws, exec, nullptr);
        ACLNN_CHECK(err, "aclnnFlatten");
    }

    // Step 3: elementwise multiply (m,1) * (1,n) -> (m,n)
    std::vector<int64_t> out_shape = {static_cast<int64_t>(a.tensorSize), static_cast<int64_t>(b.tensorSize)};

    NPUArray out(out_shape, dtype);

    {
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        auto err = aclnnMulGetWorkspaceSize(a_flat.tensorPtr, b_flat.tensorPtr, out.tensorPtr, &ws, &exec);
        ACLNN_CHECK(err, "aclnnMulGetWorkspaceSize");

        AclWorkspace workspace(ws);

        err = aclnnMul(workspace.get(), ws, exec, nullptr);
        ACLNN_CHECK(err, "aclnnMul");
    }

    auto err = aclrtSynchronizeDevice();
    ACL_RT_CHECK(err, "aclrtSynchronizeDevice");

    LOG_INFO("aclnnMul completed");
    return out;
}
