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

#include <asnumpy/array/basic.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/npu_scalar.hpp>
#include <asnumpy/utils/status_handler.hpp>
#include <fmt/format.h>

#include <aclnnop/aclnn_arange.h>
#include <aclnnop/aclnn_eye.h>
#include <aclnnop/aclnn_fill_scalar.h>
#include <aclnnop/aclnn_linspace.h>
#include <aclnnop/aclnn_ones.h>
#include <aclnnop/aclnn_zero.h>

namespace asnumpy {

NPUArray Empty(const std::vector<int64_t>& shape, py::dtype dtype) {
    LOG_DEBUG("Empty start: input_shape={}", detail::FormatShape(shape));
    try {
        auto result = NPUArray(shape, dtype);
        LOG_INFO("Empty completed");
        return result;
    } catch (const std::exception& e) {
        throw std::runtime_error(fmt::format("[basic.cpp](empty) NPUArray construction error = {}", e.what()));
    }
}

NPUArray EmptyLike(const NPUArray& prototype, py::dtype dtype) {
    LOG_DEBUG("EmptyLike start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(prototype.shape),
              prototype.tensorSize, AclDtypeName(prototype.aclDtype));
    try {
        // use prototype dtype if none specified
        py::dtype target_dtype = dtype.is_none() ? prototype.dtype() : dtype;
        // create empty array based on prototype shape and target dtype
        auto result = NPUArray(prototype.shape, target_dtype);
        LOG_INFO("EmptyLike completed");
        return result;
    } catch (const std::exception& e) {
        throw std::runtime_error(fmt::format("[basic.cpp](empty_like) NPUArray construction error = {}", e.what()));
    }
}

NPUArray Zeros(const std::vector<int64_t>& shape, py::dtype dtype) {
    LOG_DEBUG("aclnnInplaceZero start: input_shape={}", detail::FormatShape(shape));
    auto array = NPUArray(shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnInplaceZeroGetWorkspaceSize(array.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnInplaceZeroGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnInplaceZero(workspace.get(), workspace.size(), executor, nullptr);
    ACLNN_CHECK(error, "aclnnInplaceZero");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnInplaceZero completed");
    return array;
}

NPUArray Zeros_like(const NPUArray& other, py::dtype dtype) {
    LOG_DEBUG("aclnnInplaceZero start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(other.shape),
              other.tensorSize, AclDtypeName(other.aclDtype));
    auto array = NPUArray(other.shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnInplaceZeroGetWorkspaceSize(array.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnInplaceZeroGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnInplaceZero(workspace.get(), workspace.size(), executor, nullptr);
    ACLNN_CHECK(error, "aclnnInplaceZero");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnInplaceZero completed");
    return array;
}

NPUArray Full(const std::vector<int64_t>& shape, const py::object& value, py::dtype dtype) {
    LOG_DEBUG("aclnnInplaceFillScalar start: input_shape={}", detail::FormatShape(shape));
    auto array = NPUArray(shape, dtype);
    double valueDouble = 0;
    if (value.is_none()) {
        throw std::runtime_error("[basic.cpp](full) Input is None");
    }
    try {
        valueDouble = py::cast<double>(value);
    } catch (const py::cast_error& e) {
        throw std::runtime_error("[basic.cpp](full) Conversion error: " + std::string(e.what()));
    }
    aclScalar* scalar = CreateScalar(valueDouble, array.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnInplaceFillScalarGetWorkspaceSize(array.tensorPtr, scalar, &workspaceSize, &executor);
    if (error != ACL_SUCCESS) {
        aclDestroyScalar(scalar);
        ACLNN_CHECK(error, "aclnnInplaceFillScalarGetWorkspaceSize");
    }
    AclWorkspace workspace(workspaceSize);
    error = aclnnInplaceFillScalar(workspace.get(), workspace.size(), executor, nullptr);
    if (error != ACL_SUCCESS) {
        aclDestroyScalar(scalar);
        ACLNN_CHECK(error, "aclnnInplaceFillScalar");
    }
    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        aclDestroyScalar(scalar);
        ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    }
    aclDestroyScalar(scalar);
    LOG_INFO("aclnnInplaceFillScalar completed");
    return array;
}

NPUArray Full_like(const NPUArray& other, const py::object& value, py::dtype dtype) {
    LOG_DEBUG("aclnnInplaceFillScalar start: input_shape={}, tensorSize={}, aclDtype={}",
              detail::FormatShape(other.shape), other.tensorSize, AclDtypeName(other.aclDtype));
    auto array = NPUArray(other.shape, dtype);
    double valueDouble = 0;
    if (value.is_none()) {
        throw std::runtime_error("[basic.cpp](full_like) Input is None");
    }
    try {
        valueDouble = py::cast<double>(value);
    } catch (const py::cast_error& e) {
        throw std::runtime_error("[basic.cpp](full_like) Conversion error: " + std::string(e.what()));
    }
    aclScalar* scalar = CreateScalar(valueDouble, array.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnInplaceFillScalarGetWorkspaceSize(array.tensorPtr, scalar, &workspaceSize, &executor);
    if (error != ACL_SUCCESS) {
        aclDestroyScalar(scalar);
        ACLNN_CHECK(error, "aclnnInplaceFillScalarGetWorkspaceSize");
    }
    AclWorkspace workspace(workspaceSize);
    error = aclnnInplaceFillScalar(workspace.get(), workspace.size(), executor, nullptr);
    if (error != ACL_SUCCESS) {
        aclDestroyScalar(scalar);
        ACLNN_CHECK(error, "aclnnInplaceFillScalar");
    }
    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        aclDestroyScalar(scalar);
        ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    }
    aclDestroyScalar(scalar);
    LOG_INFO("aclnnInplaceFillScalar completed");
    return array;
}

NPUArray Eye(int64_t n, py::dtype dtype) {
    LOG_DEBUG("aclnnEye start: n={}", n);
    auto array = NPUArray({n, n}, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnEyeGetWorkspaceSize(n, n, array.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnEyeGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnEye(workspace.get(), workspace.size(), executor, nullptr);
    ACLNN_CHECK(error, "aclnnEye");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnEye completed");
    return array;
}

NPUArray Ones(const std::vector<int64_t>& shape, py::dtype dtype) {
    LOG_DEBUG("aclnnInplaceOne start: input_shape={}", detail::FormatShape(shape));
    auto array = NPUArray(shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnInplaceOneGetWorkspaceSize(array.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnInplaceOneGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnInplaceOne(workspace.get(), workspace.size(), executor, nullptr);
    ACLNN_CHECK(error, "aclnnInplaceOne");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnInplaceOne completed");
    return array;
}

NPUArray Identity(int64_t n, py::dtype dtype) {
    LOG_DEBUG("aclnnEye start: n={}", n);
    auto array = NPUArray({n, n}, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    auto error = aclnnEyeGetWorkspaceSize(n, n, array.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnEyeGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);

    error = aclnnEye(workspace.get(), workspace.size(), executor, nullptr);
    ACLNN_CHECK(error, "aclnnEye");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

    LOG_INFO("aclnnEye completed");
    return array;
}

NPUArray ones_like(const NPUArray& other, py::dtype dtype) {
    LOG_DEBUG("aclnnInplaceOne start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(other.shape),
              other.tensorSize, AclDtypeName(other.aclDtype));
    auto array = NPUArray(other.shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnInplaceOneGetWorkspaceSize(array.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnInplaceOneGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnInplaceOne(workspace.get(), workspace.size(), executor, nullptr);
    ACLNN_CHECK(error, "aclnnInplaceOne");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnInplaceOne completed");
    return array;
}

NPUArray Linspace(const py::object& start, const py::object& end, const py::object& steps, const py::object& dtype) {
    LOG_DEBUG("aclnnLinspace start");
    double start_val = 0.0, end_val = 0.0;
    int64_t steps_val = 0;

    try {
        start_val = py::cast<double>(start);
        end_val = py::cast<double>(end);
        steps_val = py::cast<int64_t>(steps);
    } catch (const py::cast_error& e) {
        throw std::runtime_error("[basic.cpp](linspace) Invalid start/end/steps type: " + std::string(e.what()));
    }

    if (steps_val <= 0) {
        throw std::runtime_error("[basic.cpp](linspace) steps must be > 0.");
    }

    py::dtype final_dtype;

    if (!dtype.is_none()) {
        try {
            final_dtype = py::dtype(dtype);
        } catch (...) {
            if (py::isinstance<py::str>(dtype)) {
                final_dtype = py::dtype(py::str(dtype));
            } else if (py::hasattr(dtype, "__name__")) {
                try {
                    auto numpy = py::module_::import("numpy");
                    final_dtype = numpy.attr("dtype")(dtype);
                } catch (...) {
                    throw std::runtime_error("[basic.cpp](linspace) Failed to create dtype from numpy type: " +
                                             std::string(py::str(dtype)));
                }
            } else if (py::hasattr(dtype, "dtype")) {
                final_dtype = dtype.attr("dtype");
            } else {
                try {
                    std::string dtype_str = py::cast<std::string>(dtype);
                    final_dtype = py::dtype(dtype_str);
                } catch (...) {
                    throw std::runtime_error("[basic.cpp](linspace) Unsupported dtype parameter type: " +
                                             std::string(py::str(dtype)));
                }
            }
        }
    } else {
        final_dtype = py::dtype::of<double>();
    }

    std::vector<int64_t> out_shape = {steps_val};
    NPUArray out(out_shape, final_dtype);

    if (out.tensorPtr == nullptr) {
        throw std::runtime_error("[basic.cpp](linspace) out.tensorPtr is null, failed to allocate output tensor");
    }

    aclScalar* acl_start = nullptr;
    aclScalar* acl_end = nullptr;

    try {
        acl_start = aclCreateScalar(&start_val, ACL_DOUBLE);
        acl_end = aclCreateScalar(&end_val, ACL_DOUBLE);
    } catch (...) {
        if (acl_start)
            aclDestroyScalar(acl_start);
        if (acl_end)
            aclDestroyScalar(acl_end);
        throw std::runtime_error("[basic.cpp](linspace) Failed to create ACL scalars.");
    }

    // RAII guard for scalar cleanup
    auto scalarGuard = [&]() {
        aclDestroyScalar(acl_start);
        aclDestroyScalar(acl_end);
    };

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto error = aclnnLinspaceGetWorkspaceSize(acl_start, acl_end, steps_val, out.tensorPtr, &workspaceSize, &executor);

    if (error != ACL_SUCCESS) {
        scalarGuard();
        ACLNN_CHECK(error, "aclnnLinspaceGetWorkspaceSize");
    }

    AclWorkspace workspace(workspaceSize);

    error = aclnnLinspace(workspace.get(), workspace.size(), executor, nullptr);
    if (error != ACL_SUCCESS) {
        scalarGuard();
        ACLNN_CHECK(error, "aclnnLinspace");
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        scalarGuard();
        ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    }

    scalarGuard();
    LOG_INFO("aclnnLinspace completed");
    return out;
}

} // namespace asnumpy
