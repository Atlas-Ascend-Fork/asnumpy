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

#include <asnumpy/array/manipulation.hpp>
#include <asnumpy/dtypes/promote.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/cast.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <aclnnop/aclnn_cat.h>
#include <aclnnop/aclnn_s_where.h>
#include <aclnnop/aclnn_stack.h>

#include <algorithm>
#include <fmt/format.h>
#include <optional>
#include <stdexcept>
#include <string>

namespace asnumpy {
namespace {

// One operand of a variadic op after dtype promotion: either the caller's array or a device cast
// that has to outlive the kernel call.
struct PromotedInputs {
    std::vector<const NPUArray*> arrays;
    std::vector<std::optional<NPUArray>> storage; // owns the casts, empty when no cast was needed
};

PromotedInputs PromoteAll(const std::vector<const NPUArray*>& arrays) {
    aclDataType common = arrays.front()->aclDtype;
    for (const NPUArray* a : arrays) {
        common = ResultType(common, a->aclDtype);
    }
    PromotedInputs result;
    result.arrays.resize(arrays.size());
    result.storage.resize(arrays.size());
    for (size_t i = 0; i < arrays.size(); ++i) {
        if (arrays[i]->aclDtype == common) {
            result.arrays[i] = arrays[i];
            continue;
        }
        result.storage[i] = CastTo(*arrays[i], common);
        result.arrays[i] = &*result.storage[i];
    }
    return result;
}

int64_t NormalizeAxis(int64_t axis, int64_t ndim, bool insert, const char* op) {
    const int64_t limit = insert ? ndim + 1 : ndim;
    int64_t dim = axis;
    if (dim < 0) {
        dim += limit;
    }
    if (dim < 0 || dim >= limit) {
        throw std::invalid_argument(
            fmt::format("[manipulation.cpp]({}) axis {} is out of bounds for array of dimension {}", op, axis, ndim));
    }
    return dim;
}

std::vector<int64_t> BroadcastShapes(const std::vector<const std::vector<int64_t>*>& shapes, const char* op) {
    size_t rank = 0;
    for (const auto* s : shapes) {
        rank = std::max(rank, s->size());
    }
    std::vector<int64_t> out(rank, 1);
    for (const auto* s : shapes) {
        const size_t offset = rank - s->size();
        for (size_t i = 0; i < s->size(); ++i) {
            const int64_t dim = (*s)[i];
            int64_t& target = out[offset + i];
            if (target == 1) {
                target = dim;
            } else if (dim != 1 && dim != target) {
                throw std::invalid_argument(
                    fmt::format("[manipulation.cpp]({}) operands could not be broadcast together", op));
            }
        }
    }
    return out;
}

// Build an aclTensorList from resolved operands and guarantee it is destroyed on every exit path.
//
// aclCreateTensorList takes ownership of the aclTensor objects it is given: aclDestroyTensorList
// destroys them. Handing over an NPUArray's own tensorPtr therefore freed the operand's descriptor
// and the next list-based op read a dangling tensor (CANN reported "tensor address is (nil)" /
// dtype UNDEFINED, then the process segfaulted). Build throwaway descriptors over the operands'
// device buffers instead, so only those are destroyed.
class ScopedTensorList {
  public:
    ScopedTensorList(const std::vector<const NPUArray*>& arrays, const char* op) {
        tensors_.reserve(arrays.size());
        for (const NPUArray* a : arrays) {
            aclTensor* tensor = aclCreateTensor(a->shape.data(), a->shape.size(), a->aclDtype, a->strides.data(), 0,
                                                ACL_FORMAT_ND, a->shape.data(), a->shape.size(), a->device_address());
            if (tensor == nullptr) {
                DestroyTensors();
                throw std::runtime_error(fmt::format("[manipulation.cpp]({}) aclCreateTensor failed", op));
            }
            tensors_.push_back(tensor);
        }
        list_ = aclCreateTensorList(tensors_.data(), tensors_.size());
        if (list_ == nullptr) {
            DestroyTensors();
            throw std::runtime_error(fmt::format("[manipulation.cpp]({}) aclCreateTensorList failed", op));
        }
    }

    ~ScopedTensorList() {
        if (list_ != nullptr) {
            // Also destroys the throwaway tensors created above.
            aclDestroyTensorList(list_);
            return;
        }
        DestroyTensors();
    }

    ScopedTensorList(const ScopedTensorList&) = delete;
    ScopedTensorList& operator=(const ScopedTensorList&) = delete;

    aclTensorList* get() const { return list_; }

  private:
    void DestroyTensors() {
        for (aclTensor* tensor : tensors_) {
            aclDestroyTensor(tensor);
        }
        tensors_.clear();
    }

    std::vector<aclTensor*> tensors_;
    aclTensorList* list_ = nullptr;
};

void CheckUniformShape(const std::vector<const NPUArray*>& arrays, const char* op) {
    const std::vector<int64_t>& first = arrays.front()->shape;
    for (const NPUArray* a : arrays) {
        if (a->shape != first) {
            throw std::invalid_argument(
                fmt::format("[manipulation.cpp]({}) all input arrays must have the same shape", op));
        }
    }
}

} // namespace

NPUArray Concatenate(const std::vector<const NPUArray*>& arrays, int64_t axis) {
    if (arrays.empty()) {
        throw std::invalid_argument("[manipulation.cpp](concatenate) need at least one array to concatenate");
    }
    const int64_t ndim = static_cast<int64_t>(arrays.front()->shape.size());
    if (ndim == 0) {
        throw std::invalid_argument("[manipulation.cpp](concatenate) zero-dimensional arrays cannot be concatenated");
    }
    const int64_t dim = NormalizeAxis(axis, ndim, /*insert=*/false, "concatenate");

    LOG_DEBUG("aclnnCat start: {} arrays, axis={}, first_shape={}", arrays.size(), dim,
              detail::FormatShape(arrays.front()->shape));

    for (const NPUArray* a : arrays) {
        if (static_cast<int64_t>(a->shape.size()) != ndim) {
            throw std::invalid_argument(fmt::format(
                "[manipulation.cpp](concatenate) all the input array dimensions must match exactly, but got rank {} "
                "and {}",
                ndim, a->shape.size()));
        }
        for (int64_t i = 0; i < ndim; ++i) {
            if (i != dim && a->shape[i] != arrays.front()->shape[i]) {
                throw std::invalid_argument(
                    "[manipulation.cpp](concatenate) all the input array dimensions except for the concatenation axis "
                    "must match exactly");
            }
        }
    }

    PromotedInputs inputs = PromoteAll(arrays);
    std::vector<int64_t> out_shape = inputs.arrays.front()->shape;
    int64_t axis_total = 0;
    for (const NPUArray* a : inputs.arrays) {
        axis_total += a->shape[dim];
    }
    out_shape[dim] = axis_total;

    NPUArray out(out_shape, NPUArray::GetPyDtype(inputs.arrays.front()->aclDtype));
    ScopedTensorList list(inputs.arrays, "concatenate");

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnCatGetWorkspaceSize(list.get(), dim, out.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnCatGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);
    error = aclnnCat(workspace.get(), workspace.size(), executor, nullptr);
    ACLNN_CHECK(error, "aclnnCat");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnCat completed");
    return out;
}

NPUArray Stack(const std::vector<const NPUArray*>& arrays, int64_t axis) {
    if (arrays.empty()) {
        throw std::invalid_argument("[manipulation.cpp](stack) need at least one array to stack");
    }
    const int64_t ndim = static_cast<int64_t>(arrays.front()->shape.size());
    const int64_t dim = NormalizeAxis(axis, ndim, /*insert=*/true, "stack");

    LOG_DEBUG("aclnnStack start: {} arrays, axis={}, first_shape={}", arrays.size(), dim,
              detail::FormatShape(arrays.front()->shape));

    CheckUniformShape(arrays, "stack");

    PromotedInputs inputs = PromoteAll(arrays);
    std::vector<int64_t> out_shape = inputs.arrays.front()->shape;
    out_shape.insert(out_shape.begin() + dim, static_cast<int64_t>(inputs.arrays.size()));

    NPUArray out(out_shape, NPUArray::GetPyDtype(inputs.arrays.front()->aclDtype));
    ScopedTensorList list(inputs.arrays, "stack");

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnStackGetWorkspaceSize(list.get(), dim, out.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnStackGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);
    error = aclnnStack(workspace.get(), workspace.size(), executor, nullptr);
    ACLNN_CHECK(error, "aclnnStack");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnStack completed");
    return out;
}

NPUArray Where(const NPUArray& condition, const NPUArray& self, const NPUArray& other) {
    if (condition.aclDtype != ACL_BOOL) {
        throw std::invalid_argument("[manipulation.cpp](where) condition must be a boolean array (bool dtype)");
    }

    LOG_DEBUG("aclnnSWhere start: condition_shape={}, self_shape={}, other_shape={}",
              detail::FormatShape(condition.shape), detail::FormatShape(self.shape), detail::FormatShape(other.shape));

    std::vector<const std::vector<int64_t>*> shapes{&condition.shape, &self.shape, &other.shape};
    std::vector<int64_t> out_shape = BroadcastShapes(shapes, "where");

    PromotedOperands operands(self, other);
    NPUArray out(out_shape, NPUArray::GetPyDtype(operands.common()));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnSWhereGetWorkspaceSize(condition.tensorPtr, operands.x1().tensorPtr, operands.x2().tensorPtr,
                                             out.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnSWhereGetWorkspaceSize");

    AclWorkspace workspace(workspaceSize);
    error = aclnnSWhere(workspace.get(), workspace.size(), executor, nullptr);
    ACLNN_CHECK(error, "aclnnSWhere");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnSWhere completed");
    return out;
}

} // namespace asnumpy
