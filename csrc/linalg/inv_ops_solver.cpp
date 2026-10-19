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

#include <asnumpy/linalg/inv_ops_solver.hpp>

#include <acl/acl.h>
#include <fmt/format.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <asnumpy/utils/status_handler.hpp>

#ifdef ASNUMPY_ENABLE_OPS_SOLVER
#include <cann_ops_solver.h>
#endif

using namespace asnumpy;

namespace {

#ifdef ASNUMPY_ENABLE_OPS_SOLVER

/// Largest order accepted by the ops-solver kernel (LU_MAX_GETRI_N); the library rejects larger n.
constexpr int64_t kOpsSolverMaxN = 8192;

/// Owns the ops-solver handle so every early exit still destroys it.
class ScopedSolverHandle {
  public:
    ScopedSolverHandle() = default;
    ScopedSolverHandle(const ScopedSolverHandle&) = delete;
    ScopedSolverHandle& operator=(const ScopedSolverHandle&) = delete;
    ~ScopedSolverHandle() {
        if (handle_ != nullptr) {
            aclsolverDestroy(handle_);
        }
    }

    aclsolverHandle_t* out() { return &handle_; }
    aclsolverHandle_t get() const { return handle_; }

  private:
    aclsolverHandle_t handle_ = nullptr;
};

/// Owns the pinned host staging buffer required by the library's host-pointer contract.
class ScopedHostBuffer {
  public:
    ScopedHostBuffer() = default;
    ScopedHostBuffer(const ScopedHostBuffer&) = delete;
    ScopedHostBuffer& operator=(const ScopedHostBuffer&) = delete;
    ~ScopedHostBuffer() {
        if (ptr_ != nullptr) {
            aclrtFreeHost(ptr_);
        }
    }

    bool alloc(size_t bytes) {
        bytes_ = bytes;
        return aclrtMallocHost(&ptr_, bytes) == ACL_SUCCESS;
    }
    void* get() const { return ptr_; }
    size_t bytes() const { return bytes_; }

  private:
    void* ptr_ = nullptr;
    size_t bytes_ = 0;
};

/// Validates the operand contract of the ops-solver backend and returns the matrix order.
///
/// Throws std::invalid_argument for unsupported rank/dtype/shape or sizes outside the library range.
int64_t ValidateOperand(const NPUArray& a) {
    if (a.shape.size() != 2) {
        throw std::invalid_argument(
            fmt::format("asnumpy.linalg.inv(ops-solver): only 2-D operands are supported, got {}-D", a.shape.size()));
    }
    const int64_t n = a.shape[0];
    if (a.shape[1] != n) {
        throw std::invalid_argument(fmt::format("asnumpy.linalg.inv(ops-solver): the matrix must be square, got {}x{}",
                                                a.shape[0], a.shape[1]));
    }
    if (a.aclDtype != ACL_FLOAT) {
        throw std::invalid_argument(
            fmt::format("asnumpy.linalg.inv(ops-solver): only float32 is supported, got {}", AclDtypeName(a.aclDtype)));
    }
    if (n > kOpsSolverMaxN) {
        throw std::invalid_argument(
            fmt::format("asnumpy.linalg.inv(ops-solver): n={} exceeds the backend limit of {}", n, kOpsSolverMaxN));
    }
    return n;
}

/// Runs the inversion on the NPU through ops-solver and fills `result` with it.
///
/// The library takes a HOST pointer and performs its own factorisation, so the operand is staged
/// through pinned host memory: the caller's device buffer is never handed over and never modified.
void RunOpsSolverInverse(const NPUArray& a, int64_t n, NPUArray& result) {
    const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(float);

    ScopedSolverHandle handle;
    aclError error = aclsolverCreate(handle.out());
    ACL_RT_CHECK(error, "aclsolverCreate");
    // nullptr selects the library default stream, matching the rest of asnumpy. The library
    // synchronises its stream before returning, so its result is complete at that point.
    error = aclsolverSetStream(handle.get(), nullptr);
    ACL_RT_CHECK(error, "aclsolverSetStream");

    ScopedHostBuffer staging;
    if (!staging.alloc(bytes)) {
        throw std::runtime_error(
            fmt::format("asnumpy.linalg.inv(ops-solver): aclrtMallocHost({} bytes) failed", bytes));
    }
    error = aclrtMemcpy(staging.get(), bytes, a.device_address(), bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    ACL_RT_CHECK(error, "aclrtMemcpy(device->host)");

    // 奇异检测：调用本库公开的 aclsolverSgetrf 对 host 副本做一次 LU（原地写回 LU），按 LAPACK 的
    // 精确零主元判据扫描 U 对角线。命中即抛 OpsSolverSingularMatrix（→ numpy.linalg.LinAlgError），
    // 且根本不调用 aclsolverSgetri，因此调用方的输入与结果都不会被触碰。
    // 注意：Sgetri 的输入必须是原始矩阵；这里只在副本上分解，绝不把 LU 结果传给 Sgetri。
    // 已知限制：数学奇异但 LU 未产生精确零主元的矩阵不会被判为奇异（与 LAPACK 判据一致）。
    {
        const float* stagingF = static_cast<const float*>(staging.get());
        const size_t elements = static_cast<size_t>(n) * static_cast<size_t>(n);
        std::vector<float> luCopy(stagingF, stagingF + elements);
        std::vector<int32_t> ipiv(static_cast<size_t>(n));
        int32_t luInfo = 0;
        const aclError luRet = aclsolverSgetrf(handle.get(), n, n, luCopy.data(), n, ipiv.data(), &luInfo);
        ACL_RT_CHECK(luRet, "aclsolverSgetrf");
        for (int64_t k = 0; k < n; ++k)
        {
            if (luCopy[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(k)] == 0.0f)
            {
                throw OpsSolverSingularMatrix(fmt::format(
                    "Singular matrix: asnumpy.linalg.inv(ops-solver) found U({0},{0}) exactly zero (column {0}); "
                    "the input was left unchanged",
                    k + 1));
            }
        }
    }

    int32_t info = 0;
    const aclError ret = aclsolverSgetri(handle.get(), n, static_cast<float*>(staging.get()), n, &info);
    if (ret != ACL_SUCCESS) {
        if (info > 0) {
            throw OpsSolverSingularMatrix(
                fmt::format("Singular matrix: asnumpy.linalg.inv(ops-solver) found U({0},{0}) exactly zero (info={0}); "
                            "the input was left unchanged",
                            info));
        }
        throw std::runtime_error(
            fmt::format("asnumpy.linalg.inv(ops-solver): aclsolverSgetri failed (aclError={}, info={})", ret, info));
    }

    error = aclrtMemcpy(result.device_address(), bytes, staging.get(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_RT_CHECK(error, "aclrtMemcpy(host->device)");
    // The library already synchronised, but keeping asnumpy's post-operator contract makes the
    // host staging buffer's lifetime unambiguous.
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
}

#else // ASNUMPY_ENABLE_OPS_SOLVER

/// Message shown when the optional backend was not compiled in (kept outside the entry point so that
/// function stays small, see the project's function-size rule).
std::string BackendUnavailableMessage() {
    return "asnumpy.linalg.inv(ops-solver) is not available: this asnumpy build was compiled without CANN ops-solver. "
           "Build ops-solver for your SoC (git clone https://gitcode.com/cann/ops-solver.git && "
           "cmake -S . -B build-lib -DSOC_VERSION=ascend910b -DBUILD_TEST=OFF && cmake --build build-lib -j), then "
           "rebuild asnumpy with -DASNUMPY_ENABLE_OPS_SOLVER=ON and -DASNUMPY_OPS_SOLVER_ROOT=<ops-solver root>.";
}

#endif // ASNUMPY_ENABLE_OPS_SOLVER

} // namespace

NPUArray InvOpsSolver(const NPUArray& a) {
#ifdef ASNUMPY_ENABLE_OPS_SOLVER
    const int64_t n = ValidateOperand(a);
    if (n == 0) {
        // NumPy returns an empty result for a 0x0 operand; the library rejects n == 0, so answer here.
        return NPUArray({0, 0}, ACL_FLOAT);
    }
    NPUArray result({n, n}, ACL_FLOAT);
    RunOpsSolverInverse(a, n, result);
    return result;
#else
    throw std::runtime_error(BackendUnavailableMessage());
#endif
}
