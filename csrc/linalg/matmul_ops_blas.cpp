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

#include <asnumpy/linalg/matmul_ops_blas.hpp>

#include <acl/acl.h>
#include <fmt/format.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include <asnumpy/utils/status_handler.hpp>

#ifdef ASNUMPY_ENABLE_OPS_BLAS
#include <cann_ops_blas.h>
#endif

// AclDtypeName and the status-check helpers live in namespace asnumpy, as in the other csrc units.
using namespace asnumpy;

namespace {

#ifdef ASNUMPY_ENABLE_OPS_BLAS

const char* BlasStatusName(aclblasStatus_t status) {
    switch (status) {
    case ACLBLAS_STATUS_SUCCESS:
        return "SUCCESS";
    case ACLBLAS_STATUS_NOT_INITIALIZED:
        return "NOT_INITIALIZED";
    case ACLBLAS_STATUS_ALLOC_FAILED:
        return "ALLOC_FAILED";
    case ACLBLAS_STATUS_INVALID_VALUE:
        return "INVALID_VALUE";
    case ACLBLAS_STATUS_MAPPING_ERROR:
        return "MAPPING_ERROR";
    case ACLBLAS_STATUS_EXECUTION_FAILED:
        return "EXECUTION_FAILED";
    case ACLBLAS_STATUS_INTERNAL_ERROR:
        return "INTERNAL_ERROR";
    case ACLBLAS_STATUS_NOT_SUPPORTED:
        return "NOT_SUPPORTED";
    case ACLBLAS_STATUS_ARCH_MISMATCH:
        return "ARCH_MISMATCH";
    case ACLBLAS_STATUS_HANDLE_IS_NULLPTR:
        return "HANDLE_IS_NULLPTR";
    case ACLBLAS_STATUS_INVALID_ENUM:
        return "INVALID_ENUM";
    default:
        return "UNKNOWN";
    }
}

/// Owns the ops-blas handle so that every error path still destroys it.
class ScopedBlasHandle {
  public:
    ScopedBlasHandle() = default;
    ScopedBlasHandle(const ScopedBlasHandle&) = delete;
    ScopedBlasHandle& operator=(const ScopedBlasHandle&) = delete;
    ~ScopedBlasHandle() {
        if (handle_ != nullptr) {
            aclblasDestroy(handle_);
        }
    }

    aclblasHandle_t* out() { return &handle_; }
    aclblasHandle_t get() const { return handle_; }

  private:
    aclblasHandle_t handle_ = nullptr;
};

void CheckBlas(aclblasStatus_t status, const char* api) {
    if (status == ACLBLAS_STATUS_SUCCESS) {
        return;
    }
    throw std::runtime_error(
        fmt::format("asnumpy.matmul(ops-blas): {} failed: {} (aclblasStatus_t={}); this ops-blas build may not "
                    "implement that GEMM path for the current SoC",
                    api, BlasStatusName(status), static_cast<int>(status)));
}

/// ops-blas takes int dimensions; refuse anything that cannot be represented.
int ToBlasInt(int64_t value, const char* what) {
    if (value < 0 || value > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument(
            fmt::format("asnumpy.matmul(ops-blas): {}={} is outside the supported range 0..2^31-1", what, value));
    }
    return static_cast<int>(value);
}

int64_t CheckedMul(int64_t a, int64_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<int64_t>::max() / a) {
        throw std::overflow_error(fmt::format("asnumpy.matmul(ops-blas): {} element count overflows int64", what));
    }
    return a * b;
}

/// Validated operand geometry of a 2-D float32 matmul.
struct GemmShape {
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    int64_t elements = 0;
};

/// Checks the operand contract (2-D, float32, matching inner dimension, no overflow) and returns the
/// result geometry. Throws std::invalid_argument / std::overflow_error on violation.
GemmShape ValidateOperands(const NPUArray& x1, const NPUArray& x2) {
    if (x1.shape.size() != 2 || x2.shape.size() != 2) {
        throw std::invalid_argument(
            fmt::format("asnumpy.matmul(ops-blas): only 2-D operands are supported, got {}-D and {}-D", x1.shape.size(),
                        x2.shape.size()));
    }
    if (x1.aclDtype != ACL_FLOAT || x2.aclDtype != ACL_FLOAT) {
        throw std::invalid_argument(fmt::format("asnumpy.matmul(ops-blas): only float32 operands are supported, got {} "
                                                "and {}",
                                                AclDtypeName(x1.aclDtype), AclDtypeName(x2.aclDtype)));
    }
    GemmShape shape;
    shape.m = x1.shape[0];
    shape.k = x1.shape[1];
    const int64_t k2 = x2.shape[0];
    shape.n = x2.shape[1];
    if (shape.k != k2) {
        throw std::invalid_argument(
            fmt::format("asnumpy.matmul(ops-blas): inner dimensions differ ({} vs {})", shape.k, k2));
    }
    shape.elements = CheckedMul(shape.m, shape.n, "result");
    CheckedMul(shape.elements, shape.k, "multiply-add"); // overflow guard only
    return shape;
}

/// Issues the GEMM through ops-blas and synchronises the device with asnumpy's post-operator contract.
///
/// Row-major to column-major swap mapping (docs/matmul_backend.md):
///   C_rm(m x n) = A_rm(m x k) * B_rm(k x n)   is computed as   C_cm(n x m) = B_cm(n x k) * A_cm(k x m)
/// so the BLAS "A" operand is the row-major x2 buffer (lda = n), the BLAS "B" operand is the row-major
/// x1 buffer (ldb = k) and C is the result buffer (ldc = n). The three buffers are simultaneously valid
/// row-major and column-major views, so no copy or transpose is needed.
/// Documented lower bounds: lda >= max(1, m_blas), ldb >= max(1, k_blas), ldc >= max(1, m_blas);
/// k == 0 is legal (C := beta * C) and then requires ldb >= 1 rather than 0.
void RunOpsBlasGemm(const NPUArray& x1, const NPUArray& x2, NPUArray& result, const GemmShape& shape) {
    ScopedBlasHandle handle;
    CheckBlas(aclblasCreate(handle.out()), "aclblasCreate");
    // nullptr selects the library's default stream, matching how every other asnumpy operator submits
    // its work (aclnn operators are submitted with a null stream as well).
    CheckBlas(aclblasSetStream(handle.get(), nullptr), "aclblasSetStream");

    const int64_t lda = std::max<int64_t>(1, shape.n);
    const int64_t ldb = std::max<int64_t>(1, shape.k);
    const int64_t ldc = std::max<int64_t>(1, shape.n);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int64_t stride_a = CheckedMul(shape.k, shape.n, "x2");
    const int64_t stride_b = CheckedMul(shape.m, shape.k, "x1");

    const aclblasStatus_t status = aclblasSgemmStridedBatched(
        handle.get(), ACLBLAS_OP_N, ACLBLAS_OP_N, ToBlasInt(shape.n, "n"), ToBlasInt(shape.m, "m"),
        ToBlasInt(shape.k, "k"), &alpha, static_cast<const float*>(x2.device_address()), ToBlasInt(lda, "lda"),
        stride_a, static_cast<const float*>(x1.device_address()), ToBlasInt(ldb, "ldb"), stride_b, &beta,
        static_cast<float*>(result.device_address()), ToBlasInt(ldc, "ldc"), shape.elements, 1);
    CheckBlas(status, "aclblasSgemmStridedBatched");

    // The kernels are asynchronous with respect to the handle's stream; synchronising the device keeps
    // asnumpy's established post-operator contract and guarantees the inputs and the result stay valid
    // for the whole execution.
    auto error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
}

#else // ASNUMPY_ENABLE_OPS_BLAS

/// Message shown when the optional backend was not compiled in (kept outside the entry point so that
/// function stays small, see the project's function-size rule).
std::string BackendUnavailableMessage() {
    return "asnumpy.matmul(ops-blas) is not available: this asnumpy build was compiled without CANN ops-blas. "
           "Build ops-blas for your SoC (git clone https://gitcode.com/cann/ops-blas.git && "
           "bash build.sh --soc=ascend910b3 --ops=gemm_strided_batched), then rebuild asnumpy with "
           "-DASNUMPY_ENABLE_OPS_BLAS=ON and -DASNUMPY_OPS_BLAS_ROOT=<ops-blas build or install prefix>.";
}

#endif // ASNUMPY_ENABLE_OPS_BLAS

} // namespace

NPUArray MatmulOpsBlas(const NPUArray& x1, const NPUArray& x2) {
#ifdef ASNUMPY_ENABLE_OPS_BLAS
    const GemmShape shape = ValidateOperands(x1, x2);
    // ops-blas documents m==0 / n==0 / batchCount==0 as "no computation, return SUCCESS" and leaves C
    // untouched, so an empty result is produced here instead of handing an unwritten buffer back.
    if (shape.m == 0 || shape.n == 0) {
        return NPUArray({shape.m, shape.n}, ACL_FLOAT);
    }
    NPUArray result({shape.m, shape.n}, ACL_FLOAT);
    RunOpsBlasGemm(x1, x2, result, shape);
    return result;
#else
    throw std::runtime_error(BackendUnavailableMessage());
#endif
}
