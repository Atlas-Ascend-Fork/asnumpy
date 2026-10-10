// Minimal ops-blas GEMM probe on the real NPU (issue #28 stage 3 precondition).
//
// Validates, before any asnumpy integration:
//   1. which GEMM entry points exist in the arch22 (Ascend 910B) build,
//   2. that aclblasSgemmStridedBatched with batchCount=1 computes a dense GEMM,
//   3. the row-major (asnumpy) -> column-major (BLAS) swap-operand mapping, on non-square shapes,
//   4. workspace requirements, return codes, input immutability and batching sanity.
#include <acl/acl.h>
#include <dlfcn.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "cann_ops_blas.h"

#define CHECK_RET(cond, expr)                       \
    do {                                            \
        if (!(cond)) {                              \
            std::printf("CHECK failed: %s\n", #cond); \
            expr;                                   \
        }                                           \
    } while (0)

namespace {

const char* StatusName(aclblasStatus_t s) {
    switch (s) {
    case ACLBLAS_STATUS_SUCCESS: return "SUCCESS";
    case ACLBLAS_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
    case ACLBLAS_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
    case ACLBLAS_STATUS_INVALID_VALUE: return "INVALID_VALUE";
    case ACLBLAS_STATUS_MAPPING_ERROR: return "MAPPING_ERROR";
    case ACLBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
    case ACLBLAS_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
    case ACLBLAS_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
    case ACLBLAS_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
    case ACLBLAS_STATUS_HANDLE_IS_NULLPTR: return "HANDLE_IS_NULLPTR";
    case ACLBLAS_STATUS_INVALID_ENUM: return "INVALID_ENUM";
    default: return "UNKNOWN";
    }
}

struct BlasHandle {
    aclblasHandle_t h = nullptr;
    ~BlasHandle() {
        if (h != nullptr) aclblasDestroy(h);
    }
};

struct DeviceBuf {
    void* ptr = nullptr;
    size_t bytes = 0;
    ~DeviceBuf() {
        if (ptr != nullptr) aclrtFree(ptr);
    }
    bool alloc(size_t n) {
        bytes = n;
        return aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    }
};

std::vector<float> RandomMatrix(int64_t rows, int64_t cols, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> m(static_cast<size_t>(rows) * cols);
    for (auto& v : m) v = dist(gen);
    return m;
}

// Row-major reference C = A(MxK) * B(KxN) accumulated in double.
std::vector<double> Reference(const std::vector<float>& A, const std::vector<float>& B, int64_t M, int64_t K,
                              int64_t N) {
    std::vector<double> C(static_cast<size_t>(M) * N, 0.0);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int64_t k = 0; k < K; ++k) acc += static_cast<double>(A[i * K + k]) * static_cast<double>(B[k * N + j]);
            C[i * N + j] = acc;
        }
    }
    return C;
}

// Row-major A/B/C buffers are simultaneously valid column-major buffers with the swap mapping:
//   C_rm(MxN) = A_rm(MxK) * B_rm(KxN)
//   <=> in column-major terms: C_blas(NxM) = B_blas(NxK) * A_blas(KxM)
//   m_blas = N, n_blas = M, k_blas = K, lda = N (B buffer), ldb = K (A buffer), ldc = N (C buffer).
aclblasStatus_t GemmWithSwapMapping(aclblasHandle_t handle, const DeviceBuf& A, const DeviceBuf& B, DeviceBuf& C,
                                    int64_t M, int64_t N, int64_t K, float alpha, float beta, int batchCount,
                                    int64_t strideA_rm, int64_t strideB_rm, int64_t strideC_rm) {
    return aclblasSgemmStridedBatched(handle, ACLBLAS_OP_N, ACLBLAS_OP_N, static_cast<int>(N), static_cast<int>(M),
                                      static_cast<int>(K), &alpha, static_cast<const float*>(B.ptr),
                                      static_cast<int>(N), strideA_rm, static_cast<const float*>(A.ptr),
                                      static_cast<int>(K), strideB_rm, &beta, static_cast<float*>(C.ptr),
                                      static_cast<int>(N), strideC_rm, batchCount);
}

struct Case {
    int64_t M, N, K;
};

int RunCase(aclblasHandle_t handle, const Case& c, size_t workspaceBytes, bool verbose) {
    const auto& [M, N, K] = c;
    const auto A = RandomMatrix(M, K, static_cast<uint32_t>(M * 131 + K));
    const auto B = RandomMatrix(K, N, static_cast<uint32_t>(N * 977 + K));
    const auto expected = Reference(A, B, M, K, N);

    DeviceBuf dA, dB, dC;
    if (!dA.alloc(A.size() * sizeof(float)) || !dB.alloc(B.size() * sizeof(float)) ||
        !dC.alloc(static_cast<size_t>(M) * N * sizeof(float))) {
        std::printf("  [%lldx%lldx%lld] device alloc failed\n", (long long)M, (long long)N, (long long)K);
        return 1;
    }
    aclrtMemcpy(dA.ptr, dA.bytes, A.data(), dA.bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dB.ptr, dB.bytes, B.data(), dB.bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    std::vector<float> cInit(static_cast<size_t>(M) * N, 12345.0f);  // sentinel: detects unread/beta handling
    aclrtMemcpy(dC.ptr, dC.bytes, cInit.data(), dC.bytes, ACL_MEMCPY_HOST_TO_DEVICE);

    DeviceBuf workspace;
    if (workspaceBytes > 0) {
        if (!workspace.alloc(workspaceBytes)) {
            std::printf("  workspace alloc failed\n");
            return 1;
        }
        aclblasStatus_t ws = aclblasSetWorkspace(handle, workspace.ptr, workspace.bytes);
        if (ws != ACLBLAS_STATUS_SUCCESS) {
            std::printf("  aclblasSetWorkspace -> %s\n", StatusName(ws));
            return 1;
        }
    }

    const float alpha = 1.0f, beta = 0.0f;
    aclblasStatus_t st = GemmWithSwapMapping(handle, dA, dB, dC, M, N, K, alpha, beta, 1, M * K, K * N, M * N);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        std::printf("  [%lldx%lldx%lld] ws=%zu -> %s (%d)\n", (long long)M, (long long)N, (long long)K, workspaceBytes,
                    StatusName(st), (int)st);
        return 1;
    }
    // The handle carries its own stream: the kernels are async w.r.t. it, so the caller must
    // synchronise that stream (or the device) before reading C.
    aclrtSynchronizeDevice();

    std::vector<float> got(static_cast<size_t>(M) * N);
    aclrtMemcpy(got.data(), got.size() * sizeof(float), dC.ptr, dC.bytes, ACL_MEMCPY_DEVICE_TO_HOST);

    double maxAbs = 0.0, maxRel = 0.0, scale = 1.0;
    for (size_t i = 0; i < got.size(); ++i) {
        maxAbs = std::max(maxAbs, std::fabs(static_cast<double>(got[i]) - expected[i]));
        maxRel = std::max(maxRel, std::fabs(static_cast<double>(got[i]) - expected[i]));
        scale = std::max(scale, std::fabs(expected[i]));
    }
    // Confirm the inputs were not modified.
    std::vector<float> aBack(A.size()), bBack(B.size());
    aclrtMemcpy(aBack.data(), dA.bytes, dA.ptr, dA.bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(bBack.data(), dB.bytes, dB.ptr, dB.bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    bool inputsIntact = (aBack == A) && (bBack == B);

    if (verbose) {
        std::printf("  [%4lld x %4lld x %4lld] ws=%-8zu %s  max_abs=%.3e  rel(=abs/|C|max)=%.3e  inputs_intact=%s\n",
                    (long long)M, (long long)N, (long long)K, workspaceBytes, StatusName(st), maxAbs, maxRel / scale,
                    inputsIntact ? "yes" : "NO");
    }
    return (maxRel / scale < 1e-5 && inputsIntact) ? 0 : 2;
}

} // namespace

int main(int argc, char** argv) {
    const bool verbose = (argc > 1 && std::string(argv[1]) == "-v");

    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, return 1);
    ret = aclrtSetDevice(0);
    CHECK_RET(ret == ACL_SUCCESS, return 1);
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, return 1);

    std::printf("=== 1) GEMM entry points present in this build ===\n");
    for (const char* sym : {"aclblasSgemm", "aclblasSgemmStridedBatched", "aclblasGemmEx", "aclblasSgemmBatched"}) {
        void* p = dlsym(RTLD_DEFAULT, sym);
        std::printf("  %-28s %s\n", sym, p != nullptr ? "PRESENT" : "ABSENT");
    }

    BlasHandle handle;
    aclblasStatus_t st = aclblasCreate(&handle.h);
    std::printf("=== 2) handle ===\n  aclblasCreate -> %s\n", StatusName(st));
    if (st != ACLBLAS_STATUS_SUCCESS) return 1;
    st = aclblasSetStream(handle.h, stream);
    std::printf("  aclblasSetStream(stream) -> %s\n", StatusName(st));
    int version = 0;
    if (aclblasGetVersion(handle.h, &version) == ACLBLAS_STATUS_SUCCESS) {
        std::printf("  aclblasGetVersion -> %d\n", version);
    }

    std::printf("=== 3) non-square dense GEMM via aclblasSgemmStridedBatched (batchCount=1), no workspace ===\n");
    int failures = 0;
    for (const Case& c : {Case{7, 5, 3}, Case{1, 9, 4}, Case{64, 128, 32}, Case{129, 257, 33}, Case{512, 512, 512}}) {
        failures += RunCase(handle.h, c, 0, true);
    }

    std::printf("=== 4) workspace behaviour (512x512x512) ===\n");
    for (size_t ws : {size_t{0}, size_t{128 * 4}, size_t{128 * 128 * 4}, size_t{1 << 20}}) {
        Case c{512, 512, 512};
        int rc = RunCase(handle.h, c, ws, false);
        std::printf("  workspace=%8zu bytes -> %s\n", ws, rc == 0 ? "OK" : (rc == 1 ? "FAILED" : "WRONG RESULT"));
    }

    std::printf("=== 5) batchCount=2 strided batching sanity (M=8 N=6 K=4) ===\n");
    {
        const int64_t M = 8, N = 6, K = 4;
        const auto A0 = RandomMatrix(M, K, 11), A1 = RandomMatrix(M, K, 22);
        const auto B0 = RandomMatrix(K, N, 33), B1 = RandomMatrix(K, N, 44);
        const auto E0 = Reference(A0, B0, M, K, N), E1 = Reference(A1, B1, M, K, N);
        std::vector<float> A2(A0), B2(B0);
        A2.insert(A2.end(), A1.begin(), A1.end());
        B2.insert(B2.end(), B1.begin(), B1.end());
        DeviceBuf dA, dB, dC;
        dA.alloc(A2.size() * sizeof(float));
        dB.alloc(B2.size() * sizeof(float));
        dC.alloc(2 * M * N * sizeof(float));
        aclrtMemcpy(dA.ptr, dA.bytes, A2.data(), dA.bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(dB.ptr, dB.bytes, B2.data(), dB.bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        const float alpha = 1.0f, beta = 0.0f;
        aclblasStatus_t bs =
            aclblasSgemmStridedBatched(handle.h, ACLBLAS_OP_N, ACLBLAS_OP_N, (int)N, (int)M, (int)K, &alpha,
                                       static_cast<const float*>(dB.ptr), (int)N, K * N,
                                       static_cast<const float*>(dA.ptr), (int)K, M * K, &beta,
                                       static_cast<float*>(dC.ptr), (int)N, M * N, 2);
        aclrtSynchronizeStream(stream);
        std::vector<float> got(2 * M * N);
        aclrtMemcpy(got.data(), got.size() * sizeof(float), dC.ptr, dC.bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        double e0 = 0.0, e1 = 0.0;
        for (int64_t i = 0; i < M * N; ++i) {
            e0 = std::max(e0, std::fabs(static_cast<double>(got[i]) - E0[i]));
            e1 = std::max(e1, std::fabs(static_cast<double>(got[M * N + i]) - E1[i]));
        }
        std::printf("  status=%s batch0_max_abs=%.3e batch1_max_abs=%.3e\n", StatusName(bs), e0, e1);
        if (bs != ACLBLAS_STATUS_SUCCESS || e0 > 1e-4 || e1 > 1e-4) failures += 1;
    }

    std::printf("=== 6) edge cases ===\n");
    {
        const int64_t M = 4, N = 4, K = 4;
        const auto A = RandomMatrix(M, K, 5), B = RandomMatrix(K, N, 6);
        DeviceBuf dA, dB, dC;
        dA.alloc(A.size() * sizeof(float));
        dB.alloc(B.size() * sizeof(float));
        dC.alloc(M * N * sizeof(float));
        aclrtMemcpy(dA.ptr, dA.bytes, A.data(), dA.bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(dB.ptr, dB.bytes, B.data(), dB.bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        const float alpha = 1.0f, beta = 0.0f, zero = 0.0f;

        // m == 0 -> documented immediate SUCCESS
        aclblasStatus_t s1 = aclblasSgemmStridedBatched(handle.h, ACLBLAS_OP_N, ACLBLAS_OP_N, 0, (int)N, (int)K,
                                                        &alpha, static_cast<const float*>(dB.ptr), (int)N, 0,
                                                        static_cast<const float*>(dA.ptr), (int)K, 0, &beta,
                                                        static_cast<float*>(dC.ptr), (int)N, 0, 1);
        // batchCount == 0 -> documented immediate SUCCESS
        aclblasStatus_t s2 = aclblasSgemmStridedBatched(handle.h, ACLBLAS_OP_N, ACLBLAS_OP_N, (int)N, (int)M, (int)K,
                                                        &alpha, static_cast<const float*>(dB.ptr), (int)N, K * N,
                                                        static_cast<const float*>(dA.ptr), (int)K, M * K, &beta,
                                                        static_cast<float*>(dC.ptr), (int)N, M * N, 0);
        // k == 0 with beta=0 -> documented C := 0
        std::vector<float> cInit(M * N, 7.0f);
        aclrtMemcpy(dC.ptr, dC.bytes, cInit.data(), dC.bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclblasStatus_t s3 = aclblasSgemmStridedBatched(handle.h, ACLBLAS_OP_N, ACLBLAS_OP_N, (int)N, (int)M, 0,
                                                        &alpha, static_cast<const float*>(dB.ptr), (int)N, 0,
                                                        static_cast<const float*>(dA.ptr), (int)K, 0, &beta,
                                                        static_cast<float*>(dC.ptr), (int)N, 0, 1);
        aclrtSynchronizeStream(stream);
        std::vector<float> got(M * N);
        aclrtMemcpy(got.data(), got.size() * sizeof(float), dC.ptr, dC.bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        float maxv = 0.0f;
        for (float v : got) maxv = std::max(maxv, std::fabs(v));

        // alpha == 0, beta == 1 with k > 0 -> documented C := C
        aclrtMemcpy(dC.ptr, dC.bytes, cInit.data(), dC.bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        const float one = 1.0f;
        aclblasStatus_t s4 = aclblasSgemmStridedBatched(handle.h, ACLBLAS_OP_N, ACLBLAS_OP_N, (int)N, (int)M, (int)K,
                                                        &zero, static_cast<const float*>(dB.ptr), (int)N, K * N,
                                                        static_cast<const float*>(dA.ptr), (int)K, M * K, &one,
                                                        static_cast<float*>(dC.ptr), (int)N, M * N, 1);
        aclrtSynchronizeStream(stream);
        std::vector<float> got4(M * N);
        aclrtMemcpy(got4.data(), got4.size() * sizeof(float), dC.ptr, dC.bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        bool alpha0_ok = true;
        for (int64_t i = 0; i < M * N; ++i) alpha0_ok = alpha0_ok && (got4[i] == 7.0f);

        std::printf("  m=0 -> %s ; batchCount=0 -> %s ; k=0,beta=0 -> %s (max|C|=%.1f) ; alpha=0,beta=1 keeps C -> %s\n",
                    StatusName(s1), StatusName(s2), StatusName(s3), maxv, alpha0_ok ? "yes" : "no");
        if (s1 != ACLBLAS_STATUS_SUCCESS || s2 != ACLBLAS_STATUS_SUCCESS || s3 != ACLBLAS_STATUS_SUCCESS || maxv != 0.0f ||
            !alpha0_ok) {
            failures += 1;
        }
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::printf("=== SUMMARY: %s (failures=%d) ===\n", failures == 0 ? "PROBE_OK" : "PROBE_FAILED", failures);
    return failures == 0 ? 0 : 1;
}
