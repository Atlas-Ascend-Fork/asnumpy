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

// Map where aclblasSgemmStridedBatched actually writes C on 910B (arch22).
#include <acl/acl.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "cann_ops_blas.h"

static const char* SN(aclblasStatus_t s) {
    return s == 0 ? "SUCCESS"
                  : (s == 2 ? "ALLOC_FAILED" : (s == 3 ? "INVALID_VALUE" : (s == 7 ? "NOT_SUPPORTED" : "OTHER")));
}
int main(int argc, char** argv) {
    const size_t wsBytes = (argc > 1) ? std::strtoul(argv[1], nullptr, 10) : 0;
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    aclblasHandle_t h = nullptr;
    aclblasCreate(&h);
    aclblasSetStream(h, stream);
    void* wsp = nullptr;
    if (wsBytes > 0) {
        aclrtMalloc(&wsp, wsBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        aclblasSetWorkspace(h, wsp, wsBytes);
    }

    const int sizes[][3] = {{2, 2, 2},    {4, 4, 4},    {8, 8, 8},    {13, 7, 5},   {16, 16, 16},
                            {17, 17, 17}, {20, 20, 20}, {32, 32, 32}, {64, 64, 64}, {7, 5, 3}};
    std::printf("workspace=%zu bytes\n", wsBytes);
    std::printf("  %-16s %-9s %-12s %-12s\n", "MxNxK", "status", "max|C|", "verdict");
    for (auto& s : sizes) {
        int64_t M = s[0], N = s[1], K = s[2];
        std::vector<float> A(M * K, 0.5f), B(K * N, 0.25f), C(M * N, 12345.0f);
        void *dA = nullptr, *dB = nullptr, *dC = nullptr;
        aclrtMalloc(&dA, A.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dB, B.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dC, C.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(dA, A.size() * 4, A.data(), A.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(dB, B.size() * 4, B.data(), B.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(dC, C.size() * 4, C.data(), C.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE);
        float alpha = 1.0f, beta = 0.0f;
        aclblasStatus_t st = aclblasSgemmStridedBatched(
            h, ACLBLAS_OP_N, ACLBLAS_OP_N, (int)N, (int)M, (int)K, &alpha, (const float*)dB, (int)N, (int64_t)K * N,
            (const float*)dA, (int)K, (int64_t)M * K, &beta, (float*)dC, (int)N, (int64_t)M * N, 1);
        aclrtSynchronizeStream(stream);
        std::vector<float> got(M * N);
        aclrtMemcpy(got.data(), got.size() * 4, dC, got.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST);
        // expected value with A=0.5, B=0.25: C[i][j] = 0.125*K
        double expv = 0.125 * (double)K, maxErr = 0;
        float maxAbs = 0;
        for (float v : got) {
            maxAbs = std::max(maxAbs, std::fabs(v));
            maxErr = std::max(maxErr, std::fabs((double)v - expv));
        }
        const char* verdict = (maxErr < 1e-4) ? "CORRECT" : (maxAbs == 12345.0f ? "C NOT WRITTEN" : "WRONG");
        std::printf("  %-16s %-9s %-12.4f %-12s\n",
                    (std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K)).c_str(), SN(st), maxAbs,
                    verdict);
        aclrtFree(dA);
        aclrtFree(dB);
        aclrtFree(dC);
    }
    if (wsp)
        aclrtFree(wsp);
    aclblasDestroy(h);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
