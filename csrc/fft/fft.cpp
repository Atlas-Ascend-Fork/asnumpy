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

#include <asnumpy/fft/fft.hpp>

#include <acl/acl.h>
#include <fmt/format.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <asnumpy/utils/status_handler.hpp>

#ifdef ASNUMPY_ENABLE_FFT
#include <cann_ops_fft.h>
#endif

namespace asnumpy {
namespace fft {

#ifdef ASNUMPY_ENABLE_FFT

namespace {

/// ops-fft's 1-D FFT sizes are bounded by 2^27 for every kernel path on Ascend 910B / 950.
constexpr int64_t kMaxFftSize = int64_t{1} << 27;

/// Translate an aclfftResult into a Python-visible exception carrying the library message.
void CheckAclfft(aclfftResult status, const char* api) {
    if (status == ACLFFT_SUCCESS) {
        return;
    }
    throw std::runtime_error(fmt::format("asnumpy.fft: {} failed: {} (aclfftResult={})", api,
                                         aclfftGetErrorString(status), static_cast<int>(status)));
}

/// Owns an aclfft plan so that every error path still destroys it.
class ScopedAclfftPlan {
  public:
    ScopedAclfftPlan() = default;
    ScopedAclfftPlan(const ScopedAclfftPlan&) = delete;
    ScopedAclfftPlan& operator=(const ScopedAclfftPlan&) = delete;
    ~ScopedAclfftPlan() { reset(); }

    aclfftHandle* out() { return &plan_; }
    aclfftHandle get() const { return plan_; }

    void reset() {
        if (plan_ != nullptr) {
            aclfftDestroy(plan_);
            plan_ = nullptr;
        }
    }

  private:
    aclfftHandle plan_ = nullptr;
};

} // namespace

NPUArray Fft1DC2C(const NPUArray& x, bool inverse) {
    if (x.shape.size() != 1) {
        throw std::invalid_argument(fmt::format(
            "asnumpy.fft: only one-dimensional transforms are supported, got a {}-dimensional array", x.shape.size()));
    }
    if (x.aclDtype != ACL_COMPLEX64) {
        throw std::invalid_argument(
            fmt::format("asnumpy.fft: only complex64 input is supported, got {}", AclDtypeName(x.aclDtype)));
    }

    const int64_t n = x.shape[0];
    if (n <= 0) {
        throw std::invalid_argument("asnumpy.fft: input must contain at least one element");
    }
    if (n > kMaxFftSize) {
        throw std::invalid_argument(
            fmt::format("asnumpy.fft: transform size {} exceeds the supported maximum 2^27", n));
    }

    const size_t bytes = static_cast<size_t>(n) * sizeof(aclfftComplex);
    std::vector<aclfftComplex> host_in(static_cast<size_t>(n));
    std::vector<aclfftComplex> host_out(static_cast<size_t>(n));

    auto error = aclrtMemcpy(host_in.data(), bytes, x.device_address(), bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    ACL_RT_CHECK(error, "aclrtMemcpy(device->host)");

    // batch == 1 and ACLFFT_HORIZONTAL: a one-dimensional transform over the whole array.
    ScopedAclfftPlan plan;
    CheckAclfft(aclfftPlan1d(plan.out(), static_cast<int>(n), ACLFFT_C2C, 1, ACLFFT_HORIZONTAL), "aclfftPlan1d");
    // The plan runs on the default stream, like every other asnumpy operator (aclnn operators are
    // submitted with a null stream as well); aclfftExecC2C synchronises before it returns.
    CheckAclfft(aclfftSetStream(plan.get(), nullptr), "aclfftSetStream");
    CheckAclfft(aclfftExecC2C(plan.get(), host_in.data(), host_out.data(), inverse ? ACLFFT_BACKWARD : ACLFFT_FORWARD),
                "aclfftExecC2C");
    plan.reset();

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

    if (inverse) {
        // ACLFFT_BACKWARD is unnormalised (returns N * ifft), so the 1/N that numpy.fft.ifft
        // promises is applied exactly once, here, on the host buffer.
        const float scale = 1.0f / static_cast<float>(n);
        for (auto& value : host_out) {
            value.x *= scale;
            value.y *= scale;
        }
    }

    NPUArray result({n}, ACL_COMPLEX64);
    error = aclrtMemcpy(result.device_address(), bytes, host_out.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_RT_CHECK(error, "aclrtMemcpy(host->device)");
    return result;
}

#else // ASNUMPY_ENABLE_FFT

NPUArray Fft1DC2C(const NPUArray&, bool) {
    throw std::runtime_error("asnumpy.fft is not available: this asnumpy build was compiled without CANN ops-fft. "
                             "Build ops-fft for your SoC (git clone https://gitcode.com/cann/ops-fft.git && "
                             "bash build.sh --pkg --soc=Ascend910B), install it (./build_out/*.run --full --quiet "
                             "--install-path=<prefix>), then rebuild asnumpy with -DASNUMPY_ENABLE_FFT=ON and "
                             "-DASNUMPY_OPS_FFT_ROOT=<prefix>.");
}

#endif // ASNUMPY_ENABLE_FFT

} // namespace fft
} // namespace asnumpy
