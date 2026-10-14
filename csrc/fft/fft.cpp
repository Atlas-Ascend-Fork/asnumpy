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

#include <algorithm>
#include <cstdint>
#include <limits>
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

/// ops-fft supports 1-D sizes up to 2^27 on every kernel path of the supported SoCs.
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

/// Element counts come from user shapes; keep every intermediate product checked.
int64_t CheckedMul(int64_t a, int64_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<int64_t>::max() / a) {
        throw std::overflow_error(fmt::format("asnumpy.fft: {} element count overflows int64", what));
    }
    return a * b;
}

int64_t ShapeSize(const std::vector<int64_t>& shape) {
    int64_t size = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        size = CheckedMul(size, shape[i], "array");
    }
    return size;
}

/// ops-fft takes int parameters; reject anything that cannot be represented.
int ToInt(int64_t value, const char* what) {
    if (value <= 0 || value > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument(
            fmt::format("asnumpy.fft: {}={} is outside the supported range 1..2^31-1", what, value));
    }
    return static_cast<int>(value);
}

size_t BytesFor(int64_t elements, size_t element_size) {
    return static_cast<size_t>(CheckedMul(elements, static_cast<int64_t>(element_size), "buffer"));
}

/// Factorization of an array shape around one transformed axis: the batch count is outer * inner.
struct AxisLayout {
    int64_t outer = 1;
    int64_t inner = 1;

    int64_t Batches() const { return outer * inner; }
};

AxisLayout ComputeLayout(const std::vector<int64_t>& shape, int64_t axis) {
    AxisLayout layout;
    for (int64_t i = 0; i < axis; ++i) {
        layout.outer = CheckedMul(layout.outer, shape[i], "outer");
    }
    for (int64_t i = axis + 1; i < static_cast<int64_t>(shape.size()); ++i) {
        layout.inner = CheckedMul(layout.inner, shape[i], "inner");
    }
    return layout;
}

/// Gather `src` (row-major, `src_shape`) into packed rows of `packed_len` along `axis`.
/// Positions beyond `src_len` are zero, which implements NumPy's zero padding.
void PackRows(const aclfftComplex* src, const AxisLayout& layout, int64_t src_len, int64_t packed_len,
              std::vector<aclfftComplex>& packed) {
    const aclfftComplex zero{0.0f, 0.0f};
    const int64_t copy = std::min(src_len, packed_len);
    for (int64_t o = 0; o < layout.outer; ++o) {
        for (int64_t i = 0; i < layout.inner; ++i) {
            const int64_t batch = o * layout.inner + i;
            aclfftComplex* row = packed.data() + batch * packed_len;
            const aclfftComplex* base = src + o * src_len * layout.inner + i;
            for (int64_t j = 0; j < copy; ++j) {
                row[j] = base[j * layout.inner];
            }
            for (int64_t j = copy; j < packed_len; ++j) {
                row[j] = zero;
            }
        }
    }
}

/// Scatter packed rows (row stride `dst_len`) back into a row-major array shaped `dst_shape`.
void UnpackRows(const std::vector<aclfftComplex>& packed, const AxisLayout& layout, int64_t dst_len,
                aclfftComplex* dst) {
    for (int64_t o = 0; o < layout.outer; ++o) {
        for (int64_t i = 0; i < layout.inner; ++i) {
            const int64_t batch = o * layout.inner + i;
            const aclfftComplex* row = packed.data() + batch * dst_len;
            aclfftComplex* base = dst + o * dst_len * layout.inner + i;
            for (int64_t j = 0; j < dst_len; ++j) {
                base[j * layout.inner] = row[j];
            }
        }
    }
}

/// Float counterpart of PackRows, used by the real-to-complex transform.
void PackRowsFloat(const float* src, const AxisLayout& layout, int64_t src_len, int64_t packed_len,
                   std::vector<float>& packed) {
    const int64_t copy = std::min(src_len, packed_len);
    for (int64_t o = 0; o < layout.outer; ++o) {
        for (int64_t i = 0; i < layout.inner; ++i) {
            const int64_t batch = o * layout.inner + i;
            float* row = packed.data() + batch * packed_len;
            const float* base = src + o * src_len * layout.inner + i;
            for (int64_t j = 0; j < copy; ++j) {
                row[j] = base[j * layout.inner];
            }
            for (int64_t j = copy; j < packed_len; ++j) {
                row[j] = 0.0f;
            }
        }
    }
}

/// Float counterpart of UnpackRows, used by the complex-to-real transform.
void UnpackRowsFloat(const std::vector<float>& packed, const AxisLayout& layout, int64_t dst_len, float* dst) {
    for (int64_t o = 0; o < layout.outer; ++o) {
        for (int64_t i = 0; i < layout.inner; ++i) {
            const int64_t batch = o * layout.inner + i;
            const float* row = packed.data() + batch * dst_len;
            float* base = dst + o * dst_len * layout.inner + i;
            for (int64_t j = 0; j < dst_len; ++j) {
                base[j * layout.inner] = row[j];
            }
        }
    }
}

/// Create one ops-fft plan for `batches` transforms of length `n` and run `body` with it.
/// The plan is always destroyed, including on the error paths.
template <typename Fn> void WithPlan1D(int64_t batches, int64_t n, aclfftType type, Fn&& body) {
    ScopedAclfftPlan plan;
    CheckAclfft(aclfftPlan1d(plan.out(), ToInt(n, "n"), type, ToInt(batches, "batch"), ACLFFT_HORIZONTAL),
                "aclfftPlan1d");
    // The plan runs on the default stream, exactly like the aclnn operators elsewhere in asnumpy
    // (submitted with a null stream). aclfftExecC2C synchronises internally; the device
    // synchronisation below keeps asnumpy's established post-operator contract.
    CheckAclfft(aclfftSetStream(plan.get(), nullptr), "aclfftSetStream");
    body(plan.get());
    auto error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
}

void ValidateTransformInput(const NPUArray& x, int64_t axis, aclDataType expected) {
    if (x.aclDtype != expected) {
        throw std::invalid_argument(fmt::format("asnumpy.fft: only {} input is supported, got {}",
                                                AclDtypeName(expected), AclDtypeName(x.aclDtype)));
    }
    if (x.shape.empty()) {
        throw std::invalid_argument("asnumpy.fft: a zero-dimensional array has no transform axis");
    }
    if (axis < 0 || axis >= static_cast<int64_t>(x.shape.size())) {
        throw std::invalid_argument(
            fmt::format("asnumpy.fft: axis {} is out of bounds for a {}-dimensional array", axis, x.shape.size()));
    }
    // A zero-length transform axis is not rejected here: NumPy accepts it whenever an explicit n
    // supplies the length (fft(empty, n=4) and rfft(empty, n=5) are valid), and the packing below
    // turns such an axis into an all-zero row. The public layer rejects the n=None case.
}

void ValidateLength(int64_t n) {
    if (n <= 0) {
        throw std::invalid_argument(fmt::format("asnumpy.fft: invalid number of FFT data points ({}) specified", n));
    }
    if (n > kMaxFftSize) {
        throw std::invalid_argument(
            fmt::format("asnumpy.fft: transform size {} exceeds the supported maximum 2^27", n));
    }
}

std::vector<aclfftComplex> CopyToHost(const NPUArray& x) {
    const int64_t elements = ShapeSize(x.shape);
    std::vector<aclfftComplex> host(static_cast<size_t>(elements));
    if (elements > 0) {
        const size_t bytes = BytesFor(elements, sizeof(aclfftComplex));
        auto error = aclrtMemcpy(host.data(), bytes, x.device_address(), bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        ACL_RT_CHECK(error, "aclrtMemcpy(device->host)");
    }
    return host;
}

void CopyToDevice(const std::vector<aclfftComplex>& host, const std::vector<int64_t>& shape, NPUArray& dst) {
    const int64_t elements = ShapeSize(shape);
    if (elements > 0) {
        const size_t bytes = BytesFor(elements, sizeof(aclfftComplex));
        auto error = aclrtMemcpy(dst.device_address(), bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        ACL_RT_CHECK(error, "aclrtMemcpy(host->device)");
    }
}

std::vector<float> CopyToHostFloat(const NPUArray& x) {
    const int64_t elements = ShapeSize(x.shape);
    std::vector<float> host(static_cast<size_t>(elements));
    if (elements > 0) {
        const size_t bytes = BytesFor(elements, sizeof(float));
        auto error = aclrtMemcpy(host.data(), bytes, x.device_address(), bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        ACL_RT_CHECK(error, "aclrtMemcpy(device->host)");
    }
    return host;
}

void CopyToDeviceFloat(const std::vector<float>& host, const std::vector<int64_t>& shape, NPUArray& dst) {
    const int64_t elements = ShapeSize(shape);
    if (elements > 0) {
        const size_t bytes = BytesFor(elements, sizeof(float));
        auto error = aclrtMemcpy(dst.device_address(), bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        ACL_RT_CHECK(error, "aclrtMemcpy(host->device)");
    }
}

/// Apply one batched 1-D complex transform along `axis`, producing `dst_shape`/`dst`.
void Apply1DC2C(const std::vector<aclfftComplex>& src, const std::vector<int64_t>& src_shape, int64_t axis, int64_t n,
                bool inverse, double scale, std::vector<int64_t>& dst_shape, std::vector<aclfftComplex>& dst) {
    dst_shape = src_shape;
    dst_shape[axis] = n;
    dst.assign(static_cast<size_t>(ShapeSize(dst_shape)), aclfftComplex{0.0f, 0.0f});

    const AxisLayout layout = ComputeLayout(src_shape, axis);
    const int64_t batches = layout.Batches();
    if (batches == 0) {
        return; // an empty batch dimension stays an empty result (NumPy returns an empty array)
    }

    std::vector<aclfftComplex> packed_in(static_cast<size_t>(CheckedMul(batches, n, "packed input")));
    std::vector<aclfftComplex> packed_out(static_cast<size_t>(CheckedMul(batches, n, "packed output")));
    PackRows(src.data(), layout, src_shape[axis], n, packed_in);
    WithPlan1D(batches, n, ACLFFT_C2C, [&](aclfftHandle plan) {
        CheckAclfft(
            aclfftExecC2C(plan, packed_in.data(), packed_out.data(), inverse ? ACLFFT_BACKWARD : ACLFFT_FORWARD),
            "aclfftExecC2C");
    });
    if (scale != 1.0) {
        for (auto& value : packed_out) {
            value.x = static_cast<float>(static_cast<double>(value.x) * scale);
            value.y = static_cast<float>(static_cast<double>(value.y) * scale);
        }
    }
    UnpackRows(packed_out, layout, n, dst.data());
}

/// Apply one batched 1-D real-to-complex transform: the transformed axis becomes n/2+1 complex bins.
void Apply1DR2C(const std::vector<float>& src, const std::vector<int64_t>& src_shape, int64_t axis, int64_t n,
                double scale, std::vector<int64_t>& dst_shape, std::vector<aclfftComplex>& dst) {
    const int64_t spectrum = n / 2 + 1;
    dst_shape = src_shape;
    dst_shape[axis] = spectrum;
    dst.assign(static_cast<size_t>(ShapeSize(dst_shape)), aclfftComplex{0.0f, 0.0f});

    const AxisLayout layout = ComputeLayout(src_shape, axis);
    const int64_t batches = layout.Batches();
    if (batches == 0) {
        return;
    }

    std::vector<float> packed_in(static_cast<size_t>(CheckedMul(batches, n, "packed input")));
    std::vector<aclfftComplex> packed_out(static_cast<size_t>(CheckedMul(batches, spectrum, "packed output")));
    PackRowsFloat(src.data(), layout, src_shape[axis], n, packed_in);
    WithPlan1D(batches, n, ACLFFT_R2C, [&](aclfftHandle plan) {
        CheckAclfft(aclfftExecR2C(plan, packed_in.data(), packed_out.data()), "aclfftExecR2C");
    });
    if (scale != 1.0) {
        for (auto& value : packed_out) {
            value.x = static_cast<float>(static_cast<double>(value.x) * scale);
            value.y = static_cast<float>(static_cast<double>(value.y) * scale);
        }
    }
    UnpackRows(packed_out, layout, spectrum, dst.data());
}

/// Apply one batched 1-D complex-to-real transform: the input holds n/2+1 bins, the output n reals.
void Apply1DC2R(const std::vector<aclfftComplex>& src, const std::vector<int64_t>& src_shape, int64_t axis, int64_t n,
                double scale, std::vector<int64_t>& dst_shape, std::vector<float>& dst) {
    const int64_t spectrum = n / 2 + 1;
    dst_shape = src_shape;
    dst_shape[axis] = n;
    dst.assign(static_cast<size_t>(ShapeSize(dst_shape)), 0.0f);

    const AxisLayout layout = ComputeLayout(src_shape, axis);
    const int64_t batches = layout.Batches();
    if (batches == 0) {
        return;
    }

    std::vector<aclfftComplex> packed_in(static_cast<size_t>(CheckedMul(batches, spectrum, "packed input")));
    std::vector<float> packed_out(static_cast<size_t>(CheckedMul(batches, n, "packed output")));
    // PackRows truncates a longer spectrum and zero-pads a shorter one, which is exactly NumPy's
    // rule for irfft with an explicit n.
    PackRows(src.data(), layout, src_shape[axis], spectrum, packed_in);
    WithPlan1D(batches, n, ACLFFT_C2R, [&](aclfftHandle plan) {
        CheckAclfft(aclfftExecC2R(plan, packed_in.data(), packed_out.data()), "aclfftExecC2R");
    });
    if (scale != 1.0) {
        for (auto& value : packed_out) {
            value = static_cast<float>(static_cast<double>(value) * scale);
        }
    }
    UnpackRowsFloat(packed_out, layout, n, dst.data());
}

} // namespace

NPUArray FftC2C(const NPUArray& x, int64_t axis, int64_t n, bool inverse, double scale) {
    ValidateTransformInput(x, axis, ACL_COMPLEX64);
    ValidateLength(n);

    std::vector<aclfftComplex> host_in = CopyToHost(x);
    std::vector<int64_t> out_shape;
    std::vector<aclfftComplex> host_out;
    Apply1DC2C(host_in, x.shape, axis, n, inverse, scale, out_shape, host_out);

    NPUArray result(out_shape, ACL_COMPLEX64);
    CopyToDevice(host_out, out_shape, result);
    return result;
}

NPUArray Rfft(const NPUArray& x, int64_t axis, int64_t n, double scale) {
    ValidateTransformInput(x, axis, ACL_FLOAT);
    ValidateLength(n);

    std::vector<float> host_in = CopyToHostFloat(x);
    std::vector<int64_t> out_shape;
    std::vector<aclfftComplex> host_out;
    Apply1DR2C(host_in, x.shape, axis, n, scale, out_shape, host_out);

    NPUArray result(out_shape, ACL_COMPLEX64);
    CopyToDevice(host_out, out_shape, result);
    return result;
}

NPUArray Irfft(const NPUArray& x, int64_t axis, int64_t n, double scale) {
    ValidateTransformInput(x, axis, ACL_COMPLEX64);
    ValidateLength(n);

    std::vector<aclfftComplex> host_in = CopyToHost(x);
    std::vector<int64_t> out_shape;
    std::vector<float> host_out;
    Apply1DC2R(host_in, x.shape, axis, n, scale, out_shape, host_out);

    NPUArray result(out_shape, ACL_FLOAT);
    CopyToDeviceFloat(host_out, out_shape, result);
    return result;
}

NPUArray Fft2C2C(const NPUArray& x, int64_t axis0, int64_t axis1, int64_t n0, int64_t n1, bool inverse,
                 double scale0, double scale1) {
    ValidateTransformInput(x, axis0, ACL_COMPLEX64);
    ValidateTransformInput(x, axis1, ACL_COMPLEX64);
    ValidateLength(n0);
    ValidateLength(n1);

    // ops-fft's native two-dimensional operator (fft2_d) is restricted to nx, ny in {32, 64, 128} on
    // Ascend 910B and is not implemented on Ascend 950, so the general path applies two verified
    // one-dimensional ops-fft transforms, one per axis; both passes execute on the NPU. The array is
    // bridged through the host once: the passes ping-pong between host buffers and a single H2D copy
    // publishes the result. This is *not* the native two-dimensional kernel, and it carries the
    // host-side layout overhead documented in docs/fft.md.
    //
    // The passes run in reverse axis order because that is what NumPy's _raw_fftnd does; the order
    // is observable when the same axis is given twice with different lengths, where NumPy ends up
    // with the length of the first entry.
    std::vector<int64_t> shape1;
    std::vector<int64_t> shape2;
    std::vector<aclfftComplex> buffer1;
    std::vector<aclfftComplex> buffer2;
    Apply1DC2C(CopyToHost(x), x.shape, axis1, n1, inverse, scale1, shape1, buffer1);
    Apply1DC2C(buffer1, shape1, axis0, n0, inverse, scale0, shape2, buffer2);

    NPUArray result(shape2, ACL_COMPLEX64);
    CopyToDevice(buffer2, shape2, result);
    return result;
}

#else // ASNUMPY_ENABLE_FFT

namespace {
[[noreturn]] void BackendUnavailable() {
    throw std::runtime_error("asnumpy.fft is not available: this asnumpy build was compiled without CANN ops-fft. "
                             "Build ops-fft for your SoC (git clone https://gitcode.com/cann/ops-fft.git && "
                             "bash build.sh --pkg --soc=Ascend910B), install it (./build_out/*.run --full --quiet "
                             "--install-path=<prefix>), then rebuild asnumpy with -DASNUMPY_ENABLE_FFT=ON and "
                             "-DASNUMPY_OPS_FFT_ROOT=<prefix>.");
}
} // namespace

NPUArray FftC2C(const NPUArray&, int64_t, int64_t, bool, double) { BackendUnavailable(); }
NPUArray Rfft(const NPUArray&, int64_t, int64_t, double) { BackendUnavailable(); }
NPUArray Irfft(const NPUArray&, int64_t, int64_t, double) { BackendUnavailable(); }
NPUArray Fft2C2C(const NPUArray&, int64_t, int64_t, int64_t, int64_t, bool, double, double) { BackendUnavailable(); }

#endif // ASNUMPY_ENABLE_FFT

} // namespace fft
} // namespace asnumpy
