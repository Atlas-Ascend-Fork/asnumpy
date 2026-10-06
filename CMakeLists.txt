# *****************************************************************************
# Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *****************************************************************************

cmake_minimum_required(VERSION 3.22...3.30)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

# CMake policies
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Symbol visibility. Do not remove, and do not change the value -- see the footgun below.
#
# asnumpy's public ABI is exactly one symbol: PyInit__core. The build produces no linkable artifact
# (every csrc/ target is an OBJECT lib; there is no SHARED/STATIC lib, no install(EXPORT), no
# installed header), and the wheel ships only asnumpy/_core*.so. So every C++ symbol here is an
# implementation detail and belongs hidden. pybind11 already exports the one symbol that must be
# exported, via PYBIND11_EXPORT on the module init function -- which is why no export macro of our
# own (ASNUMPY_API / generate_export_header) is warranted: it would advertise a C++ API contract
# that deliberately does not exist.
#
# The concrete hazard this closes, measured on this tree: without these two lines _core.so exported
# 1027 dynamic symbols, of which 808 were fmt::/spdlog:: vague-linkage template and inline copies
# (79% of the total), and 26 of those were *weak* symbols whose names also exist in libfmt.so.8 /
# libspdlog.so.1 -- the very libraries _core.so dynamically links (readelf -d shows both as NEEDED).
# Weak symbols merge by name at load time, first one wins, so any other extension in the same
# interpreter carrying a different fmt/spdlog version could silently bind our calls to its copy, or
# ours to its. It also exported 90 NPUArray symbols. With the lines: 31 symbols, 0 fmt, 0 spdlog,
# 0 NPUArray. Those template copies are inherent to a header-heavy C++ stack -- Cython-based CuPy
# has no fmt/spdlog and so has no equivalent exposure; do not cite CuPy's lack of this flag here.
#
# CMAKE_VISIBILITY_INLINES_HIDDEN is the half that removes those weak inline copies -- it is not
# redundant with the line above it. pybind11Config.cmake prescribes exactly this pair.
#
# pybind11 also forces hidden on its own types (PYBIND11_NAMESPACE carries
# visibility("hidden")), so a default-visibility class holding a pybind type -- NPUArray::dtype is
# one -- additionally trips -Wattributes in every TU. That warning is the symptom pybind11's own
# CMake comment names ("mixed hidden/non-hidden types"); it is evidence here, not the reason.
# pybind11_add_module() sets visibility on the _core target only, never on the csrc/ OBJECT libs
# that feed it, so a global set() is the only thing that reaches them.
#
# FOOTGUN: pybind11's guard is `if(NOT DEFINED CMAKE_CXX_VISIBILITY_PRESET)` -- an EXISTENCE check.
# Leaving this unset is safe (pybind11 forces hidden on _core itself); setting it to `hidden` is the
# supported override. But setting it to any OTHER value, `default` included, silently disables
# pybind11's fallback and leaves _core exporting everything -- worse than having no line at all.
#
# Size gain is real but secondary (-7%), and almost entirely dynamic-linking metadata: .dynstr -81%,
# .text only -0.4%. There is no measurable link-time gain. Do not justify these lines on size.
#
# Must precede the add_subdirectory() calls below: the target property is initialized from this
# variable at target-creation time.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# FetchContent module (for fmt, spdlog, etc.)
include(FetchContent)

# ========== Python with NumPy ==========
# Find Python BEFORE pybind11. With PYBIND11_FINDPYTHON=ON, pybind11 runs its own find_package(Python);
# resolving it first here means that call reuses this result instead of re-detecting, which is what
# produced the second, duplicate "Found Python" line at configure time.
#
# Request Development.Module, NOT the full Development. `Development` pulls in Development.Embed, whose
# artifacts (libpython) exist only to embed the interpreter in a host program -- something asnumpy never
# does. The build consumes exactly two Python things: Python::NumPy and Python_NumPy_INCLUDE_DIRS
# (bindings/python/CMakeLists.txt), plus Python::Module via pybind11_add_module; none needs Embed, and
# readelf confirms _core.so has no libpython in NEEDED. Requesting Development.Embed makes configure
# fail on interpreters that ship no libpython (manylinux wheel images, python-build-standalone) for a
# library the module never links -- a real hazard for the v0.4.0 wheel release, not a hypothetical.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)

# ========== pybind11 (from pip, scikit-build-core auto-configures CMAKE_PREFIX_PATH) ==========
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# ========== CANN SDK (environment variable lookup) ==========
if(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(ASCEND_CANN_PATH "$ENV{ASCEND_TOOLKIT_HOME}")
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PATH "$ENV{ASCEND_HOME_PATH}")
else()
    set(ASCEND_CANN_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    message(WARNING "CANN SDK path not set. Using default: ${ASCEND_CANN_PATH}. "
                    "Set ASCEND_TOOLKIT_HOME or ASCEND_HOME_PATH environment variable.")
endif()
message(STATUS "CANN SDK: ${ASCEND_CANN_PATH}")

include_directories(${ASCEND_CANN_PATH}/include)
link_directories(${ASCEND_CANN_PATH}/lib64)

# NumPy include
include_directories(${Python_NumPy_INCLUDE_DIRS})

# ========== fmt (system package only) ==========
find_package(fmt REQUIRED PATHS /usr/lib64/cmake /usr/lib/cmake /usr/lib/aarch64-linux-gnu/cmake NO_DEFAULT_PATH)
message(STATUS "Using system fmt: ${fmt_VERSION}")

# ========== spdlog (system package only) ==========
find_package(spdlog REQUIRED PATHS /usr/lib64/cmake /usr/lib/cmake /usr/lib/aarch64-linux-gnu/cmake NO_DEFAULT_PATH)
message(STATUS "Using system spdlog: ${spdlog_VERSION}")
add_compile_definitions(SPDLOG_FMT_EXTERNAL)

# ========== ops-fft (optional FFT backend) ==========
# asnumpy.fft is an optional feature: it is only built when ASNUMPY_ENABLE_FFT=ON *and* an existing
# ops-fft installation is pointed at through ASNUMPY_OPS_FFT_ROOT (or the finer-grained
# include/library variables below). The dependency is never downloaded or hardcoded: ops-fft is a
# separate CANN operator package that must be built for the target SoC first, see
# https://gitcode.com/cann/ops-fft. Without it the rest of asnumpy builds and imports unchanged and
# asnumpy.fft raises a RuntimeError explaining how to enable it.
option(ASNUMPY_ENABLE_FFT "Build asnumpy.fft on top of a CANN ops-fft installation" OFF)
set(ASNUMPY_OPS_FFT_ROOT "$ENV{ASNUMPY_OPS_FFT_ROOT}" CACHE PATH
    "ops-fft install prefix, containing include/cann_ops_fft.h and lib64/libcann_ops_fft.so")
# ASNUMPY_OPS_FFT_INCLUDE_DIR (directory holding cann_ops_fft.h) and ASNUMPY_OPS_FFT_LIBRARY (full
# path to libcann_ops_fft.so) may be set directly to override the search below. They are deliberately
# NOT pre-declared here: find_path/find_library skip their search when the result variable is already
# defined -- even to an empty string -- so an empty cache entry would silently disable the lookup.

if(ASNUMPY_ENABLE_FFT)
    # ops-fft installs in the CANN layout (a private --install-path prefix yields
    # <prefix>/cann/<arch>-linux/{include/math_libs,lib64}), while a bare ops-fft checkout keeps
    # include/ and build/ at its root; accept both without guessing anywhere else on the system.
    set(_asnumpy_ops_fft_hints
        "${ASNUMPY_OPS_FFT_ROOT}"
        "${ASNUMPY_OPS_FFT_ROOT}/cann"
        "${ASNUMPY_OPS_FFT_ROOT}/cann/aarch64-linux"
        "${ASNUMPY_OPS_FFT_ROOT}/aarch64-linux")
    if(NOT ASNUMPY_OPS_FFT_INCLUDE_DIR AND ASNUMPY_OPS_FFT_ROOT)
        find_path(ASNUMPY_OPS_FFT_INCLUDE_DIR cann_ops_fft.h
                  PATHS ${_asnumpy_ops_fft_hints}
                  PATH_SUFFIXES include include/math_libs
                  NO_DEFAULT_PATH)
    endif()
    if(NOT ASNUMPY_OPS_FFT_LIBRARY AND ASNUMPY_OPS_FFT_ROOT)
        find_library(ASNUMPY_OPS_FFT_LIBRARY cann_ops_fft
                     PATHS ${_asnumpy_ops_fft_hints}
                     PATH_SUFFIXES lib64 lib build
                     NO_DEFAULT_PATH)
    endif()
    if(NOT ASNUMPY_OPS_FFT_INCLUDE_DIR OR NOT ASNUMPY_OPS_FFT_LIBRARY)
        message(FATAL_ERROR
            "ASNUMPY_ENABLE_FFT=ON, but ops-fft was not found (include='${ASNUMPY_OPS_FFT_INCLUDE_DIR}', "
            "library='${ASNUMPY_OPS_FFT_LIBRARY}').\n"
            "Build and install ops-fft for the target SoC first:\n"
            "  git clone https://gitcode.com/cann/ops-fft.git && cd ops-fft\n"
            "  bash build.sh --pkg --soc=Ascend910B\n"
            "  ./build_out/cann-910b-ops-fft_*.run --full --quiet --install-path=<prefix>\n"
            "then configure with -DASNUMPY_ENABLE_FFT=ON -DASNUMPY_OPS_FFT_ROOT=<prefix>, or point "
            "-DASNUMPY_OPS_FFT_INCLUDE_DIR=<dir> and -DASNUMPY_OPS_FFT_LIBRARY=<path> at an existing build.")
    endif()
    message(STATUS "ops-fft enabled: ${ASNUMPY_OPS_FFT_LIBRARY} (include: ${ASNUMPY_OPS_FFT_INCLUDE_DIR})")
else()
    message(STATUS "ops-fft disabled: asnumpy.fft will report that the backend is unavailable")
endif()

# ========== Source directories ==========
include_directories(include)
add_subdirectory(csrc)
add_subdirectory(bindings/python)

if(SKBUILD)
    message(STATUS "Building with scikit-build-core")
endif()
