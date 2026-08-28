#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -e

DP_ASSERT_EQUAL()
{
    local actual="$1"
    local expected="$2"
    local msg="$3"
    if [ "${actual}" != "${expected}" ]; then
        echo "::error::ASSERT FAILED: ${msg} (expected=${expected}, actual=${actual})"
        exit 1
    fi
}

echo "OS VERSION_ID: $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)"

# ascendhub/cann images run as root and may not ship sudo; legacy ci_cann images
# run as a jenkins user and need it.
SUDO="sudo"
if [ "$(id -u)" = "0" ] && ! command -v sudo >/dev/null 2>&1; then
    SUDO=""
fi

#########
# CANN toolkit #
#########
# Ubuntu 22.04 + CANN 9.0.0 images bake the toolkit into /usr/local/Ascend.
# Legacy ci_cann images mount it from the host instead.
if [ ! -e /usr/local/Ascend/ascend-toolkit/latest ] && [ -e /home/jenkins/Ascend/ascend-toolkit/latest ]; then
    ${SUDO} mkdir -p /usr/local/Ascend/ascend-toolkit
    ${SUDO} ln -s /home/jenkins/Ascend/ascend-toolkit/latest /usr/local/Ascend/ascend-toolkit/latest
fi
if [ -f /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash ]; then
    source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash 2>&1 || true
elif [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>&1 || true
fi

#########
# toolchain #
#########
${SUDO} apt-get update
# The ascendhub/cann devel image ships the CANN toolchain but not these C++ build deps;
# on images that already have them this is a cheap no-op.
${SUDO} apt-get install -y build-essential ccache cmake git libfmt-dev libspdlog-dev ninja-build

${SUDO} mkdir -p /usr/local/ccache/bin
${SUDO} ln -sf /usr/bin/ccache /usr/local/ccache/bin/ccache

# Huawei Cloud PyPI mirror for the build-frontend install and build isolation below.
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://repo.huaweicloud.com/repository/pypi/simple}"

#########
# Build #
#########

cd ${WORKSPACE} || exit
pwd
case $(uname -m) in
    x86_64) LIB_ARCH="x86_64-linux-gnu" ;;
    aarch64|arm64) LIB_ARCH="aarch64-linux-gnu" ;;
    *) echo "Unsupported arch: $(uname -m)"; exit 1 ;;
esac

${SUDO} mkdir -p /usr/lib/cmake /usr/lib64/cmake

if [ -d "/usr/lib/${LIB_ARCH}/cmake/fmt" ]; then
    ${SUDO} ln -sf "/usr/lib/${LIB_ARCH}/cmake/fmt" /usr/lib/cmake/fmt
    ${SUDO} ln -sf "/usr/lib/${LIB_ARCH}/cmake/fmt" /usr/lib64/cmake/fmt
fi

if [ -d "/usr/lib/${LIB_ARCH}/cmake/spdlog" ]; then
    ${SUDO} ln -sf "/usr/lib/${LIB_ARCH}/cmake/spdlog" /usr/lib/cmake/spdlog
    ${SUDO} ln -sf "/usr/lib/${LIB_ARCH}/cmake/spdlog" /usr/lib64/cmake/spdlog
elif [ -d "/usr/share/cmake/spdlog" ]; then
    ${SUDO} ln -sf /usr/share/cmake/spdlog /usr/lib/cmake/spdlog
    ${SUDO} ln -sf /usr/share/cmake/spdlog /usr/lib64/cmake/spdlog
fi

export fmt_DIR="/usr/lib/${LIB_ARCH}/cmake/fmt"
export CMAKE_PREFIX_PATH="${fmt_DIR}:${CMAKE_PREFIX_PATH}"

# Build with the image's own interpreter (python3 = 3.11 on the ascendhub/cann
# 9.0.0-910b-ubuntu22.04-py3.11 image). pybind11 extensions are CPython-ABI specific,
# so the wheel inherits this interpreter's tag (cp311 there).
PYTHON_BIN="${PYTHON_BIN:-python3}"
"${PYTHON_BIN}" --version

if ! "${PYTHON_BIN}" -m build --version >/dev/null 2>&1; then
    "${PYTHON_BIN}" -m pip install build
fi

# sdist + wheel; the wheel is built from the sdist, doubling as an sdist-completeness check.
set +e
"${PYTHON_BIN}" -m build --outdir "${WORKSPACE}/dist"
ret=$?
set -e

DP_ASSERT_EQUAL "$ret" "0" "build asnumpy ${task_name} with ${PYTHON_BIN}"
ls -l "${WORKSPACE}/dist"
