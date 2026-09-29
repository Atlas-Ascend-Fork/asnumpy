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


################################ TOOLCHAIN #####################################
apt-get update
apt-get install -y build-essential ccache cmake git libfmt-dev libspdlog-dev ninja-build
mkdir -p /usr/local/ccache/bin
ln -sf /usr/bin/ccache /usr/local/ccache/bin/ccache

export PIP_INDEX_URL="${PIP_INDEX_URL:-https://repo.huaweicloud.com/repository/pypi/simple}"

################################## BUILD #######################################

cd ${WORKSPACE} || exit
pwd
case $(uname -m) in
    x86_64) LIB_ARCH="x86_64-linux-gnu" ;;
    aarch64|arm64) LIB_ARCH="aarch64-linux-gnu" ;;
    *) echo "Unsupported arch: $(uname -m)"; exit 1 ;;
esac

mkdir -p /usr/lib/cmake /usr/lib64/cmake

if [ -d "/usr/lib/${LIB_ARCH}/cmake/fmt" ]; then
    ln -sf "/usr/lib/${LIB_ARCH}/cmake/fmt" /usr/lib/cmake/fmt
    ln -sf "/usr/lib/${LIB_ARCH}/cmake/fmt" /usr/lib64/cmake/fmt
fi

if [ -d "/usr/lib/${LIB_ARCH}/cmake/spdlog" ]; then
    ln -sf "/usr/lib/${LIB_ARCH}/cmake/spdlog" /usr/lib/cmake/spdlog
    ln -sf "/usr/lib/${LIB_ARCH}/cmake/spdlog" /usr/lib64/cmake/spdlog
elif [ -d "/usr/share/cmake/spdlog" ]; then
    ln -sf /usr/share/cmake/spdlog /usr/lib/cmake/spdlog
    ln -sf /usr/share/cmake/spdlog /usr/lib64/cmake/spdlog
fi

export fmt_DIR="/usr/lib/${LIB_ARCH}/cmake/fmt"
export CMAKE_PREFIX_PATH="${fmt_DIR}:${CMAKE_PREFIX_PATH}"


PYTHON_BIN="${PYTHON_BIN:-python3}"
"${PYTHON_BIN}" --version

if ! "${PYTHON_BIN}" -m build --version >/dev/null 2>&1; then
    "${PYTHON_BIN}" -m pip install build
fi

set +e
"${PYTHON_BIN}" -m build --outdir "${WORKSPACE}/dist"
ret=$?
set -e

DP_ASSERT_EQUAL "$ret" "0" "build asnumpy ${task_name} with ${PYTHON_BIN}"
ls -l "${WORKSPACE}/dist"
