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

echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
#########
# install #
#########

sudo mkdir -p /usr/local/Ascend/ascend-toolkit
sudo ln -s /home/jenkins/Ascend/ascend-toolkit/latest /usr/local/Ascend/ascend-toolkit/latest
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash 2>&1 || true

sudo apt-get update
sudo apt-get install ccache -y
python3.10 -m pip install build

sudo mkdir -p /usr/local/ccache/bin
sudo ln -sf /usr/bin/ccache /usr/local/ccache/bin/ccache

#########
# Build #
#########
# Build

cd ${WORKSPACE} || exit
pwd
case $(uname -m) in
    x86_64) LIB_ARCH="x86_64-linux-gnu" ;;
    aarch64|arm64) LIB_ARCH="aarch64-linux-gnu" ;;
    *) echo "Unsupported arch: $(uname -m)"; exit 1 ;;
esac

sudo mkdir -p /usr/lib/cmake /usr/lib64/cmake

if [ -d "/usr/lib/${LIB_ARCH}/cmake/fmt" ]; then
    sudo ln -sf "/usr/lib/${LIB_ARCH}/cmake/fmt" /usr/lib/cmake/fmt
    sudo ln -sf "/usr/lib/${LIB_ARCH}/cmake/fmt" /usr/lib64/cmake/fmt
fi

if [ -d "/usr/lib/${LIB_ARCH}/cmake/spdlog" ]; then
    sudo ln -sf "/usr/lib/${LIB_ARCH}/cmake/spdlog" /usr/lib/cmake/spdlog
    sudo ln -sf "/usr/lib/${LIB_ARCH}/cmake/spdlog" /usr/lib64/cmake/spdlog
elif [ -d "/usr/share/cmake/spdlog" ]; then
    sudo ln -sf /usr/share/cmake/spdlog /usr/lib/cmake/spdlog
    sudo ln -sf /usr/share/cmake/spdlog /usr/lib64/cmake/spdlog
fi

export fmt_DIR="/usr/lib/${LIB_ARCH}/cmake/fmt"
export CMAKE_PREFIX_PATH="${fmt_DIR}:${CMAKE_PREFIX_PATH}"

python3.10 -m pip install -e .

python3.10 -m build
ret=$?

DP_ASSERT_EQUAL "$ret" "0" "build asnumpy ${task_name}"
