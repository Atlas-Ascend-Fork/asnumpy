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

set -Eeuo pipefail

echo "start run test case, please wait ..."
cd ${WORKSPACE}

export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_SLOG_PRINT_TO_STDOUT=0
source /usr/local/Ascend/cann/set_env.sh

log() {
  local dt
  dt=$(date '+%Y%m%d.%H%M%S')
  echo "===================================================================="
  echo "$dt : $*"
  echo "===================================================================="
}

log "init test case, please wait ..."


source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash;
export PATH=/usr/bin:/usr/local/bin:$PATH
export PATH=/usr/local/python/python310/bin:$PATH

sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    dpkg-dev \
    git \
    libfmt-dev \
    libspdlog-dev \
    ninja-build

ARCH="$(dpkg-architecture -qDEB_HOST_MULTIARCH)"
export fmt_DIR="/usr/lib/${ARCH}/cmake/fmt"
export spdlog_DIR="/usr/lib/${ARCH}/cmake/spdlog"

# Keep build isolation enabled so pip honors [build-system].requires.
python3.10 -m pip install --upgrade pip
# python3.10 -m pip install -e ".[test]"
wget https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${obs_path}/cann-asnumpy-linux-arm.whl
ls
VERSION=$(unzip -p cann-asnumpy-linux-arm.whl *.dist-info/METADATA | grep "^Version:" | head -1 | cut -d' ' -f2 | tr -d '\r')
echo "版本号: $VERSION"
ORIGINAL_NAME="asnumpy-${VERSION}-cp310-cp310-linux_aarch64.whl"
echo "原文件名: $ORIGINAL_NAME"
mv cann-asnumpy-linux-arm.whl "$ORIGINAL_NAME"
# python3.10 -m pip install cann-asnumpy-linux-arm.whl
python3.10 -m pip install asnumpy*.whl

set +e
python3.10 -m pytest tests/ -n 8 -v 2>&1 | tee -a ./run_test.log
pytest_exit_code=${PIPESTATUS[0]}
set -e

echo "pip3 install obs"
python3.10 -m pip install esdk-obs-python \
    --trusted-host repo.huaweicloud.com \
    -i https://repo.huaweicloud.com/repository/pypi/simple

# ==============================
# 打包log
# ==============================
mkdir -p /root/ascend
slog_name="slog.tar.gz"
tar -zcf "${slog_name}" -C /root/ascend log

# ==============================
# 检查测试结果
# ==============================
log "checking test results ..."

date_time=`date +%Y%m%d`"."`date +%H%M%S`
if grep -w -e "passed" "./run_test.log"; then
  echo "$date_time : run test case success"
else
  echo "$date_time : run test case failed"
  exit 1
fi
