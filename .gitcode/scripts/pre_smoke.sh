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

log() {
  local dt
  dt=$(date '+%Y%m%d.%H%M%S')
  echo "===================================================================="
  echo "$dt : $*"
  echo "===================================================================="
}

log "init test case, please wait ..."

PYTHON_BIN="${PYTHON_BIN:?The workflow must select Python}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://repo.huaweicloud.com/repository/pypi/simple}"

apt-get update
apt-get install -y wget unzip
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install pytest pytest-xdist

package_name="${package_name:-cann-asnumpy-linux-arm.whl}"
wget "https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${obs_path}/${package_name}"
ls

# pip requires the canonical wheel filename; recover it from the wheel metadata
# (version + full ABI/platform tag) so this works for any Python/arch combination.
VERSION=$(unzip -p "${package_name}" '*.dist-info/METADATA' | grep -m1 "^Version:" | awk '{print $2}' | tr -d '\r')
TAG=$(unzip -p "${package_name}" '*.dist-info/WHEEL' | grep -m1 "^Tag:" | awk '{print $2}' | tr -d '\r')
echo "版本号: ${VERSION}  标签: ${TAG}"
wheel_file="asnumpy-${VERSION}-${TAG}.whl"
mv "${package_name}" "${wheel_file}"
"${PYTHON_BIN}" -m pip install "${wheel_file}"

set +e
"${PYTHON_BIN}" -m pytest tests/ -n 8 -v 2>&1 | tee -a ./run_test.log
pytest_exit_code=${PIPESTATUS[0]}
set -e

echo "pip3 install obs"
"${PYTHON_BIN}" -m pip install esdk-obs-python \
    --trusted-host repo.huaweicloud.com \
    -i https://repo.huaweicloud.com/repository/pypi/simple

# ==============================
# 打包log
# ==============================
mkdir -p /root/ascend/log
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
