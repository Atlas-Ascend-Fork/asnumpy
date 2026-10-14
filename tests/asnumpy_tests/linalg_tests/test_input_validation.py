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

"""linalg 族非法输入校验测试

覆盖（对照 NumPy 报错行为，upstream issues #36-#38 #47 #76）：
1. 矩阵算子低维输入（qr/det/slogdet/inv）→ ValueError
2. einsum 操作数数量、下标白名单、维度匹配校验
3. dot/vdot 尺寸不匹配 → ValueError
4. 前端 ArrayLike 转换回归
"""

import numpy
import pytest

import asnumpy


def _from_numpy(values, dtype=numpy.float32):
    return asnumpy.ndarray.from_numpy(numpy.asarray(values, dtype=dtype))


# --- 低维输入：矩阵算子需要 >= 2 维 ---


@pytest.mark.parametrize(
    "call",
    [
        lambda a: asnumpy.linalg.qr(a),
        lambda a: asnumpy.linalg.det(a),
        lambda a: asnumpy.linalg.slogdet(a),
        lambda a: asnumpy.linalg.inv(a),
    ],
)
def test_matrix_ops_reject_1d_input(call):
    host = numpy.arange(4.0, dtype=numpy.float32)

    # NumPy 同样拒绝 1 维输入
    with pytest.raises(numpy.linalg.LinAlgError):
        numpy.linalg.det(host)

    with pytest.raises(ValueError, match="at least 2 dimensional"):
        call(_from_numpy(host))


def test_matrix_ops_reject_0d_input():
    scalar = _from_numpy(3.0)

    with pytest.raises(ValueError, match="at least 2 dimensional"):
        asnumpy.linalg.det(scalar)


# --- einsum 校验 ---


def test_einsum_rejects_single_operand():
    a = _from_numpy([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="exactly 2 operands"):
        asnumpy.einsum("i->i", a)


def test_einsum_rejects_unsupported_subscripts():
    a = _from_numpy([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="not supported"):
        asnumpy.einsum("i,i->", a, a)


def test_einsum_rejects_rank_mismatch():
    a2d = _from_numpy(numpy.ones((2, 3)))

    with pytest.raises(ValueError, match="two 1-D operands"):
        asnumpy.einsum("a,b->ab", a2d, a2d)


def test_einsum_outer_supported_path():
    host_a = numpy.arange(1.0, 4.0, dtype=numpy.float32)
    host_b = numpy.arange(1.0, 3.0, dtype=numpy.float32)

    result = asnumpy.einsum("a,b->ab", _from_numpy(host_a), _from_numpy(host_b))

    numpy.testing.assert_allclose(result.to_numpy(), numpy.outer(host_a, host_b), rtol=1e-5)


# --- dot/vdot 尺寸校验（float32 保证走 NPU 路径而非 fp64 host fallback） ---


def test_dot_rejects_misaligned_1d():
    a = _from_numpy([1.0, 2.0, 3.0])
    b = _from_numpy([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="not aligned"):
        asnumpy.dot(a, b)


def test_dot_rejects_misaligned_2d():
    a = _from_numpy(numpy.ones((2, 3)))
    b = _from_numpy(numpy.ones((4, 5)))

    with pytest.raises(ValueError, match="not aligned"):
        asnumpy.dot(a, b)


def test_vdot_rejects_size_mismatch():
    a = _from_numpy(numpy.ones(8))
    b = _from_numpy(numpy.ones(2))

    with pytest.raises(ValueError, match="equal size"):
        asnumpy.vdot(a, b)


# --- 前端 ArrayLike 转换回归 ---


def test_linalg_accepts_numpy_input():
    host = numpy.array([[2.0, 0.0], [0.0, 3.0]], dtype=numpy.float32)

    det = asnumpy.linalg.det(host)
    numpy.testing.assert_allclose(det.to_numpy(), 6.0, rtol=1e-5)

    sign, logdet = asnumpy.linalg.slogdet(host)
    assert sign.to_numpy().item() == 1.0

    q, r = asnumpy.linalg.qr(host)
    assert q.shape == (2, 2) and r.shape == (2, 2)
