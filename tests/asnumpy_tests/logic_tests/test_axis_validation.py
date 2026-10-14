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

"""all/any 的 axis 校验与语义测试

覆盖（upstream issue #40）：
1. 越界 / 重复 axis → ValueError
2. 负 axis 按 NumPy 语义从末维归一化
3. int 形式 axis 可用
4. 多轴 keepdims 两种取值的结果正确性对照 NumPy
"""

import numpy
import pytest

import asnumpy


def _from_numpy(values):
    return asnumpy.ndarray.from_numpy(numpy.asarray(values))


HOST = numpy.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])


@pytest.mark.parametrize("reduce", [asnumpy.all, asnumpy.any])
@pytest.mark.parametrize("axis", [5, [5], [-3], [0, 0]])
def test_rejects_invalid_axis(reduce, axis):
    # np.all 不接受 list 形式 axis，转 tuple 等价对照
    numpy_reduce = getattr(numpy, reduce.__name__)
    numpy_axis = tuple(axis) if isinstance(axis, list) else axis
    with pytest.raises((ValueError, IndexError)):
        numpy_reduce(HOST, axis=numpy_axis)

    with pytest.raises(ValueError):
        reduce(_from_numpy(HOST), axis=axis)


@pytest.mark.parametrize("reduce", [asnumpy.all, asnumpy.any])
def test_negative_axis_matches_numpy(reduce):
    x = _from_numpy(HOST)
    numpy_reduce = getattr(numpy, reduce.__name__)

    result = reduce(x, axis=-1)
    numpy.testing.assert_array_equal(result.to_numpy(), numpy_reduce(HOST, axis=-1))

    result_keepdims = reduce(x, axis=-1, keepdims=True)
    assert result_keepdims.shape == numpy_reduce(HOST, axis=-1, keepdims=True).shape


@pytest.mark.parametrize("reduce", [asnumpy.all, asnumpy.any])
def test_int_axis_matches_numpy(reduce):
    # int axis 是 NumPy 最常用形式
    x = _from_numpy(HOST)
    numpy_reduce = getattr(numpy, reduce.__name__)

    result = reduce(x, axis=0)
    numpy.testing.assert_array_equal(result.to_numpy(), numpy_reduce(HOST, axis=0))


@pytest.mark.parametrize("reduce", [asnumpy.all, asnumpy.any])
@pytest.mark.parametrize("keepdims", [False, True])
def test_multiple_axes_matches_numpy(reduce, keepdims):
    # 多轴（含负轴混合）验证 erase 降序删除的顺序正确性
    host = numpy.arange(24.0).reshape(2, 3, 4)
    x = _from_numpy(host)
    numpy_reduce = getattr(numpy, reduce.__name__)

    result = reduce(x, axis=[2, 0], keepdims=keepdims)
    expected = numpy_reduce(host, axis=(2, 0), keepdims=keepdims)
    assert result.shape == expected.shape
    numpy.testing.assert_array_equal(result.to_numpy(), expected)


def test_accepts_numpy_input():
    # 前端 ArrayLike 转换回归
    result = asnumpy.all(HOST)
    assert bool(result.to_numpy().item()) == bool(numpy.all(HOST))
