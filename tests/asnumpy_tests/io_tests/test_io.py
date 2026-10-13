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

"""I/O 模块测试

包含：
1. 单数组保存/加载: save, load
2. 多数组归档: savez, savez_compressed
3. NPUArray 自动转换

优化维度：
- save/load 往返一致性
- .npy/.npz/.压缩 .npz 格式兼容性
- NPUArray 输入自动转 numpy 存储
- npz 懒加载容器行为
"""

import io

import numpy

from asnumpy import testing


# ========== 辅助函数 ==========
def _create_array(xp, data, dtype):
    """辅助函数：创建数组"""
    np_arr = numpy.array(data, dtype=dtype)
    if xp is numpy:
        return np_arr
    return xp.ndarray.from_numpy(np_arr)


# ==========================================================================
# 1. 单数组保存/加载测试
# ==========================================================================


@testing.for_dtypes([numpy.float32, numpy.float64])
def test_save_load_roundtrip_file_object(dtype):
    """测试 save/load 往返一致性: 文件对象"""
    import asnumpy as ap

    data = numpy.array([[1.5, -2.0, 3.25], [4.5, 0.0, -6.75]], dtype=dtype)

    buffer = io.BytesIO()
    ap.save(buffer, ap.ndarray.from_numpy(data))
    buffer.seek(0)

    loaded = ap.load(buffer)
    numpy.testing.assert_allclose(loaded.to_numpy(), data)


@testing.for_dtypes([numpy.int32, numpy.float32])
def test_load_npy_returns_npu_array(dtype):
    """测试 load .npy 返回 NPUArray"""
    import asnumpy as ap

    data = numpy.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
    buffer = io.BytesIO()
    numpy.save(buffer, data)
    buffer.seek(0)

    loaded = ap.load(buffer)
    assert isinstance(loaded, ap.ndarray)
    numpy.testing.assert_array_equal(loaded.to_numpy(), data)


# ==========================================================================
# 2. NPUArray 自动转换测试
# ==========================================================================


@testing.for_dtypes([numpy.float32, numpy.int64])
def test_save_auto_converts_npu_array(dtype):
    """测试 save 自动将 NPUArray 转为 numpy 后存储"""
    import asnumpy as ap

    data = numpy.array([[1, 2], [3, 4]], dtype=dtype)
    arr = ap.ndarray.from_numpy(data)
    buffer = io.BytesIO()

    ap.save(buffer, arr)
    buffer.seek(0)

    reloaded = numpy.load(buffer)
    numpy.testing.assert_array_equal(reloaded, data)


@testing.for_dtypes([numpy.float32])
def test_savez_auto_converts_npu_array(dtype):
    """测试 savez 自动转换位置参数和关键字参数中的 NPUArray"""
    import asnumpy as ap

    first = ap.ndarray.from_numpy(numpy.array([1.0, 2.0, 3.0], dtype=dtype))
    second = ap.ndarray.from_numpy(numpy.array([[4.0, 5.0], [6.0, 7.0]], dtype=dtype))

    buffer = io.BytesIO()
    ap.savez(buffer, first, named=second)
    buffer.seek(0)

    with numpy.load(buffer) as loaded:
        numpy.testing.assert_array_equal(loaded["arr_0"], first.to_numpy())
        numpy.testing.assert_array_equal(loaded["named"], second.to_numpy())


# ==========================================================================
# 3. .npz 归档与压缩格式兼容性测试
# ==========================================================================


@testing.for_dtypes([numpy.float32, numpy.float64])
def test_load_npz_roundtrip_returns_lazy_npu_arrays(dtype):
    """测试 load .npz 后可懒加载并返回 NPUArray"""
    import asnumpy as ap

    left = numpy.array([1.0, 2.0, 3.0], dtype=dtype)
    right = numpy.array([[4.0, 5.0], [6.0, 7.0]], dtype=dtype)

    buffer = io.BytesIO()
    ap.savez(buffer, left=left, right=right)
    buffer.seek(0)

    with ap.load(buffer) as loaded:
        assert loaded.files == ["left", "right"]
        left_arr = loaded["left"]
        right_arr = loaded["right"]
        assert isinstance(left_arr, ap.ndarray)
        assert isinstance(right_arr, ap.ndarray)
        numpy.testing.assert_allclose(left_arr.to_numpy(), left)
        numpy.testing.assert_allclose(right_arr.to_numpy(), right)


@testing.for_dtypes([numpy.float32])
def test_load_npz_cached_item(dtype):
    """测试 npz 同一 key 重复访问命中缓存"""
    import asnumpy as ap

    data = numpy.array([1.0, 2.0, 3.0], dtype=dtype)
    buffer = io.BytesIO()
    ap.savez(buffer, values=data)
    buffer.seek(0)

    with ap.load(buffer) as loaded:
        first = loaded["values"]
        second = loaded["values"]
        assert first is second
        numpy.testing.assert_allclose(first.to_numpy(), data)


@testing.for_dtypes([numpy.int32, numpy.float32])
def test_savez_compressed_compatible_with_numpy_load(dtype):
    """测试压缩 npz 格式与 numpy.load 兼容"""
    import asnumpy as ap

    data = numpy.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
    extra = numpy.array([7, 8, 9], dtype=dtype)
    buffer = io.BytesIO()

    ap.savez_compressed(buffer, data=data, extra=ap.ndarray.from_numpy(extra))
    buffer.seek(0)

    with numpy.load(buffer) as loaded:
        numpy.testing.assert_array_equal(loaded["data"], data)
        numpy.testing.assert_array_equal(loaded["extra"], extra)


@testing.for_dtypes([numpy.float32])
def test_load_numpy_compressed_npz_returns_npu_array(dtype):
    """测试读取 numpy 生成的压缩 npz 仍返回 NPUArray"""
    import asnumpy as ap

    first = numpy.array([1.0, 2.0, 3.0], dtype=dtype)
    second = numpy.array([[4.0], [5.0]], dtype=dtype)
    buffer = io.BytesIO()

    numpy.savez_compressed(buffer, first=first, second=second)
    buffer.seek(0)

    with ap.load(buffer) as loaded:
        numpy.testing.assert_allclose(loaded["first"].to_numpy(), first)
        numpy.testing.assert_allclose(loaded["second"].to_numpy(), second)
