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

"""全局归约 keepdims 行为测试

覆盖 issue #118：sum/prod/nansum/nanprod 与 max/min/amax/amin/nanmax/nanmin
在 axis=None 时应保留 keepdims 语义（每个输入轴保留一个 singleton 维度），
sum/prod 系同时向原生层透传 dtype。
"""

import numpy

from asnumpy import testing


def _create_array(xp, data, dtype):
    np_arr = numpy.array(data, dtype=dtype)
    if xp is numpy:
        return np_arr
    return xp.ndarray.from_numpy(np_arr)


# ========== sum / prod / nansum / nanprod ==========


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_sum_global_keepdims(xp, dtype):
    """Global sum retains one singleton dimension per input axis."""
    a = _create_array(xp, [[1, 2], [3, 4]], dtype)
    result = xp.sum(a, keepdims=True)
    assert result.shape == (1, 1)
    return result


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_prod_global_keepdims(xp, dtype):
    """Global prod retains one singleton dimension per input axis."""
    a = _create_array(xp, [[1, 2], [3, 4]], dtype)
    result = xp.prod(a, keepdims=True)
    assert result.shape == (1, 1)
    return result


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_nansum_global_keepdims(xp, dtype):
    """Global nansum retains one singleton dimension per input axis."""
    a = _create_array(xp, [[1.0, numpy.nan], [3.0, 4.0]], dtype)
    result = xp.nansum(a, keepdims=True)
    assert result.shape == (1, 1)
    return result


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_nanprod_global_keepdims(xp, dtype):
    """Global nanprod retains one singleton dimension per input axis."""
    a = _create_array(xp, [[1.0, numpy.nan], [3.0, 4.0]], dtype)
    result = xp.nanprod(a, keepdims=True)
    assert result.shape == (1, 1)
    return result


@testing.for_dtypes([numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_sum_global_dtype(xp, dtype):
    """Global sum honours the requested accumulator dtype when axis=None."""
    a = _create_array(xp, [[1, 2], [3, 4]], dtype)
    result = xp.sum(a, dtype=numpy.float32)
    assert numpy.asarray(result).dtype == numpy.float32
    return result


# ========== max / min / amax / amin / nanmax / nanmin ==========


@testing.for_all_dtypes(no_complex=True)
@testing.numpy_asnumpy_array_equal()
def test_max_global_keepdims(xp, dtype):
    """Global max retains one singleton dimension per input axis."""
    a = _create_array(xp, [[1, 5], [3, 2]], dtype)
    result = xp.max(a, keepdims=True)
    assert result.shape == (1, 1)
    return result


@testing.for_all_dtypes(no_complex=True)
@testing.numpy_asnumpy_array_equal()
def test_min_global_keepdims(xp, dtype):
    """Global min retains one singleton dimension per input axis."""
    a = _create_array(xp, [[1, 5], [3, 2]], dtype)
    result = xp.min(a, keepdims=True)
    assert result.shape == (1, 1)
    return result


@testing.for_all_dtypes(no_complex=True)
@testing.numpy_asnumpy_array_equal()
def test_amax_global_keepdims(xp, dtype):
    """Global amax retains one singleton dimension per input axis."""
    a = _create_array(xp, [[1, 5], [3, 2]], dtype)
    result = xp.amax(a, keepdims=True)
    assert result.shape == (1, 1)
    return result


@testing.for_all_dtypes(no_complex=True)
@testing.numpy_asnumpy_array_equal()
def test_amin_global_keepdims(xp, dtype):
    """Global amin retains one singleton dimension per input axis."""
    a = _create_array(xp, [[1, 5], [3, 2]], dtype)
    result = xp.amin(a, keepdims=True)
    assert result.shape == (1, 1)
    return result


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_nanmax_global_keepdims(xp, dtype):
    """Global nanmax retains one singleton dimension per input axis."""
    a = _create_array(xp, [[1.0, numpy.nan], [3.0, 2.0]], dtype)
    result = xp.nanmax(a, keepdims=True)
    assert result.shape == (1, 1)
    return result


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_array_equal()
def test_nanmin_global_keepdims(xp, dtype):
    """Global nanmin retains one singleton dimension per input axis."""
    a = _create_array(xp, [[1.0, numpy.nan], [3.0, 2.0]], dtype)
    result = xp.nanmin(a, keepdims=True)
    assert result.shape == (1, 1)
    return result
