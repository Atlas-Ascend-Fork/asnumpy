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


from ._core.statistics import mean as _mean
from ._types import ArrayLike, AxisLike, DTypeLike
from .utils import _convert_dtype, ndarray


def mean(
    a: ArrayLike,
    axis: AxisLike = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> ndarray | float:
    if axis is None:
        converted_dtype = _convert_dtype(dtype)
        if keepdims and getattr(a, "ndim", 0) > 0:
            return ndarray(_mean(a, keepdims, converted_dtype))
        return _mean(a, converted_dtype)  # type: ignore[no-any-return]
    return ndarray(_mean(a, axis, keepdims, _convert_dtype(dtype)))
