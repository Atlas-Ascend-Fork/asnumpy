# *****************************************************************************
# Copyright (c) 2025 AISS and ISE Group at Harbin Institute of Technology. All Rights Reserved.
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

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias, TypeVar, Union

import numpy as np

if TYPE_CHECKING:
    from .utils import ndarray  # noqa: F401

ArrayLike = Union[  # noqa: UP007
    "ndarray",  # NPUArray — string forward ref to avoid circular import
    np.ndarray,
    int,
    float,
    complex,
    bool,
    Sequence,
]

DTypeLike: TypeAlias = np.dtype | str | type | None

ShapeLike = int | Sequence[int]

AxisLike = int | Sequence[int] | None

AxisOptional = int | Sequence[int] | None

ScalarLike = int | float | complex | bool

T = TypeVar("T")


__all__ = [
    "ArrayLike",
    "DTypeLike",
    "ShapeLike",
    "AxisLike",
    "AxisOptional",
    "ScalarLike",
    "T",
]
