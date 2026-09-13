# *****************************************************************************
# Copyright (c) 2025 ISE Group at Harbin Institute of Technology. All Rights Reserved.
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

from loguru import logger

from ._core.cann import (
    finalize as _finalize,
)
from ._core.cann import (
    get_device_count as _get_device_count,
)
from ._core.cann import (
    init as _init,
)
from ._core.cann import (
    reset_device as _reset_device,
)
from ._core.cann import (
    reset_device_force as _reset_device_force,
)
from ._core.cann import (
    set_device as _set_device,
)


@logger.catch
def set_device(device_id: int) -> None:
    logger.info(f"Setting device to {device_id}")
    _set_device(device_id)


@logger.catch
def reset_device(device_id: int) -> None:
    logger.info(f"Resetting device {device_id}")
    _reset_device(device_id)


@logger.catch
def reset_device_force(device_id: int) -> None:
    logger.info(f"Force resetting device {device_id}")
    _reset_device_force(device_id)


@logger.catch
def get_device_count() -> int:
    logger.info("Querying NPU device count")
    return _get_device_count()


@logger.catch
def init() -> None:
    logger.info("Initializing CANN backend")
    _init()


@logger.catch
def finalize() -> None:
    logger.info("Finalizing CANN backend")
    _finalize()
