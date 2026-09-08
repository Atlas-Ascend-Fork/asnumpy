"""Regression tests for Issue #37: reject invalid matrix shapes before native work."""

import numpy as np
import pytest

import asnumpy as ap
from asnumpy import _core

INVALID_SHAPES = [(), (3,), (0,), (2, 3), (3, 2), (4, 2, 3)]


@pytest.mark.parametrize("name", ["det", "slogdet", "inv"])
@pytest.mark.parametrize("shape", INVALID_SHAPES)
@pytest.mark.parametrize("input_kind", ["npu", "numpy", "list"])
def test_public_invalid_matrix_shape(name, shape, input_kind):
    host = np.ones(shape, dtype=np.float32)
    if input_kind == "npu":
        value = ap.ndarray.from_numpy(host)
    elif input_kind == "list":
        value = host.tolist()
    else:
        value = host

    with pytest.raises(np.linalg.LinAlgError):
        getattr(np.linalg, name)(host)
    message = "at least two dimensions" if len(shape) < 2 else "must be square"
    with pytest.raises(np.linalg.LinAlgError, match=message):
        getattr(ap.linalg, name)(value)


@pytest.mark.parametrize("name", ["det", "slogdet", "inv"])
@pytest.mark.parametrize("shape", INVALID_SHAPES)
def test_native_invalid_matrix_shape(name, shape):
    # Bypass the Python guard, including inv's C++ call into Linalg_Slogdet.
    value = ap.ndarray.from_numpy(np.ones(shape, dtype=np.float32))
    message = "at least two dimensions" if len(shape) < 2 else "must be square"
    with pytest.raises(ValueError, match=message):
        getattr(_core.linalg, name)(value)


@pytest.mark.parametrize("name", ["det", "slogdet", "inv"])
def test_invalid_shape_checked_before_host_transfer(name):
    class InvalidDeviceArray:
        shape = (3,)

        def to_numpy(self):
            pytest.fail("Invalid matrix shape must be rejected before host transfer")

    with pytest.raises(np.linalg.LinAlgError, match="at least two dimensions"):
        getattr(ap.linalg, name)(InvalidDeviceArray())


@pytest.mark.parametrize("shape", [(), (3,), (2, 3)])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_slogdet_invalid_shape_with_nonfinite_values(shape, value):
    array = ap.ndarray.from_numpy(np.full(shape, value, dtype=np.float32))
    with pytest.raises(np.linalg.LinAlgError):
        ap.linalg.slogdet(array)
