import numpy as np
import pytest

import asnumpy as ap


@pytest.fixture(scope="module", autouse=True)
def npu_context():
    ap.init()
    ap.set_device(0)


@pytest.mark.parametrize("endpoint", [False, True])
@pytest.mark.parametrize("steps", [0, 1, 2, 5, 17])
@pytest.mark.parametrize("bounds", [(0.0, 1.0), (3.5, -2.0), (-1.0, -1.0)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_linspace_endpoint(endpoint, steps, bounds, dtype):
    actual = ap.linspace(*bounds, steps, dtype, endpoint=endpoint)
    expected = np.linspace(*bounds, steps, dtype=dtype, endpoint=endpoint)
    assert tuple(actual.shape) == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual.to_numpy(), expected, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("dtype", [None, np.float32, np.float64, np.int32, np.int64])
def test_linspace_empty_dtype(dtype):
    actual = ap.linspace(0, 1, 0, dtype=dtype)
    expected = np.linspace(0, 1, 0, dtype=dtype)
    assert tuple(actual.shape) == (0,)
    assert actual.dtype == expected.dtype


def test_linspace_default_endpoint():
    np.testing.assert_allclose(ap.linspace(0, 1, 5).to_numpy(), np.linspace(0, 1, 5))


@pytest.mark.parametrize("endpoint", [False, True])
def test_linspace_negative_steps(endpoint):
    with pytest.raises(ValueError, match="steps must be >= 0"):
        ap.linspace(0, 1, -1, endpoint=endpoint)
