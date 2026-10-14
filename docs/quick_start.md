# Quick Start

> Back to [README](../README.md) | See also: [Examples](../examples/) | [API Documentation](.)

This guide walks you through the basic usage of AsNumpy and shows how to migrate existing NumPy code to run on Ascend NPU with minimal changes.

## Prerequisites

- AsNumpy installed ([Installation Guide](../README.md#installation))
- Ascend 910B NPU with CANN 8.2.RC1.alpha003+
- Python 3.10+

## NumPy vs AsNumpy

The key idea: **change the import, keep the rest**.

<table>
<tr>
<th width="50%">NumPy (CPU)</th>
<th width="50%">AsNumpy (NPU)</th>
</tr>
<tr>
<td>

```python
import numpy as np

rows, cols = 20000, 20000
m1 = np.random.normal(0, 1, (rows, cols))
m2 = np.random.normal(0, 1, (rows, cols))

# Compute on CPU
product = np.multiply(m1, m2)
result = np.sum(product)
print(result)
```

</td>
<td>

```python
import numpy as np
import asnumpy as ap

rows, cols = 20000, 20000
m1 = np.random.normal(0, 1, (rows, cols))
m2 = np.random.normal(0, 1, (rows, cols))

# Transfer to NPU
m1_npu = ap.ndarray.from_numpy(m1)
m2_npu = ap.ndarray.from_numpy(m2)

# Compute on NPU
product = ap.multiply(m1_npu, m2_npu)
result = ap.sum(product)
print(result.to_numpy())
```

</td>
</tr>
</table>

## End-to-End Example

```python
import numpy as np
import asnumpy as ap

# AsNumpy auto-initializes the NPU device on import
# and releases it on exit (no manual init/finalize needed)

# 1. Create data on CPU (NumPy)
np_a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
np_b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

# 2. Transfer to NPU
npu_a = ap.ndarray.from_numpy(np_a)
npu_b = ap.ndarray.from_numpy(np_b)

# 3. Run operations on NPU
npu_sum   = ap.add(npu_a, npu_b)
npu_prod  = ap.multiply(npu_a, npu_b)
npu_total = ap.sum(npu_prod)

# 4. Transfer results back to CPU
print("Sum:   ", npu_sum.to_numpy())    # [11. 22. 33. 44.]
print("Prod:  ", npu_prod.to_numpy())   # [ 10.  40.  90. 160.]
print("Total: ", npu_total.to_numpy())  # 300.0

# 5. Verify against NumPy
assert np.allclose(npu_sum.to_numpy(), np.add(np_a, np_b))
assert np.allclose(npu_prod.to_numpy(), np.multiply(np_a, np_b))
print("Verification passed.")
```

## Checking the Device

```python
import asnumpy as ap

# Query available NPU devices
print(ap.get_device_count())  # e.g. 8

# Switch to a specific NPU (default is 0)
ap.set_device(1)
```

## More Examples

Runnable scripts are available in [`examples/`](../examples/):

| Script | Operation |
|--------|-----------|
| [`01_add.py`](../examples/01_add.py) | Element-wise addition |
| [`02_exp2.py`](../examples/02_exp2.py) | Exponentiation (2^x) |
| [`03_multiply.py`](../examples/03_multiply.py) | Element-wise multiply (with benchmark) |
| [`04_all.py`](../examples/04_all.py) | Combined operations |
| [`05_divide.py`](../examples/05_divide.py) | Element-wise division |
