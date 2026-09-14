# FAQ

> Back to [README](../README.md)

Frequently asked questions about installing and using AsNumpy.

---

<details>
<summary><b>How do I check if CANN is correctly installed?</b></summary>

```bash
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
```

</details>

<details>
<summary><b>What should I do if I encounter a compilation error?</b></summary>

1. Confirm CMake version >= 3.26: `cmake --version`
2. Confirm GCC version >= 11.2: `gcc --version`
3. Ensure the CANN environment variable is set:
   ```bash
   export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
   ```
4. Try a clean rebuild:
   ```bash
   pip install -e . --no-build-isolation
   ```

</details>

<details>
<summary><b>How compatible is AsNumpy with NumPy?</b></summary>

AsNumpy is designed to be API-compatible with NumPy, but operator coverage is not yet complete. The current version (v0.2.0) covers the most common math, logic, sorting, and array-creation APIs.

The roadmap target is to cover the **top 100 most-used NumPy APIs** by v1.0. See the [README roadmap](../README.md#roadmap) for the full plan.

</details>

<details>
<summary><b>Why is AsNumpy slower than NumPy for small arrays?</b></summary>

For small tensors (e.g., 500×500), NPU kernel launch overhead dominates the measured time. NPU acceleration becomes significant starting around 1000×1000 (`float32`). See [Benchmarks](benchmarks.md) for detailed data.

</details>

<details>
<summary><b>Do I need to manually initialize or finalize the NPU?</b></summary>

No. AsNumpy handles device initialization automatically on `import asnumpy` and releases the device on program exit. You only need to call `ap.set_device(n)` if you want to select a specific NPU other than device 0.

</details>

<details>
<summary><b>Can I pass an AsNumpy array directly to pandas?</b></summary>

Yes. `NPUArray` implements the NumPy array protocol (`__array__`, `__iter__`, `__len__`), so pandas accepts it the same way it accepts a `numpy.ndarray`:

```python
import numpy as np
import pandas as pd
import asnumpy as ap

a = ap.ndarray.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))

print(pd.Series(a))                      # 0  1.0 / 1  2.0 / 2  3.0
print(pd.DataFrame({"x": a}))            # 3 rows, column x
```

pandas converts the array by copying it from NPU to host memory, exactly like `np.asarray(a)` or `a.to_numpy()`. If you want the conversion to be explicit — or you are on an older release that predates the protocol — call `.to_numpy()` yourself:

```python
frame = pd.DataFrame(a.to_numpy())
```

</details>
