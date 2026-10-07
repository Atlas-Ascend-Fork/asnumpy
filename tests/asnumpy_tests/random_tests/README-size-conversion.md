# NumPy integer random sizes

The random distribution wrappers share `asnumpy.utils._convert_size`.
Their native bindings accept `std::vector<int64_t>` shapes. A Python
integer was wrapped in a one-element tuple, but a NumPy integer scalar
was passed through, so `np.int64(5)` failed the sequence conversion.

Accept NumPy integer scalars in the existing scalar branch and convert
them with `operator.index`. Existing sequence and non-integer inputs
continue to reach the original binding validation. This does not change
distribution algorithms, random seeds or supported output dtypes.

With the installed Asnumpy package, run:

```bash
python3 -m pytest tests/asnumpy_tests/random_tests/test_size_conversion.py -q
```

The tests cover all eight NumPy integer types at zero, one and five,
compare the vector shape with NumPy's real random API, and retain Python
integer/bool, tuple/list and non-integer controls. They exercise shape
conversion only, without invoking a distribution kernel.

Host validation may load the complete `utils.py` under a namespace
package with an import-only `_core` placeholder. That verifies the real
conversion helper without the CANN SDK; it does not verify NPU random
numbers. A separate real Pybind `std::vector<int64_t>` probe can check
the scalar-versus-sequence binding contract without loading CANN.
