<div align="center">

<img src="docs/images/AsNumpy Logo.png" alt="AsNumpy Logo" width="280">

# AsNumpy

### NumPy for Ascend NPU

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![CANN](https://img.shields.io/badge/CANN-8.2RC1+-orange.svg)]()
[![Platform](https://img.shields.io/badge/platform-Ascend%20910B-green.svg)]()
[![Version](https://img.shields.io/badge/version-0.2.0-brightgreen.svg)]()

[Docs](docs/) | [Installation](#installation) | [Quick Start](docs/quick_start.md) | [Examples](examples/) | [Architecture](docs/architecture.md) | [Issues](https://gitcode.com/cann/asnumpy/issues) | [OpenBOAT](https://gitcode.com/HIT1920/OpenBOAT)

</div>

AsNumpy is a lightweight Python library for scientific computing on Ascend NPU, fully compatible with the NumPy API.
It wraps Huawei CANN operators through a pybind11 binding layer, exposing them via the `NPUArray` data structure that mirrors `numpy.ndarray`.
Developed by the AISS Group and the ISE Group at Harbin Institute of Technology, in collaboration with the Huawei CANN team.

```python
import numpy as np
import asnumpy as ap

# Generate data on CPU
m1 = np.random.normal(0, 1, (3000, 3000)).astype(np.float32)
m2 = np.random.normal(0, 1, (3000, 3000)).astype(np.float32)

# Transfer to NPU
m1_npu = ap.ndarray.from_numpy(m1)
m2_npu = ap.ndarray.from_numpy(m2)

# Compute on NPU — same API as NumPy
result = ap.multiply(m1_npu, m2_npu)

# Transfer back to CPU
print(result.to_numpy())
```

> **Note**: The Performance section below benchmarks `ap.mean()` (a reduction operation),
> which differs from the `ap.multiply()` (element-wise) demo above. See [benchmarks.md](docs/benchmarks.md)
> for full reproduction steps.

<!-- toc -->

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Performance](#performance)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Resources](#resources)
- [License](#license)
- [Acknowledgements](#acknowledgements)

<!-- tocstop -->

## Features

- **NumPy-compatible API** — function names and signatures match NumPy; migrate existing code with minimal changes
- **Ascend 910B native acceleration** — operators run directly on NPU via CANN ACLNN without framework overhead
- **Automatic resource management** — `NPUArray` destructor releases device memory automatically (RAII)
- **Bidirectional data transfer** — `from_numpy()` and `to_numpy()` for seamless CPU↔NPU exchange
- **Broadcasting support** — built-in `GetBroadcastShape()` follows NumPy broadcasting semantics
- **Operator extensibility** — missing CANN operators are supplemented by [OpenBOAT](https://gitcode.com/HIT1920/OpenBOAT)

## Installation

**Requirements:** GCC >= 11.2, CMake >= 3.26, Python >= 3.10, CANN >= 8.2.RC1.alpha003, Ascend 910B NPU.

Set the CANN environment variable before building:

```bash
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
```

**Using uv (recommended)**

```bash
git clone --recursive https://gitcode.com/cann/asnumpy.git
cd asnumpy
uv sync
```

**Using pip**

```bash
git clone --recursive https://gitcode.com/cann/asnumpy.git
cd asnumpy

# Upgrade pip, setuptools, and wheel to their latest versions
pip install --upgrade pip setuptools wheel

# Install the 'build' package, which provides a simple build front-end for Python packages
pip install build

# Build the current Python project (create source distribution and wheel)
python -m build

# Install all generated wheel (.whl) files from the dist/ directory
pip install dist/*.whl
```

**Verify the installation**

```python
import asnumpy as ap
arr = ap.ones((1000, 1000), dtype=ap.float32)
print(arr.shape)  # (1000, 1000)
```

## Quick Start

A minimal example is shown above. For a full side-by-side comparison of NumPy vs AsNumpy code and more usage patterns, see the [Quick Start guide](docs/quick_start.md).

Runnable scripts are in [`examples/`](examples/).

## Performance

At 3000×3000 `float32`, `ap.mean()` runs **35.70× faster** than `np.mean()` on the same machine.
As data scale increases, the NPU's parallel computing advantage becomes more pronounced.

> `ap.mean()` is a reduction operation; the top-section demo uses `ap.multiply()` (element-wise).

| Shape | Data Size | AsNumpy (ms) | NumPy (ms) | Speedup |
|-------|-----------|-------------|------------|---------|
| (500, 500) | 250,000 | 0.1446 | 0.1429 | 0.99× |
| (1000, 1000) | 1,000,000 | 0.1510 | 0.4904 | 3.25× |
| (2000, 2000) | 4,000,000 | 0.1636 | 1.9372 | 11.84× |
| (3000, 3000) | 9,000,000 | 0.1857 | 6.6303 | **35.70×** |

Full test environment, controlled variables, and reproduction instructions: [benchmarks.md](docs/benchmarks.md).

## Roadmap

| Release | Quarter | Key Deliverables |
|---------|---------|-----------------|
| **v0.3.0** | 26Q1 | Documentation site (Docsify + GitCode Pages), CI/CD pipeline with hardware whitelist, code quality overhaul (spdlog, clang-format), PyPI release |
| **v0.4.0** | 26Q2 | Mathematical functions 100% API coverage, Ascend 950 validation, triton-ascend operator integration (10 ops), memory pool (experimental) |
| **v0.5.0** | 26Q2 | Linear algebra full coverage, triton-ascend operator library expanded to 20 ops, multi-NPU distributed computing research |

## Contributing

Contributions are welcome. Small fixes can be submitted directly as pull requests. For larger features, please open an issue first to discuss the design.

See the [Developer Guide](docs/developer_guide.md) for build instructions, coding conventions, and how to add new operators.

## Resources

- [Documentation](docs/)
- [Quick Start](docs/quick_start.md)
- [Architecture](docs/architecture.md)
- [Benchmarks](docs/benchmarks.md)
- [FAQ](docs/faq.md)
- [Developer Guide](docs/developer_guide.md)
- [Examples](examples/)
- [Issue Tracker](https://gitcode.com/cann/asnumpy/issues)
- [OpenBOAT Operator Library](https://gitcode.com/HIT1920/OpenBOAT)

## License

Apache License, Version 2.0 — see [LICENSE](LICENSE).

AsNumpy is designed based on NumPy's API (see [NumPy license](https://github.com/numpy/numpy/blob/main/LICENSE.txt)).

AsNumpy is being developed and maintained by the [School of Computer Science, Harbin Institute of Technology](https://gitcode.com/cann/asnumpy) in deep collaboration with the [Huawei CANN team](https://www.hiascend.com/), together with [community contributors](https://gitcode.com/cann/asnumpy/graphs/contributors).

## Acknowledgements

- AISS Group, School of Computer Science, Harbin Institute of Technology — Prof. Su Tonghua's team
- ISE Group, School of Computer Science, Harbin Institute of Technology — Prof. Wang Tiantian's team
- Huawei CANN team

---

<div align="center">

If AsNumpy is useful to you, please give us a star.

</div>
