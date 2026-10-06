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

"""
asnumpy.io
----------
I/O module for saving and loading arrays.

Implements:
- save
- savez
- savez_compressed
- load
"""

import numpy as _np

from ._core import ndarray as NPUArray
from .utils import ndarray as PublicArray


def _to_numpy(x):
    """
    Convert NPUArray to numpy.ndarray if needed.
    - If input is NPUArray: call x.to_numpy() (implemented in the C++ layer).
    - If input is already numpy.ndarray: return as-is.
    """
    if isinstance(x, NPUArray):
        return x.to_numpy()
    return x


def save(file, arr, allow_pickle=False):
    """
    Save a single array to a .npy file.
    """
    host = _to_numpy(arr)
    return _np.save(file, host, allow_pickle=allow_pickle)


def savez(file, *args, **kwargs):
    """
    Save multiple arrays into an .npz archive (uncompressed).
    Positional args are saved as arr_0, arr_1, ...
    Keyword args are saved with their given names.
    """
    conv_args = [_to_numpy(a) for a in args]
    conv_kwargs = {k: _to_numpy(v) for k, v in kwargs.items()}
    return _np.savez(file, *conv_args, **conv_kwargs)


def savez_compressed(file, *args, **kwargs):
    """
    Save multiple arrays into a compressed .npz archive.
    """
    conv_args = [_to_numpy(a) for a in args]
    conv_kwargs = {k: _to_numpy(v) for k, v in kwargs.items()}
    return _np.savez_compressed(file, *conv_args, **conv_kwargs)


class _AsnpNpz:
    """
    Lazy-loading container for .npz files.
    Works like a dictionary:
        with asnumpy.load("data.npz") as f:
            arr = f["key"]
    Values are returned as the public asnumpy.ndarray wrapper.
    """

    def __init__(self, npz):
        self._npz = npz  # numpy.lib.npyio.NpzFile object
        self._cache = {}  # dict to store already converted NPUArray

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._npz.close()

    @property
    def files(self):
        """List of keys (array names) in this npz archive."""
        return list(self._npz.files)

    def keys(self):
        return self.files

    def __iter__(self):
        return iter(self.files)

    def __getitem__(self, key):
        if key not in self._cache:
            host = self._npz[key]  # lazy unzip → numpy.ndarray
            self._cache[key] = PublicArray.from_numpy(host)  # upload to NPU
        return self._cache[key]


def load(file, mmap_mode=None, allow_pickle=False, **kwargs):
    """
    Load array(s) from .npy or .npz file.

    Arguments
    ----------
    file : str or file-like
        File path.
    mmap_mode : {None, 'r+', 'r', 'w+', 'c'}, optional
        If not None, memory-map the file (host side only).
    allow_pickle : bool, default False
        Allow loading pickled object arrays.
    kwargs : dict
        Other keyword args passed to numpy.load.
    """
    obj = _np.load(file, mmap_mode=mmap_mode, allow_pickle=allow_pickle, **kwargs)

    if isinstance(obj, _np.ndarray):
        # .npy file → host ndarray → public asnumpy.ndarray
        return PublicArray.from_numpy(obj)
    else:
        # .npz file → wrap in lazy container
        return _AsnpNpz(obj)
