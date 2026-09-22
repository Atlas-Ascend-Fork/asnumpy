/******************************************************************************
 * Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include <asnumpy/utils/cast.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <utility>

namespace {

// The NPUArray binding is assembled from these helpers so that no single function exceeds CodeArts
// Check's 50-line NBNC (non-blank, non-comment) limit -- the "超大函数[C++]" rule that a
// single-chained bind_utils() used to trip. bind_utils() itself only wires the pieces together.

void BindArrayCore(py::class_<NPUArray>& cls) {
    cls.def(py::init<const std::vector<int64_t>&, py::dtype>(), py::arg("shape"), py::arg("dtype"),
            "Constructs an empty NPUArray with the given shape and dtype.")
        // One-argument construction must remain a deep copy.
        .def(py::init<const NPUArray&>(), "Copy constructor for NPUArray")
        // Keep _move required and keyword-only: ndarray(other) selects the copy constructor, while
        // ndarray(other, _move=True) selects this consuming constructor.
        .def(py::init([](NPUArray& other, bool move) {
                 if (!move) {
                     // The project's exception translator maps std::invalid_argument to
                     // Python ValueError; py::value_error would be caught as std::runtime_error.
                     throw std::invalid_argument("_move must be true for the internal consuming constructor");
                 }
                 return NPUArray(std::move(other));
             }),
             py::arg("other"), py::kw_only(), py::arg("_move"),
             "Internal consuming constructor; the source must not be used afterwards")
        .def("to_numpy", &NPUArray::ToNumpy)
        .def_static("from_numpy", &NPUArray::FromNumpy, py::arg("host_data"))
        .def(
            "astype",
            // Takes py::object, not py::dtype: py::dtype's caster is a strict isinstance check with
            // no conversion, so it would reject the scalar-type spellings NumPy users write
            // (np.float32) and accept only np.dtype("float32"). Normalizing here keeps direct
            // _core.ndarray users -- e.g. whatever ap.load() returns -- working.
            [](const NPUArray& self, const py::object& dtype) {
                return asnumpy::CastTo(self, NPUArray::GetACLDataType(py::dtype::from_args(dtype)));
            },
            py::arg("dtype"), "Cast the array to the given dtype on device, returning a new array.");
}

void BindArrayProperties(py::class_<NPUArray>& cls) {
    cls.def_property_readonly("shape",
                              [](const NPUArray& self) {
                                  py::tuple shape_tuple(self.shape.size());
                                  for (size_t i = 0; i < self.shape.size(); ++i) {
                                      shape_tuple[i] = self.shape[i];
                                  }
                                  return shape_tuple;
                              })
        .def_property_readonly("dtype", [](const NPUArray& self) { return self.dtype; })
        .def_property_readonly("aclDtype", [](const NPUArray& self) { return static_cast<int>(self.aclDtype); })
        .def_property_readonly("ndim", [](const NPUArray& self) { return self.shape.size(); })
        .def_property_readonly("itemsize",
                               [](const NPUArray& self) { return NPUArray::GetDataTypeSize(self.aclDtype); })
        .def_property_readonly(
            "nbytes", [](const NPUArray& self) { return self.tensorSize * NPUArray::GetDataTypeSize(self.aclDtype); })
        .def_property_readonly("strides", [](const NPUArray& self) {
            auto itemsize = NPUArray::GetDataTypeSize(self.aclDtype);
            py::tuple byte_strides(self.strides.size());
            for (size_t i = 0; i < self.strides.size(); ++i) {
                byte_strides[i] = self.strides[i] * itemsize;
            }
            return byte_strides;
        });
}

// NumPy array protocol. Without __array__, numpy.asarray(npu_array) produced a 0-d object array and
// libraries built on NumPy (pandas, matplotlib, scikit-learn, ...) rejected the array as an
// unrecognized type. Defined on the core type rather than on the Python-level asnumpy.ndarray
// subclass so that every array asnumpy hands out -- including the raw _core.ndarray returned by
// asnumpy.load() -- is consumable by NumPy.
void BindArrayProtocol(py::class_<NPUArray>& cls) {
    cls.def(
           "__array__",
           [](const NPUArray& self, py::object dtype, py::object /*copy*/) -> py::object {
               // NumPy 2.x passes `copy`; it needs no handling because ToNumpy() always allocates a
               // fresh host array. A requested dtype is applied on the host, exactly as
               // numpy.ndarray.__array__ does.
               py::array host = self.ToNumpy();
               if (dtype.is_none()) {
                   return host;
               }
               return host.attr("astype")(dtype);
           },
           py::arg("dtype") = py::none(), py::arg("copy") = py::none(),
           "Return the array contents as a numpy.ndarray (NumPy array protocol).")
        // pandas decides whether a value is array-like through the iteration protocol, so
        // __array__ alone is not enough: pd.Series(array) and pd.DataFrame(array) treat the array
        // as a scalar object without __iter__. Iterating the host view keeps NumPy's semantics
        // (rows for a 2-d array, TypeError for a 0-d array).
        .def(
            "__iter__", [](const NPUArray& self) { return py::iter(self.ToNumpy()); },
            "Iterate over the first axis, yielding numpy arrays (NumPy iteration protocol).")
        .def(
            "__len__",
            [](const NPUArray& self) {
                if (self.shape.empty()) {
                    // Matches NumPy: len() of a 0-d array is a TypeError, not 0. Raised through
                    // error_already_set rather than py::type_error because the module's exception
                    // translator catches std::runtime_error first, and py::type_error derives from
                    // it -- that path would report a RuntimeError instead.
                    PyErr_SetString(PyExc_TypeError, "len() of unsized object");
                    throw py::error_already_set();
                }
                return static_cast<py::ssize_t>(self.shape.front());
            },
            "Return the length of the first dimension (NumPy len() protocol).");
}

} // namespace

void bind_utils(pybind11::module_& utils) {
    py::class_<NPUArray> cls(utils, "ndarray");
    BindArrayCore(cls);
    BindArrayProperties(cls);
    BindArrayProtocol(cls);
    utils.def("broadcast_shape", &GetBroadcastShape, py::arg("a"), py::arg("b"));
}
