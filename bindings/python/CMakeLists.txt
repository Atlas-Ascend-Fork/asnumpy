# ******************************************************************************
#  * Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  * you may not use this file except in compliance with the License.
#  * You may obtain a copy of the License at
#  *
#  * http://www.apache.org/licenses/LICENSE-2.0
#  *
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS,
#  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  * See the License for the specific language governing permissions and
#  * limitations under the License.
#  ******************************************************************************

pybind11_add_module(_core
    module.cpp
    bind_array.cpp
    bind_cann.cpp
    bind_fft.cpp
    bind_linalg.cpp
    bind_math.cpp
    bind_logic.cpp
    bind_random.cpp
    bind_sorting.cpp
    bind_statistics.cpp
    bind_nn.cpp
    bind_testing.cpp
    bind_utils.cpp)

target_include_directories(_core
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${Python_NumPy_INCLUDE_DIRS}
)

target_link_libraries(_core
    PUBLIC
    ascendcl
    runtime
    nnopbase
    opapi
    Python::NumPy
)
target_link_libraries(_core
    PUBLIC
    array
    cann
    dtypes
    utils
    linalg
    random
    math
    logic
    sorting
    statistics
    nn
)

# libops_blas.so lives in its own build/install prefix, so the installed _core.so must carry that
# directory in its run path; otherwise importing asnumpy fails with a loader error. The directory
# comes from the user-supplied dependency path, never from a hardcoded constant.
if(ASNUMPY_ENABLE_OPS_BLAS)
    get_filename_component(_asnumpy_ops_blas_libdir "${ASNUMPY_OPS_BLAS_LIBRARY}" DIRECTORY)
    set_property(TARGET _core APPEND PROPERTY INSTALL_RPATH "${_asnumpy_ops_blas_libdir}")
endif()

# Install the compiled module
install(TARGETS _core
        LIBRARY DESTINATION "asnumpy"
        COMPONENT python)
