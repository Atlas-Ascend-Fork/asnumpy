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

add_library(fft OBJECT fft.cpp)

target_include_directories(fft PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(fft PUBLIC fmt::fmt ascend_sdk pybind11::pybind11)

if(ASNUMPY_ENABLE_FFT)
    target_compile_definitions(fft PUBLIC ASNUMPY_ENABLE_FFT=1)
    target_include_directories(fft PUBLIC ${ASNUMPY_OPS_FFT_INCLUDE_DIR})
    target_link_libraries(fft PUBLIC ${ASNUMPY_OPS_FFT_LIBRARY})
endif()
