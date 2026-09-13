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

add_library(math OBJECT arithmetic_operations.cpp exponents_and_logarithms.cpp floating_point_routines.cpp
            handling_complex_numbers.cpp hyperbolic_functions.cpp miscellaneous.cpp other_special_functions.cpp
            rational_routines.cpp rounding.cpp sums_products_differences.cpp trigonometric_functions.cpp extrema_finding.cpp)
#             rational_routines.cpp rounding.cpp sums_products_differences.cpp )

target_include_directories(math PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(math PUBLIC fmt::fmt ascend_sdk pybind11::pybind11)
