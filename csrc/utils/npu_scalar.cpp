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
 *****************************************************************************/

#include "asnumpy/utils/npu_scalar.hpp"
#include <asnumpy/utils/status_handler.hpp>

#include <cstdint>
#include <cstring>

#include <fmt/format.h>


/*
    Converts an IEEE-754 binary32 value to IEEE-754 binary16 bits with
    round-to-nearest, ties-to-even, including subnormal, overflow and NaN
    handling. aclCreateScalar expects the raw binary16 bit pattern for
    ACL_FLOAT16, so an explicit conversion is required here.
*/
static uint16_t FloatToFloat16Bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000U);
    const uint32_t exponent = (bits >> 23) & 0xFFU;
    const uint32_t mantissa = bits & 0x7FFFFFU;

    if (exponent == 0xFFU) {
        // Infinities keep their sign; any NaN becomes a canonical quiet NaN.
        return static_cast<uint16_t>(mantissa == 0U ? (sign | 0x7C00U) : (sign | 0x7E00U));
    }

    const int32_t newExponent = static_cast<int32_t>(exponent) - 127 + 15;
    if (newExponent >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }

    uint16_t half = 0;
    if (newExponent <= 0) {
        if (newExponent < -10) {
            return sign;
        }
        const uint32_t subnormal = mantissa | 0x800000U;
        const uint32_t shift = static_cast<uint32_t>(14 - newExponent);
        half = static_cast<uint16_t>(subnormal >> shift);
        const uint32_t remainder = subnormal & ((1U << shift) - 1U);
        const uint32_t halfway = 1U << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half & 1U) != 0U)) {
            ++half;
        }
        return static_cast<uint16_t>(sign | half);
    }

    half = static_cast<uint16_t>((static_cast<uint32_t>(newExponent) << 10) | (mantissa >> 13));
    const uint32_t remainder = mantissa & 0x1FFFU;
    if (remainder > 0x1000U || (remainder == 0x1000U && (half & 1U) != 0U)) {
        // Incrementing can carry into the exponent field, which correctly
        // yields infinity on overflow.
        ++half;
    }
    return static_cast<uint16_t>(sign | half);
}

/*
    Creates an aclScalar object by automatically determining the appropriate ACL data type
    based on the C++ type of the input value. Uses TypeToACLDtype for compile-time mapping.
*/
template <typename T> aclScalar* CreateScalar(T value) {
    return CreateScalar(value, TypeToACLDtype<std::decay_t<T>>::value);
}

/*
    Creates an aclScalar object for a given value with explicit data type control.
    Performs optimized value conversion when the input type matches the target data type.
    Falls back to static_cast conversion when types differ.
*/
template <typename ValueType> aclScalar* CreateScalar(ValueType value, aclDataType dtype) {
    // Helper lambda to check aclCreateScalar return value
    auto checkScalar = [](aclScalar* scalar, const char* api_name) -> aclScalar* {
        if (scalar == nullptr) {
            throw std::runtime_error(fmt::format("[npu_scalar.cpp]({}) {} returned nullptr", __func__, api_name));
        }
        return scalar;
    };

    switch (dtype) {
    case ACL_FLOAT: {
        if constexpr (std::is_same_v<std::decay_t<ValueType>, float>) {
            return checkScalar(aclCreateScalar(&value, ACL_FLOAT), "aclCreateScalar");
        } else {
            auto converted = static_cast<float>(value);
            return checkScalar(aclCreateScalar(&converted, ACL_FLOAT), "aclCreateScalar");
        }
    }
    case ACL_DOUBLE: {
        if constexpr (std::is_same_v<std::decay_t<ValueType>, double>) {
            return checkScalar(aclCreateScalar(&value, ACL_DOUBLE), "aclCreateScalar");
        } else {
            auto converted = static_cast<double>(value);
            return checkScalar(aclCreateScalar(&converted, ACL_DOUBLE), "aclCreateScalar");
        }
    }
    case ACL_INT32: {
        if constexpr (std::is_same_v<std::decay_t<ValueType>, int32_t>) {
            return checkScalar(aclCreateScalar(&value, ACL_INT32), "aclCreateScalar");
        } else {
            auto converted = static_cast<int32_t>(value);
            return checkScalar(aclCreateScalar(&converted, ACL_INT32), "aclCreateScalar");
        }
    }
    case ACL_INT64: {
        if constexpr (std::is_same_v<std::decay_t<ValueType>, int64_t>) {
            return checkScalar(aclCreateScalar(&value, ACL_INT64), "aclCreateScalar");
        } else {
            auto converted = static_cast<int64_t>(value);
            return checkScalar(aclCreateScalar(&converted, ACL_INT64), "aclCreateScalar");
        }
    }
    case ACL_INT8: {
        if constexpr (std::is_same_v<std::decay_t<ValueType>, int8_t>) {
            return checkScalar(aclCreateScalar(&value, ACL_INT8), "aclCreateScalar");
        } else {
            auto converted = static_cast<int8_t>(value);
            return checkScalar(aclCreateScalar(&converted, ACL_INT8), "aclCreateScalar");
        }
    }
    case ACL_INT16: {
        if constexpr (std::is_same_v<std::decay_t<ValueType>, int16_t>) {
            return checkScalar(aclCreateScalar(&value, ACL_INT16), "aclCreateScalar");
        } else {
            auto converted = static_cast<int16_t>(value);
            return checkScalar(aclCreateScalar(&converted, ACL_INT16), "aclCreateScalar");
        }
    }
    case ACL_UINT8: {
        if constexpr (std::is_same_v<std::decay_t<ValueType>, uint8_t>) {
            return checkScalar(aclCreateScalar(&value, ACL_UINT8), "aclCreateScalar");
        } else {
            auto converted = static_cast<uint8_t>(value);
            return checkScalar(aclCreateScalar(&converted, ACL_UINT8), "aclCreateScalar");
        }
    }
    case ACL_UINT16: {
        if constexpr (std::is_same_v<std::decay_t<ValueType>, uint16_t>) {
            return checkScalar(aclCreateScalar(&value, ACL_UINT16), "aclCreateScalar");
        } else {
            auto converted = static_cast<uint16_t>(value);
            return checkScalar(aclCreateScalar(&converted, ACL_UINT16), "aclCreateScalar");
        }
    }
    case ACL_UINT32: {
        if constexpr (std::is_same_v<std::decay_t<ValueType>, uint32_t>) {
            return checkScalar(aclCreateScalar(&value, ACL_UINT32), "aclCreateScalar");
        } else {
            auto converted = static_cast<uint32_t>(value);
            return checkScalar(aclCreateScalar(&converted, ACL_UINT32), "aclCreateScalar");
        }
    }
    case ACL_UINT64: {
        if constexpr (std::is_same_v<std::decay_t<ValueType>, uint64_t>) {
            return checkScalar(aclCreateScalar(&value, ACL_UINT64), "aclCreateScalar");
        } else {
            auto converted = static_cast<uint64_t>(value);
            return checkScalar(aclCreateScalar(&converted, ACL_UINT64), "aclCreateScalar");
        }
    }
    case ACL_BOOL: {
        if constexpr (std::is_same_v<std::decay_t<ValueType>, bool>) {
            return checkScalar(aclCreateScalar(&value, ACL_BOOL), "aclCreateScalar");
        } else {
            auto converted = static_cast<bool>(value);
            return checkScalar(aclCreateScalar(&converted, ACL_BOOL), "aclCreateScalar");
        }
    }
    case ACL_FLOAT16: {
        auto converted = FloatToFloat16Bits(static_cast<float>(value));
        return checkScalar(aclCreateScalar(&converted, ACL_FLOAT16), "aclCreateScalar");
    }
    case ACL_BF16: {
        auto converted = static_cast<uint16_t>(static_cast<float>(value));
        return checkScalar(aclCreateScalar(&converted, ACL_BF16), "aclCreateScalar");
    }
    case ACL_INT4: {
        auto converted = static_cast<int8_t>(value);
        return checkScalar(aclCreateScalar(&converted, ACL_INT4), "aclCreateScalar");
    }
    case ACL_UINT1: {
        auto converted = static_cast<uint8_t>(value);
        return checkScalar(aclCreateScalar(&converted, ACL_UINT1), "aclCreateScalar");
    }
    case ACL_COMPLEX64: {
        auto converted = std::complex<float>(static_cast<float>(value), 0.0f);
        return checkScalar(aclCreateScalar(&converted, ACL_COMPLEX64), "aclCreateScalar");
    }
    case ACL_COMPLEX128: {
        auto converted = std::complex<double>(static_cast<double>(value), 0.0);
        return checkScalar(aclCreateScalar(&converted, ACL_COMPLEX128), "aclCreateScalar");
    }
    case ACL_COMPLEX32: {
        auto converted = std::complex<float>(static_cast<float>(value), 0.0f);
        return checkScalar(aclCreateScalar(&converted, ACL_COMPLEX32), "aclCreateScalar");
    }
    case ACL_HIFLOAT8: {
        auto converted = static_cast<uint8_t>(static_cast<float>(value));
        return checkScalar(aclCreateScalar(&converted, ACL_HIFLOAT8), "aclCreateScalar");
    }
    case ACL_FLOAT8_E5M2: {
        auto converted = static_cast<uint8_t>(static_cast<float>(value));
        return checkScalar(aclCreateScalar(&converted, ACL_FLOAT8_E5M2), "aclCreateScalar");
    }
    case ACL_FLOAT8_E4M3FN: {
        auto converted = static_cast<uint8_t>(static_cast<float>(value));
        return checkScalar(aclCreateScalar(&converted, ACL_FLOAT8_E4M3FN), "aclCreateScalar");
    }
    case ACL_FLOAT8_E8M0: {
        auto converted = static_cast<uint8_t>(static_cast<float>(value));
        return checkScalar(aclCreateScalar(&converted, ACL_FLOAT8_E8M0), "aclCreateScalar");
    }
    case ACL_FLOAT6_E3M2: {
        auto converted = static_cast<uint8_t>(static_cast<float>(value));
        return checkScalar(aclCreateScalar(&converted, ACL_FLOAT6_E3M2), "aclCreateScalar");
    }
    case ACL_FLOAT6_E2M3: {
        auto converted = static_cast<uint8_t>(static_cast<float>(value));
        return checkScalar(aclCreateScalar(&converted, ACL_FLOAT6_E2M3), "aclCreateScalar");
    }
    case ACL_FLOAT4_E2M1: {
        auto converted = static_cast<uint8_t>(static_cast<float>(value));
        return checkScalar(aclCreateScalar(&converted, ACL_FLOAT4_E2M1), "aclCreateScalar");
    }
    case ACL_FLOAT4_E1M2: {
        auto converted = static_cast<uint8_t>(static_cast<float>(value));
        return checkScalar(aclCreateScalar(&converted, ACL_FLOAT4_E1M2), "aclCreateScalar");
    }
    case ACL_STRING: {
        static std::string str_value;
        str_value = std::to_string(static_cast<double>(value));
        return checkScalar(aclCreateScalar(str_value.data(), ACL_STRING), "aclCreateScalar");
    }
    case ACL_DT_UNDEFINED: {
        // use default value 0 for undefined types
        auto converted = static_cast<int32_t>(0);
        return checkScalar(aclCreateScalar(&converted, ACL_INT32), "aclCreateScalar");
    }
    default:
        throw std::runtime_error(
            fmt::format("[npu_scalar.cpp]({}) unsupported dtype: {}", __func__, static_cast<int>(dtype)));
    }
}

// =====================
// Explicit instantiations
// =====================

// 1) CreateScalar(T value) -- auto type deduction version
template aclScalar* CreateScalar<float>(float);
template aclScalar* CreateScalar<double>(double);
template aclScalar* CreateScalar<int32_t>(int32_t);
template aclScalar* CreateScalar<int64_t>(int64_t);
template aclScalar* CreateScalar<int16_t>(int16_t);
template aclScalar* CreateScalar<int8_t>(int8_t);
template aclScalar* CreateScalar<uint64_t>(uint64_t);
template aclScalar* CreateScalar<uint32_t>(uint32_t);
template aclScalar* CreateScalar<uint16_t>(uint16_t);
template aclScalar* CreateScalar<uint8_t>(uint8_t);
template aclScalar* CreateScalar<bool>(bool);

// 2) CreateScalar(ValueType value, aclDataType dtype) -- explicit dtype version
template aclScalar* CreateScalar<float>(float, aclDataType);
template aclScalar* CreateScalar<double>(double, aclDataType);
template aclScalar* CreateScalar<int32_t>(int32_t, aclDataType);
template aclScalar* CreateScalar<int64_t>(int64_t, aclDataType);
template aclScalar* CreateScalar<int16_t>(int16_t, aclDataType);
template aclScalar* CreateScalar<int8_t>(int8_t, aclDataType);
template aclScalar* CreateScalar<uint64_t>(uint64_t, aclDataType);
template aclScalar* CreateScalar<uint32_t>(uint32_t, aclDataType);
template aclScalar* CreateScalar<uint16_t>(uint16_t, aclDataType);
template aclScalar* CreateScalar<uint8_t>(uint8_t, aclDataType);
template aclScalar* CreateScalar<bool>(bool, aclDataType);

// 3) CreateScalar(py::object scalar, aclDataType dtype) -- Python object version
template <typename T> aclScalar* CreateScalarFromPython(const py::object& scalar, aclDataType dtype) {
    try {
        T value = py::cast<T>(scalar);
        return CreateScalar(value, dtype);
    } catch (const py::cast_error& e) {
        throw std::runtime_error(fmt::format("[npu_scalar.cpp]({}) Failed to convert Python object to {}: {}", __func__,
                                             typeid(T).name(), e.what()));
    }
}

// type mapping table
using TypeHandler = std::function<aclScalar*(const py::object&, aclDataType)>;
static const std::unordered_map<aclDataType, TypeHandler> type_handlers = {
    {ACL_FLOAT, CreateScalarFromPython<float>},     {ACL_DOUBLE, CreateScalarFromPython<double>},
    {ACL_INT32, CreateScalarFromPython<int32_t>},   {ACL_INT64, CreateScalarFromPython<int64_t>},
    {ACL_INT8, CreateScalarFromPython<int8_t>},     {ACL_INT16, CreateScalarFromPython<int16_t>},
    {ACL_UINT8, CreateScalarFromPython<uint8_t>},   {ACL_UINT16, CreateScalarFromPython<uint16_t>},
    {ACL_UINT32, CreateScalarFromPython<uint32_t>}, {ACL_UINT64, CreateScalarFromPython<uint64_t>},
    {ACL_BOOL, CreateScalarFromPython<bool>},
};

aclScalar* CreateScalar(const py::object& scalar, aclDataType dtype) {
    auto it = type_handlers.find(dtype);
    if (it != type_handlers.end()) {
        return it->second(scalar, dtype);
    }

    // default
    try {
        double value = py::cast<double>(scalar);
        return CreateScalar(value, dtype);
    } catch (const py::cast_error& e) {
        throw std::runtime_error(
            fmt::format("[npu_scalar.cpp]({}) Failed to convert Python object to appropriate type for dtype {}: {}",
                        __func__, static_cast<int>(dtype), e.what()));
    }
}
