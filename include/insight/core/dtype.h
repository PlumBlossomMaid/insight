// insight/core/dtype.h
#pragma once
#include <cstddef>
#include <cstdint>
#include <complex>

namespace ins {

    /**
     * @brief Data type enumeration.
     *
     * Defines all supported data types including floating point, integer,
     * complex, and experimental types like float8.
     * Order follows promotion priority (higher priority later in the list).
     */
    enum class DType {
        UNKNOWN = 0,
        BOOL,           ///< Boolean
        U8,             ///< 8-bit unsigned integer
        I8,             ///< 8-bit signed integer
        I16,            ///< 16-bit signed integer
        I32,            ///< 32-bit signed integer
        I64,            ///< 64-bit signed integer
        F16,            ///< 16-bit floating point (IEEE 754 half)
        BF16,           ///< bfloat16
        F32,            ///< 32-bit floating point (IEEE 754 single)
        F64,            ///< 64-bit floating point (IEEE 754 double)
        C32,            ///< Complex64 (two 32-bit floats)
        C64,            ///< Complex128 (two 64-bit doubles)
        F8_E4M3,        ///< 8-bit float, 4 exponent bits, 3 mantissa bits
        F8_E5M2,        ///< 8-bit float, 5 exponent bits, 2 mantissa bits
        U16,            ///< 16-bit unsigned integer
        U32,            ///< 32-bit unsigned integer
        U64,            ///< 64-bit unsigned integer
        DTYPE_COUNT     ///< Number of data types (must be last)
    };

    /**
     * @brief Returns the string name of a data type.
     * @param dtype The data type enumerator
     * @return String name (e.g., "float32", "int64", never returns nullptr)
     */
    const char* dtype_name(DType dtype);

    /**
     * @brief Get DType from string name.
     * @param name Type name (e.g., "float32", "int64")
     * @return Corresponding DType, or DType::UNKNOWN if not found
     */
    DType dtype_from_name(const std::string& name);

    /**
     * @brief Returns the size in bytes of a data type.
     * @param dtype The data type enumerator
     * @return Size in bytes (0 for UNKNOWN)
     */
    size_t dtype_size(DType dtype);

    /**
     * @brief Checks if a data type is floating point.
     * @param dtype The data type enumerator
     * @return true if dtype is a floating point type (including complex)
     */
    bool is_floating_point(DType dtype);

    /**
     * @brief Checks if a data type is integer (signed or unsigned).
     * @param dtype The data type enumerator
     * @return true if dtype is an integer type (including bool)
     */
    bool is_integer(DType dtype);

    /**
     * @brief Checks if a data type is complex (float-based complex numbers).
     * @param dtype The data type enumerator
     * @return true if dtype is complex64 or complex128
     */
    bool is_complex(DType dtype);

    /**
     * @brief Checks if a data type is signed (e.g., int32, float64).
     * @note Unsigned types: U8, U16, U32, U64
     * @param dtype The data type enumerator
     * @return true if dtype can represent negative values
     */
    bool is_signed(DType dtype);

    /**
     * @brief Outputs dtype name to stream.
     * @param os Output stream
     * @param dtype Data type enumerator
     * @return Reference to the output stream
     */
    std::ostream& operator<<(std::ostream& os, DType dtype);


    /**
     * @brief Get DType from C++ type at compile time.
     *
     * Usage:
     * @code
     *   DType dtype = dtype_of<float>();  // returns DType::F32
     * @endcode
     *
     * @tparam T C++ type
     * @return Corresponding DType enumerator
     */
    template<typename T>
    constexpr DType dtype_of() {
        if constexpr (std::is_same_v<T, float>) return DType::F32;
        else if constexpr (std::is_same_v<T, double>) return DType::F64;
        else if constexpr (std::is_same_v<T, int32_t>) return DType::I32;
        else if constexpr (std::is_same_v<T, int64_t>) return DType::I64;
        else if constexpr (std::is_same_v<T, uint8_t>) return DType::U8;
        else if constexpr (std::is_same_v<T, bool>) return DType::BOOL;
        else if constexpr (std::is_same_v<T, int8_t>) return DType::I8;
        else if constexpr (std::is_same_v<T, int16_t>) return DType::I16;
        else if constexpr (std::is_same_v<T, uint16_t>) return DType::U16;
        else if constexpr (std::is_same_v<T, uint32_t>) return DType::U32;
        else if constexpr (std::is_same_v<T, uint64_t>) return DType::U64;
        else if constexpr (std::is_same_v<T, std::complex<float>>) return DType::C32;
        else if constexpr (std::is_same_v<T, std::complex<double>>) return DType::C64;
        else {
            static_assert(sizeof(T) == 0, "Unsupported type for dtype_of");
            return DType::UNKNOWN;
        }
    }

} // namespace ins