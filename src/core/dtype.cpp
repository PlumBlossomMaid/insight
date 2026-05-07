// src/core/dtype.cpp
#include "insight/core/dtype.h"

namespace ins {

    static const struct DTypeInfo {
        const char* name;
        size_t size;
        bool is_float;
        bool is_int;
        bool is_complex;
        bool is_signed;
    } dtype_infos[] = {
        {"unknown", 0, false, false, false, false},                               // UNKNOWN
        {"bool",    sizeof(bool), false, false, false, false},                    // BOOL
        {"uint8",   sizeof(uint8_t), false, true, false, false},                  // U8
        {"int8",    sizeof(int8_t), false, true, false, true},                    // I8
        {"int16",   sizeof(int16_t), false, true, false, true},                   // I16
        {"int32",   sizeof(int32_t), false, true, false, true},                   // I32
        {"int64",   sizeof(int64_t), false, true, false, true},                   // I64
        {"float16", sizeof(uint16_t), true, false, false, true},                  // F16
        {"bfloat16", sizeof(uint16_t), true, false, false, true},                 // BF16
        {"float32", sizeof(float), true, false, false, true},                     // F32
        {"float64", sizeof(double), true, false, false, true},                    // F64
        {"complex64", sizeof(std::complex<float>), true, false, true, true},      // C32
        {"complex128", sizeof(std::complex<double>), true, false, true, true},    // C64
        {"float8_e4m3", sizeof(uint8_t), true, false, false, true},               // F8_E4M3
        {"float8_e5m2", sizeof(uint8_t), true, false, false, true},               // F8_E5M2
        {"uint16",  sizeof(uint16_t), false, true, false, false},                 // U16
        {"uint32",  sizeof(uint32_t), false, true, false, false},                 // U32
        {"uint64",  sizeof(uint64_t), false, true, false, false},                 // U64
    };

    static_assert(sizeof(dtype_infos) / sizeof(dtype_infos[0]) == static_cast<size_t>(DType::DTYPE_COUNT),
        "dtype_infos size mismatch");

    const char* dtype_name(DType dtype) {
        int idx = static_cast<int>(dtype);
        if (idx < 0 || idx >= static_cast<int>(DType::DTYPE_COUNT))
            return dtype_infos[0].name;
        return dtype_infos[idx].name;
    }

    DType dtype_from_name(const std::string& name) {
        for (int i = 0; i < static_cast<int>(DType::DTYPE_COUNT); ++i) {
            if (name == dtype_infos[i].name) {
                return static_cast<DType>(i);
            }
        }
        return DType::UNKNOWN;
    }

    size_t dtype_size(DType dtype) {
        int idx = static_cast<int>(dtype);
        if (idx < 0 || idx >= static_cast<int>(DType::DTYPE_COUNT))
            return 0;
        return dtype_infos[idx].size;
    }

    bool is_floating_point(DType dtype) {
        int idx = static_cast<int>(dtype);
        if (idx < 0 || idx >= static_cast<int>(DType::DTYPE_COUNT))
            return false;
        return dtype_infos[idx].is_float;
    }

    bool is_integer(DType dtype) {
        int idx = static_cast<int>(dtype);
        if (idx < 0 || idx >= static_cast<int>(DType::DTYPE_COUNT))
            return false;
        return dtype_infos[idx].is_int;
    }

    bool is_complex(DType dtype) {
        int idx = static_cast<int>(dtype);
        if (idx < 0 || idx >= static_cast<int>(DType::DTYPE_COUNT))
            return false;
        return dtype_infos[idx].is_complex;
    }

    bool is_signed(DType dtype) {
        int idx = static_cast<int>(dtype);
        if (idx < 0 || idx >= static_cast<int>(DType::DTYPE_COUNT))
            return false;
        return dtype_infos[idx].is_signed;
    }

    std::ostream& operator<<(std::ostream& os, DType dtype) {
        os << dtype_name(dtype);
        return os;
    }

} // namespace ins