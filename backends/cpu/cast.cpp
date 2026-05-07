// backends/cpu/cast.cpp
#include "insight/plugin/op_registry.h"
#include "insight/utils/promotion.h"
#include "insight/core/array.h"
#include <complex>

namespace ins::cpu {

    // Generic cast helper
    template<typename SrcT, typename DstT>
    static Array cast_impl(const Array& x) {
        Array out(x.shape(), dtype_of<DstT>(), x.place());

        const SrcT* src = x.data<SrcT>();
        DstT* dst = out.data<DstT>();
        int64_t n = x.numel();

#pragma omp parallel for
        for (int64_t i = 0; i < n; ++i) {
            dst[i] = static_cast<DstT>(src[i]);
        }

        return out;
    }

    // Specialization: bool -> other (false=0, true=1)
    template<typename DstT>
    static Array cast_from_bool_impl(const Array& x) {
        Array out(x.shape(), dtype_of<DstT>(), x.place());

        const bool* src = x.data<bool>();
        DstT* dst = out.data<DstT>();
        int64_t n = x.numel();

#pragma omp parallel for
        for (int64_t i = 0; i < n; ++i) {
            dst[i] = src[i] ? static_cast<DstT>(1) : static_cast<DstT>(0);
        }

        return out;
    }

    // Specialization: other -> bool
    template<typename SrcT>
    static Array cast_to_bool_impl(const Array& x) {
        Array out(x.shape(), DType::BOOL, x.place());

        const SrcT* src = x.data<SrcT>();
        bool* dst = out.data<bool>();
        int64_t n = x.numel();

#pragma omp parallel for
        for (int64_t i = 0; i < n; ++i) {
            dst[i] = (src[i] != SrcT(0));
        }

        return out;
    }

    // Specialization: float -> complex
    template<typename T>
    static Array cast_to_complex_impl(const Array& x) {
        using ComplexT = std::complex<T>;
        Array out(x.shape(), dtype_of<ComplexT>(), x.place());

        const T* src = x.data<T>();
        ComplexT* dst = out.data<ComplexT>();
        int64_t n = x.numel();

#pragma omp parallel for
        for (int64_t i = 0; i < n; ++i) {
            dst[i] = ComplexT(src[i], 0);
        }

        return out;
    }

    // Specialization: complex -> float (取实部)
    template<typename T>
    static Array cast_complex_to_float_impl(const Array& x) {
        using ComplexT = std::complex<T>;
        Array out(x.shape(), dtype_of<T>(), x.place());

        const ComplexT* src = x.data<ComplexT>();
        T* dst = out.data<T>();
        int64_t n = x.numel();

#pragma omp parallel for
        for (int64_t i = 0; i < n; ++i) {
            dst[i] = src[i].real();
        }

        return out;
    }

    // Double dispatch for cast
    static OpArgs cast_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        DType target_dtype = std::any_cast<DType>(args[1]);
        DType src_dtype = x.dtype();

        if (src_dtype == target_dtype) {
            return { x };
        }

        Array result;

        // Dispatch all possible conversions
        switch (src_dtype) {
            // ========== From BOOL ==========
        case DType::BOOL: {
            using SrcT = bool;
            switch (target_dtype) {
            case DType::U8:   result = cast_impl<SrcT, uint8_t>(x); break;
            case DType::I8:   result = cast_impl<SrcT, int8_t>(x); break;
            case DType::I16:  result = cast_impl<SrcT, int16_t>(x); break;
            case DType::I32:  result = cast_impl<SrcT, int32_t>(x); break;
            case DType::I64:  result = cast_impl<SrcT, int64_t>(x); break;
            case DType::U16:  result = cast_impl<SrcT, uint16_t>(x); break;
            case DType::U32:  result = cast_impl<SrcT, uint32_t>(x); break;
            case DType::U64:  result = cast_impl<SrcT, uint64_t>(x); break;
            case DType::F16:  result = cast_impl<SrcT, uint16_t>(x); break;  // placeholder
            case DType::BF16: result = cast_impl<SrcT, uint16_t>(x); break;
            case DType::F32:  result = cast_impl<SrcT, float>(x); break;
            case DType::F64:  result = cast_impl<SrcT, double>(x); break;
            case DType::C32:  result = cast_impl<SrcT, std::complex<float>>(x); break;
            case DType::C64:  result = cast_impl<SrcT, std::complex<double>>(x); break;
            default: INS_THROW("cast: unsupported target type from bool");
            }
            break;
        }

                        // ========== From U8 ==========
        case DType::U8: {
            using SrcT = uint8_t;
            switch (target_dtype) {
            case DType::BOOL: result = cast_to_bool_impl<SrcT>(x); break;
            case DType::I8:   result = cast_impl<SrcT, int8_t>(x); break;
            case DType::I16:  result = cast_impl<SrcT, int16_t>(x); break;
            case DType::I32:  result = cast_impl<SrcT, int32_t>(x); break;
            case DType::I64:  result = cast_impl<SrcT, int64_t>(x); break;
            case DType::U16:  result = cast_impl<SrcT, uint16_t>(x); break;
            case DType::U32:  result = cast_impl<SrcT, uint32_t>(x); break;
            case DType::U64:  result = cast_impl<SrcT, uint64_t>(x); break;
            case DType::F32:  result = cast_impl<SrcT, float>(x); break;
            case DType::F64:  result = cast_impl<SrcT, double>(x); break;
            case DType::C32:  result = cast_impl<SrcT, std::complex<float>>(x); break;
            case DType::C64:  result = cast_impl<SrcT, std::complex<double>>(x); break;
            default: INS_THROW("cast: unsupported target type from uint8");
            }
            break;
        }

                      // ========== From I8 ==========
        case DType::I8: {
            using SrcT = int8_t;
            switch (target_dtype) {
            case DType::BOOL: result = cast_to_bool_impl<SrcT>(x); break;
            case DType::U8:   result = cast_impl<SrcT, uint8_t>(x); break;
            case DType::I16:  result = cast_impl<SrcT, int16_t>(x); break;
            case DType::I32:  result = cast_impl<SrcT, int32_t>(x); break;
            case DType::I64:  result = cast_impl<SrcT, int64_t>(x); break;
            case DType::U16:  result = cast_impl<SrcT, uint16_t>(x); break;
            case DType::U32:  result = cast_impl<SrcT, uint32_t>(x); break;
            case DType::U64:  result = cast_impl<SrcT, uint64_t>(x); break;
            case DType::F32:  result = cast_impl<SrcT, float>(x); break;
            case DType::F64:  result = cast_impl<SrcT, double>(x); break;
            case DType::C32:  result = cast_impl<SrcT, std::complex<float>>(x); break;
            case DType::C64:  result = cast_impl<SrcT, std::complex<double>>(x); break;
            default: INS_THROW("cast: unsupported target type from int8");
            }
            break;
        }

                      // ========== From I16 ==========
        case DType::I16: {
            using SrcT = int16_t;
            switch (target_dtype) {
            case DType::BOOL: result = cast_to_bool_impl<SrcT>(x); break;
            case DType::U8:   result = cast_impl<SrcT, uint8_t>(x); break;
            case DType::I8:   result = cast_impl<SrcT, int8_t>(x); break;
            case DType::I32:  result = cast_impl<SrcT, int32_t>(x); break;
            case DType::I64:  result = cast_impl<SrcT, int64_t>(x); break;
            case DType::U16:  result = cast_impl<SrcT, uint16_t>(x); break;
            case DType::U32:  result = cast_impl<SrcT, uint32_t>(x); break;
            case DType::U64:  result = cast_impl<SrcT, uint64_t>(x); break;
            case DType::F32:  result = cast_impl<SrcT, float>(x); break;
            case DType::F64:  result = cast_impl<SrcT, double>(x); break;
            case DType::C32:  result = cast_impl<SrcT, std::complex<float>>(x); break;
            case DType::C64:  result = cast_impl<SrcT, std::complex<double>>(x); break;
            default: INS_THROW("cast: unsupported target type from int16");
            }
            break;
        }

                       // ========== From I32 ==========
        case DType::I32: {
            using SrcT = int32_t;
            switch (target_dtype) {
            case DType::BOOL: result = cast_to_bool_impl<SrcT>(x); break;
            case DType::U8:   result = cast_impl<SrcT, uint8_t>(x); break;
            case DType::I8:   result = cast_impl<SrcT, int8_t>(x); break;
            case DType::I16:  result = cast_impl<SrcT, int16_t>(x); break;
            case DType::I64:  result = cast_impl<SrcT, int64_t>(x); break;
            case DType::U16:  result = cast_impl<SrcT, uint16_t>(x); break;
            case DType::U32:  result = cast_impl<SrcT, uint32_t>(x); break;
            case DType::U64:  result = cast_impl<SrcT, uint64_t>(x); break;
            case DType::F32:  result = cast_impl<SrcT, float>(x); break;
            case DType::F64:  result = cast_impl<SrcT, double>(x); break;
            case DType::C32:  result = cast_impl<SrcT, std::complex<float>>(x); break;
            case DType::C64:  result = cast_impl<SrcT, std::complex<double>>(x); break;
            default: INS_THROW("cast: unsupported target type from int32");
            }
            break;
        }

                       // ========== From I64 ==========
        case DType::I64: {
            using SrcT = int64_t;
            switch (target_dtype) {
            case DType::BOOL: result = cast_to_bool_impl<SrcT>(x); break;
            case DType::U8:   result = cast_impl<SrcT, uint8_t>(x); break;
            case DType::I8:   result = cast_impl<SrcT, int8_t>(x); break;
            case DType::I16:  result = cast_impl<SrcT, int16_t>(x); break;
            case DType::I32:  result = cast_impl<SrcT, int32_t>(x); break;
            case DType::U16:  result = cast_impl<SrcT, uint16_t>(x); break;
            case DType::U32:  result = cast_impl<SrcT, uint32_t>(x); break;
            case DType::U64:  result = cast_impl<SrcT, uint64_t>(x); break;
            case DType::F32:  result = cast_impl<SrcT, float>(x); break;
            case DType::F64:  result = cast_impl<SrcT, double>(x); break;
            case DType::C32:  result = cast_impl<SrcT, std::complex<float>>(x); break;
            case DType::C64:  result = cast_impl<SrcT, std::complex<double>>(x); break;
            default: INS_THROW("cast: unsupported target type from int64");
            }
            break;
        }

                       // ========== From U16 ==========
        case DType::U16: {
            using SrcT = uint16_t;
            switch (target_dtype) {
            case DType::BOOL: result = cast_to_bool_impl<SrcT>(x); break;
            case DType::U8:   result = cast_impl<SrcT, uint8_t>(x); break;
            case DType::I8:   result = cast_impl<SrcT, int8_t>(x); break;
            case DType::I16:  result = cast_impl<SrcT, int16_t>(x); break;
            case DType::I32:  result = cast_impl<SrcT, int32_t>(x); break;
            case DType::I64:  result = cast_impl<SrcT, int64_t>(x); break;
            case DType::U32:  result = cast_impl<SrcT, uint32_t>(x); break;
            case DType::U64:  result = cast_impl<SrcT, uint64_t>(x); break;
            case DType::F32:  result = cast_impl<SrcT, float>(x); break;
            case DType::F64:  result = cast_impl<SrcT, double>(x); break;
            case DType::C32:  result = cast_impl<SrcT, std::complex<float>>(x); break;
            case DType::C64:  result = cast_impl<SrcT, std::complex<double>>(x); break;
            default: INS_THROW("cast: unsupported target type from uint16");
            }
            break;
        }

                       // ========== From U32 ==========
        case DType::U32: {
            using SrcT = uint32_t;
            switch (target_dtype) {
            case DType::BOOL: result = cast_to_bool_impl<SrcT>(x); break;
            case DType::U8:   result = cast_impl<SrcT, uint8_t>(x); break;
            case DType::I8:   result = cast_impl<SrcT, int8_t>(x); break;
            case DType::I16:  result = cast_impl<SrcT, int16_t>(x); break;
            case DType::I32:  result = cast_impl<SrcT, int32_t>(x); break;
            case DType::I64:  result = cast_impl<SrcT, int64_t>(x); break;
            case DType::U16:  result = cast_impl<SrcT, uint16_t>(x); break;
            case DType::U64:  result = cast_impl<SrcT, uint64_t>(x); break;
            case DType::F32:  result = cast_impl<SrcT, float>(x); break;
            case DType::F64:  result = cast_impl<SrcT, double>(x); break;
            case DType::C32:  result = cast_impl<SrcT, std::complex<float>>(x); break;
            case DType::C64:  result = cast_impl<SrcT, std::complex<double>>(x); break;
            default: INS_THROW("cast: unsupported target type from uint32");
            }
            break;
        }

                       // ========== From U64 ==========
        case DType::U64: {
            using SrcT = uint64_t;
            switch (target_dtype) {
            case DType::BOOL: result = cast_to_bool_impl<SrcT>(x); break;
            case DType::U8:   result = cast_impl<SrcT, uint8_t>(x); break;
            case DType::I8:   result = cast_impl<SrcT, int8_t>(x); break;
            case DType::I16:  result = cast_impl<SrcT, int16_t>(x); break;
            case DType::I32:  result = cast_impl<SrcT, int32_t>(x); break;
            case DType::I64:  result = cast_impl<SrcT, int64_t>(x); break;
            case DType::U16:  result = cast_impl<SrcT, uint16_t>(x); break;
            case DType::U32:  result = cast_impl<SrcT, uint32_t>(x); break;
            case DType::F32:  result = cast_impl<SrcT, float>(x); break;
            case DType::F64:  result = cast_impl<SrcT, double>(x); break;
            case DType::C32:  result = cast_impl<SrcT, std::complex<float>>(x); break;
            case DType::C64:  result = cast_impl<SrcT, std::complex<double>>(x); break;
            default: INS_THROW("cast: unsupported target type from uint64");
            }
            break;
        }

                       // ========== From F32 ==========
        case DType::F32: {
            using SrcT = float;
            switch (target_dtype) {
            case DType::BOOL: result = cast_to_bool_impl<SrcT>(x); break;
            case DType::U8:   result = cast_impl<SrcT, uint8_t>(x); break;
            case DType::I8:   result = cast_impl<SrcT, int8_t>(x); break;
            case DType::I16:  result = cast_impl<SrcT, int16_t>(x); break;
            case DType::I32:  result = cast_impl<SrcT, int32_t>(x); break;
            case DType::I64:  result = cast_impl<SrcT, int64_t>(x); break;
            case DType::U16:  result = cast_impl<SrcT, uint16_t>(x); break;
            case DType::U32:  result = cast_impl<SrcT, uint32_t>(x); break;
            case DType::U64:  result = cast_impl<SrcT, uint64_t>(x); break;
            case DType::F64:  result = cast_impl<SrcT, double>(x); break;
            case DType::C32:  result = cast_to_complex_impl<float>(x); break;
            case DType::C64:  result = cast_impl<SrcT, std::complex<double>>(x); break;
            default: INS_THROW("cast: unsupported target type from float32");
            }
            break;
        }

                       // ========== From F64 ==========
        case DType::F64: {
            using SrcT = double;
            switch (target_dtype) {
            case DType::BOOL: result = cast_to_bool_impl<SrcT>(x); break;
            case DType::U8:   result = cast_impl<SrcT, uint8_t>(x); break;
            case DType::I8:   result = cast_impl<SrcT, int8_t>(x); break;
            case DType::I16:  result = cast_impl<SrcT, int16_t>(x); break;
            case DType::I32:  result = cast_impl<SrcT, int32_t>(x); break;
            case DType::I64:  result = cast_impl<SrcT, int64_t>(x); break;
            case DType::U16:  result = cast_impl<SrcT, uint16_t>(x); break;
            case DType::U32:  result = cast_impl<SrcT, uint32_t>(x); break;
            case DType::U64:  result = cast_impl<SrcT, uint64_t>(x); break;
            case DType::F32:  result = cast_impl<SrcT, float>(x); break;
            case DType::C32:  result = cast_impl<SrcT, std::complex<float>>(x); break;
            case DType::C64:  result = cast_to_complex_impl<double>(x); break;
            default: INS_THROW("cast: unsupported target type from float64");
            }
            break;
        }

                       // ========== From C32 ==========
        case DType::C32: {
            using SrcT = std::complex<float>;
            switch (target_dtype) {
            case DType::BOOL: result = cast_to_bool_impl<SrcT>(x); break;
            case DType::F32:  result = cast_complex_to_float_impl<float>(x); break;
            case DType::C64:  result = cast_impl<SrcT, std::complex<double>>(x); break;
            default: INS_THROW("cast: unsupported target type from complex64, got target type: ", dtype_name(target_dtype));
            }
            break;
        }

                       // ========== From C64 ==========
        case DType::C64: {
            using SrcT = std::complex<double>;
            switch (target_dtype) {
            case DType::BOOL: result = cast_to_bool_impl<SrcT>(x); break;
            case DType::F64:  result = cast_complex_to_float_impl<double>(x); break;
            case DType::C32:  result = cast_impl<SrcT, std::complex<float>>(x); break;
            default: INS_THROW("cast: unsupported target type from complex128, got target type: ", dtype_name(target_dtype));
            }
            break;
        }

        default: INS_THROW("cast: unsupported source type");
        }

        return { result };
    }

    // Register cast kernel (source dtype is used for dispatch, target via argument)
    REGISTER_KERNEL(cast, CPU, BOOL, cast_kernel);
    REGISTER_KERNEL(cast, CPU, U8, cast_kernel);
    REGISTER_KERNEL(cast, CPU, I8, cast_kernel);
    REGISTER_KERNEL(cast, CPU, I16, cast_kernel);
    REGISTER_KERNEL(cast, CPU, I32, cast_kernel);
    REGISTER_KERNEL(cast, CPU, I64, cast_kernel);
    REGISTER_KERNEL(cast, CPU, U16, cast_kernel);
    REGISTER_KERNEL(cast, CPU, U32, cast_kernel);
    REGISTER_KERNEL(cast, CPU, U64, cast_kernel);
    REGISTER_KERNEL(cast, CPU, F32, cast_kernel);
    REGISTER_KERNEL(cast, CPU, F64, cast_kernel);
    REGISTER_KERNEL(cast, CPU, C32, cast_kernel);
    REGISTER_KERNEL(cast, CPU, C64, cast_kernel);

    REGISTER_MODULE(cast, CPU);

} // namespace ins::cpu