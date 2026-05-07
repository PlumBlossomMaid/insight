// backends/cpu/unary.cpp
#include "insight/plugin/op_registry.h"
#include "insight/core/array.h"
#include <cmath>
#include <complex>

namespace ins::cpu {

    // ============================================================================
    // Generic unary operation helper
    // ============================================================================

    template<typename T, typename Op>
    static Array unary_impl(const Array& x, Op op) {
        int64_t n = x.numel();
        int ndim = x.shape().ndim();

        std::vector<int64_t> dims(ndim);
        std::vector<int64_t> strides(ndim);
        for (int i = 0; i < ndim; ++i) {
            dims[i] = x.shape().dim(i);
            strides[i] = x.strides()[i];
        }

        Array out(x.shape(), x.dtype(), x.place());
        const T* x_data = x.data<T>();
        T* out_data = out.data<T>();

        #pragma omp parallel for
        for (int64_t linear = 0; linear < n; ++linear) {
            int64_t tmp = linear;
            int64_t indices[4] = { 0 };
            for (int d = ndim - 1; d >= 0; --d) {
                indices[d] = tmp % dims[d];
                tmp /= dims[d];
            }

            int64_t offset = 0;
            for (int d = 0; d < ndim; ++d) {
                offset += indices[d] * strides[d];
            }

            out_data[linear] = op(x_data[offset]);
        }

        return out;
    }

    // ============================================================================
    // Registration macro for all numeric types (abs, negative, square)
    // ============================================================================

#define REGISTER_UNARY_ALL_OP(op_name, func) \
    static OpArgs op_name##_kernel(const OpArgs& args) { \
        const Array& x = std::any_cast<const Array&>(args[0]); \
        DType dtype = x.dtype(); \
        Array result; \
        switch (dtype) { \
            case DType::U8:   result = unary_impl<uint8_t>(x, func<uint8_t>); break; \
            case DType::I8:   result = unary_impl<int8_t>(x, func<int8_t>); break; \
            case DType::I16:  result = unary_impl<int16_t>(x, func<int16_t>); break; \
            case DType::I32:  result = unary_impl<int32_t>(x, func<int32_t>); break; \
            case DType::I64:  result = unary_impl<int64_t>(x, func<int64_t>); break; \
            case DType::U16:  result = unary_impl<uint16_t>(x, func<uint16_t>); break; \
            case DType::U32:  result = unary_impl<uint32_t>(x, func<uint32_t>); break; \
            case DType::U64:  result = unary_impl<uint64_t>(x, func<uint64_t>); break; \
            case DType::F32:  result = unary_impl<float>(x, func<float>); break; \
            case DType::F64:  result = unary_impl<double>(x, func<double>); break; \
            case DType::C32:  result = unary_impl<std::complex<float>>(x, func<std::complex<float>>); break; \
            case DType::C64:  result = unary_impl<std::complex<double>>(x, func<std::complex<double>>); break; \
            case DType::BOOL:  result = unary_impl<bool>(x, func<bool>); break; \
            default: INS_THROW(#op_name ": unsupported dtype"); \
        } \
        return {result}; \
    } \
    REGISTER_KERNEL(op_name, CPU, U8,  op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, I8,  op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, I16, op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, I32, op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, I64, op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, U16, op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, U32, op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, U64, op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, F32, op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, F64, op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, C32, op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, C64, op_name##_kernel)

// ============================================================================
// Registration macro for floating-point only ops
// ============================================================================

#define REGISTER_UNARY_FLOAT_OP(op_name, func) \
    static OpArgs op_name##_kernel(const OpArgs& args) { \
        const Array& x = std::any_cast<const Array&>(args[0]); \
        DType dtype = x.dtype(); \
        Array result; \
        switch (dtype) { \
            case DType::F32: result = unary_impl<float>(x, [](float v) { return func(v); }); break; \
            case DType::F64: result = unary_impl<double>(x, [](double v) { return func(v); }); break; \
            default: INS_THROW(#op_name ": only float32 and float64 are supported"); \
        } \
        return {result}; \
    } \
    REGISTER_KERNEL(op_name, CPU, F32, op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, F64, op_name##_kernel)

// ============================================================================
// abs: absolute value
// ============================================================================

    template<typename T>
    T abs_func(T v) {
        if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
            std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>) {
            return v;
        }
        else if constexpr (std::is_same_v<T, std::complex<float>> ||
            std::is_same_v<T, std::complex<double>>) {
            return std::abs(v);
        }
        else {
            return v < 0 ? -v : v;
        }
    }

    template<>
    float abs_func<float>(float v) { return std::fabs(v); }

    template<>
    double abs_func<double>(double v) { return std::fabs(v); }

    REGISTER_UNARY_ALL_OP(abs, abs_func);

    // ============================================================================
    // negative: unary minus
    // ============================================================================

    template<typename T>
    T neg_func(T v) {
        if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
            std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>) {
            INS_THROW("negative: unsigned integers not supported");
        }
        else {
            return -v;
        }
    }

    REGISTER_UNARY_ALL_OP(negative, neg_func);

    // ============================================================================
    // square: x * x
    // ============================================================================

    template<typename T>
    T square_func(T v) {
        return v * v;
    }

    REGISTER_UNARY_ALL_OP(square, square_func);

    // ============================================================================
    // Floating-point math functions
    // ============================================================================

    REGISTER_UNARY_FLOAT_OP(exp, std::exp);
    REGISTER_UNARY_FLOAT_OP(exp2, std::exp2);
    REGISTER_UNARY_FLOAT_OP(expm1, std::expm1);
    REGISTER_UNARY_FLOAT_OP(log, std::log);
    REGISTER_UNARY_FLOAT_OP(log2, std::log2);
    REGISTER_UNARY_FLOAT_OP(log10, std::log10);
    REGISTER_UNARY_FLOAT_OP(log1p, std::log1p);
    REGISTER_UNARY_FLOAT_OP(sqrt, std::sqrt);
    REGISTER_UNARY_FLOAT_OP(cbrt, std::cbrt);
    namespace {
        template<typename T>
        T reciprocal_func(T v) {
            return T(1) / v;
        }
	}
    REGISTER_UNARY_FLOAT_OP(reciprocal, reciprocal_func);

    // ============================================================================
    // Trigonometric functions
    // ============================================================================

    REGISTER_UNARY_FLOAT_OP(sin, std::sin);
    REGISTER_UNARY_FLOAT_OP(cos, std::cos);
    REGISTER_UNARY_FLOAT_OP(tan, std::tan);
    REGISTER_UNARY_FLOAT_OP(asin, std::asin);
    REGISTER_UNARY_FLOAT_OP(acos, std::acos);
    REGISTER_UNARY_FLOAT_OP(atan, std::atan);
    REGISTER_UNARY_FLOAT_OP(sinh, std::sinh);
    REGISTER_UNARY_FLOAT_OP(cosh, std::cosh);
    REGISTER_UNARY_FLOAT_OP(tanh, std::tanh);
    REGISTER_UNARY_FLOAT_OP(asinh, std::asinh);
    REGISTER_UNARY_FLOAT_OP(acosh, std::acosh);
    REGISTER_UNARY_FLOAT_OP(atanh, std::atanh);

    // ============================================================================
    // Rounding functions
    // ============================================================================

    REGISTER_UNARY_FLOAT_OP(floor, std::floor);
    REGISTER_UNARY_FLOAT_OP(ceil, std::ceil);
    REGISTER_UNARY_FLOAT_OP(trunc, std::trunc);
    REGISTER_UNARY_FLOAT_OP(rint, std::rint);

    // ============================================================================
    // sign: sign function (-1, 0, 1)
    // ============================================================================

    static OpArgs sign_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        DType dtype = x.dtype();
        Array result;
        switch (dtype) {
            // Unsigned integers: always 0 or 1
        case DType::U8:
            result = unary_impl<uint8_t>(x, [](uint8_t v) { return v > 0 ? 1 : 0; });
            break;
        case DType::U16:
            result = unary_impl<uint16_t>(x, [](uint16_t v) { return v > 0 ? 1 : 0; });
            break;
        case DType::U32:
            result = unary_impl<uint32_t>(x, [](uint32_t v) { return v > 0 ? 1 : 0; });
            break;
        case DType::U64:
            result = unary_impl<uint64_t>(x, [](uint64_t v) { return v > 0 ? 1 : 0; });
            break;

            // Signed integers: -1, 0, 1
        case DType::I8:
            result = unary_impl<int8_t>(x, [](int8_t v) { return v > 0 ? 1 : (v < 0 ? -1 : 0); });
            break;
        case DType::I16:
            result = unary_impl<int16_t>(x, [](int16_t v) { return v > 0 ? 1 : (v < 0 ? -1 : 0); });
            break;
        case DType::I32:
            result = unary_impl<int32_t>(x, [](int32_t v) { return v > 0 ? 1 : (v < 0 ? -1 : 0); });
            break;
        case DType::I64:
            result = unary_impl<int64_t>(x, [](int64_t v) { return v > 0 ? 1 : (v < 0 ? -1 : 0); });
            break;

            // Floating point: -1.0, 0.0, 1.0
        case DType::F16:
            result = unary_impl<uint16_t>(x, [](uint16_t v) {
                // F16 storage as uint16_t, need to interpret as half
                // For now, treat based on storage value (simplified)
                return v > 0 ? static_cast<uint16_t>(1) : (v < 0 ? static_cast<uint16_t>(0x8000 | 1) : 0);
                });
            break;
        case DType::BF16:
            result = unary_impl<uint16_t>(x, [](uint16_t v) {
                // Similar to F16
                return v > 0 ? static_cast<uint16_t>(1) : (v < 0 ? static_cast<uint16_t>(0x8000 | 1) : 0);
                });
            break;
        case DType::F32:
            result = unary_impl<float>(x, [](float v) { return v > 0 ? 1.0f : (v < 0 ? -1.0f : 0.0f); });
            break;
        case DType::F64:
            result = unary_impl<double>(x, [](double v) { return v > 0 ? 1.0 : (v < 0 ? -1.0 : 0.0); });
            break;

            // Complex: return unit complex (cosθ + i·sinθ) or 0+0j
        case DType::C32:
            result = unary_impl<std::complex<float>>(x, [](std::complex<float> v) {
                float norm = std::abs(v);
                if (norm == 0) return std::complex<float>(0, 0);
                return v / norm;
                });
            break;
        case DType::C64:
            result = unary_impl<std::complex<double>>(x, [](std::complex<double> v) {
                double norm = std::abs(v);
                if (norm == 0) return std::complex<double>(0, 0);
                return v / norm;
                });
            break;

        case DType::BOOL:
            result = unary_impl<bool>(x, [](bool v) { return v ? 1 : 0; });
            break;

        default:
            INS_THROW("sign: unsupported dtype ", static_cast<int>(dtype));
        }
        return { result };
    }

    // Register for all supported types
    REGISTER_KERNEL(sign, CPU, BOOL, sign_kernel);
    REGISTER_KERNEL(sign, CPU, U8, sign_kernel);
    REGISTER_KERNEL(sign, CPU, U16, sign_kernel);
    REGISTER_KERNEL(sign, CPU, U32, sign_kernel);
    REGISTER_KERNEL(sign, CPU, U64, sign_kernel);
    REGISTER_KERNEL(sign, CPU, I8, sign_kernel);
    REGISTER_KERNEL(sign, CPU, I16, sign_kernel);
    REGISTER_KERNEL(sign, CPU, I32, sign_kernel);
    REGISTER_KERNEL(sign, CPU, I64, sign_kernel);
    REGISTER_KERNEL(sign, CPU, F16, sign_kernel);
    REGISTER_KERNEL(sign, CPU, BF16, sign_kernel);
    REGISTER_KERNEL(sign, CPU, F32, sign_kernel);
    REGISTER_KERNEL(sign, CPU, F64, sign_kernel);
    REGISTER_KERNEL(sign, CPU, C32, sign_kernel);
    REGISTER_KERNEL(sign, CPU, C64, sign_kernel);

    // ============================================================================
    // conjugate: complex conjugate
    // ============================================================================

    static OpArgs conj_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        DType dtype = x.dtype();
        Array result;
        switch (dtype) {
        case DType::C32: result = unary_impl<std::complex<float>>(x, [](std::complex<float> v) { return std::conj(v); }); break;
        case DType::C64: result = unary_impl<std::complex<double>>(x, [](std::complex<double> v) { return std::conj(v); }); break;
        default: result = x; break;
        }
        return { result };
    }
    REGISTER_KERNEL(conj, CPU, C32, conj_kernel);
    REGISTER_KERNEL(conj, CPU, C64, conj_kernel);

    // ============================================================================
    // deg2rad / rad2deg
    // ============================================================================

    static OpArgs deg2rad_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        DType dtype = x.dtype();
        Array result;
        const float pi_f = 3.141592653589793f / 180.0f;
        const double pi_d = 3.141592653589793 / 180.0;
        switch (dtype) {
        case DType::F32: result = unary_impl<float>(x, [pi_f](float v) { return v * pi_f; }); break;
        case DType::F64: result = unary_impl<double>(x, [pi_d](double v) { return v * pi_d; }); break;
        default: INS_THROW("deg2rad: only float32 and float64 are supported");
        }
        return { result };
    }
    REGISTER_KERNEL(deg2rad, CPU, F32, deg2rad_kernel);
    REGISTER_KERNEL(deg2rad, CPU, F64, deg2rad_kernel);

    static OpArgs rad2deg_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        DType dtype = x.dtype();
        Array result;
        const float r2d_f = 180.0f / 3.141592653589793f;
        const double r2d_d = 180.0 / 3.141592653589793;
        switch (dtype) {
        case DType::F32: result = unary_impl<float>(x, [r2d_f](float v) { return v * r2d_f; }); break;
        case DType::F64: result = unary_impl<double>(x, [r2d_d](double v) { return v * r2d_d; }); break;
        default: INS_THROW("rad2deg: only float32 and float64 are supported");
        }
        return { result };
    }
    REGISTER_KERNEL(rad2deg, CPU, F32, rad2deg_kernel);
    REGISTER_KERNEL(rad2deg, CPU, F64, rad2deg_kernel);

    // ============================================================================
    // logical_not: returns bool (true if input is zero)
    // ============================================================================

    template<typename T>
    static Array logical_not_impl(const Array& x) {
        int64_t n = x.numel();
        int ndim = x.shape().ndim();

        std::vector<int64_t> dims(ndim);
        std::vector<int64_t> strides(ndim);
        for (int i = 0; i < ndim; ++i) {
            dims[i] = x.shape().dim(i);
            strides[i] = x.strides()[i];
        }

        Array out(x.shape(), DType::BOOL, x.place());
        const T* x_data = x.data<T>();
        bool* out_data = out.data<bool>();

        #pragma omp parallel for
        for (int64_t linear = 0; linear < n; ++linear) {
            int64_t tmp = linear;
            int64_t indices[4] = { 0 };
            for (int d = ndim - 1; d >= 0; --d) {
                indices[d] = tmp % dims[d];
                tmp /= dims[d];
            }

            int64_t offset = 0;
            for (int d = 0; d < ndim; ++d) {
                offset += indices[d] * strides[d];
            }

            out_data[linear] = (x_data[offset] == T(0));
        }

        return out;
    }

    static OpArgs logical_not_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        DType dtype = x.dtype();
        Array result;
        switch (dtype) {
        case DType::BOOL: result = logical_not_impl<bool>(x); break;
        case DType::U8:   result = logical_not_impl<uint8_t>(x); break;
        case DType::I8:   result = logical_not_impl<int8_t>(x); break;
        case DType::I16:  result = logical_not_impl<int16_t>(x); break;
        case DType::I32:  result = logical_not_impl<int32_t>(x); break;
        case DType::I64:  result = logical_not_impl<int64_t>(x); break;
        case DType::U16:  result = logical_not_impl<uint16_t>(x); break;
        case DType::U32:  result = logical_not_impl<uint32_t>(x); break;
        case DType::U64:  result = logical_not_impl<uint64_t>(x); break;
        case DType::F32:  result = logical_not_impl<float>(x); break;
        case DType::F64:  result = logical_not_impl<double>(x); break;
        case DType::C32:  result = logical_not_impl<std::complex<float>>(x); break;
        case DType::C64:  result = logical_not_impl<std::complex<double>>(x); break;
        default: INS_THROW("logical_not: unsupported dtype");
        }
        return { result };
    }

    // Register all dtypes for logical_not
    REGISTER_KERNEL(logical_not, CPU, BOOL, logical_not_kernel);
    REGISTER_KERNEL(logical_not, CPU, U8, logical_not_kernel);
    REGISTER_KERNEL(logical_not, CPU, I8, logical_not_kernel);
    REGISTER_KERNEL(logical_not, CPU, I16, logical_not_kernel);
    REGISTER_KERNEL(logical_not, CPU, I32, logical_not_kernel);
    REGISTER_KERNEL(logical_not, CPU, I64, logical_not_kernel);
    REGISTER_KERNEL(logical_not, CPU, U16, logical_not_kernel);
    REGISTER_KERNEL(logical_not, CPU, U32, logical_not_kernel);
    REGISTER_KERNEL(logical_not, CPU, U64, logical_not_kernel);
    REGISTER_KERNEL(logical_not, CPU, F32, logical_not_kernel);
    REGISTER_KERNEL(logical_not, CPU, F64, logical_not_kernel);
    REGISTER_KERNEL(logical_not, CPU, C32, logical_not_kernel);
    REGISTER_KERNEL(logical_not, CPU, C64, logical_not_kernel);

    // ============================================================================
    // bitwise_not
    // ============================================================================

#define REGISTER_BITWISE_NOT_KERNEL(dtype, T) \
    static OpArgs bitwise_not_##dtype##_kernel(const OpArgs& args) { \
        const Array& x = std::any_cast<const Array&>(args[0]); \
        Array result = unary_impl<T>(x, [](T v) { return ~v; }); \
        return {result}; \
    } \
    REGISTER_KERNEL(bitwise_not, CPU, dtype, bitwise_not_##dtype##_kernel)

    REGISTER_BITWISE_NOT_KERNEL(U8, uint8_t);
    REGISTER_BITWISE_NOT_KERNEL(I8, int8_t);
    REGISTER_BITWISE_NOT_KERNEL(I16, int16_t);
    REGISTER_BITWISE_NOT_KERNEL(I32, int32_t);
    REGISTER_BITWISE_NOT_KERNEL(I64, int64_t);
    REGISTER_BITWISE_NOT_KERNEL(U16, uint16_t);
    REGISTER_BITWISE_NOT_KERNEL(U32, uint32_t);
    REGISTER_BITWISE_NOT_KERNEL(U64, uint64_t);
	REGISTER_KERNEL(bitwise_not, CPU, BOOL, logical_not_kernel); // align with NumPy behavior: ~True = False, ~False = True

#undef REGISTER_BITWISE_NOT_KERNEL

    // ============================================================================
    // Module registration
    // ============================================================================

    REGISTER_MODULE(unary, CPU);

} // namespace ins::cpu