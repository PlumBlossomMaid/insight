// backends/cpu/elementwise.cpp
#include "insight/plugin/op_registry.h"
#include "insight/core/array.h"
#include <complex>
#include <cmath>
#include <iostream>

namespace ins::cpu {

    // ============================================================================
    // Generic elementwise operation helper
    // ============================================================================

    // Comparison helper: returns bool array
    template<typename T, typename Op>
    static Array compare_impl(const Array& a, const Array& b, Op op) {
        if (a.shape().ndim() == 0 && b.shape().ndim() == 0) {
            Array out(Shape({}), DType::BOOL, a.place());
            const T* a_data = a.data<T>();
            const T* b_data = b.data<T>();
            *out.data<bool>() = op(*a_data, *b_data);
            return out;
        }

        if (a.shape().ndim() == 0) {
            Array a_broadcasted(b.shape(), a.dtype(), a.place());
            T* a_bc_data = a_broadcasted.data<T>();
            T a_val = *a.data<T>();
#pragma omp parallel for
            for (int64_t i = 0; i < a_broadcasted.numel(); ++i) {
                a_bc_data[i] = a_val;
            }
            return compare_impl<T>(a_broadcasted, b, op);
        }

        if (b.shape().ndim() == 0) {
            Array b_broadcasted(a.shape(), b.dtype(), b.place());
            T* b_bc_data = b_broadcasted.data<T>();
            T b_val = *b.data<T>();
#pragma omp parallel for
            for (int64_t i = 0; i < b_broadcasted.numel(); ++i) {
                b_bc_data[i] = b_val;
            }
            return compare_impl<T>(a, b_broadcasted, op);
        }

        int64_t n = a.numel();
        int ndim = a.shape().ndim();

        std::vector<int64_t> dims(ndim);
        std::vector<int64_t> a_strides(ndim);
        std::vector<int64_t> b_strides(ndim);
        std::vector<int64_t> out_strides(ndim);

        for (int i = 0; i < ndim; ++i) {
            dims[i] = a.shape().dim(i);
            a_strides[i] = a.strides()[i];
            b_strides[i] = b.strides()[i];
        }

        // Output is contiguous
        out_strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            out_strides[i] = out_strides[i + 1] * dims[i + 1];
        }

        Array out(a.shape(), DType::BOOL, a.place());
        const T* a_data = a.data<T>();
        const T* b_data = b.data<T>();
        bool* out_data = out.data<bool>();

#pragma omp parallel for
        for (int64_t linear = 0; linear < n; ++linear) {
            int64_t tmp = linear;
            std::vector<int64_t> indices(ndim);
            for (int d = ndim - 1; d >= 0; --d) {
                indices[d] = tmp % dims[d];
                tmp /= dims[d];
            }

            int64_t a_offset = 0, b_offset = 0, out_offset = 0;
            for (int d = 0; d < ndim; ++d) {
                a_offset += indices[d] * a_strides[d];
                b_offset += indices[d] * b_strides[d];
                out_offset += indices[d] * out_strides[d];
            }

            out_data[out_offset] = op(a_data[a_offset], b_data[b_offset]);
        }

        return out;
    }

    template<typename T, typename Op>
    static Array elementwise_impl(const Array& a, const Array& b, Op op) {
        int64_t n = a.numel();
        int ndim = a.shape().ndim();

        // Special case: scalar (0-dimension)
        if (ndim == 0) {
            Array out(a.shape(), a.dtype(), a.place());
            const T* a_data = a.data<T>();
            const T* b_data = b.data<T>();
            T* out_data = out.data<T>();
            out_data[0] = op(a_data[0], b_data[0]);
            return out;
        }

        int64_t dims[INSIGHT_MAX_NDIM] = { 0 };
        int64_t a_strides[INSIGHT_MAX_NDIM] = { 0 };
        int64_t b_strides[INSIGHT_MAX_NDIM] = { 0 };
        int64_t out_strides[INSIGHT_MAX_NDIM] = { 0 };

        for (int i = 0; i < ndim; ++i) {
            dims[i] = a.shape().dim(i);
            a_strides[i] = a.strides()[i];
            b_strides[i] = b.strides()[i];
        }

        out_strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            out_strides[i] = out_strides[i + 1] * dims[i + 1];
        }

        Array out(a.shape(), a.dtype(), a.place());
        const T* a_data = a.data<T>();
        const T* b_data = b.data<T>();
        T* out_data = out.data<T>();

#pragma omp parallel for
        for (int64_t linear = 0; linear < n; ++linear) {
            int64_t tmp = linear;
            int64_t indices[INSIGHT_MAX_NDIM] = { 0 };
            for (int d = ndim - 1; d >= 0; --d) {
                indices[d] = tmp % dims[d];
                tmp /= dims[d];
            }

            int64_t a_offset = 0, b_offset = 0, out_offset = 0;
            for (int d = 0; d < ndim; ++d) {
                a_offset += indices[d] * a_strides[d];
                b_offset += indices[d] * b_strides[d];
                out_offset += indices[d] * out_strides[d];
            }

            out_data[out_offset] = op(a_data[a_offset], b_data[b_offset]);
        }

        return out;
    }

    // ============================================================================
    // Comparison helpers for complex numbers (only compare real part)
    // ============================================================================

    template<typename T>
    static bool complex_equal(const std::complex<T>& a, const std::complex<T>& b) {
        return a.real() == b.real() && a.imag() == b.imag();
    }

    template<typename T>
    static bool complex_not_equal(const std::complex<T>& a, const std::complex<T>& b) {
        return !complex_equal(a, b);
    }

    template<typename T>
    static bool complex_greater(const std::complex<T>& a, const std::complex<T>& b) {
        return a.real() > b.real();
    }

    template<typename T>
    static bool complex_less(const std::complex<T>& a, const std::complex<T>& b) {
        return a.real() < b.real();
    }

    template<typename T>
    static bool complex_greater_equal(const std::complex<T>& a, const std::complex<T>& b) {
        return a.real() >= b.real();
    }

    template<typename T>
    static bool complex_less_equal(const std::complex<T>& a, const std::complex<T>& b) {
        return a.real() <= b.real();
    }

    // ============================================================================
    // Kernel registration macro for arithmetic operations
    // ============================================================================

#define REGISTER_ARITH_KERNEL(op_name, op) \
    static OpArgs op_name##_kernel(const OpArgs& args) { \
        const Array& a = std::any_cast<const Array&>(args[0]); \
        const Array& b = std::any_cast<const Array&>(args[1]); \
        DType dtype = a.dtype(); \
        Array result; \
        switch (dtype) { \
            case DType::BOOL: result = elementwise_impl<bool>(a, b, op<bool>()); break; \
            case DType::U8:   result = elementwise_impl<uint8_t>(a, b, op<uint8_t>()); break; \
            case DType::I8:   result = elementwise_impl<int8_t>(a, b, op<int8_t>()); break; \
            case DType::I16:  result = elementwise_impl<int16_t>(a, b, op<int16_t>()); break; \
            case DType::I32:  result = elementwise_impl<int32_t>(a, b, op<int32_t>()); break; \
            case DType::I64:  result = elementwise_impl<int64_t>(a, b, op<int64_t>()); break; \
            case DType::U16:  result = elementwise_impl<uint16_t>(a, b, op<uint16_t>()); break; \
            case DType::U32:  result = elementwise_impl<uint32_t>(a, b, op<uint32_t>()); break; \
            case DType::U64:  result = elementwise_impl<uint64_t>(a, b, op<uint64_t>()); break; \
            case DType::F32:  result = elementwise_impl<float>(a, b, op<float>()); break; \
            case DType::F64:  result = elementwise_impl<double>(a, b, op<double>()); break; \
            case DType::C32:  result = elementwise_impl<std::complex<float>>(a, b, op<std::complex<float>>()); break; \
            case DType::C64:  result = elementwise_impl<std::complex<double>>(a, b, op<std::complex<double>>()); break; \
            default: INS_THROW(#op_name ": unsupported dtype"); \
        } \
        return {result}; \
    } \
    REGISTER_KERNEL(op_name, CPU, BOOL, op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, U8,   op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, I8,   op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, I16,  op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, I32,  op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, I64,  op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, U16,  op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, U32,  op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, U64,  op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, F32,  op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, F64,  op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, C32,  op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, C64,  op_name##_kernel)

// ============================================================================
// Arithmetic operations
// ============================================================================

    REGISTER_ARITH_KERNEL(add, std::plus);
    REGISTER_ARITH_KERNEL(sub, std::minus);
    REGISTER_ARITH_KERNEL(mul, std::multiplies);
    REGISTER_ARITH_KERNEL(div, std::divides);

    // ============================================================================
    // Power operation (float only)
    // ============================================================================

    static OpArgs pow_kernel(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        DType dtype = a.dtype();
        Array result;
        switch (dtype) {
        case DType::F32: result = elementwise_impl<float>(a, b, [](float x, float y) { return std::pow(x, y); }); break;
        case DType::F64: result = elementwise_impl<double>(a, b, [](double x, double y) { return std::pow(x, y); }); break;
        default: INS_THROW("pow: only float32 and float64 are supported");
        }
        return { result };
    }
    REGISTER_KERNEL(pow, CPU, F32, pow_kernel);
    REGISTER_KERNEL(pow, CPU, F64, pow_kernel);

    // ============================================================================
    // Modulo operation (integer: %, float: fmod)
    // ============================================================================

    static OpArgs mod_kernel(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        DType dtype = a.dtype();
        Array result;
        switch (dtype) {
        case DType::U8:   result = elementwise_impl<uint8_t>(a, b, [](uint8_t x, uint8_t y) { return x % y; }); break;
        case DType::I8:   result = elementwise_impl<int8_t>(a, b, [](int8_t x, int8_t y) { return x % y; }); break;
        case DType::I16:  result = elementwise_impl<int16_t>(a, b, [](int16_t x, int16_t y) { return x % y; }); break;
        case DType::I32:  result = elementwise_impl<int32_t>(a, b, [](int32_t x, int32_t y) { return x % y; }); break;
        case DType::I64:  result = elementwise_impl<int64_t>(a, b, [](int64_t x, int64_t y) { return x % y; }); break;
        case DType::U16:  result = elementwise_impl<uint16_t>(a, b, [](uint16_t x, uint16_t y) { return x % y; }); break;
        case DType::U32:  result = elementwise_impl<uint32_t>(a, b, [](uint32_t x, uint32_t y) { return x % y; }); break;
        case DType::U64:  result = elementwise_impl<uint64_t>(a, b, [](uint64_t x, uint64_t y) { return x % y; }); break;
        case DType::F32:  result = elementwise_impl<float>(a, b, [](float x, float y) { return std::fmod(x, y); }); break;
        case DType::F64:  result = elementwise_impl<double>(a, b, [](double x, double y) { return std::fmod(x, y); }); break;
        default: INS_THROW("mod: unsupported dtype");
        }
        return { result };
    }
    REGISTER_KERNEL(mod, CPU, U8, mod_kernel);
    REGISTER_KERNEL(mod, CPU, I8, mod_kernel);
    REGISTER_KERNEL(mod, CPU, I16, mod_kernel);
    REGISTER_KERNEL(mod, CPU, I32, mod_kernel);
    REGISTER_KERNEL(mod, CPU, I64, mod_kernel);
    REGISTER_KERNEL(mod, CPU, U16, mod_kernel);
    REGISTER_KERNEL(mod, CPU, U32, mod_kernel);
    REGISTER_KERNEL(mod, CPU, U64, mod_kernel);
    REGISTER_KERNEL(mod, CPU, F32, mod_kernel);
    REGISTER_KERNEL(mod, CPU, F64, mod_kernel);

    // ============================================================================
    // Comparison operations (complex uses special helpers)
    // ============================================================================

#define REGISTER_CMP_KERNEL_COMPLEX(op_name, fp_op, complex_op) \
    static OpArgs op_name##_kernel(const OpArgs& args) { \
        const Array& a = std::any_cast<const Array&>(args[0]); \
        const Array& b = std::any_cast<const Array&>(args[1]); \
        DType dtype = a.dtype(); \
        Array result; \
        switch (dtype) { \
            case DType::BOOL: result = compare_impl<bool>(a, b, fp_op<bool>()); break; \
            case DType::U8:   result = compare_impl<uint8_t>(a, b, fp_op<uint8_t>()); break; \
            case DType::I8:   result = compare_impl<int8_t>(a, b, fp_op<int8_t>()); break; \
            case DType::I16:  result = compare_impl<int16_t>(a, b, fp_op<int16_t>()); break; \
            case DType::I32:  result = compare_impl<int32_t>(a, b, fp_op<int32_t>()); break; \
            case DType::I64:  result = compare_impl<int64_t>(a, b, fp_op<int64_t>()); break; \
            case DType::U16:  result = compare_impl<uint16_t>(a, b, fp_op<uint16_t>()); break; \
            case DType::U32:  result = compare_impl<uint32_t>(a, b, fp_op<uint32_t>()); break; \
            case DType::U64:  result = compare_impl<uint64_t>(a, b, fp_op<uint64_t>()); break; \
            case DType::F32:  result = compare_impl<float>(a, b, fp_op<float>()); break; \
            case DType::F64:  result = compare_impl<double>(a, b, fp_op<double>()); break; \
            case DType::C32:  result = compare_impl<std::complex<float>>(a, b, complex_op<float>); break; \
            case DType::C64:  result = compare_impl<std::complex<double>>(a, b, complex_op<double>); break; \
            default: INS_THROW(#op_name ": unsupported dtype"); \
        } \
        return {result}; \
    } \
    REGISTER_KERNEL(op_name, CPU, BOOL, op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, U8,   op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, I8,   op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, I16,  op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, I32,  op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, I64,  op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, U16,  op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, U32,  op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, U64,  op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, F32,  op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, F64,  op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, C32,  op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, C64,  op_name##_kernel)

// equal and not_equal compare both real and imaginary parts
    REGISTER_CMP_KERNEL_COMPLEX(equal, std::equal_to, complex_equal);
    REGISTER_CMP_KERNEL_COMPLEX(not_equal, std::not_equal_to, complex_not_equal);

    // greater, less, etc. only compare real part
    REGISTER_CMP_KERNEL_COMPLEX(greater, std::greater, complex_greater);
    REGISTER_CMP_KERNEL_COMPLEX(less, std::less, complex_less);
    REGISTER_CMP_KERNEL_COMPLEX(greater_equal, std::greater_equal, complex_greater_equal);
    REGISTER_CMP_KERNEL_COMPLEX(less_equal, std::less_equal, complex_less_equal);

    // ============================================================================
    // Logical operations (bool only)
    // ============================================================================

#define REGISTER_LOGICAL_KERNEL(op_name, op) \
    static OpArgs op_name##_kernel(const OpArgs& args) { \
        const Array& a = std::any_cast<const Array&>(args[0]); \
        const Array& b = std::any_cast<const Array&>(args[1]); \
        Array result; \
        result = elementwise_impl<bool>(a, b, op<bool>()); \
        return {result}; \
    } \
    REGISTER_KERNEL(op_name, CPU, BOOL, op_name##_kernel)

    REGISTER_LOGICAL_KERNEL(logical_and, std::logical_and);
    REGISTER_LOGICAL_KERNEL(logical_or, std::logical_or);
    REGISTER_LOGICAL_KERNEL(logical_xor, std::bit_xor);

    // ============================================================================
    // Bitwise operations (integer types only)
    // ============================================================================

#define REGISTER_BITWISE_KERNEL(op_name, op) \
    static OpArgs op_name##_kernel(const OpArgs& args) { \
        const Array& a = std::any_cast<const Array&>(args[0]); \
        const Array& b = std::any_cast<const Array&>(args[1]); \
        DType dtype = a.dtype(); \
        Array result; \
        switch (dtype) { \
            case DType::U8:   result = elementwise_impl<uint8_t>(a, b, op<uint8_t>()); break; \
            case DType::I8:   result = elementwise_impl<int8_t>(a, b, op<int8_t>()); break; \
            case DType::I16:  result = elementwise_impl<int16_t>(a, b, op<int16_t>()); break; \
            case DType::I32:  result = elementwise_impl<int32_t>(a, b, op<int32_t>()); break; \
            case DType::I64:  result = elementwise_impl<int64_t>(a, b, op<int64_t>()); break; \
            default: INS_THROW(#op_name ": only integer types are supported"); \
        } \
        return {result}; \
    } \
    REGISTER_KERNEL(op_name, CPU, U8,  op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, I8,  op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, I16, op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, I32, op_name##_kernel); \
    REGISTER_KERNEL(op_name, CPU, I64, op_name##_kernel)

    REGISTER_BITWISE_KERNEL(bitwise_and, std::bit_and);
    REGISTER_BITWISE_KERNEL(bitwise_or, std::bit_or);
    REGISTER_BITWISE_KERNEL(bitwise_xor, std::bit_xor);

	// align with NumPy: for bool, bitwise_and/or/xor are the same as logical_and/or/xor
    REGISTER_KERNEL(bitwise_and, CPU, BOOL, logical_and_kernel);
    REGISTER_KERNEL(bitwise_or, CPU, BOOL, logical_or_kernel);
    REGISTER_KERNEL(bitwise_xor, CPU, BOOL, logical_xor_kernel);

    static OpArgs left_shift_kernel(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        DType dtype = a.dtype();
        Array result;
        switch (dtype) {
        case DType::U8:   result = elementwise_impl<uint8_t>(a, b, [](uint8_t x, uint8_t y) { return x << y; }); break;
        case DType::I8:   result = elementwise_impl<int8_t>(a, b, [](int8_t x, int8_t y) { return x << y; }); break;
        case DType::I16:  result = elementwise_impl<int16_t>(a, b, [](int16_t x, int16_t y) { return x << y; }); break;
        case DType::I32:  result = elementwise_impl<int32_t>(a, b, [](int32_t x, int32_t y) { return x << y; }); break;
        case DType::I64:  result = elementwise_impl<int64_t>(a, b, [](int64_t x, int64_t y) { return x << y; }); break;
        default: INS_THROW("bitwise_left_shift: only integer types are supported");
        }
        return { result };
    }
    REGISTER_KERNEL(bitwise_left_shift, CPU, U8, left_shift_kernel);
    REGISTER_KERNEL(bitwise_left_shift, CPU, I8, left_shift_kernel);
    REGISTER_KERNEL(bitwise_left_shift, CPU, I16, left_shift_kernel);
    REGISTER_KERNEL(bitwise_left_shift, CPU, I32, left_shift_kernel);
    REGISTER_KERNEL(bitwise_left_shift, CPU, I64, left_shift_kernel);

    static OpArgs right_shift_kernel(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        DType dtype = a.dtype();
        Array result;
        switch (dtype) {
        case DType::U8:   result = elementwise_impl<uint8_t>(a, b, [](uint8_t x, uint8_t y) { return x >> y; }); break;
        case DType::I8:   result = elementwise_impl<int8_t>(a, b, [](int8_t x, int8_t y) { return x >> y; }); break;
        case DType::I16:  result = elementwise_impl<int16_t>(a, b, [](int16_t x, int16_t y) { return x >> y; }); break;
        case DType::I32:  result = elementwise_impl<int32_t>(a, b, [](int32_t x, int32_t y) { return x >> y; }); break;
        case DType::I64:  result = elementwise_impl<int64_t>(a, b, [](int64_t x, int64_t y) { return x >> y; }); break;
        default: INS_THROW("bitwise_right_shift: only integer types are supported");
        }
        return { result };
    }
    REGISTER_KERNEL(bitwise_right_shift, CPU, U8, right_shift_kernel);
    REGISTER_KERNEL(bitwise_right_shift, CPU, I8, right_shift_kernel);
    REGISTER_KERNEL(bitwise_right_shift, CPU, I16, right_shift_kernel);
    REGISTER_KERNEL(bitwise_right_shift, CPU, I32, right_shift_kernel);
    REGISTER_KERNEL(bitwise_right_shift, CPU, I64, right_shift_kernel);

    // ============================================================================
    // Maximum / Minimum operations
    // ============================================================================

    namespace inner_patch {

        template<typename T>
        struct maximum {
            T operator()(T x, T y) const {
                if constexpr (std::is_same_v<T, std::complex<float>> ||
                    std::is_same_v<T, std::complex<double>>) {
                    // For complex numbers, compare real parts only (aligns with NumPy)
                    return (x.real() > y.real()) ? x : y;
                }
                else {
                    return std::max(x, y);
                }
            }
        };

        template<typename T>
        struct minimum {
            T operator()(T x, T y) const {
                if constexpr (std::is_same_v<T, std::complex<float>> ||
                    std::is_same_v<T, std::complex<double>>) {
                    // For complex numbers, compare real parts only (aligns with NumPy)
                    return (x.real() < y.real()) ? x : y;
                }
                else {
                    return std::min(x, y);
                }
            }
        };

    } // namespace inner_patch

    REGISTER_ARITH_KERNEL(maximum, inner_patch::maximum);
    REGISTER_ARITH_KERNEL(minimum, inner_patch::minimum);

    // ============================================================================
    // Module registration
    // ============================================================================

    REGISTER_MODULE(elementwise, CPU);

} // namespace ins::cpu