// backends/cpu/reduction.cpp
#include "insight/plugin/op_registry.h"
#include "insight/core/array.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>
#include <limits>
#include <iostream>
namespace ins::cpu {

    // ========== Helper: Check NaN ==========
    template<typename T>
    static inline bool is_nan(T val) {
        return false;
    }

    template<>
    inline bool is_nan<float>(float val) {
        return std::isnan(val);
    }

    template<>
    inline bool is_nan<double>(double val) {
        return std::isnan(val);
    }

    // ========== sum ==========
    template<typename T>
    static Array sum_impl(const Array& out, const Array& prepared, int64_t batch_size, int64_t reduce_size) {
        T* dst = const_cast<T*>(out.data<T>());
        const T* src = prepared.data<T>();
        int64_t total_out = out.numel();

        #pragma omp parallel for
        for (int64_t i = 0; i < total_out; ++i) {
            T sum = T(0);
            for (int64_t j = 0; j < reduce_size; ++j) {
                sum += src[i * reduce_size + j];
            }
            dst[i] = sum;
        }

        return out;
    }

    static OpArgs sum_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);
        int64_t batch_size = std::any_cast<int64_t>(args[2]);
        int64_t reduce_size = std::any_cast<int64_t>(args[3]);

        switch (out.dtype()) {
        case DType::F32: return { sum_impl<float>(out, prepared, batch_size, reduce_size) };
        case DType::F64: return { sum_impl<double>(out, prepared, batch_size, reduce_size) };
        case DType::I32: return { sum_impl<int32_t>(out, prepared, batch_size, reduce_size) };
        case DType::I64: return { sum_impl<int64_t>(out, prepared, batch_size, reduce_size) };
        case DType::U8:  return { sum_impl<uint8_t>(out, prepared, batch_size, reduce_size) };
        default: INS_THROW("sum: unsupported dtype");
        }
    }

    REGISTER_KERNEL(sum, CPU, F32, sum_kernel);
    REGISTER_KERNEL(sum, CPU, F64, sum_kernel);
    REGISTER_KERNEL(sum, CPU, I32, sum_kernel);
    REGISTER_KERNEL(sum, CPU, I64, sum_kernel);
    REGISTER_KERNEL(sum, CPU, U8, sum_kernel);

    // ========== max ==========
    template<typename T>
    static Array max_impl(const Array& out, const Array& prepared, int64_t batch_size, int64_t reduce_size) {
        T* dst = const_cast<T*>(out.data<T>());
        const T* src = prepared.data<T>();
        int64_t total_out = out.numel();

#pragma omp parallel for
        for (int64_t i = 0; i < total_out; ++i) {
            T max_val = src[i * reduce_size];
            for (int64_t j = 1; j < reduce_size; ++j) {
                T val = src[i * reduce_size + j];
                if (val > max_val) max_val = val;
            }
            dst[i] = max_val;
        }
        return out;
    }

    static OpArgs max_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);
        int64_t batch_size = std::any_cast<int64_t>(args[2]);
        int64_t reduce_size = std::any_cast<int64_t>(args[3]);

        switch (out.dtype()) {
        case DType::F32: return { max_impl<float>(out, prepared, batch_size, reduce_size) };
        case DType::F64: return { max_impl<double>(out, prepared, batch_size, reduce_size) };
        case DType::I32: return { max_impl<int32_t>(out, prepared, batch_size, reduce_size) };
        case DType::I64: return { max_impl<int64_t>(out, prepared, batch_size, reduce_size) };
        case DType::U8:  return { max_impl<uint8_t>(out, prepared, batch_size, reduce_size) };
        default: INS_THROW("max: unsupported dtype");
        }
    }

    REGISTER_KERNEL(max, CPU, F32, max_kernel);
    REGISTER_KERNEL(max, CPU, F64, max_kernel);
    REGISTER_KERNEL(max, CPU, I32, max_kernel);
    REGISTER_KERNEL(max, CPU, I64, max_kernel);
    REGISTER_KERNEL(max, CPU, U8, max_kernel);

    // ========== min ==========
    template<typename T>
    static Array min_impl(const Array& out, const Array& prepared, int64_t batch_size, int64_t reduce_size) {
        T* dst = const_cast<T*>(out.data<T>());
        const T* src = prepared.data<T>();
        int64_t total_out = out.numel();

#pragma omp parallel for
        for (int64_t i = 0; i < total_out; ++i) {
            T min_val = src[i * reduce_size];
            for (int64_t j = 1; j < reduce_size; ++j) {
                T val = src[i * reduce_size + j];
                if (val < min_val) min_val = val;
            }
            dst[i] = min_val;
        }
        return out;
    }

    static OpArgs min_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);
        int64_t batch_size = std::any_cast<int64_t>(args[2]);
        int64_t reduce_size = std::any_cast<int64_t>(args[3]);

        switch (out.dtype()) {
        case DType::F32: return { min_impl<float>(out, prepared, batch_size, reduce_size) };
        case DType::F64: return { min_impl<double>(out, prepared, batch_size, reduce_size) };
        case DType::I32: return { min_impl<int32_t>(out, prepared, batch_size, reduce_size) };
        case DType::I64: return { min_impl<int64_t>(out, prepared, batch_size, reduce_size) };
        case DType::U8:  return { min_impl<uint8_t>(out, prepared, batch_size, reduce_size) };
        default: INS_THROW("min: unsupported dtype");
        }
    }

    REGISTER_KERNEL(min, CPU, F32, min_kernel);
    REGISTER_KERNEL(min, CPU, F64, min_kernel);
    REGISTER_KERNEL(min, CPU, I32, min_kernel);
    REGISTER_KERNEL(min, CPU, I64, min_kernel);
    REGISTER_KERNEL(min, CPU, U8, min_kernel);

    // ========== prod ==========
    template<typename T>
    static Array prod_impl(const Array& out, const Array& prepared, int64_t batch_size, int64_t reduce_size) {
        T* dst = const_cast<T*>(out.data<T>());
        const T* src = prepared.data<T>();
        int64_t total_out = out.numel();

#pragma omp parallel for
        for (int64_t i = 0; i < total_out; ++i) {
            T prod = T(1);
            for (int64_t j = 0; j < reduce_size; ++j) {
                prod *= src[i * reduce_size + j];
            }
            dst[i] = prod;
        }
        return out;
    }

    static OpArgs prod_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);
        int64_t batch_size = std::any_cast<int64_t>(args[2]);
        int64_t reduce_size = std::any_cast<int64_t>(args[3]);

        switch (out.dtype()) {
        case DType::F32: return { prod_impl<float>(out, prepared, batch_size, reduce_size) };
        case DType::F64: return { prod_impl<double>(out, prepared, batch_size, reduce_size) };
        case DType::I32: return { prod_impl<int32_t>(out, prepared, batch_size, reduce_size) };
        case DType::I64: return { prod_impl<int64_t>(out, prepared, batch_size, reduce_size) };
        case DType::U8:  return { prod_impl<uint8_t>(out, prepared, batch_size, reduce_size) };
        default: INS_THROW("prod: unsupported dtype");
        }
    }

    REGISTER_KERNEL(prod, CPU, F32, prod_kernel);
    REGISTER_KERNEL(prod, CPU, F64, prod_kernel);
    REGISTER_KERNEL(prod, CPU, I32, prod_kernel);
    REGISTER_KERNEL(prod, CPU, I64, prod_kernel);
    REGISTER_KERNEL(prod, CPU, U8, prod_kernel);

    // ========== any ==========
    template<typename T>
    static Array any_impl(const Array& out, const Array& prepared, int64_t batch_size, int64_t reduce_size) {
        bool* dst = const_cast<bool*>(out.data<bool>());
        const T* src = prepared.data<T>();
        int64_t total_out = out.numel();

#pragma omp parallel for
        for (int64_t i = 0; i < total_out; ++i) {
            bool any = false;
            for (int64_t j = 0; j < reduce_size; ++j) {
                if (src[i * reduce_size + j] != T(0)) {
                    any = true;
                    break;
                }
            }
            dst[i] = any;
        }
        return out;
    }

    static OpArgs any_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);
        int64_t batch_size = std::any_cast<int64_t>(args[2]);
        int64_t reduce_size = std::any_cast<int64_t>(args[3]);
        DType src_dtype = prepared.dtype();

        switch (src_dtype) {
        case DType::F32 : return { any_impl<float   >(out, prepared, batch_size, reduce_size) };
        case DType::F64 : return { any_impl<double  >(out, prepared, batch_size, reduce_size) };
        case DType::I8  : return { any_impl<int8_t  >(out, prepared, batch_size, reduce_size) };
        case DType::I16 : return { any_impl<int8_t  >(out, prepared, batch_size, reduce_size) };
        case DType::I32 : return { any_impl<int32_t >(out, prepared, batch_size, reduce_size) };
        case DType::I64 : return { any_impl<int64_t >(out, prepared, batch_size, reduce_size) };
        case DType::U8  : return { any_impl<uint8_t >(out, prepared, batch_size, reduce_size) };
        case DType::U16 : return { any_impl<uint16_t>(out, prepared, batch_size, reduce_size) };
        case DType::U32 : return { any_impl<uint32_t>(out, prepared, batch_size, reduce_size) };
        case DType::U64 : return { any_impl<uint64_t>(out, prepared, batch_size, reduce_size) };
        case DType::BOOL: return { any_impl<bool    >(out, prepared, batch_size, reduce_size) };
        default: INS_THROW("any: unsupported dtype, got dtype: ", dtype_name(prepared.dtype()));
        }
    }

    REGISTER_KERNEL(any, CPU, F32, any_kernel);
    REGISTER_KERNEL(any, CPU, F64, any_kernel);
    REGISTER_KERNEL(any, CPU, I8, any_kernel);
    REGISTER_KERNEL(any, CPU, I16, any_kernel);
    REGISTER_KERNEL(any, CPU, I32, any_kernel);
    REGISTER_KERNEL(any, CPU, I64, any_kernel);
    REGISTER_KERNEL(any, CPU, U8, any_kernel);
    REGISTER_KERNEL(any, CPU, U16, any_kernel);
    REGISTER_KERNEL(any, CPU, U32, any_kernel);
    REGISTER_KERNEL(any, CPU, U64, any_kernel);
    REGISTER_KERNEL(any, CPU, BOOL, any_kernel);

    // ========== all ==========
    template<typename T>
    static Array all_impl(const Array& out, const Array& prepared, int64_t batch_size, int64_t reduce_size) {
        bool* dst = const_cast<bool*>(out.data<bool>());
        const T* src = prepared.data<T>();
        int64_t total_out = out.numel();

#pragma omp parallel for
        for (int64_t i = 0; i < total_out; ++i) {
            bool all_true = true;
            for (int64_t j = 0; j < reduce_size; ++j) {
                if (src[i * reduce_size + j] == T(0)) {
                    all_true = false;
                    break;
                }
            }
            dst[i] = all_true;
        }
        return out;
    }

    static OpArgs all_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);
        int64_t batch_size = std::any_cast<int64_t>(args[2]);
        int64_t reduce_size = std::any_cast<int64_t>(args[3]);
        DType src_dtype = prepared.dtype();

        switch (src_dtype) {
        case DType::F32 : return { all_impl<float   >(out, prepared, batch_size, reduce_size) };
        case DType::F64 : return { all_impl<double  >(out, prepared, batch_size, reduce_size) };
        case DType::I8  : return { all_impl<int8_t  >(out, prepared, batch_size, reduce_size) };
        case DType::I16 : return { all_impl<int8_t  >(out, prepared, batch_size, reduce_size) };
        case DType::I32 : return { all_impl<int32_t >(out, prepared, batch_size, reduce_size) };
        case DType::I64 : return { all_impl<int64_t >(out, prepared, batch_size, reduce_size) };
        case DType::U8  : return { all_impl<uint8_t >(out, prepared, batch_size, reduce_size) };
        case DType::U16 : return { all_impl<uint16_t>(out, prepared, batch_size, reduce_size) };
        case DType::U32 : return { all_impl<uint32_t>(out, prepared, batch_size, reduce_size) };
        case DType::U64 : return { all_impl<uint64_t>(out, prepared, batch_size, reduce_size) };
        case DType::BOOL: return { all_impl<bool    >(out, prepared, batch_size, reduce_size) };
        default: INS_THROW("all: unsupported dtype, got dtype: ", dtype_name(prepared.dtype()));
        }
    }

    REGISTER_KERNEL(all, CPU, F32, all_kernel);
    REGISTER_KERNEL(all, CPU, F64, all_kernel);
    REGISTER_KERNEL(all, CPU, I8, all_kernel);
    REGISTER_KERNEL(all, CPU, I16, all_kernel);
    REGISTER_KERNEL(all, CPU, I32, all_kernel);
    REGISTER_KERNEL(all, CPU, I64, all_kernel);
    REGISTER_KERNEL(all, CPU, U8, all_kernel);
    REGISTER_KERNEL(all, CPU, U16, all_kernel);
    REGISTER_KERNEL(all, CPU, U32, all_kernel);
    REGISTER_KERNEL(all, CPU, U64, all_kernel);
    REGISTER_KERNEL(all, CPU, BOOL, all_kernel);

    // ========== argmax ==========
    template<typename T>
    static Array argmax_impl(const Array& out, const Array& prepared, int64_t batch_size, int64_t reduce_size) {
        int64_t* dst = const_cast<int64_t*>(out.data<int64_t>());
        const T* src = prepared.data<T>();
        int64_t total_out = out.numel();

#pragma omp parallel for
        for (int64_t i = 0; i < total_out; ++i) {
            int64_t max_idx = 0;
            T max_val = src[i * reduce_size];
            for (int64_t j = 1; j < reduce_size; ++j) {
                T val = src[i * reduce_size + j];
                if (val > max_val) {
                    max_val = val;
                    max_idx = j;
                }
            }
            dst[i] = max_idx;
        }
        return out;
    }

    static OpArgs argmax_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);
        int64_t batch_size = std::any_cast<int64_t>(args[2]);
        int64_t reduce_size = std::any_cast<int64_t>(args[3]);
        DType src_dtype = prepared.dtype();

        switch (src_dtype) {
        case DType::F32: return { argmax_impl<float>(out, prepared, batch_size, reduce_size) };
        case DType::F64: return { argmax_impl<double>(out, prepared, batch_size, reduce_size) };
        case DType::I32: return { argmax_impl<int32_t>(out, prepared, batch_size, reduce_size) };
        case DType::I64: return { argmax_impl<int64_t>(out, prepared, batch_size, reduce_size) };
        case DType::U8:  return { argmax_impl<uint8_t>(out, prepared, batch_size, reduce_size) };
        default: INS_THROW("argmax: unsupported dtype");
        }
    }

    REGISTER_KERNEL(argmax, CPU, F32, argmax_kernel);
    REGISTER_KERNEL(argmax, CPU, F64, argmax_kernel);
    REGISTER_KERNEL(argmax, CPU, I32, argmax_kernel);
    REGISTER_KERNEL(argmax, CPU, I64, argmax_kernel);
    REGISTER_KERNEL(argmax, CPU, U8, argmax_kernel);

    // ========== argmin ==========
    template<typename T>
    static Array argmin_impl(const Array& out, const Array& prepared, int64_t batch_size, int64_t reduce_size) {
        int64_t* dst = const_cast<int64_t*>(out.data<int64_t>());
        const T* src = prepared.data<T>();
        int64_t total_out = out.numel();

#pragma omp parallel for
        for (int64_t i = 0; i < total_out; ++i) {
            int64_t min_idx = 0;
            T min_val = src[i * reduce_size];
            for (int64_t j = 1; j < reduce_size; ++j) {
                T val = src[i * reduce_size + j];
                if (val < min_val) {
                    min_val = val;
                    min_idx = j;
                }
            }
            dst[i] = min_idx;
        }
        return out;
    }

    static OpArgs argmin_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);
        int64_t batch_size = std::any_cast<int64_t>(args[2]);
        int64_t reduce_size = std::any_cast<int64_t>(args[3]);
        DType src_dtype = prepared.dtype();

        switch (src_dtype) {
        case DType::F32: return { argmin_impl<float>(out, prepared, batch_size, reduce_size) };
        case DType::F64: return { argmin_impl<double>(out, prepared, batch_size, reduce_size) };
        case DType::I32: return { argmin_impl<int32_t>(out, prepared, batch_size, reduce_size) };
        case DType::I64: return { argmin_impl<int64_t>(out, prepared, batch_size, reduce_size) };
        case DType::U8:  return { argmin_impl<uint8_t>(out, prepared, batch_size, reduce_size) };
        default: INS_THROW("argmin: unsupported dtype");
        }
    }

    REGISTER_KERNEL(argmin, CPU, F32, argmin_kernel);
    REGISTER_KERNEL(argmin, CPU, F64, argmin_kernel);
    REGISTER_KERNEL(argmin, CPU, I32, argmin_kernel);
    REGISTER_KERNEL(argmin, CPU, I64, argmin_kernel);
    REGISTER_KERNEL(argmin, CPU, U8, argmin_kernel);

    // ========== count_nonzero ==========

    template<typename T>
    static Array count_nonzero_impl(const Array& out, const Array& prepared, int64_t batch_size, int64_t reduce_size) {
        int64_t* dst = const_cast<int64_t*>(out.data<int64_t>());
        const T* src = prepared.data<T>();
        int64_t total_out = out.numel();

#ifdef INSIGHT_WITH_OPENMP
#pragma omp parallel for
#endif
        for (int64_t i = 0; i < total_out; ++i) {
            int64_t count = 0;
            for (int64_t j = 0; j < reduce_size; ++j) {
                if (src[i * reduce_size + j] != T(0)) {
                    ++count;
                }
            }
            dst[i] = count;
        }
        return out;
    }

    static OpArgs count_nonzero_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);
        int64_t batch_size = std::any_cast<int64_t>(args[2]);
        int64_t reduce_size = std::any_cast<int64_t>(args[3]);
        DType src_dtype = prepared.dtype();

        switch (src_dtype) {
        case DType::BOOL: return { count_nonzero_impl<bool>(out, prepared, batch_size, reduce_size) };
        case DType::U8:   return { count_nonzero_impl<uint8_t>(out, prepared, batch_size, reduce_size) };
        case DType::I8:   return { count_nonzero_impl<int8_t>(out, prepared, batch_size, reduce_size) };
        case DType::I16:  return { count_nonzero_impl<int16_t>(out, prepared, batch_size, reduce_size) };
        case DType::I32:  return { count_nonzero_impl<int32_t>(out, prepared, batch_size, reduce_size) };
        case DType::I64:  return { count_nonzero_impl<int64_t>(out, prepared, batch_size, reduce_size) };
        case DType::U16:  return { count_nonzero_impl<uint16_t>(out, prepared, batch_size, reduce_size) };
        case DType::U32:  return { count_nonzero_impl<uint32_t>(out, prepared, batch_size, reduce_size) };
        case DType::U64:  return { count_nonzero_impl<uint64_t>(out, prepared, batch_size, reduce_size) };
        case DType::F32:  return { count_nonzero_impl<float>(out, prepared, batch_size, reduce_size) };
        case DType::F64:  return { count_nonzero_impl<double>(out, prepared, batch_size, reduce_size) };
        case DType::C32:  return { count_nonzero_impl<std::complex<float>>(out, prepared, batch_size, reduce_size) };
        case DType::C64:  return { count_nonzero_impl<std::complex<double>>(out, prepared, batch_size, reduce_size) };
        default: INS_THROW("count_nonzero: unsupported dtype");
        }
    }

    REGISTER_KERNEL(count_nonzero, CPU, BOOL, count_nonzero_kernel);
    REGISTER_KERNEL(count_nonzero, CPU, U8, count_nonzero_kernel);
    REGISTER_KERNEL(count_nonzero, CPU, I8, count_nonzero_kernel);
    REGISTER_KERNEL(count_nonzero, CPU, I16, count_nonzero_kernel);
    REGISTER_KERNEL(count_nonzero, CPU, I32, count_nonzero_kernel);
    REGISTER_KERNEL(count_nonzero, CPU, I64, count_nonzero_kernel);
    REGISTER_KERNEL(count_nonzero, CPU, U16, count_nonzero_kernel);
    REGISTER_KERNEL(count_nonzero, CPU, U32, count_nonzero_kernel);
    REGISTER_KERNEL(count_nonzero, CPU, U64, count_nonzero_kernel);
    REGISTER_KERNEL(count_nonzero, CPU, F32, count_nonzero_kernel);
    REGISTER_KERNEL(count_nonzero, CPU, F64, count_nonzero_kernel);
    REGISTER_KERNEL(count_nonzero, CPU, C32, count_nonzero_kernel);
    REGISTER_KERNEL(count_nonzero, CPU, C64, count_nonzero_kernel);

    // ========== cumsum ==========
    template<typename InT, typename OutT>
    static Array cumsum_impl(const Array& out, const Array& x, int axis) {
        OutT* dst = const_cast<OutT*>(out.data<OutT>());
        const InT* src = x.data<InT>();
        const Shape& shape = x.shape();
        int ndim = shape.ndim();

        std::vector<int64_t> dims(ndim);
        std::vector<int64_t> strides(ndim);
        for (int i = 0; i < ndim; ++i) {
            dims[i] = shape.dim(i);
        }
        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * dims[i + 1];
        }

        int64_t total = x.numel();
        int64_t axis_stride = strides[axis];

        for (int64_t linear = 0; linear < total; ++linear) {
            std::vector<int64_t> coords(ndim);
            int64_t tmp = linear;
            for (int d = ndim - 1; d >= 0; --d) {
                coords[d] = tmp % dims[d];
                tmp /= dims[d];
            }

            int64_t base_idx = 0;
            for (int d = 0; d < ndim; ++d) {
                if (d != axis) {
                    base_idx += coords[d] * strides[d];
                }
            }

            OutT sum = OutT(0);
            for (int64_t k = 0; k <= coords[axis]; ++k) {
                int64_t idx = base_idx + k * axis_stride;
                sum += static_cast<OutT>(src[idx]);
            }
            dst[linear] = sum;
        }

        return out;
    }

    // ========== cumsum_kernel ==========
    static OpArgs cumsum_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& x = std::any_cast<const Array&>(args[1]);
        int axis = std::any_cast<int>(args[2]);

        DType in_dtype = x.dtype();
        DType out_dtype = out.dtype();

        // 根据输入输出类型组合选择正确的模板实例
        if (in_dtype == DType::F32 && out_dtype == DType::F64) {
            return { cumsum_impl<float, double>(out, x, axis) };
        }
        else if (in_dtype == DType::F32 && out_dtype == DType::F32) {
            return { cumsum_impl<float, float>(out, x, axis) };
        }
        else if (in_dtype == DType::F64 && out_dtype == DType::F64) {
            return { cumsum_impl<double, double>(out, x, axis) };
        }
        else if (in_dtype == DType::I32 && out_dtype == DType::I32) {
            return { cumsum_impl<int32_t, int32_t>(out, x, axis) };
        }
        else if (in_dtype == DType::I64 && out_dtype == DType::I64) {
            return { cumsum_impl<int64_t, int64_t>(out, x, axis) };
        }
        else if (in_dtype == DType::I32 && out_dtype == DType::F64) {
            return { cumsum_impl<int32_t, double>(out, x, axis) };
        }
        else if (in_dtype == DType::I64 && out_dtype == DType::F64) {
            return { cumsum_impl<int64_t, double>(out, x, axis) };
        }
        else {
            INS_THROW("cumsum: unsupported dtype combination: ",
                dtype_name(in_dtype), " -> ", dtype_name(out_dtype));
        }
    }

    // 注册 kernel
    REGISTER_KERNEL(cumsum, CPU, F32, cumsum_kernel);
    REGISTER_KERNEL(cumsum, CPU, F64, cumsum_kernel);
    REGISTER_KERNEL(cumsum, CPU, I32, cumsum_kernel);
    REGISTER_KERNEL(cumsum, CPU, I64, cumsum_kernel);

    // ========== cumprod ==========
    template<typename InT, typename OutT>
    static Array cumprod_impl(const Array& out, const Array& x, int axis) {
        OutT* dst = const_cast<OutT*>(out.data<OutT>());
        const InT* src = x.data<InT>();
        const Shape& shape = x.shape();
        int ndim = shape.ndim();

        std::vector<int64_t> dims(ndim);
        std::vector<int64_t> strides(ndim);
        for (int i = 0; i < ndim; ++i) {
            dims[i] = shape.dim(i);
        }
        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * dims[i + 1];
        }

        int64_t total = x.numel();
        int64_t axis_stride = strides[axis];

        for (int64_t linear = 0; linear < total; ++linear) {
            std::vector<int64_t> coords(ndim);
            int64_t tmp = linear;
            for (int d = ndim - 1; d >= 0; --d) {
                coords[d] = tmp % dims[d];
                tmp /= dims[d];
            }

            int64_t base_idx = 0;
            for (int d = 0; d < ndim; ++d) {
                if (d != axis) {
                    base_idx += coords[d] * strides[d];
                }
            }

            OutT prod = OutT(1);
            for (int64_t k = 0; k <= coords[axis]; ++k) {
                int64_t idx = base_idx + k * axis_stride;
                prod *= static_cast<OutT>(src[idx]);
            }
            dst[linear] = prod;
        }

        return out;
    }

    static OpArgs cumprod_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& x = std::any_cast<const Array&>(args[1]);
        int axis = std::any_cast<int>(args[2]);

        DType in_dtype = x.dtype();
        DType out_dtype = out.dtype();

        if (in_dtype == DType::F32 && out_dtype == DType::F64) {
            return { cumprod_impl<float, double>(out, x, axis) };
        }
        else if (in_dtype == DType::F32 && out_dtype == DType::F32) {
            return { cumprod_impl<float, float>(out, x, axis) };
        }
        else if (in_dtype == DType::F64 && out_dtype == DType::F64) {
            return { cumprod_impl<double, double>(out, x, axis) };
        }
        else if (in_dtype == DType::I32 && out_dtype == DType::I32) {
            return { cumprod_impl<int32_t, int32_t>(out, x, axis) };
        }
        else if (in_dtype == DType::I64 && out_dtype == DType::I64) {
            return { cumprod_impl<int64_t, int64_t>(out, x, axis) };
        }
        else if (in_dtype == DType::I32 && out_dtype == DType::F64) {
            return { cumprod_impl<int32_t, double>(out, x, axis) };
        }
        else if (in_dtype == DType::I64 && out_dtype == DType::F64) {
            return { cumprod_impl<int64_t, double>(out, x, axis) };
        }
        else {
            INS_THROW("cumprod: unsupported dtype combination");
        }
    }

    REGISTER_KERNEL(cumprod, CPU, F32, cumprod_kernel);
    REGISTER_KERNEL(cumprod, CPU, F64, cumprod_kernel);
    REGISTER_KERNEL(cumprod, CPU, I32, cumprod_kernel);
    REGISTER_KERNEL(cumprod, CPU, I64, cumprod_kernel);

    // ========== cummax ==========
    template<typename T>
    static Array cummax_impl(const Array& out, const Array& x, int axis) {
        T* dst = const_cast<T*>(out.data<T>());
        const T* src = x.data<T>();
        const Shape& shape = x.shape();
        int ndim = shape.ndim();

        std::vector<int64_t> dims(ndim);
        for (int i = 0; i < ndim; ++i) {
            dims[i] = shape.dim(i);
        }
        std::vector<int64_t> strides(ndim);
        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * dims[i + 1];
        }

        int64_t total = x.numel();

#pragma omp parallel for
        for (int64_t linear = 0; linear < total; ++linear) {
            std::vector<int64_t> coords(ndim);
            int64_t tmp = linear;
            for (int d = ndim - 1; d >= 0; --d) {
                coords[d] = tmp % dims[d];
                tmp /= dims[d];
            }

            T max_val = src[linear];
            for (int64_t k = 0; k <= coords[axis]; ++k) {
                int64_t idx = 0;
                int64_t stride_mult = 1;
                for (int d = ndim - 1; d >= 0; --d) {
                    int64_t coord = (d == axis) ? k : coords[d];
                    idx += coord * stride_mult;
                    stride_mult *= dims[d];
                }
                if (src[idx] > max_val) max_val = src[idx];
            }
            dst[linear] = max_val;
        }
        return out;
    }

    static OpArgs cummax_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& x = std::any_cast<const Array&>(args[1]);
        int axis = std::any_cast<int>(args[2]);

        switch (out.dtype()) {
        case DType::F32: return { cummax_impl<float>(out, x, axis) };
        case DType::F64: return { cummax_impl<double>(out, x, axis) };
        case DType::I32: return { cummax_impl<int32_t>(out, x, axis) };
        case DType::I64: return { cummax_impl<int64_t>(out, x, axis) };
        default: INS_THROW("cummax: unsupported dtype");
        }
    }

    REGISTER_KERNEL(cummax, CPU, F32, cummax_kernel);
    REGISTER_KERNEL(cummax, CPU, F64, cummax_kernel);
    REGISTER_KERNEL(cummax, CPU, I32, cummax_kernel);
    REGISTER_KERNEL(cummax, CPU, I64, cummax_kernel);

    // ========== cummin ==========
    template<typename T>
    static Array cummin_impl(const Array& out, const Array& x, int axis) {
        T* dst = const_cast<T*>(out.data<T>());
        const T* src = x.data<T>();
        const Shape& shape = x.shape();
        int ndim = shape.ndim();

        std::vector<int64_t> dims(ndim);
        for (int i = 0; i < ndim; ++i) {
            dims[i] = shape.dim(i);
        }
        std::vector<int64_t> strides(ndim);
        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * dims[i + 1];
        }

        int64_t total = x.numel();

#pragma omp parallel for
        for (int64_t linear = 0; linear < total; ++linear) {
            std::vector<int64_t> coords(ndim);
            int64_t tmp = linear;
            for (int d = ndim - 1; d >= 0; --d) {
                coords[d] = tmp % dims[d];
                tmp /= dims[d];
            }

            T min_val = src[linear];
            for (int64_t k = 0; k <= coords[axis]; ++k) {
                int64_t idx = 0;
                int64_t stride_mult = 1;
                for (int d = ndim - 1; d >= 0; --d) {
                    int64_t coord = (d == axis) ? k : coords[d];
                    idx += coord * stride_mult;
                    stride_mult *= dims[d];
                }
                if (src[idx] < min_val) min_val = src[idx];
            }
            dst[linear] = min_val;
        }
        return out;
    }

    static OpArgs cummin_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& x = std::any_cast<const Array&>(args[1]);
        int axis = std::any_cast<int>(args[2]);

        switch (out.dtype()) {
        case DType::F32: return { cummin_impl<float>(out, x, axis) };
        case DType::F64: return { cummin_impl<double>(out, x, axis) };
        case DType::I32: return { cummin_impl<int32_t>(out, x, axis) };
        case DType::I64: return { cummin_impl<int64_t>(out, x, axis) };
        default: INS_THROW("cummin: unsupported dtype");
        }
    }

    REGISTER_KERNEL(cummin, CPU, F32, cummin_kernel);
    REGISTER_KERNEL(cummin, CPU, F64, cummin_kernel);
    REGISTER_KERNEL(cummin, CPU, I32, cummin_kernel);
    REGISTER_KERNEL(cummin, CPU, I64, cummin_kernel);

    // ========== quantile ==========
    template<typename InT, typename OutT>
    static Array quantile_impl(const Array& out, const Array& prepared, int64_t batch_size, int64_t reduce_size, double q) {
        OutT* dst = const_cast<OutT*>(out.data<OutT>());
        const InT* src = prepared.data<InT>();
        int64_t total_out = out.numel();

        for (int64_t i = 0; i < total_out; ++i) {
            std::vector<InT> row(reduce_size);
            for (int64_t j = 0; j < reduce_size; ++j) {
                row[j] = src[i * reduce_size + j];
            }
            std::sort(row.begin(), row.end());

            if (q == 0.5) {
                // median
                if (reduce_size % 2 == 1) {
                    dst[i] = static_cast<OutT>(row[reduce_size / 2]);
                }
                else {
                    dst[i] = static_cast<OutT>((row[reduce_size / 2 - 1] + row[reduce_size / 2]) / static_cast<InT>(2));
                }
            }
            else {
                // quantile with linear interpolation
                double idx = q * (reduce_size - 1);
                int64_t lo = static_cast<int64_t>(idx);
                int64_t hi = lo + 1;
                if (hi >= reduce_size) {
                    dst[i] = static_cast<OutT>(row[lo]);
                }
                else {
                    double frac = idx - lo;
                    dst[i] = static_cast<OutT>(row[lo] * (1 - frac) + row[hi] * frac);
                }
            }
        }

        return out;
    }

    static OpArgs quantile_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);
        int64_t batch_size = std::any_cast<int64_t>(args[2]);
        int64_t reduce_size = std::any_cast<int64_t>(args[3]);
        double q = std::any_cast<double>(args[4]);

        DType in_dtype = prepared.dtype();
        DType out_dtype = out.dtype();

        Array& mutable_out = const_cast<Array&>(out);

        if (in_dtype == DType::F32 && out_dtype == DType::F64) {
            return { quantile_impl<float, double>(mutable_out, prepared, batch_size, reduce_size, q) };
        }
        else if (in_dtype == DType::F32 && out_dtype == DType::F32) {
            return { quantile_impl<float, float>(mutable_out, prepared, batch_size, reduce_size, q) };
        }
        else if (in_dtype == DType::F64 && out_dtype == DType::F64) {
            return { quantile_impl<double, double>(mutable_out, prepared, batch_size, reduce_size, q) };
        }
        else {
            INS_THROW("quantile: unsupported dtype combination");
        }
    }

    REGISTER_KERNEL(quantile, CPU, F32, quantile_kernel);
    REGISTER_KERNEL(quantile, CPU, F64, quantile_kernel);

    // ========== nansum ==========
    template<typename T>
    static std::pair<Array, Array> nansum_impl(const Array& sum_out, const Array& count_out,
        const Array& prepared, int64_t batch_size, int64_t reduce_size) {
        T* sum_dst = const_cast<T*>(sum_out.data<T>());
        int64_t* count_dst = const_cast<int64_t*>(count_out.data<int64_t>());
        const T* src = prepared.data<T>();
        int64_t total_out = sum_out.numel();

#pragma omp parallel for
        for (int64_t i = 0; i < total_out; ++i) {
            T sum = T(0);
            int64_t cnt = 0;
            for (int64_t j = 0; j < reduce_size; ++j) {
                T val = src[i * reduce_size + j];
                if (!std::isnan(val)) {
                    sum += val;
                    ++cnt;
                }
            }
            sum_dst[i] = sum;
            count_dst[i] = cnt;
        }
        return { sum_out, count_out };
    }

    static OpArgs nansum_kernel(const OpArgs& args) {
        const Array& sum_out = std::any_cast<const Array&>(args[0]);
        const Array& count_out = std::any_cast<const Array&>(args[1]);
        const Array& prepared = std::any_cast<const Array&>(args[2]);
        int64_t batch_size = std::any_cast<int64_t>(args[3]);
        int64_t reduce_size = std::any_cast<int64_t>(args[4]);

        // Check size consistency
        int64_t expected_prepared_numel = batch_size * reduce_size;
        if (prepared.numel() != expected_prepared_numel) {
            INS_THROW("nansum: prepared size mismatch");
        }

        int64_t expected_output_numel = batch_size;
        if (sum_out.numel() != expected_output_numel) {
            INS_THROW("nansum: output size mismatch");
        }

        Array& mutable_sum = const_cast<Array&>(sum_out);
        Array& mutable_count = const_cast<Array&>(count_out);
        DType src_dtype = prepared.dtype();

        switch (src_dtype) {
        case DType::F32: {
            auto [s, c] = nansum_impl<float>(mutable_sum, mutable_count, prepared, batch_size, reduce_size);
            return { s, c };
        }
        case DType::F64: {
            auto [s, c] = nansum_impl<double>(mutable_sum, mutable_count, prepared, batch_size, reduce_size);
            return { s, c };
        }
        default: INS_THROW("nansum: only float32/64 supported");
        }
    }

    REGISTER_KERNEL(nansum, CPU, F32, nansum_kernel);
    REGISTER_KERNEL(nansum, CPU, F64, nansum_kernel);

    // ========== nanmax ==========
    template<typename T>
    static Array nanmax_impl(const Array& out, const Array& prepared, int64_t batch_size, int64_t reduce_size) {
        T* dst = const_cast<T*>(out.data<T>());
        const T* src = prepared.data<T>();
        int64_t total_out = out.numel();

#pragma omp parallel for
        for (int64_t i = 0; i < total_out; ++i) {
            T max_val = T(0);
            bool found = false;
            for (int64_t j = 0; j < reduce_size; ++j) {
                T val = src[i * reduce_size + j];
                if (!is_nan(val)) {
                    if (!found) {
                        max_val = val;
                        found = true;
                    }
                    else if (val > max_val) {
                        max_val = val;
                    }
                }
            }
            dst[i] = found ? max_val : std::numeric_limits<T>::quiet_NaN();
        }
        return out;
    }

    static OpArgs nanmax_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);
        int64_t batch_size = std::any_cast<int64_t>(args[2]);
        int64_t reduce_size = std::any_cast<int64_t>(args[3]);
        DType src_dtype = prepared.dtype();

        switch (src_dtype) {
        case DType::F32: return { nanmax_impl<float>(out, prepared, batch_size, reduce_size) };
        case DType::F64: return { nanmax_impl<double>(out, prepared, batch_size, reduce_size) };
        default: INS_THROW("nanmax: only float32/64 supported");
        }
    }

    REGISTER_KERNEL(nanmax, CPU, F32, nanmax_kernel);
    REGISTER_KERNEL(nanmax, CPU, F64, nanmax_kernel);

    // ========== nanmin ==========
    template<typename T>
    static Array nanmin_impl(const Array& out, const Array& prepared, int64_t batch_size, int64_t reduce_size) {
        T* dst = const_cast<T*>(out.data<T>());
        const T* src = prepared.data<T>();
        int64_t total_out = out.numel();

#pragma omp parallel for
        for (int64_t i = 0; i < total_out; ++i) {
            T min_val = T(0);
            bool found = false;
            for (int64_t j = 0; j < reduce_size; ++j) {
                T val = src[i * reduce_size + j];
                if (!is_nan(val)) {
                    if (!found) {
                        min_val = val;
                        found = true;
                    }
                    else if (val < min_val) {
                        min_val = val;
                    }
                }
            }
            dst[i] = found ? min_val : std::numeric_limits<T>::quiet_NaN();
        }
        return out;
    }

    static OpArgs nanmin_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);
        int64_t batch_size = std::any_cast<int64_t>(args[2]);
        int64_t reduce_size = std::any_cast<int64_t>(args[3]);
        DType src_dtype = prepared.dtype();

        switch (src_dtype) {
        case DType::F32: return { nanmin_impl<float>(out, prepared, batch_size, reduce_size) };
        case DType::F64: return { nanmin_impl<double>(out, prepared, batch_size, reduce_size) };
        default: INS_THROW("nanmin: only float32/64 supported");
        }
    }

    REGISTER_KERNEL(nanmin, CPU, F32, nanmin_kernel);
    REGISTER_KERNEL(nanmin, CPU, F64, nanmin_kernel);

    // ========== nanvar ==========
    template<typename T>
    static Array nanvar_impl(const Array& out, const Array& prepared, int64_t batch_size, int64_t reduce_size, int ddof) {
        T* dst = const_cast<T*>(out.data<T>());
        const T* src = prepared.data<T>();
        int64_t total_out = out.numel();

        for (int64_t i = 0; i < total_out; ++i) {
            const T* row = src + i * reduce_size;
            int64_t count = 0;
            T sum = T(0);
            for (int64_t j = 0; j < reduce_size; ++j) {
                T val = row[j];
                if (!std::isnan(val)) {
                    sum += val;
                    ++count;
                }
            }
            if (count == 0) {
                dst[i] = std::numeric_limits<T>::quiet_NaN();
                continue;
            }
            T mean = sum / static_cast<T>(count);
            T sq_sum = T(0);
            for (int64_t j = 0; j < reduce_size; ++j) {
                T val = row[j];
                if (!std::isnan(val)) {
                    T diff = val - mean;
                    sq_sum += diff * diff;
                }
            }
            T divisor = static_cast<T>(count - ddof);
            if (divisor <= T(0)) divisor = T(1);
            dst[i] = sq_sum / divisor;
        }

        return out;
    }

    static OpArgs nanvar_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);
        int64_t batch_size = std::any_cast<int64_t>(args[2]);
        int64_t reduce_size = std::any_cast<int64_t>(args[3]);
        int ddof = std::any_cast<int>(args[4]);
        DType src_dtype = prepared.dtype();

        switch (src_dtype) {
        case DType::F32: return { nanvar_impl<float>(out, prepared, batch_size, reduce_size, ddof) };
        case DType::F64: return { nanvar_impl<double>(out, prepared, batch_size, reduce_size, ddof) };
        default: INS_THROW("nanvar: only float32/64 supported");
        }
    }

    REGISTER_KERNEL(nanvar, CPU, F32, nanvar_kernel);
    REGISTER_KERNEL(nanvar, CPU, F64, nanvar_kernel);

    // ========== nanquantile ==========
    template<typename T>
    static Array nanquantile_impl(const Array& out, const Array& prepared, int64_t batch_size, int64_t reduce_size, double q) {
        T* dst = const_cast<T*>(out.data<T>());
        const T* src = prepared.data<T>();
        int64_t total_out = out.numel();

#pragma omp parallel for
        for (int64_t i = 0; i < total_out; ++i) {
            const T* row = src + i * reduce_size;
            std::vector<T> valid;
            valid.reserve(reduce_size);
            for (int64_t j = 0; j < reduce_size; ++j) {
                T val = row[j];
                if (!is_nan(val)) {
                    valid.push_back(val);
                }
            }

            if (valid.empty()) {
                dst[i] = std::numeric_limits<T>::quiet_NaN();
                continue;
            }

            std::sort(valid.begin(), valid.end());
            int64_t n = static_cast<int64_t>(valid.size());

            if (q == 0.5) {
                if (n % 2 == 1) {
                    dst[i] = valid[n / 2];
                }
                else {
                    dst[i] = (valid[n / 2 - 1] + valid[n / 2]) / static_cast<T>(2);
                }
            }
            else {
                double idx = q * (n - 1);
                int64_t lo = static_cast<int64_t>(idx);
                int64_t hi = lo + 1;
                if (hi >= n) {
                    dst[i] = valid[lo];
                }
                else {
                    double frac = idx - lo;
                    dst[i] = static_cast<T>(valid[lo] * (1 - frac) + valid[hi] * frac);
                }
            }
        }
        return out;
    }

    static OpArgs nanquantile_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);
        int64_t batch_size = std::any_cast<int64_t>(args[2]);
        int64_t reduce_size = std::any_cast<int64_t>(args[3]);
        double q = std::any_cast<double>(args[4]);
        DType src_dtype = prepared.dtype();

        switch (src_dtype) {
        case DType::F32: return { nanquantile_impl<float>(out, prepared, batch_size, reduce_size, q) };
        case DType::F64: return { nanquantile_impl<double>(out, prepared, batch_size, reduce_size, q) };
        default: INS_THROW("nanquantile: only float32/64 supported");
        }
    }

    REGISTER_KERNEL(nanquantile, CPU, F32, nanquantile_kernel);
    REGISTER_KERNEL(nanquantile, CPU, F64, nanquantile_kernel);

    // ========== Module Registration ==========
    REGISTER_MODULE(reduction, CPU);

} // namespace ins::cpu