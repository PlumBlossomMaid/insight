// backends/cpu/creation.cpp
#include "insight/plugin/op_registry.h"
#include "insight/core/array.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <complex>

namespace ins::cpu {

    // ========== full ==========

    template<typename T>
    static Array full_impl(const Array& out, double fill_value) {
        T* data = (T*)out.data<T>();
        T val = static_cast<T>(fill_value);
        int64_t n = out.numel();
#pragma omp parallel for
        for (int64_t i = 0; i < n; ++i) {
            data[i] = val;
        }
        return out;
    }

    static OpArgs full_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        double fill_value = std::any_cast<double>(args[1]);
        DType dtype = out.dtype();
        Array result;
        switch (dtype) {
        case DType::BOOL: result = full_impl<bool>(out, fill_value); break;
        case DType::U8:   result = full_impl<uint8_t>(out, fill_value); break;
        case DType::I8:   result = full_impl<int8_t>(out, fill_value); break;
        case DType::I16:  result = full_impl<int16_t>(out, fill_value); break;
        case DType::I32:  result = full_impl<int32_t>(out, fill_value); break;
        case DType::I64:  result = full_impl<int64_t>(out, fill_value); break;
        case DType::U16:  result = full_impl<uint16_t>(out, fill_value); break;
        case DType::U32:  result = full_impl<uint32_t>(out, fill_value); break;
        case DType::U64:  result = full_impl<uint64_t>(out, fill_value); break;
        case DType::F32:  result = full_impl<float>(out, fill_value); break;
        case DType::F64:  result = full_impl<double>(out, fill_value); break;
        case DType::C32:  result = full_impl<std::complex<float>>(out, fill_value); break;
        case DType::C64:  result = full_impl<std::complex<double>>(out, fill_value); break;
        default: INS_THROW("full: unsupported dtype");
        }
        return { result };
    }

    // Register full for all types
    REGISTER_KERNEL(full, CPU, BOOL, full_kernel);
    REGISTER_KERNEL(full, CPU, U8, full_kernel);
    REGISTER_KERNEL(full, CPU, I8, full_kernel);
    REGISTER_KERNEL(full, CPU, I16, full_kernel);
    REGISTER_KERNEL(full, CPU, I32, full_kernel);
    REGISTER_KERNEL(full, CPU, I64, full_kernel);
    REGISTER_KERNEL(full, CPU, U16, full_kernel);
    REGISTER_KERNEL(full, CPU, U32, full_kernel);
    REGISTER_KERNEL(full, CPU, U64, full_kernel);
    REGISTER_KERNEL(full, CPU, F32, full_kernel);
    REGISTER_KERNEL(full, CPU, F64, full_kernel);
    REGISTER_KERNEL(full, CPU, C32, full_kernel);
    REGISTER_KERNEL(full, CPU, C64, full_kernel);

    // ========== eye ==========
    template<typename T>
    static Array eye_impl(const Array& out, int64_t k) {
        T* data = (T*)out.data<T>();
        int64_t n = out.shape().dim(0);
        int64_t m = out.shape().dim(1);
        int64_t total = n * m;

        // Initialize to zero
#pragma omp parallel for
        for (int64_t idx = 0; idx < total; ++idx) {
            data[idx] = T(0);
        }

        // Set diagonal
        for (int64_t i = 0; i < n; ++i) {
            int64_t j = i + k;
            if (j >= 0 && j < m) {
                data[i * m + j] = T(1);
            }
        }
        return out;
    }

    static OpArgs eye_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        int64_t k = std::any_cast<int64_t>(args[1]);
        DType dtype = out.dtype();
        Array result;
        switch (dtype) {
        case DType::BOOL: result = eye_impl<bool>(out, k); break;
        case DType::U8:   result = eye_impl<uint8_t>(out, k); break;
        case DType::I8:   result = eye_impl<int8_t>(out, k); break;
        case DType::I16:  result = eye_impl<int16_t>(out, k); break;
        case DType::I32:  result = eye_impl<int32_t>(out, k); break;
        case DType::I64:  result = eye_impl<int64_t>(out, k); break;
        case DType::U16:  result = eye_impl<uint16_t>(out, k); break;
        case DType::U32:  result = eye_impl<uint32_t>(out, k); break;
        case DType::U64:  result = eye_impl<uint64_t>(out, k); break;
        case DType::F16:  result = eye_impl<uint16_t>(out, k); break;  // placeholder
        case DType::BF16: result = eye_impl<uint16_t>(out, k); break;  // placeholder
        case DType::F32:  result = eye_impl<float>(out, k); break;
        case DType::F64:  result = eye_impl<double>(out, k); break;
        case DType::C32:  result = eye_impl<std::complex<float>>(out, k); break;
        case DType::C64:  result = eye_impl<std::complex<double>>(out, k); break;
        default: INS_THROW("eye: unsupported dtype");
        }
        return { result };
    }

    // Register eye for all types
    REGISTER_KERNEL(eye, CPU, BOOL, eye_kernel);
    REGISTER_KERNEL(eye, CPU, U8, eye_kernel);
    REGISTER_KERNEL(eye, CPU, I8, eye_kernel);
    REGISTER_KERNEL(eye, CPU, I16, eye_kernel);
    REGISTER_KERNEL(eye, CPU, I32, eye_kernel);
    REGISTER_KERNEL(eye, CPU, I64, eye_kernel);
    REGISTER_KERNEL(eye, CPU, U16, eye_kernel);
    REGISTER_KERNEL(eye, CPU, U32, eye_kernel);
    REGISTER_KERNEL(eye, CPU, U64, eye_kernel);
    REGISTER_KERNEL(eye, CPU, F16, eye_kernel);
    REGISTER_KERNEL(eye, CPU, BF16, eye_kernel);
    REGISTER_KERNEL(eye, CPU, F32, eye_kernel);
    REGISTER_KERNEL(eye, CPU, F64, eye_kernel);
    REGISTER_KERNEL(eye, CPU, C32, eye_kernel);
    REGISTER_KERNEL(eye, CPU, C64, eye_kernel);

    // ========== arange ==========

    template<typename T>
    static Array arange_impl(const Array& out, double start, double step) {
        T* data = (T*)out.data<T>();
        int64_t n = out.numel();
#pragma omp parallel for
        for (int64_t i = 0; i < n; ++i) {
            data[i] = static_cast<T>(start + i * step);
        }
        return out;
    }

    static OpArgs arange_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        double start = std::any_cast<double>(args[1]);
        double step = std::any_cast<double>(args[2]);
        DType dtype = out.dtype();
        Array result;
        switch (dtype) {
        case DType::F32: result = arange_impl<float>(out, start, step); break;
        case DType::F64: result = arange_impl<double>(out, start, step); break;
        case DType::I32: result = arange_impl<int32_t>(out, start, step); break;
        case DType::I64: result = arange_impl<int64_t>(out, start, step); break;
        default: INS_THROW("arange: unsupported dtype (only float32/64, int32/64)");
        }
        return { result };
    }

    REGISTER_KERNEL(arange, CPU, F32, arange_kernel);
    REGISTER_KERNEL(arange, CPU, F64, arange_kernel);
    REGISTER_KERNEL(arange, CPU, I32, arange_kernel);
    REGISTER_KERNEL(arange, CPU, I64, arange_kernel);

    // ========== linspace ==========

    template<typename T>
    static Array linspace_impl(const Array& out, double start, double stop) {
        T* data = (T*)out.data<T>();
        int64_t n = out.numel();
        if (n == 1) {
            data[0] = static_cast<T>(start);
        }
        else {
            double step = (stop - start) / (n - 1);
#pragma omp parallel for
            for (int64_t i = 0; i < n; ++i) {
                data[i] = static_cast<T>(start + i * step);
            }
        }
        return out;
    }

    static OpArgs linspace_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        double start = std::any_cast<double>(args[1]);
        double stop = std::any_cast<double>(args[2]);
        DType dtype = out.dtype();
        Array result;
        switch (dtype) {
        case DType::F32: result = linspace_impl<float>(out, start, stop); break;
        case DType::F64: result = linspace_impl<double>(out, start, stop); break;
        default: INS_THROW("linspace: only float32/64 supported");
        }
        return { result };
    }

    REGISTER_KERNEL(linspace, CPU, F32, linspace_kernel);
    REGISTER_KERNEL(linspace, CPU, F64, linspace_kernel);

    // ========== logspace ==========

    template<typename T>
    static Array logspace_impl(const Array& out, double start, double stop, double base) {
        T* data = (T*)out.data<T>();
        int64_t n = out.numel();
        if (n == 1) {
            data[0] = static_cast<T>(std::pow(base, start));
        }
        else {
            double step = (stop - start) / (n - 1);
#pragma omp parallel for
            for (int64_t i = 0; i < n; ++i) {
                data[i] = static_cast<T>(std::pow(base, start + i * step));
            }
        }
        return out;
    }

    static OpArgs logspace_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        double start = std::any_cast<double>(args[1]);
        double stop = std::any_cast<double>(args[2]);
        double base = std::any_cast<double>(args[3]);
        DType dtype = out.dtype();
        Array result;
        switch (dtype) {
        case DType::F32: result = logspace_impl<float>(out, start, stop, base); break;
        case DType::F64: result = logspace_impl<double>(out, start, stop, base); break;
        default: INS_THROW("logspace: only float32/64 supported");
        }
        return { result };
    }

    REGISTER_KERNEL(logspace, CPU, F32, logspace_kernel);
    REGISTER_KERNEL(logspace, CPU, F64, logspace_kernel);

    // ========== Module registration ==========

    REGISTER_MODULE(creation, CPU);

} // namespace ins::cpu