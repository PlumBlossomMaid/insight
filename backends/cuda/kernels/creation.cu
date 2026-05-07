// backends/cuda/kernels/creation.cu
#include "insight/plugin/op_registry.h"
#include "insight/core/array.h"
#include "insight/core/exception.h"
#include "insight/core/launch_config.h"
#include <cmath>
#include <complex>

namespace ins::gpu {

    // ============================================================================
    // Device helper functions for complex numbers
    // ============================================================================

    __device__ inline float real(const std::complex<float>& z) {
        return reinterpret_cast<const float*>(&z)[0];
    }

    __device__ inline float imag(const std::complex<float>& z) {
        return reinterpret_cast<const float*>(&z)[1];
    }

    __device__ inline double real(const std::complex<double>& z) {
        return reinterpret_cast<const double*>(&z)[0];
    }

    __device__ inline double imag(const std::complex<double>& z) {
        return reinterpret_cast<const double*>(&z)[1];
    }

    __device__ inline void set_c32(std::complex<float>* out, float r, float i) {
        reinterpret_cast<float*>(out)[0] = r;
        reinterpret_cast<float*>(out)[1] = i;
    }

    __device__ inline void set_c64(std::complex<double>* out, double r, double i) {
        reinterpret_cast<double*>(out)[0] = r;
        reinterpret_cast<double*>(out)[1] = i;
    }

    // ============================================================================
    // full kernel
    // ============================================================================

    template<typename T>
    __global__ void full_kernel(T* out, int64_t n, T val) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = val;
        }
    }

    // full - complex (separate to avoid constexpr __host__ issues)
    __global__ void full_c32_kernel(std::complex<float>* out, int64_t n, float real_val, float imag_val) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            set_c32(&out[idx], real_val, imag_val);
        }
    }

    __global__ void full_c64_kernel(std::complex<double>* out, int64_t n, double real_val, double imag_val) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            set_c64(&out[idx], real_val, imag_val);
        }
    }

    template<typename T>
    static void full_impl(const Array& out, double fill_value) {
        int64_t n = out.numel();
        LaunchConfig config(n);
        T val = static_cast<T>(fill_value);
        full_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            (T*)out.data<T>(), n, val
            );
    }

    // Complex specializations
    static void full_c32_impl(const Array& out, double fill_value) {
        int64_t n = out.numel();
        LaunchConfig config(n);
        float val = static_cast<float>(fill_value);
        full_c32_kernel << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            (std::complex<float>*)out.data<std::complex<float>>(), n, val, 0.0f
            );
    }

    static void full_c64_impl(const Array& out, double fill_value) {
        int64_t n = out.numel();
        LaunchConfig config(n);
        double val = static_cast<double>(fill_value);
        full_c64_kernel << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            (std::complex<double>*)out.data<std::complex<double>>(), n, val, 0.0
            );
    }

    static OpArgs full_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        double fill_value = std::any_cast<double>(args[1]);
        DType dtype = out.dtype();

        switch (dtype) {
        case DType::BOOL: full_impl<bool>(out, fill_value); break;
        case DType::U8:   full_impl<uint8_t>(out, fill_value); break;
        case DType::I8:   full_impl<int8_t>(out, fill_value); break;
        case DType::I16:  full_impl<int16_t>(out, fill_value); break;
        case DType::I32:  full_impl<int32_t>(out, fill_value); break;
        case DType::I64:  full_impl<int64_t>(out, fill_value); break;
        case DType::U16:  full_impl<uint16_t>(out, fill_value); break;
        case DType::U32:  full_impl<uint32_t>(out, fill_value); break;
        case DType::U64:  full_impl<uint64_t>(out, fill_value); break;
        case DType::F32:  full_impl<float>(out, fill_value); break;
        case DType::F64:  full_impl<double>(out, fill_value); break;
        case DType::C32:  full_c32_impl(out, fill_value); break;
        case DType::C64:  full_c64_impl(out, fill_value); break;
        default: INS_THROW("full: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(full, GPU, BOOL, full_wrapper);
    REGISTER_KERNEL(full, GPU, U8, full_wrapper);
    REGISTER_KERNEL(full, GPU, I8, full_wrapper);
    REGISTER_KERNEL(full, GPU, I16, full_wrapper);
    REGISTER_KERNEL(full, GPU, I32, full_wrapper);
    REGISTER_KERNEL(full, GPU, I64, full_wrapper);
    REGISTER_KERNEL(full, GPU, U16, full_wrapper);
    REGISTER_KERNEL(full, GPU, U32, full_wrapper);
    REGISTER_KERNEL(full, GPU, U64, full_wrapper);
    REGISTER_KERNEL(full, GPU, F32, full_wrapper);
    REGISTER_KERNEL(full, GPU, F64, full_wrapper);
    REGISTER_KERNEL(full, GPU, C32, full_wrapper);
    REGISTER_KERNEL(full, GPU, C64, full_wrapper);

    // ============================================================================
    // eye kernel - regular types
    // ============================================================================

    template<typename T>
    __global__ void eye_kernel(T* out, int64_t n, int64_t m, int64_t k) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        int64_t total = n * m;
        if (idx < total) {
            out[idx] = T(0);
        }
    }

    template<typename T>
    __global__ void eye_diag_kernel(T* out, int64_t n, int64_t m, int64_t k) {
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            int64_t j = i + k;
            if (j >= 0 && j < m) {
                out[i * m + j] = T(1);
            }
        }
    }

    // eye - complex (separate to avoid constexpr __host__ issues)
    __global__ void eye_c32_kernel(std::complex<float>* out, int64_t n, int64_t m, int64_t k) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        int64_t total = n * m;
        if (idx < total) {
            set_c32(&out[idx], 0.0f, 0.0f);
        }
    }

    __global__ void eye_c32_diag_kernel(std::complex<float>* out, int64_t n, int64_t m, int64_t k) {
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            int64_t j = i + k;
            if (j >= 0 && j < m) {
                set_c32(&out[i * m + j], 1.0f, 0.0f);
            }
        }
    }

    __global__ void eye_c64_kernel(std::complex<double>* out, int64_t n, int64_t m, int64_t k) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        int64_t total = n * m;
        if (idx < total) {
            set_c64(&out[idx], 0.0, 0.0);
        }
    }

    __global__ void eye_c64_diag_kernel(std::complex<double>* out, int64_t n, int64_t m, int64_t k) {
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            int64_t j = i + k;
            if (j >= 0 && j < m) {
                set_c64(&out[i * m + j], 1.0, 0.0);
            }
        }
    }

    template<typename T>
    static void eye_impl(const Array& out, int64_t k) {
        int64_t n = out.shape().dim(0);
        int64_t m = out.shape().dim(1);
        int64_t total = n * m;

        // Zero fill
        LaunchConfig config_zero(total);
        eye_kernel<T> << <config_zero.blocks, config_zero.threads, config_zero.shmSize, config_zero.stream >> > (
            (T*)out.data<T>(), n, m, k
            );

        // Set diagonal
        LaunchConfig config_diag(n);
        eye_diag_kernel<T> << <config_diag.blocks, config_diag.threads, config_diag.shmSize, config_diag.stream >> > (
            (T*)out.data<T>(), n, m, k
            );
    }

    // Complex specializations
    static void eye_c32_impl(const Array& out, int64_t k) {
        int64_t n = out.shape().dim(0);
        int64_t m = out.shape().dim(1);
        int64_t total = n * m;

        LaunchConfig config_zero(total);
        eye_c32_kernel << <config_zero.blocks, config_zero.threads, config_zero.shmSize, config_zero.stream >> > (
            (std::complex<float>*)out.data<std::complex<float>>(), n, m, k
            );

        LaunchConfig config_diag(n);
        eye_c32_diag_kernel << <config_diag.blocks, config_diag.threads, config_diag.shmSize, config_diag.stream >> > (
            (std::complex<float>*)out.data<std::complex<float>>(), n, m, k
            );
    }

    static void eye_c64_impl(const Array& out, int64_t k) {
        int64_t n = out.shape().dim(0);
        int64_t m = out.shape().dim(1);
        int64_t total = n * m;

        LaunchConfig config_zero(total);
        eye_c64_kernel << <config_zero.blocks, config_zero.threads, config_zero.shmSize, config_zero.stream >> > (
            (std::complex<double>*)out.data<std::complex<double>>(), n, m, k
            );

        LaunchConfig config_diag(n);
        eye_c64_diag_kernel << <config_diag.blocks, config_diag.threads, config_diag.shmSize, config_diag.stream >> > (
            (std::complex<double>*)out.data<std::complex<double>>(), n, m, k
            );
    }

    static OpArgs eye_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        int64_t k = std::any_cast<int64_t>(args[1]);
        DType dtype = out.dtype();

        switch (dtype) {
        case DType::BOOL: eye_impl<bool>(out, k); break;
        case DType::U8:   eye_impl<uint8_t>(out, k); break;
        case DType::I8:   eye_impl<int8_t>(out, k); break;
        case DType::I16:  eye_impl<int16_t>(out, k); break;
        case DType::I32:  eye_impl<int32_t>(out, k); break;
        case DType::I64:  eye_impl<int64_t>(out, k); break;
        case DType::U16:  eye_impl<uint16_t>(out, k); break;
        case DType::U32:  eye_impl<uint32_t>(out, k); break;
        case DType::U64:  eye_impl<uint64_t>(out, k); break;
        case DType::F32:  eye_impl<float>(out, k); break;
        case DType::F64:  eye_impl<double>(out, k); break;
        case DType::C32:  eye_c32_impl(out, k); break;
        case DType::C64:  eye_c64_impl(out, k); break;
        default: INS_THROW("eye: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(eye, GPU, BOOL, eye_wrapper);
    REGISTER_KERNEL(eye, GPU, U8, eye_wrapper);
    REGISTER_KERNEL(eye, GPU, I8, eye_wrapper);
    REGISTER_KERNEL(eye, GPU, I16, eye_wrapper);
    REGISTER_KERNEL(eye, GPU, I32, eye_wrapper);
    REGISTER_KERNEL(eye, GPU, I64, eye_wrapper);
    REGISTER_KERNEL(eye, GPU, U16, eye_wrapper);
    REGISTER_KERNEL(eye, GPU, U32, eye_wrapper);
    REGISTER_KERNEL(eye, GPU, U64, eye_wrapper);
    REGISTER_KERNEL(eye, GPU, F32, eye_wrapper);
    REGISTER_KERNEL(eye, GPU, F64, eye_wrapper);
    REGISTER_KERNEL(eye, GPU, C32, eye_wrapper);
    REGISTER_KERNEL(eye, GPU, C64, eye_wrapper);

    // ============================================================================
    // arange kernel
    // ============================================================================

    template<typename T>
    __global__ void arange_kernel(T* out, int64_t n, double start, double step) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = static_cast<T>(start + idx * step);
        }
    }

    template<typename T>
    static void arange_impl(const Array& out, double start, double step) {
        int64_t n = out.numel();
        LaunchConfig config(n);
        arange_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            (T*)out.data<T>(), n, start, step
            );
    }

    static OpArgs arange_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        double start = std::any_cast<double>(args[1]);
        double step = std::any_cast<double>(args[2]);
        DType dtype = out.dtype();

        switch (dtype) {
        case DType::F32: arange_impl<float>(out, start, step); break;
        case DType::F64: arange_impl<double>(out, start, step); break;
        case DType::I32: arange_impl<int32_t>(out, start, step); break;
        case DType::I64: arange_impl<int64_t>(out, start, step); break;
        default: INS_THROW("arange: unsupported dtype (only float32/64, int32/64)");
        }
        return { out };
    }

    REGISTER_KERNEL(arange, GPU, F32, arange_wrapper);
    REGISTER_KERNEL(arange, GPU, F64, arange_wrapper);
    REGISTER_KERNEL(arange, GPU, I32, arange_wrapper);
    REGISTER_KERNEL(arange, GPU, I64, arange_wrapper);

    // ============================================================================
    // linspace kernel
    // ============================================================================

    template<typename T>
    __global__ void linspace_kernel(T* out, int64_t n, double start, double step) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = static_cast<T>(start + idx * step);
        }
    }

    template<typename T>
    static void linspace_impl(const Array& out, double start, double stop) {
        int64_t n = out.numel();
        LaunchConfig config(n);
        double step = (n == 1) ? 0.0 : (stop - start) / (n - 1);
        linspace_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            (T*)out.data<T>(), n, start, step
            );
    }

    static OpArgs linspace_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        double start = std::any_cast<double>(args[1]);
        double stop = std::any_cast<double>(args[2]);
        DType dtype = out.dtype();

        switch (dtype) {
        case DType::F32: linspace_impl<float>(out, start, stop); break;
        case DType::F64: linspace_impl<double>(out, start, stop); break;
        default: INS_THROW("linspace: only float32/64 supported");
        }
        return { out };
    }

    REGISTER_KERNEL(linspace, GPU, F32, linspace_wrapper);
    REGISTER_KERNEL(linspace, GPU, F64, linspace_wrapper);

    // ============================================================================
    // logspace kernel
    // ============================================================================

    template<typename T>
    __global__ void logspace_kernel(T* out, int64_t n, double start, double step, double base) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = static_cast<T>(pow(base, start + idx * step));
        }
    }

    template<typename T>
    static void logspace_impl(const Array& out, double start, double stop, double base) {
        int64_t n = out.numel();
        LaunchConfig config(n);
        double step = (n == 1) ? 0.0 : (stop - start) / (n - 1);
        logspace_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            (T*)out.data<T>(), n, start, step, base
            );
    }

    static OpArgs logspace_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        double start = std::any_cast<double>(args[1]);
        double stop = std::any_cast<double>(args[2]);
        double base = std::any_cast<double>(args[3]);
        DType dtype = out.dtype();

        switch (dtype) {
        case DType::F32: logspace_impl<float>(out, start, stop, base); break;
        case DType::F64: logspace_impl<double>(out, start, stop, base); break;
        default: INS_THROW("logspace: only float32/64 supported");
        }
        return { out };
    }

    REGISTER_KERNEL(logspace, GPU, F32, logspace_wrapper);
    REGISTER_KERNEL(logspace, GPU, F64, logspace_wrapper);

} // namespace ins::gpu

// ============================================================================
// Module registration
// ============================================================================

REGISTER_MODULE(creation, GPU);