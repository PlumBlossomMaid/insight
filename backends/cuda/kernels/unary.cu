// backends/cuda/kernels/unary.cu
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
    // Device helper: compute offset from linear index using strides
    // Assumes ndim <= 4 (practical for most tensor operations)
    // ============================================================================

    __device__ inline int64_t compute_offset(
        int64_t linear, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ strides
    ) {
        int64_t offset = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            offset += (linear % dims[d]) * strides[d];
            linear /= dims[d];
        }
        return offset;
    }

    // ============================================================================
    // Basic math kernels (abs, negative, square) - with stride support
    // ============================================================================

    // abs - signed integer
    template<typename T>
    __global__ void abs_signed_kernel(
        const T* __restrict__ x, T* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            T v = x[offset];
            out[idx] = v < 0 ? -v : v;
        }
    }

    // abs - unsigned (no-op)
    template<typename T>
    __global__ void abs_unsigned_kernel(
        const T* __restrict__ x, T* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            out[idx] = x[offset];
        }
    }

    // abs - float
    __global__ void abs_f32_kernel(
        const float* __restrict__ x, float* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            out[idx] = fabsf(x[offset]);
        }
    }

    __global__ void abs_f64_kernel(
        const double* __restrict__ x, double* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            out[idx] = fabs(x[offset]);
        }
    }

    // abs - complex
    __global__ void abs_c32_kernel(
        const std::complex<float>* __restrict__ x, std::complex<float>* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            float r = real(x[offset]);
            float im = imag(x[offset]);
            float norm = sqrtf(r * r + im * im);
            set_c32(&out[idx], norm, 0.0f);
        }
    }

    __global__ void abs_c64_kernel(
        const std::complex<double>* __restrict__ x, std::complex<double>* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            double r = real(x[offset]);
            double im = imag(x[offset]);
            double norm = sqrt(r * r + im * im);
            set_c64(&out[idx], norm, 0.0);
        }
    }

    // negative - regular types
    template<typename T>
    __global__ void negative_kernel(
        const T* __restrict__ x, T* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            out[idx] = -x[offset];
        }
    }

    // negative - complex (separate to avoid constexpr __host__ issues)
    __global__ void negative_c32_kernel(
        const std::complex<float>* __restrict__ x, std::complex<float>* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            set_c32(&out[idx], -real(x[offset]), -imag(x[offset]));
        }
    }

    __global__ void negative_c64_kernel(
        const std::complex<double>* __restrict__ x, std::complex<double>* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            set_c64(&out[idx], -real(x[offset]), -imag(x[offset]));
        }
    }

    // square - regular types
    template<typename T>
    __global__ void square_kernel(
        const T* __restrict__ x, T* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            out[idx] = x[offset] * x[offset];
        }
    }

    // square - complex
    __global__ void square_c32_kernel(
        const std::complex<float>* __restrict__ x, std::complex<float>* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            float r = real(x[offset]);
            float im = imag(x[offset]);
            set_c32(&out[idx], r * r - im * im, 2.0f * r * im);
        }
    }

    __global__ void square_c64_kernel(
        const std::complex<double>* __restrict__ x, std::complex<double>* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            double r = real(x[offset]);
            double im = imag(x[offset]);
            set_c64(&out[idx], r * r - im * im, 2.0 * r * im);
        }
    }

    // ============================================================================
    // Floating-point math kernels - with stride support
    // ============================================================================

#define DEFINE_FLOAT_UNARY_KERNEL(name, f32_func, f64_func) \
        __global__ void name##_f32_kernel( \
            const float* __restrict__ x, float* __restrict__ out, int64_t n, \
            int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides \
        ) { \
            int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; \
            if (idx < n) { \
                int64_t offset = compute_offset(idx, ndim, dims, strides); \
                out[idx] = f32_func(x[offset]); \
            } \
        } \
        __global__ void name##_f64_kernel( \
            const double* __restrict__ x, double* __restrict__ out, int64_t n, \
            int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides \
        ) { \
            int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; \
            if (idx < n) { \
                int64_t offset = compute_offset(idx, ndim, dims, strides); \
                out[idx] = f64_func(x[offset]); \
            } \
        }

    DEFINE_FLOAT_UNARY_KERNEL(exp, expf, exp)
        DEFINE_FLOAT_UNARY_KERNEL(exp2, exp2f, exp2)
        DEFINE_FLOAT_UNARY_KERNEL(expm1, expm1f, expm1)
        DEFINE_FLOAT_UNARY_KERNEL(log, logf, log)
        DEFINE_FLOAT_UNARY_KERNEL(log2, log2f, log2)
        DEFINE_FLOAT_UNARY_KERNEL(log10, log10f, log10)
        DEFINE_FLOAT_UNARY_KERNEL(log1p, log1pf, log1p)
        DEFINE_FLOAT_UNARY_KERNEL(sqrt, sqrtf, sqrt)
        DEFINE_FLOAT_UNARY_KERNEL(cbrt, cbrtf, cbrt)
        DEFINE_FLOAT_UNARY_KERNEL(sin, sinf, sin)
        DEFINE_FLOAT_UNARY_KERNEL(cos, cosf, cos)
        DEFINE_FLOAT_UNARY_KERNEL(tan, tanf, tan)
        DEFINE_FLOAT_UNARY_KERNEL(asin, asinf, asin)
        DEFINE_FLOAT_UNARY_KERNEL(acos, acosf, acos)
        DEFINE_FLOAT_UNARY_KERNEL(atan, atanf, atan)
        DEFINE_FLOAT_UNARY_KERNEL(sinh, sinhf, sinh)
        DEFINE_FLOAT_UNARY_KERNEL(cosh, coshf, cosh)
        DEFINE_FLOAT_UNARY_KERNEL(tanh, tanhf, tanh)
        DEFINE_FLOAT_UNARY_KERNEL(asinh, asinhf, asinh)
        DEFINE_FLOAT_UNARY_KERNEL(acosh, acoshf, acosh)
        DEFINE_FLOAT_UNARY_KERNEL(atanh, atanhf, atanh)
        DEFINE_FLOAT_UNARY_KERNEL(floor, floorf, floor)
        DEFINE_FLOAT_UNARY_KERNEL(ceil, ceilf, ceil)
        DEFINE_FLOAT_UNARY_KERNEL(trunc, truncf, trunc)
        DEFINE_FLOAT_UNARY_KERNEL(rint, rintf, rint)

#undef DEFINE_FLOAT_UNARY_KERNEL

        // reciprocal (separate because it's not a math function)
        __global__ void reciprocal_f32_kernel(
            const float* __restrict__ x, float* __restrict__ out, int64_t n,
            int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
        ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            out[idx] = 1.0f / x[offset];
        }
    }

    __global__ void reciprocal_f64_kernel(
        const double* __restrict__ x, double* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            out[idx] = 1.0 / x[offset];
        }
    }

    // ============================================================================
    // Sign kernels - with stride support
    // ============================================================================

    template<typename T>
    __global__ void sign_unsigned_kernel(
        const T* __restrict__ x, T* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            out[idx] = x[offset] > 0 ? 1 : 0;
        }
    }

    template<typename T>
    __global__ void sign_signed_kernel(
        const T* __restrict__ x, T* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            T v = x[offset];
            out[idx] = v > 0 ? 1 : (v < 0 ? -1 : 0);
        }
    }

    __global__ void sign_f32_kernel(
        const float* __restrict__ x, float* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            float v = x[offset];
            out[idx] = v > 0 ? 1.0f : (v < 0 ? -1.0f : 0.0f);
        }
    }

    __global__ void sign_f64_kernel(
        const double* __restrict__ x, double* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            double v = x[offset];
            out[idx] = v > 0 ? 1.0 : (v < 0 ? -1.0 : 0.0);
        }
    }

    __global__ void sign_c32_kernel(
        const std::complex<float>* __restrict__ x, std::complex<float>* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            float r = real(x[offset]);
            float im = imag(x[offset]);
            float norm = sqrtf(r * r + im * im);
            if (norm == 0) {
                set_c32(&out[idx], 0.0f, 0.0f);
            }
            else {
                set_c32(&out[idx], r / norm, im / norm);
            }
        }
    }

    __global__ void sign_c64_kernel(
        const std::complex<double>* __restrict__ x, std::complex<double>* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            double r = real(x[offset]);
            double im = imag(x[offset]);
            double norm = sqrt(r * r + im * im);
            if (norm == 0) {
                set_c64(&out[idx], 0.0, 0.0);
            }
            else {
                set_c64(&out[idx], r / norm, im / norm);
            }
        }
    }

    __global__ void sign_bool_kernel(
        const bool* __restrict__ x, int32_t* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            out[idx] = x[offset] ? 1 : 0;
        }
    }

    // ============================================================================
    // Conjugate kernels - with stride support
    // ============================================================================

    __global__ void conj_c32_kernel(
        const std::complex<float>* __restrict__ x, std::complex<float>* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            set_c32(&out[idx], real(x[offset]), -imag(x[offset]));
        }
    }

    __global__ void conj_c64_kernel(
        const std::complex<double>* __restrict__ x, std::complex<double>* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            set_c64(&out[idx], real(x[offset]), -imag(x[offset]));
        }
    }

    template<typename T>
    __global__ void conj_real_kernel(
        const T* __restrict__ x, T* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            out[idx] = x[offset];
        }
    }

    // ============================================================================
    // deg2rad / rad2deg kernels - with stride support
    // ============================================================================

    __global__ void deg2rad_f32_kernel(
        const float* __restrict__ x, float* __restrict__ out, int64_t n, float factor,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            out[idx] = x[offset] * factor;
        }
    }

    __global__ void deg2rad_f64_kernel(
        const double* __restrict__ x, double* __restrict__ out, int64_t n, double factor,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            out[idx] = x[offset] * factor;
        }
    }

    __global__ void rad2deg_f32_kernel(
        const float* __restrict__ x, float* __restrict__ out, int64_t n, float factor,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            out[idx] = x[offset] * factor;
        }
    }

    __global__ void rad2deg_f64_kernel(
        const double* __restrict__ x, double* __restrict__ out, int64_t n, double factor,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            out[idx] = x[offset] * factor;
        }
    }

    // ============================================================================
    // logical_not kernels - with stride support
    // ============================================================================

    template<typename T>
    __global__ void logical_not_kernel(
        const T* __restrict__ x, bool* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            out[idx] = (x[offset] == T(0));
        }
    }

    __global__ void logical_not_c32_kernel(
        const std::complex<float>* __restrict__ x, bool* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            out[idx] = (real(x[offset]) == 0.0f && imag(x[offset]) == 0.0f);
        }
    }

    __global__ void logical_not_c64_kernel(
        const std::complex<double>* __restrict__ x, bool* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            out[idx] = (real(x[offset]) == 0.0 && imag(x[offset]) == 0.0);
        }
    }

    // ============================================================================
    // bitwise_not kernels - with stride support
    // ============================================================================

    template<typename T>
    __global__ void bitwise_not_kernel(
        const T* __restrict__ x, T* __restrict__ out, int64_t n,
        int ndim, const int64_t* __restrict__ dims, const int64_t* __restrict__ strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t offset = compute_offset(idx, ndim, dims, strides);
            out[idx] = ~x[offset];
        }
    }

    // ============================================================================
    // Helper: prepare dims and strides arrays on device
    // ============================================================================

    struct StrideInfo {
        int ndim;
        int64_t dims[4];
        int64_t strides[4];
    };

    static StrideInfo prepare_stride_info(const Array& x) {
        StrideInfo info;
        info.ndim = x.shape().ndim();
        for (int i = 0; i < info.ndim; ++i) {
            info.dims[i] = x.shape().dim(i);
            info.strides[i] = x.strides()[i];
        }
        // Fill remaining with 1 to avoid uninitialized reads
        for (int i = info.ndim; i < 4; ++i) {
            info.dims[i] = 1;
            info.strides[i] = 0;
        }
        return info;
    }

    // ============================================================================
    // Helper: launch kernel with stride info
    // ============================================================================

    template<typename KernelFunc, typename... Args>
    static void launch_with_strides(KernelFunc kernel, const Array& x, Array& out, Args... args) {
        int64_t n = x.numel();
        LaunchConfig config(n);
        StrideInfo info = prepare_stride_info(x);

        // Copy dims and strides to device
        int64_t* d_dims;
        int64_t* d_strides;
        cudaMalloc(&d_dims, 4 * sizeof(int64_t));
        cudaMalloc(&d_strides, 4 * sizeof(int64_t));
        cudaMemcpy(d_dims, info.dims, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_strides, info.strides, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);

        kernel << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            args..., n, info.ndim, d_dims, d_strides);

        cudaFree(d_dims);
        cudaFree(d_strides);
    }

    // ============================================================================
    // Helper macro for stride-based float kernels
    // ============================================================================

#define LAUNCH_FLOAT_STRIDE_KERNEL(kernel_f32, kernel_f64, x, out) \
        do { \
            int64_t n = (x).numel(); \
            LaunchConfig config(n); \
            StrideInfo info = prepare_stride_info(x); \
            int64_t* d_dims; \
            int64_t* d_strides; \
            cudaMalloc(&d_dims, 4 * sizeof(int64_t)); \
            cudaMalloc(&d_strides, 4 * sizeof(int64_t)); \
            cudaMemcpy(d_dims, info.dims, 4 * sizeof(int64_t), cudaMemcpyHostToDevice); \
            cudaMemcpy(d_strides, info.strides, 4 * sizeof(int64_t), cudaMemcpyHostToDevice); \
            kernel_f32<<<config.blocks, config.threads, config.shmSize, config.stream>>>( \
                (x).data<float>(), (out).data<float>(), n, info.ndim, d_dims, d_strides); \
            cudaFree(d_dims); \
            cudaFree(d_strides); \
        } while(0)

    // ============================================================================
    // Implementation functions - abs
    // ============================================================================

    template<typename T>
    static void abs_signed_impl(const Array& x, Array& out) {
        launch_with_strides(abs_signed_kernel<T>, x, out, x.data<T>(), out.data<T>());
    }

    template<typename T>
    static void abs_unsigned_impl(const Array& x, Array& out) {
        launch_with_strides(abs_unsigned_kernel<T>, x, out, x.data<T>(), out.data<T>());
    }

    static void abs_f32_impl(const Array& x, Array& out) {
        launch_with_strides(abs_f32_kernel, x, out, x.data<float>(), out.data<float>());
    }

    static void abs_f64_impl(const Array& x, Array& out) {
        launch_with_strides(abs_f64_kernel, x, out, x.data<double>(), out.data<double>());
    }

    static void abs_c32_impl(const Array& x, Array& out) {
        launch_with_strides(abs_c32_kernel, x, out, x.data<std::complex<float>>(), out.data<std::complex<float>>());
    }

    static void abs_c64_impl(const Array& x, Array& out) {
        launch_with_strides(abs_c64_kernel, x, out, x.data<std::complex<double>>(), out.data<std::complex<double>>());
    }

    // ============================================================================
    // Implementation functions - negative
    // ============================================================================

    template<typename T>
    static void negative_impl(const Array& x, Array& out) {
        launch_with_strides(negative_kernel<T>, x, out, x.data<T>(), out.data<T>());
    }

    static void negative_c32_impl(const Array& x, Array& out) {
        launch_with_strides(negative_c32_kernel, x, out, x.data<std::complex<float>>(), out.data<std::complex<float>>());
    }

    static void negative_c64_impl(const Array& x, Array& out) {
        launch_with_strides(negative_c64_kernel, x, out, x.data<std::complex<double>>(), out.data<std::complex<double>>());
    }

    // ============================================================================
    // Implementation functions - square
    // ============================================================================

    template<typename T>
    static void square_impl(const Array& x, Array& out) {
        launch_with_strides(square_kernel<T>, x, out, x.data<T>(), out.data<T>());
    }

    static void square_c32_impl(const Array& x, Array& out) {
        launch_with_strides(square_c32_kernel, x, out, x.data<std::complex<float>>(), out.data<std::complex<float>>());
    }

    static void square_c64_impl(const Array& x, Array& out) {
        launch_with_strides(square_c64_kernel, x, out, x.data<std::complex<double>>(), out.data<std::complex<double>>());
    }

    // ============================================================================
    // Implementation functions - float math (using macros)
    // ============================================================================

#define DEFINE_FLOAT_IMPL(name) \
        static void name##_f32_impl(const Array& x, Array& out) { \
            launch_with_strides(name##_f32_kernel, x, out, x.data<float>(), out.data<float>()); \
        } \
        static void name##_f64_impl(const Array& x, Array& out) { \
            launch_with_strides(name##_f64_kernel, x, out, x.data<double>(), out.data<double>()); \
        }

    DEFINE_FLOAT_IMPL(exp)
        DEFINE_FLOAT_IMPL(exp2)
        DEFINE_FLOAT_IMPL(expm1)
        DEFINE_FLOAT_IMPL(log)
        DEFINE_FLOAT_IMPL(log2)
        DEFINE_FLOAT_IMPL(log10)
        DEFINE_FLOAT_IMPL(log1p)
        DEFINE_FLOAT_IMPL(sqrt)
        DEFINE_FLOAT_IMPL(cbrt)
        DEFINE_FLOAT_IMPL(sin)
        DEFINE_FLOAT_IMPL(cos)
        DEFINE_FLOAT_IMPL(tan)
        DEFINE_FLOAT_IMPL(asin)
        DEFINE_FLOAT_IMPL(acos)
        DEFINE_FLOAT_IMPL(atan)
        DEFINE_FLOAT_IMPL(sinh)
        DEFINE_FLOAT_IMPL(cosh)
        DEFINE_FLOAT_IMPL(tanh)
        DEFINE_FLOAT_IMPL(asinh)
        DEFINE_FLOAT_IMPL(acosh)
        DEFINE_FLOAT_IMPL(atanh)
        DEFINE_FLOAT_IMPL(floor)
        DEFINE_FLOAT_IMPL(ceil)
        DEFINE_FLOAT_IMPL(trunc)
        DEFINE_FLOAT_IMPL(rint)

#undef DEFINE_FLOAT_IMPL

        static void reciprocal_f32_impl(const Array& x, Array& out) {
        launch_with_strides(reciprocal_f32_kernel, x, out, x.data<float>(), out.data<float>());
    }

    static void reciprocal_f64_impl(const Array& x, Array& out) {
        launch_with_strides(reciprocal_f64_kernel, x, out, x.data<double>(), out.data<double>());
    }

    // ============================================================================
    // Implementation functions - sign
    // ============================================================================

    template<typename T>
    static void sign_unsigned_impl(const Array& x, Array& out) {
        launch_with_strides(sign_unsigned_kernel<T>, x, out, x.data<T>(), out.data<T>());
    }

    template<typename T>
    static void sign_signed_impl(const Array& x, Array& out) {
        launch_with_strides(sign_signed_kernel<T>, x, out, x.data<T>(), out.data<T>());
    }

    static void sign_f32_impl(const Array& x, Array& out) {
        launch_with_strides(sign_f32_kernel, x, out, x.data<float>(), out.data<float>());
    }

    static void sign_f64_impl(const Array& x, Array& out) {
        launch_with_strides(sign_f64_kernel, x, out, x.data<double>(), out.data<double>());
    }

    static void sign_c32_impl(const Array& x, Array& out) {
        launch_with_strides(sign_c32_kernel, x, out, x.data<std::complex<float>>(), out.data<std::complex<float>>());
    }

    static void sign_c64_impl(const Array& x, Array& out) {
        launch_with_strides(sign_c64_kernel, x, out, x.data<std::complex<double>>(), out.data<std::complex<double>>());
    }

    static void sign_bool_impl(const Array& x, Array& out) {
        launch_with_strides(sign_bool_kernel, x, out, x.data<bool>(), out.data<int32_t>());
    }

    // ============================================================================
    // Implementation functions - conj
    // ============================================================================

    static void conj_c32_impl(const Array& x, Array& out) {
        launch_with_strides(conj_c32_kernel, x, out, x.data<std::complex<float>>(), out.data<std::complex<float>>());
    }

    static void conj_c64_impl(const Array& x, Array& out) {
        launch_with_strides(conj_c64_kernel, x, out, x.data<std::complex<double>>(), out.data<std::complex<double>>());
    }

    template<typename T>
    static void conj_real_impl(const Array& x, Array& out) {
        launch_with_strides(conj_real_kernel<T>, x, out, x.data<T>(), out.data<T>());
    }

    // ============================================================================
    // Implementation functions - logical_not
    // ============================================================================

    template<typename T>
    static void logical_not_impl(const Array& x, Array& out) {
        launch_with_strides(logical_not_kernel<T>, x, out, x.data<T>(), out.data<bool>());
    }

    static void logical_not_c32_impl(const Array& x, Array& out) {
        launch_with_strides(logical_not_c32_kernel, x, out, x.data<std::complex<float>>(), out.data<bool>());
    }

    static void logical_not_c64_impl(const Array& x, Array& out) {
        launch_with_strides(logical_not_c64_kernel, x, out, x.data<std::complex<double>>(), out.data<bool>());
    }

    // ============================================================================
    // Implementation functions - bitwise_not
    // ============================================================================

    template<typename T>
    static void bitwise_not_impl(const Array& x, Array& out) {
        launch_with_strides(bitwise_not_kernel<T>, x, out, x.data<T>(), out.data<T>());
    }

    // ============================================================================
    // Wrapper functions (for operator registration)
    // ============================================================================

    // abs wrapper
    static OpArgs abs_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        Array out(x.shape(), x.dtype(), x.place());

        switch (x.dtype()) {
        case DType::I8:    abs_signed_impl<int8_t>(x, out); break;
        case DType::I16:   abs_signed_impl<int16_t>(x, out); break;
        case DType::I32:   abs_signed_impl<int32_t>(x, out); break;
        case DType::I64:   abs_signed_impl<int64_t>(x, out); break;
        case DType::U8:    abs_unsigned_impl<uint8_t>(x, out); break;
        case DType::U16:   abs_unsigned_impl<uint16_t>(x, out); break;
        case DType::U32:   abs_unsigned_impl<uint32_t>(x, out); break;
        case DType::U64:   abs_unsigned_impl<uint64_t>(x, out); break;
        case DType::F32:   abs_f32_impl(x, out); break;
        case DType::F64:   abs_f64_impl(x, out); break;
        case DType::C32:   abs_c32_impl(x, out); break;
        case DType::C64:   abs_c64_impl(x, out); break;
        case DType::BOOL:  abs_unsigned_impl<bool>(x, out); break;
        default: INS_THROW("abs: unsupported dtype ", dtype_name(x.dtype()));
        }
        return { out };
    }

    // negative wrapper
    static OpArgs negative_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        Array out(x.shape(), x.dtype(), x.place());

        switch (x.dtype()) {
        case DType::I8:    negative_impl<int8_t>(x, out); break;
        case DType::I16:   negative_impl<int16_t>(x, out); break;
        case DType::I32:   negative_impl<int32_t>(x, out); break;
        case DType::I64:   negative_impl<int64_t>(x, out); break;
        case DType::F32:   negative_impl<float>(x, out); break;
        case DType::F64:   negative_impl<double>(x, out); break;
        case DType::C32:   negative_c32_impl(x, out); break;
        case DType::C64:   negative_c64_impl(x, out); break;
        default: INS_THROW("negative: unsupported dtype ", dtype_name(x.dtype()));
        }
        return { out };
    }

    // square wrapper
    static OpArgs square_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        Array out(x.shape(), x.dtype(), x.place());

        switch (x.dtype()) {
        case DType::I8:    square_impl<int8_t>(x, out); break;
        case DType::I16:   square_impl<int16_t>(x, out); break;
        case DType::I32:   square_impl<int32_t>(x, out); break;
        case DType::I64:   square_impl<int64_t>(x, out); break;
        case DType::U8:    square_impl<uint8_t>(x, out); break;
        case DType::U16:   square_impl<uint16_t>(x, out); break;
        case DType::U32:   square_impl<uint32_t>(x, out); break;
        case DType::U64:   square_impl<uint64_t>(x, out); break;
        case DType::F32:   square_impl<float>(x, out); break;
        case DType::F64:   square_impl<double>(x, out); break;
        case DType::C32:   square_c32_impl(x, out); break;
        case DType::C64:   square_c64_impl(x, out); break;
        case DType::BOOL:  square_impl<bool>(x, out); break;
        default: INS_THROW("square: unsupported dtype ", dtype_name(x.dtype()));
        }
        return { out };
    }

    // Float math wrappers
#define DEFINE_FLOAT_WRAPPER(name) \
        static OpArgs name##_wrapper(const OpArgs& args) { \
            const Array& x = std::any_cast<const Array&>(args[0]); \
            Array out(x.shape(), x.dtype(), x.place()); \
            switch (x.dtype()) { \
            case DType::F32: name##_f32_impl(x, out); break; \
            case DType::F64: name##_f64_impl(x, out); break; \
            default: INS_THROW(#name ": only float32 and float64 are supported"); \
            } \
            return { out }; \
        }

    DEFINE_FLOAT_WRAPPER(exp)
        DEFINE_FLOAT_WRAPPER(exp2)
        DEFINE_FLOAT_WRAPPER(expm1)
        DEFINE_FLOAT_WRAPPER(log)
        DEFINE_FLOAT_WRAPPER(log2)
        DEFINE_FLOAT_WRAPPER(log10)
        DEFINE_FLOAT_WRAPPER(log1p)
        DEFINE_FLOAT_WRAPPER(sqrt)
        DEFINE_FLOAT_WRAPPER(cbrt)

        static OpArgs reciprocal_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        Array out(x.shape(), x.dtype(), x.place());
        switch (x.dtype()) {
        case DType::F32: reciprocal_f32_impl(x, out); break;
        case DType::F64: reciprocal_f64_impl(x, out); break;
        default: INS_THROW("reciprocal: only float32 and float64 are supported");
        }
        return { out };
    }

    DEFINE_FLOAT_WRAPPER(sin)
        DEFINE_FLOAT_WRAPPER(cos)
        DEFINE_FLOAT_WRAPPER(tan)
        DEFINE_FLOAT_WRAPPER(asin)
        DEFINE_FLOAT_WRAPPER(acos)
        DEFINE_FLOAT_WRAPPER(atan)
        DEFINE_FLOAT_WRAPPER(sinh)
        DEFINE_FLOAT_WRAPPER(cosh)
        DEFINE_FLOAT_WRAPPER(tanh)
        DEFINE_FLOAT_WRAPPER(asinh)
        DEFINE_FLOAT_WRAPPER(acosh)
        DEFINE_FLOAT_WRAPPER(atanh)
        DEFINE_FLOAT_WRAPPER(floor)
        DEFINE_FLOAT_WRAPPER(ceil)
        DEFINE_FLOAT_WRAPPER(trunc)
        DEFINE_FLOAT_WRAPPER(rint)

#undef DEFINE_FLOAT_WRAPPER

        // sign wrapper
        static OpArgs sign_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        Array out(x.shape(), x.dtype(), x.place());

        switch (x.dtype()) {
        case DType::U8:    sign_unsigned_impl<uint8_t>(x, out); break;
        case DType::U16:   sign_unsigned_impl<uint16_t>(x, out); break;
        case DType::U32:   sign_unsigned_impl<uint32_t>(x, out); break;
        case DType::U64:   sign_unsigned_impl<uint64_t>(x, out); break;
        case DType::I8:    sign_signed_impl<int8_t>(x, out); break;
        case DType::I16:   sign_signed_impl<int16_t>(x, out); break;
        case DType::I32:   sign_signed_impl<int32_t>(x, out); break;
        case DType::I64:   sign_signed_impl<int64_t>(x, out); break;
        case DType::F32:   sign_f32_impl(x, out); break;
        case DType::F64:   sign_f64_impl(x, out); break;
        case DType::C32:   sign_c32_impl(x, out); break;
        case DType::C64:   sign_c64_impl(x, out); break;
        case DType::BOOL:  sign_bool_impl(x, out); break;
        default: INS_THROW("sign: unsupported dtype ", dtype_name(x.dtype()));
        }
        return { out };
    }

    // conj wrapper
    static OpArgs conj_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        Array out(x.shape(), x.dtype(), x.place());

        switch (x.dtype()) {
        case DType::C32:   conj_c32_impl(x, out); break;
        case DType::C64:   conj_c64_impl(x, out); break;
        case DType::F32:   conj_real_impl<float>(x, out); break;
        case DType::F64:   conj_real_impl<double>(x, out); break;
        case DType::I8:    conj_real_impl<int8_t>(x, out); break;
        case DType::I16:   conj_real_impl<int16_t>(x, out); break;
        case DType::I32:   conj_real_impl<int32_t>(x, out); break;
        case DType::I64:   conj_real_impl<int64_t>(x, out); break;
        case DType::U8:    conj_real_impl<uint8_t>(x, out); break;
        case DType::U16:   conj_real_impl<uint16_t>(x, out); break;
        case DType::U32:   conj_real_impl<uint32_t>(x, out); break;
        case DType::U64:   conj_real_impl<uint64_t>(x, out); break;
        case DType::BOOL:  conj_real_impl<bool>(x, out); break;
        default: INS_THROW("conj: unsupported dtype ", dtype_name(x.dtype()));
        }
        return { out };
    }

    // deg2rad wrapper
    static OpArgs deg2rad_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        Array out(x.shape(), x.dtype(), x.place());
        int64_t n = x.numel();
        LaunchConfig config(n);
        StrideInfo info = prepare_stride_info(x);

        int64_t* d_dims;
        int64_t* d_strides;
        cudaMalloc(&d_dims, 4 * sizeof(int64_t));
        cudaMalloc(&d_strides, 4 * sizeof(int64_t));
        cudaMemcpy(d_dims, info.dims, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_strides, info.strides, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);

        switch (x.dtype()) {
        case DType::F32: {
            float factor = 3.141592653589793f / 180.0f;
            deg2rad_f32_kernel << <config.blocks, config.threads, config.shmSize, config.stream >> > (
                x.data<float>(), out.data<float>(), n, factor, info.ndim, d_dims, d_strides);
            break;
        }
        case DType::F64: {
            double factor = 3.141592653589793 / 180.0;
            deg2rad_f64_kernel << <config.blocks, config.threads, config.shmSize, config.stream >> > (
                x.data<double>(), out.data<double>(), n, factor, info.ndim, d_dims, d_strides);
            break;
        }
        default:
            cudaFree(d_dims);
            cudaFree(d_strides);
            INS_THROW("deg2rad: only float32 and float64 are supported");
        }

        cudaFree(d_dims);
        cudaFree(d_strides);
        return { out };
    }

    // rad2deg wrapper
    static OpArgs rad2deg_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        Array out(x.shape(), x.dtype(), x.place());
        int64_t n = x.numel();
        LaunchConfig config(n);
        StrideInfo info = prepare_stride_info(x);

        int64_t* d_dims;
        int64_t* d_strides;
        cudaMalloc(&d_dims, 4 * sizeof(int64_t));
        cudaMalloc(&d_strides, 4 * sizeof(int64_t));
        cudaMemcpy(d_dims, info.dims, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_strides, info.strides, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);

        switch (x.dtype()) {
        case DType::F32: {
            float factor = 180.0f / 3.141592653589793f;
            rad2deg_f32_kernel << <config.blocks, config.threads, config.shmSize, config.stream >> > (
                x.data<float>(), out.data<float>(), n, factor, info.ndim, d_dims, d_strides);
            break;
        }
        case DType::F64: {
            double factor = 180.0 / 3.141592653589793;
            rad2deg_f64_kernel << <config.blocks, config.threads, config.shmSize, config.stream >> > (
                x.data<double>(), out.data<double>(), n, factor, info.ndim, d_dims, d_strides);
            break;
        }
        default:
            cudaFree(d_dims);
            cudaFree(d_strides);
            INS_THROW("rad2deg: only float32 and float64 are supported");
        }

        cudaFree(d_dims);
        cudaFree(d_strides);
        return { out };
    }

    // logical_not wrapper
    static OpArgs logical_not_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        Array out(x.shape(), DType::BOOL, x.place());

        switch (x.dtype()) {
        case DType::BOOL: logical_not_impl<bool>(x, out); break;
        case DType::U8:   logical_not_impl<uint8_t>(x, out); break;
        case DType::I8:   logical_not_impl<int8_t>(x, out); break;
        case DType::I16:  logical_not_impl<int16_t>(x, out); break;
        case DType::I32:  logical_not_impl<int32_t>(x, out); break;
        case DType::I64:  logical_not_impl<int64_t>(x, out); break;
        case DType::U16:  logical_not_impl<uint16_t>(x, out); break;
        case DType::U32:  logical_not_impl<uint32_t>(x, out); break;
        case DType::U64:  logical_not_impl<uint64_t>(x, out); break;
        case DType::F32:  logical_not_impl<float>(x, out); break;
        case DType::F64:  logical_not_impl<double>(x, out); break;
        case DType::C32:  logical_not_c32_impl(x, out); break;
        case DType::C64:  logical_not_c64_impl(x, out); break;
        default: INS_THROW("logical_not: unsupported dtype ", dtype_name(x.dtype()));
        }
        return { out };
    }

    // bitwise_not wrapper
    static OpArgs bitwise_not_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        Array out(x.shape(), x.dtype(), x.place());

        switch (x.dtype()) {
        case DType::U8:   bitwise_not_impl<uint8_t>(x, out); break;
        case DType::I8:   bitwise_not_impl<int8_t>(x, out); break;
        case DType::I16:  bitwise_not_impl<int16_t>(x, out); break;
        case DType::I32:  bitwise_not_impl<int32_t>(x, out); break;
        case DType::I64:  bitwise_not_impl<int64_t>(x, out); break;
        case DType::U16:  bitwise_not_impl<uint16_t>(x, out); break;
        case DType::U32:  bitwise_not_impl<uint32_t>(x, out); break;
        case DType::U64:  bitwise_not_impl<uint64_t>(x, out); break;
        case DType::BOOL: bitwise_not_impl<bool>(x, out); break;
        default: INS_THROW("bitwise_not: unsupported dtype ", dtype_name(x.dtype()));
        }
        return { out };
    }

    // ============================================================================
    // Kernel Registration
    // ============================================================================

    // abs
    REGISTER_KERNEL(abs, GPU, I8, abs_wrapper);
    REGISTER_KERNEL(abs, GPU, I16, abs_wrapper);
    REGISTER_KERNEL(abs, GPU, I32, abs_wrapper);
    REGISTER_KERNEL(abs, GPU, I64, abs_wrapper);
    REGISTER_KERNEL(abs, GPU, U8, abs_wrapper);
    REGISTER_KERNEL(abs, GPU, U16, abs_wrapper);
    REGISTER_KERNEL(abs, GPU, U32, abs_wrapper);
    REGISTER_KERNEL(abs, GPU, U64, abs_wrapper);
    REGISTER_KERNEL(abs, GPU, F32, abs_wrapper);
    REGISTER_KERNEL(abs, GPU, F64, abs_wrapper);
    REGISTER_KERNEL(abs, GPU, C32, abs_wrapper);
    REGISTER_KERNEL(abs, GPU, C64, abs_wrapper);
    REGISTER_KERNEL(abs, GPU, BOOL, abs_wrapper);

    // negative
    REGISTER_KERNEL(negative, GPU, I8, negative_wrapper);
    REGISTER_KERNEL(negative, GPU, I16, negative_wrapper);
    REGISTER_KERNEL(negative, GPU, I32, negative_wrapper);
    REGISTER_KERNEL(negative, GPU, I64, negative_wrapper);
    REGISTER_KERNEL(negative, GPU, F32, negative_wrapper);
    REGISTER_KERNEL(negative, GPU, F64, negative_wrapper);
    REGISTER_KERNEL(negative, GPU, C32, negative_wrapper);
    REGISTER_KERNEL(negative, GPU, C64, negative_wrapper);

    // square
    REGISTER_KERNEL(square, GPU, I8, square_wrapper);
    REGISTER_KERNEL(square, GPU, I16, square_wrapper);
    REGISTER_KERNEL(square, GPU, I32, square_wrapper);
    REGISTER_KERNEL(square, GPU, I64, square_wrapper);
    REGISTER_KERNEL(square, GPU, U8, square_wrapper);
    REGISTER_KERNEL(square, GPU, U16, square_wrapper);
    REGISTER_KERNEL(square, GPU, U32, square_wrapper);
    REGISTER_KERNEL(square, GPU, U64, square_wrapper);
    REGISTER_KERNEL(square, GPU, F32, square_wrapper);
    REGISTER_KERNEL(square, GPU, F64, square_wrapper);
    REGISTER_KERNEL(square, GPU, C32, square_wrapper);
    REGISTER_KERNEL(square, GPU, C64, square_wrapper);
    REGISTER_KERNEL(square, GPU, BOOL, square_wrapper);

    // float math
#define REGISTER_FLOAT_OP(name) \
        REGISTER_KERNEL(name, GPU, F32, name##_wrapper); \
        REGISTER_KERNEL(name, GPU, F64, name##_wrapper)

    REGISTER_FLOAT_OP(exp);
    REGISTER_FLOAT_OP(exp2);
    REGISTER_FLOAT_OP(expm1);
    REGISTER_FLOAT_OP(log);
    REGISTER_FLOAT_OP(log2);
    REGISTER_FLOAT_OP(log10);
    REGISTER_FLOAT_OP(log1p);
    REGISTER_FLOAT_OP(sqrt);
    REGISTER_FLOAT_OP(cbrt);
    REGISTER_FLOAT_OP(reciprocal);

    REGISTER_FLOAT_OP(sin);
    REGISTER_FLOAT_OP(cos);
    REGISTER_FLOAT_OP(tan);
    REGISTER_FLOAT_OP(asin);
    REGISTER_FLOAT_OP(acos);
    REGISTER_FLOAT_OP(atan);
    REGISTER_FLOAT_OP(sinh);
    REGISTER_FLOAT_OP(cosh);
    REGISTER_FLOAT_OP(tanh);
    REGISTER_FLOAT_OP(asinh);
    REGISTER_FLOAT_OP(acosh);
    REGISTER_FLOAT_OP(atanh);

    REGISTER_FLOAT_OP(floor);
    REGISTER_FLOAT_OP(ceil);
    REGISTER_FLOAT_OP(trunc);
    REGISTER_FLOAT_OP(rint);

#undef REGISTER_FLOAT_OP

    // sign
    REGISTER_KERNEL(sign, GPU, BOOL, sign_wrapper);
    REGISTER_KERNEL(sign, GPU, U8, sign_wrapper);
    REGISTER_KERNEL(sign, GPU, U16, sign_wrapper);
    REGISTER_KERNEL(sign, GPU, U32, sign_wrapper);
    REGISTER_KERNEL(sign, GPU, U64, sign_wrapper);
    REGISTER_KERNEL(sign, GPU, I8, sign_wrapper);
    REGISTER_KERNEL(sign, GPU, I16, sign_wrapper);
    REGISTER_KERNEL(sign, GPU, I32, sign_wrapper);
    REGISTER_KERNEL(sign, GPU, I64, sign_wrapper);
    REGISTER_KERNEL(sign, GPU, F32, sign_wrapper);
    REGISTER_KERNEL(sign, GPU, F64, sign_wrapper);
    REGISTER_KERNEL(sign, GPU, C32, sign_wrapper);
    REGISTER_KERNEL(sign, GPU, C64, sign_wrapper);

    // conj
    REGISTER_KERNEL(conj, GPU, C32, conj_wrapper);
    REGISTER_KERNEL(conj, GPU, C64, conj_wrapper);
    REGISTER_KERNEL(conj, GPU, F32, conj_wrapper);
    REGISTER_KERNEL(conj, GPU, F64, conj_wrapper);
    REGISTER_KERNEL(conj, GPU, I8, conj_wrapper);
    REGISTER_KERNEL(conj, GPU, I16, conj_wrapper);
    REGISTER_KERNEL(conj, GPU, I32, conj_wrapper);
    REGISTER_KERNEL(conj, GPU, I64, conj_wrapper);
    REGISTER_KERNEL(conj, GPU, U8, conj_wrapper);
    REGISTER_KERNEL(conj, GPU, U16, conj_wrapper);
    REGISTER_KERNEL(conj, GPU, U32, conj_wrapper);
    REGISTER_KERNEL(conj, GPU, U64, conj_wrapper);
    REGISTER_KERNEL(conj, GPU, BOOL, conj_wrapper);

    // deg2rad
    REGISTER_KERNEL(deg2rad, GPU, F32, deg2rad_wrapper);
    REGISTER_KERNEL(deg2rad, GPU, F64, deg2rad_wrapper);

    // rad2deg
    REGISTER_KERNEL(rad2deg, GPU, F32, rad2deg_wrapper);
    REGISTER_KERNEL(rad2deg, GPU, F64, rad2deg_wrapper);

    // logical_not
    REGISTER_KERNEL(logical_not, GPU, BOOL, logical_not_wrapper);
    REGISTER_KERNEL(logical_not, GPU, U8, logical_not_wrapper);
    REGISTER_KERNEL(logical_not, GPU, I8, logical_not_wrapper);
    REGISTER_KERNEL(logical_not, GPU, I16, logical_not_wrapper);
    REGISTER_KERNEL(logical_not, GPU, I32, logical_not_wrapper);
    REGISTER_KERNEL(logical_not, GPU, I64, logical_not_wrapper);
    REGISTER_KERNEL(logical_not, GPU, U16, logical_not_wrapper);
    REGISTER_KERNEL(logical_not, GPU, U32, logical_not_wrapper);
    REGISTER_KERNEL(logical_not, GPU, U64, logical_not_wrapper);
    REGISTER_KERNEL(logical_not, GPU, F32, logical_not_wrapper);
    REGISTER_KERNEL(logical_not, GPU, F64, logical_not_wrapper);
    REGISTER_KERNEL(logical_not, GPU, C32, logical_not_wrapper);
    REGISTER_KERNEL(logical_not, GPU, C64, logical_not_wrapper);

    // bitwise_not
    REGISTER_KERNEL(bitwise_not, GPU, U8, bitwise_not_wrapper);
    REGISTER_KERNEL(bitwise_not, GPU, I8, bitwise_not_wrapper);
    REGISTER_KERNEL(bitwise_not, GPU, I16, bitwise_not_wrapper);
    REGISTER_KERNEL(bitwise_not, GPU, I32, bitwise_not_wrapper);
    REGISTER_KERNEL(bitwise_not, GPU, I64, bitwise_not_wrapper);
    REGISTER_KERNEL(bitwise_not, GPU, U16, bitwise_not_wrapper);
    REGISTER_KERNEL(bitwise_not, GPU, U32, bitwise_not_wrapper);
    REGISTER_KERNEL(bitwise_not, GPU, U64, bitwise_not_wrapper);
    REGISTER_KERNEL(bitwise_not, GPU, BOOL, logical_not_wrapper);

} // namespace ins::gpu

REGISTER_MODULE(unary, GPU);