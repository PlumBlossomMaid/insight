// backends/cuda/kernels/elementwise.cu
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
    // Helper: prepare stride info for binary elementwise ops
    // ============================================================================

    struct StrideInfo {
        int ndim;
        std::vector<int64_t> dims;
        std::vector<int64_t> a_strides;
        std::vector<int64_t> b_strides;
    };

    static StrideInfo prepare_stride_info(const Array& a, const Array& b) {
        StrideInfo info;
        info.ndim = a.shape().ndim();
        int b_ndim = b.shape().ndim();
        int b_offset = info.ndim - b_ndim;

        info.dims.resize(info.ndim);
        info.a_strides.resize(info.ndim);
        info.b_strides.resize(info.ndim);

        for (int i = 0; i < info.ndim; ++i) {
            info.dims[i] = a.shape().dim(i);
            info.a_strides[i] = a.strides()[i];

            int b_i = i - b_offset;
            if (b_i >= 0 && b.shape().dim(b_i) == a.shape().dim(i)) {
                info.b_strides[i] = b.strides()[b_i];
            }
            else {
                info.b_strides[i] = 0;  // broadcasting
            }
        }
        return info;
    }

    // ============================================================================
    // Basic arithmetic kernels (add, sub, mul, div) - with stride support
    // ============================================================================

    template<typename T>
    __global__ void add_kernel(
        const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out,
        int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = a[a_off] + b[b_off];
        }
    }

    template<typename T>
    __global__ void sub_kernel(
        const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out,
        int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = a[a_off] - b[b_off];
        }
    }

    template<typename T>
    __global__ void mul_kernel(
        const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out,
        int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = a[a_off] * b[b_off];
        }
    }

    template<typename T>
    __global__ void div_kernel(
        const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out,
        int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = a[a_off] / b[b_off];
        }
    }

    // ============================================================================
    // Complex arithmetic kernels (separate to avoid constexpr __host__ issues)
    // ============================================================================

    __global__ void add_c32_kernel(
        const std::complex<float>* __restrict__ a, const std::complex<float>* __restrict__ b,
        std::complex<float>* __restrict__ out, int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            float ar = real(a[a_off]), ai = imag(a[a_off]);
            float br = real(b[b_off]), bi = imag(b[b_off]);
            set_c32(&out[idx], ar + br, ai + bi);
        }
    }

    __global__ void add_c64_kernel(
        const std::complex<double>* __restrict__ a, const std::complex<double>* __restrict__ b,
        std::complex<double>* __restrict__ out, int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            double ar = real(a[a_off]), ai = imag(a[a_off]);
            double br = real(b[b_off]), bi = imag(b[b_off]);
            set_c64(&out[idx], ar + br, ai + bi);
        }
    }

    __global__ void sub_c32_kernel(
        const std::complex<float>* __restrict__ a, const std::complex<float>* __restrict__ b,
        std::complex<float>* __restrict__ out, int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            float ar = real(a[a_off]), ai = imag(a[a_off]);
            float br = real(b[b_off]), bi = imag(b[b_off]);
            set_c32(&out[idx], ar - br, ai - bi);
        }
    }

    __global__ void sub_c64_kernel(
        const std::complex<double>* __restrict__ a, const std::complex<double>* __restrict__ b,
        std::complex<double>* __restrict__ out, int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            double ar = real(a[a_off]), ai = imag(a[a_off]);
            double br = real(b[b_off]), bi = imag(b[b_off]);
            set_c64(&out[idx], ar - br, ai - bi);
        }
    }

    __global__ void mul_c32_kernel(
        const std::complex<float>* __restrict__ a, const std::complex<float>* __restrict__ b,
        std::complex<float>* __restrict__ out, int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            float ar = real(a[a_off]), ai = imag(a[a_off]);
            float br = real(b[b_off]), bi = imag(b[b_off]);
            set_c32(&out[idx], ar * br - ai * bi, ar * bi + ai * br);
        }
    }

    __global__ void mul_c64_kernel(
        const std::complex<double>* __restrict__ a, const std::complex<double>* __restrict__ b,
        std::complex<double>* __restrict__ out, int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            double ar = real(a[a_off]), ai = imag(a[a_off]);
            double br = real(b[b_off]), bi = imag(b[b_off]);
            set_c64(&out[idx], ar * br - ai * bi, ar * bi + ai * br);
        }
    }

    __global__ void div_c32_kernel(
        const std::complex<float>* __restrict__ a, const std::complex<float>* __restrict__ b,
        std::complex<float>* __restrict__ out, int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            float ar = real(a[a_off]), ai = imag(a[a_off]);
            float br = real(b[b_off]), bi = imag(b[b_off]);
            float denom = br * br + bi * bi;
            set_c32(&out[idx], (ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom);
        }
    }

    __global__ void div_c64_kernel(
        const std::complex<double>* __restrict__ a, const std::complex<double>* __restrict__ b,
        std::complex<double>* __restrict__ out, int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            double ar = real(a[a_off]), ai = imag(a[a_off]);
            double br = real(b[b_off]), bi = imag(b[b_off]);
            double denom = br * br + bi * bi;
            set_c64(&out[idx], (ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom);
        }
    }

    // ============================================================================
    // Comparison kernels - with stride support
    // ============================================================================

    template<typename T>
    __global__ void equal_kernel(
        const T* __restrict__ a, const T* __restrict__ b, bool* __restrict__ out,
        int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = a[a_off] == b[b_off];
        }
    }

    __global__ void equal_c32_kernel(
        const std::complex<float>* __restrict__ a, const std::complex<float>* __restrict__ b,
        bool* __restrict__ out, int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = (real(a[a_off]) == real(b[b_off])) && (imag(a[a_off]) == imag(b[b_off]));
        }
    }

    __global__ void equal_c64_kernel(
        const std::complex<double>* __restrict__ a, const std::complex<double>* __restrict__ b,
        bool* __restrict__ out, int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = (real(a[a_off]) == real(b[b_off])) && (imag(a[a_off]) == imag(b[b_off]));
        }
    }

    template<typename T>
    __global__ void not_equal_kernel(
        const T* __restrict__ a, const T* __restrict__ b, bool* __restrict__ out,
        int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = a[a_off] != b[b_off];
        }
    }

    __global__ void not_equal_c32_kernel(
        const std::complex<float>* __restrict__ a, const std::complex<float>* __restrict__ b,
        bool* __restrict__ out, int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = (real(a[a_off]) != real(b[b_off])) || (imag(a[a_off]) != imag(b[b_off]));
        }
    }

    __global__ void not_equal_c64_kernel(
        const std::complex<double>* __restrict__ a, const std::complex<double>* __restrict__ b,
        bool* __restrict__ out, int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = (real(a[a_off]) != real(b[b_off])) || (imag(a[a_off]) != imag(b[b_off]));
        }
    }

    template<typename T>
    __global__ void greater_kernel(
        const T* __restrict__ a, const T* __restrict__ b, bool* __restrict__ out,
        int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = a[a_off] > b[b_off];
        }
    }

    __global__ void greater_c32_kernel(
        const std::complex<float>* __restrict__ a, const std::complex<float>* __restrict__ b,
        bool* __restrict__ out, int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = real(a[a_off]) > real(b[b_off]);
        }
    }

    __global__ void greater_c64_kernel(
        const std::complex<double>* __restrict__ a, const std::complex<double>* __restrict__ b,
        bool* __restrict__ out, int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = real(a[a_off]) > real(b[b_off]);
        }
    }

    template<typename T>
    __global__ void less_kernel(
        const T* __restrict__ a, const T* __restrict__ b, bool* __restrict__ out,
        int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = a[a_off] < b[b_off];
        }
    }

    __global__ void less_c32_kernel(
        const std::complex<float>* __restrict__ a, const std::complex<float>* __restrict__ b,
        bool* __restrict__ out, int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = real(a[a_off]) < real(b[b_off]);
        }
    }

    __global__ void less_c64_kernel(
        const std::complex<double>* __restrict__ a, const std::complex<double>* __restrict__ b,
        bool* __restrict__ out, int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = real(a[a_off]) < real(b[b_off]);
        }
    }

    template<typename T>
    __global__ void greater_equal_kernel(
        const T* __restrict__ a, const T* __restrict__ b, bool* __restrict__ out,
        int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = a[a_off] >= b[b_off];
        }
    }

    __global__ void greater_equal_c32_kernel(
        const std::complex<float>* __restrict__ a, const std::complex<float>* __restrict__ b,
        bool* __restrict__ out, int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = real(a[a_off]) >= real(b[b_off]);
        }
    }

    __global__ void greater_equal_c64_kernel(
        const std::complex<double>* __restrict__ a, const std::complex<double>* __restrict__ b,
        bool* __restrict__ out, int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = real(a[a_off]) >= real(b[b_off]);
        }
    }

    template<typename T>
    __global__ void less_equal_kernel(
        const T* __restrict__ a, const T* __restrict__ b, bool* __restrict__ out,
        int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = a[a_off] <= b[b_off];
        }
    }

    __global__ void less_equal_c32_kernel(
        const std::complex<float>* __restrict__ a, const std::complex<float>* __restrict__ b,
        bool* __restrict__ out, int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = real(a[a_off]) <= real(b[b_off]);
        }
    }

    __global__ void less_equal_c64_kernel(
        const std::complex<double>* __restrict__ a, const std::complex<double>* __restrict__ b,
        bool* __restrict__ out, int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = real(a[a_off]) <= real(b[b_off]);
        }
    }

    // ============================================================================
    // Logical operation kernels - with stride support
    // ============================================================================

    template<typename T>
    __global__ void logical_and_kernel(
        const T* __restrict__ a, const T* __restrict__ b, bool* __restrict__ out,
        int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = (a[a_off] != T(0)) && (b[b_off] != T(0));
        }
    }

    template<typename T>
    __global__ void logical_or_kernel(
        const T* __restrict__ a, const T* __restrict__ b, bool* __restrict__ out,
        int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = (a[a_off] != T(0)) || (b[b_off] != T(0));
        }
    }

    template<typename T>
    __global__ void logical_xor_kernel(
        const T* __restrict__ a, const T* __restrict__ b, bool* __restrict__ out,
        int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = (a[a_off] != T(0)) != (b[b_off] != T(0));
        }
    }

    // ============================================================================
    // Maximum/Minimum kernels - with stride support
    // ============================================================================

    template<typename T>
    __global__ void maximum_kernel(
        const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out,
        int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = (a[a_off] > b[b_off]) ? a[a_off] : b[b_off];
        }
    }

    __global__ void maximum_c32_kernel(
        const std::complex<float>* __restrict__ a, const std::complex<float>* __restrict__ b,
        std::complex<float>* __restrict__ out, int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = (real(a[a_off]) > real(b[b_off])) ? a[a_off] : b[b_off];
        }
    }

    __global__ void maximum_c64_kernel(
        const std::complex<double>* __restrict__ a, const std::complex<double>* __restrict__ b,
        std::complex<double>* __restrict__ out, int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = (real(a[a_off]) > real(b[b_off])) ? a[a_off] : b[b_off];
        }
    }

    template<typename T>
    __global__ void minimum_kernel(
        const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out,
        int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = (a[a_off] < b[b_off]) ? a[a_off] : b[b_off];
        }
    }

    __global__ void minimum_c32_kernel(
        const std::complex<float>* __restrict__ a, const std::complex<float>* __restrict__ b,
        std::complex<float>* __restrict__ out, int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = (real(a[a_off]) < real(b[b_off])) ? a[a_off] : b[b_off];
        }
    }

    __global__ void minimum_c64_kernel(
        const std::complex<double>* __restrict__ a, const std::complex<double>* __restrict__ b,
        std::complex<double>* __restrict__ out, int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = (real(a[a_off]) < real(b[b_off])) ? a[a_off] : b[b_off];
        }
    }

    // ============================================================================
    // Power kernel - with stride support
    // ============================================================================

    template<typename T>
    __global__ void power_kernel(
        const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out,
        int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = std::pow(a[a_off], b[b_off]);
        }
    }

    template<typename T>
    __global__ void power_int_kernel(
        const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out,
        int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = static_cast<T>(std::pow(static_cast<double>(a[a_off]), static_cast<double>(b[b_off])));
        }
    }

    // ============================================================================
    // Bitwise operation kernels - with stride support
    // ============================================================================

    template<typename T>
    __global__ void bitwise_and_kernel(
        const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out,
        int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = a[a_off] & b[b_off];
        }
    }

    template<typename T>
    __global__ void bitwise_or_kernel(
        const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out,
        int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = a[a_off] | b[b_off];
        }
    }

    template<typename T>
    __global__ void bitwise_xor_kernel(
        const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out,
        int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = a[a_off] ^ b[b_off];
        }
    }

    template<typename T>
    __global__ void left_shift_kernel(
        const T* __restrict__ a, const int64_t* __restrict__ b, T* __restrict__ out,
        int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = a[a_off] << b[b_off];
        }
    }

    template<typename T>
    __global__ void right_shift_kernel(
        const T* __restrict__ a, const int64_t* __restrict__ b, T* __restrict__ out,
        int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = a[a_off] >> b[b_off];
        }
    }

    // ============================================================================
    // Modulo kernels - with stride support
    // ============================================================================

    template<typename T>
    __global__ void mod_kernel_integer(
        const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out,
        int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = a[a_off] % b[b_off];
        }
    }

    __global__ void mod_kernel_float(
        const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out,
        int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = fmodf(a[a_off], b[b_off]);
        }
    }

    __global__ void mod_kernel_double(
        const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ out,
        int64_t n, int ndim,
        const int64_t* __restrict__ dims,
        const int64_t* __restrict__ a_strides,
        const int64_t* __restrict__ b_strides
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int64_t a_off = compute_offset(idx, ndim, dims, a_strides);
            int64_t b_off = compute_offset(idx, ndim, dims, b_strides);
            out[idx] = fmod(a[a_off], b[b_off]);
        }
    }

    // ============================================================================
    // Implementation functions - all use the unified launch pattern
    // ============================================================================

    template<typename KernelFunc, typename... Args>
    static void launch_elementwise_kernel(
        KernelFunc kernel, const Array& a, const Array& b, Array& out, Args... args
    ) {
        int64_t n = out.numel();
        LaunchConfig config(n);
        StrideInfo info = prepare_stride_info(a, b);

        int64_t* d_dims;
        int64_t* d_a_strides;
        int64_t* d_b_strides;
        cudaMalloc(&d_dims, info.ndim * sizeof(int64_t));
        cudaMalloc(&d_a_strides, info.ndim * sizeof(int64_t));
        cudaMalloc(&d_b_strides, info.ndim * sizeof(int64_t));
        cudaMemcpy(d_dims, info.dims.data(), info.ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_a_strides, info.a_strides.data(), info.ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b_strides, info.b_strides.data(), info.ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

        kernel << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            args..., n, info.ndim, d_dims, d_a_strides, d_b_strides
            );

        cudaFree(d_dims);
        cudaFree(d_a_strides);
        cudaFree(d_b_strides);
    }

    // ============================================================================
    // Wrapper functions
    // ============================================================================

    static OpArgs add_wrapper(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        Array out(a.shape(), a.dtype(), a.place());

        switch (a.dtype()) {
        case DType::F32:   launch_elementwise_kernel(add_kernel<float>, a, b, out, a.data<float>(), b.data<float>(), out.data<float>()); break;
        case DType::F64:   launch_elementwise_kernel(add_kernel<double>, a, b, out, a.data<double>(), b.data<double>(), out.data<double>()); break;
        case DType::I8:    launch_elementwise_kernel(add_kernel<int8_t>, a, b, out, a.data<int8_t>(), b.data<int8_t>(), out.data<int8_t>()); break;
        case DType::I16:   launch_elementwise_kernel(add_kernel<int16_t>, a, b, out, a.data<int16_t>(), b.data<int16_t>(), out.data<int16_t>()); break;
        case DType::I32:   launch_elementwise_kernel(add_kernel<int32_t>, a, b, out, a.data<int32_t>(), b.data<int32_t>(), out.data<int32_t>()); break;
        case DType::I64:   launch_elementwise_kernel(add_kernel<int64_t>, a, b, out, a.data<int64_t>(), b.data<int64_t>(), out.data<int64_t>()); break;
        case DType::U8:    launch_elementwise_kernel(add_kernel<uint8_t>, a, b, out, a.data<uint8_t>(), b.data<uint8_t>(), out.data<uint8_t>()); break;
        case DType::U16:   launch_elementwise_kernel(add_kernel<uint16_t>, a, b, out, a.data<uint16_t>(), b.data<uint16_t>(), out.data<uint16_t>()); break;
        case DType::U32:   launch_elementwise_kernel(add_kernel<uint32_t>, a, b, out, a.data<uint32_t>(), b.data<uint32_t>(), out.data<uint32_t>()); break;
        case DType::U64:   launch_elementwise_kernel(add_kernel<uint64_t>, a, b, out, a.data<uint64_t>(), b.data<uint64_t>(), out.data<uint64_t>()); break;
        case DType::C32:   launch_elementwise_kernel(add_c32_kernel, a, b, out, a.data<std::complex<float>>(), b.data<std::complex<float>>(), out.data<std::complex<float>>()); break;
        case DType::C64:   launch_elementwise_kernel(add_c64_kernel, a, b, out, a.data<std::complex<double>>(), b.data<std::complex<double>>(), out.data<std::complex<double>>()); break;
        case DType::BOOL:  launch_elementwise_kernel(add_kernel<bool>, a, b, out, a.data<bool>(), b.data<bool>(), out.data<bool>()); break;
        default: INS_THROW("add: unsupported dtype ", dtype_name(a.dtype()));
        }
        return { out };
    }

    static OpArgs sub_wrapper(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        Array out(a.shape(), a.dtype(), a.place());

        switch (a.dtype()) {
        case DType::F32:   launch_elementwise_kernel(sub_kernel<float>, a, b, out, a.data<float>(), b.data<float>(), out.data<float>()); break;
        case DType::F64:   launch_elementwise_kernel(sub_kernel<double>, a, b, out, a.data<double>(), b.data<double>(), out.data<double>()); break;
        case DType::I8:    launch_elementwise_kernel(sub_kernel<int8_t>, a, b, out, a.data<int8_t>(), b.data<int8_t>(), out.data<int8_t>()); break;
        case DType::I16:   launch_elementwise_kernel(sub_kernel<int16_t>, a, b, out, a.data<int16_t>(), b.data<int16_t>(), out.data<int16_t>()); break;
        case DType::I32:   launch_elementwise_kernel(sub_kernel<int32_t>, a, b, out, a.data<int32_t>(), b.data<int32_t>(), out.data<int32_t>()); break;
        case DType::I64:   launch_elementwise_kernel(sub_kernel<int64_t>, a, b, out, a.data<int64_t>(), b.data<int64_t>(), out.data<int64_t>()); break;
        case DType::U8:    launch_elementwise_kernel(sub_kernel<uint8_t>, a, b, out, a.data<uint8_t>(), b.data<uint8_t>(), out.data<uint8_t>()); break;
        case DType::U16:   launch_elementwise_kernel(sub_kernel<uint16_t>, a, b, out, a.data<uint16_t>(), b.data<uint16_t>(), out.data<uint16_t>()); break;
        case DType::U32:   launch_elementwise_kernel(sub_kernel<uint32_t>, a, b, out, a.data<uint32_t>(), b.data<uint32_t>(), out.data<uint32_t>()); break;
        case DType::U64:   launch_elementwise_kernel(sub_kernel<uint64_t>, a, b, out, a.data<uint64_t>(), b.data<uint64_t>(), out.data<uint64_t>()); break;
        case DType::C32:   launch_elementwise_kernel(sub_c32_kernel, a, b, out, a.data<std::complex<float>>(), b.data<std::complex<float>>(), out.data<std::complex<float>>()); break;
        case DType::C64:   launch_elementwise_kernel(sub_c64_kernel, a, b, out, a.data<std::complex<double>>(), b.data<std::complex<double>>(), out.data<std::complex<double>>()); break;
        case DType::BOOL:  launch_elementwise_kernel(sub_kernel<bool>, a, b, out, a.data<bool>(), b.data<bool>(), out.data<bool>()); break;
        default: INS_THROW("sub: unsupported dtype ", dtype_name(a.dtype()));
        }
        return { out };
    }

    static OpArgs mul_wrapper(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        Array out(a.shape(), a.dtype(), a.place());

        switch (a.dtype()) {
        case DType::F32:   launch_elementwise_kernel(mul_kernel<float>, a, b, out, a.data<float>(), b.data<float>(), out.data<float>()); break;
        case DType::F64:   launch_elementwise_kernel(mul_kernel<double>, a, b, out, a.data<double>(), b.data<double>(), out.data<double>()); break;
        case DType::I8:    launch_elementwise_kernel(mul_kernel<int8_t>, a, b, out, a.data<int8_t>(), b.data<int8_t>(), out.data<int8_t>()); break;
        case DType::I16:   launch_elementwise_kernel(mul_kernel<int16_t>, a, b, out, a.data<int16_t>(), b.data<int16_t>(), out.data<int16_t>()); break;
        case DType::I32:   launch_elementwise_kernel(mul_kernel<int32_t>, a, b, out, a.data<int32_t>(), b.data<int32_t>(), out.data<int32_t>()); break;
        case DType::I64:   launch_elementwise_kernel(mul_kernel<int64_t>, a, b, out, a.data<int64_t>(), b.data<int64_t>(), out.data<int64_t>()); break;
        case DType::U8:    launch_elementwise_kernel(mul_kernel<uint8_t>, a, b, out, a.data<uint8_t>(), b.data<uint8_t>(), out.data<uint8_t>()); break;
        case DType::U16:   launch_elementwise_kernel(mul_kernel<uint16_t>, a, b, out, a.data<uint16_t>(), b.data<uint16_t>(), out.data<uint16_t>()); break;
        case DType::U32:   launch_elementwise_kernel(mul_kernel<uint32_t>, a, b, out, a.data<uint32_t>(), b.data<uint32_t>(), out.data<uint32_t>()); break;
        case DType::U64:   launch_elementwise_kernel(mul_kernel<uint64_t>, a, b, out, a.data<uint64_t>(), b.data<uint64_t>(), out.data<uint64_t>()); break;
        case DType::C32:   launch_elementwise_kernel(mul_c32_kernel, a, b, out, a.data<std::complex<float>>(), b.data<std::complex<float>>(), out.data<std::complex<float>>()); break;
        case DType::C64:   launch_elementwise_kernel(mul_c64_kernel, a, b, out, a.data<std::complex<double>>(), b.data<std::complex<double>>(), out.data<std::complex<double>>()); break;
        case DType::BOOL:  launch_elementwise_kernel(mul_kernel<bool>, a, b, out, a.data<bool>(), b.data<bool>(), out.data<bool>()); break;
        default: INS_THROW("mul: unsupported dtype ", dtype_name(a.dtype()));
        }
        return { out };
    }

    static OpArgs div_wrapper(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        Array out(a.shape(), a.dtype(), a.place());

        switch (a.dtype()) {
        case DType::F32:   launch_elementwise_kernel(div_kernel<float>, a, b, out, a.data<float>(), b.data<float>(), out.data<float>()); break;
        case DType::F64:   launch_elementwise_kernel(div_kernel<double>, a, b, out, a.data<double>(), b.data<double>(), out.data<double>()); break;
        case DType::I8:    launch_elementwise_kernel(div_kernel<int8_t>, a, b, out, a.data<int8_t>(), b.data<int8_t>(), out.data<int8_t>()); break;
        case DType::I16:   launch_elementwise_kernel(div_kernel<int16_t>, a, b, out, a.data<int16_t>(), b.data<int16_t>(), out.data<int16_t>()); break;
        case DType::I32:   launch_elementwise_kernel(div_kernel<int32_t>, a, b, out, a.data<int32_t>(), b.data<int32_t>(), out.data<int32_t>()); break;
        case DType::I64:   launch_elementwise_kernel(div_kernel<int64_t>, a, b, out, a.data<int64_t>(), b.data<int64_t>(), out.data<int64_t>()); break;
        case DType::U8:    launch_elementwise_kernel(div_kernel<uint8_t>, a, b, out, a.data<uint8_t>(), b.data<uint8_t>(), out.data<uint8_t>()); break;
        case DType::U16:   launch_elementwise_kernel(div_kernel<uint16_t>, a, b, out, a.data<uint16_t>(), b.data<uint16_t>(), out.data<uint16_t>()); break;
        case DType::U32:   launch_elementwise_kernel(div_kernel<uint32_t>, a, b, out, a.data<uint32_t>(), b.data<uint32_t>(), out.data<uint32_t>()); break;
        case DType::U64:   launch_elementwise_kernel(div_kernel<uint64_t>, a, b, out, a.data<uint64_t>(), b.data<uint64_t>(), out.data<uint64_t>()); break;
        case DType::C32:   launch_elementwise_kernel(div_c32_kernel, a, b, out, a.data<std::complex<float>>(), b.data<std::complex<float>>(), out.data<std::complex<float>>()); break;
        case DType::C64:   launch_elementwise_kernel(div_c64_kernel, a, b, out, a.data<std::complex<double>>(), b.data<std::complex<double>>(), out.data<std::complex<double>>()); break;
        case DType::BOOL:  launch_elementwise_kernel(div_kernel<bool>, a, b, out, a.data<bool>(), b.data<bool>(), out.data<bool>()); break;
        default: INS_THROW("div: unsupported dtype ", dtype_name(a.dtype()));
        }
        return { out };
    }

    static OpArgs mod_wrapper(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        Array out(a.shape(), a.dtype(), a.place());

        switch (a.dtype()) {
        case DType::I8:    launch_elementwise_kernel(mod_kernel_integer<int8_t>, a, b, out, a.data<int8_t>(), b.data<int8_t>(), out.data<int8_t>()); break;
        case DType::I16:   launch_elementwise_kernel(mod_kernel_integer<int16_t>, a, b, out, a.data<int16_t>(), b.data<int16_t>(), out.data<int16_t>()); break;
        case DType::I32:   launch_elementwise_kernel(mod_kernel_integer<int32_t>, a, b, out, a.data<int32_t>(), b.data<int32_t>(), out.data<int32_t>()); break;
        case DType::I64:   launch_elementwise_kernel(mod_kernel_integer<int64_t>, a, b, out, a.data<int64_t>(), b.data<int64_t>(), out.data<int64_t>()); break;
        case DType::U8:    launch_elementwise_kernel(mod_kernel_integer<uint8_t>, a, b, out, a.data<uint8_t>(), b.data<uint8_t>(), out.data<uint8_t>()); break;
        case DType::U16:   launch_elementwise_kernel(mod_kernel_integer<uint16_t>, a, b, out, a.data<uint16_t>(), b.data<uint16_t>(), out.data<uint16_t>()); break;
        case DType::U32:   launch_elementwise_kernel(mod_kernel_integer<uint32_t>, a, b, out, a.data<uint32_t>(), b.data<uint32_t>(), out.data<uint32_t>()); break;
        case DType::U64:   launch_elementwise_kernel(mod_kernel_integer<uint64_t>, a, b, out, a.data<uint64_t>(), b.data<uint64_t>(), out.data<uint64_t>()); break;
        case DType::F32:   launch_elementwise_kernel(mod_kernel_float, a, b, out, a.data<float>(), b.data<float>(), out.data<float>()); break;
        case DType::F64:   launch_elementwise_kernel(mod_kernel_double, a, b, out, a.data<double>(), b.data<double>(), out.data<double>()); break;
        default: INS_THROW("mod: unsupported dtype ", dtype_name(a.dtype()));
        }
        return { out };
    }

    static OpArgs equal_wrapper(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        Array out(a.shape(), DType::BOOL, a.place());

        switch (a.dtype()) {
        case DType::F32:   launch_elementwise_kernel(equal_kernel<float>, a, b, out, a.data<float>(), b.data<float>(), out.data<bool>()); break;
        case DType::F64:   launch_elementwise_kernel(equal_kernel<double>, a, b, out, a.data<double>(), b.data<double>(), out.data<bool>()); break;
        case DType::I8:    launch_elementwise_kernel(equal_kernel<int8_t>, a, b, out, a.data<int8_t>(), b.data<int8_t>(), out.data<bool>()); break;
        case DType::I16:   launch_elementwise_kernel(equal_kernel<int16_t>, a, b, out, a.data<int16_t>(), b.data<int16_t>(), out.data<bool>()); break;
        case DType::I32:   launch_elementwise_kernel(equal_kernel<int32_t>, a, b, out, a.data<int32_t>(), b.data<int32_t>(), out.data<bool>()); break;
        case DType::I64:   launch_elementwise_kernel(equal_kernel<int64_t>, a, b, out, a.data<int64_t>(), b.data<int64_t>(), out.data<bool>()); break;
        case DType::U8:    launch_elementwise_kernel(equal_kernel<uint8_t>, a, b, out, a.data<uint8_t>(), b.data<uint8_t>(), out.data<bool>()); break;
        case DType::U16:   launch_elementwise_kernel(equal_kernel<uint16_t>, a, b, out, a.data<uint16_t>(), b.data<uint16_t>(), out.data<bool>()); break;
        case DType::U32:   launch_elementwise_kernel(equal_kernel<uint32_t>, a, b, out, a.data<uint32_t>(), b.data<uint32_t>(), out.data<bool>()); break;
        case DType::U64:   launch_elementwise_kernel(equal_kernel<uint64_t>, a, b, out, a.data<uint64_t>(), b.data<uint64_t>(), out.data<bool>()); break;
        case DType::C32:   launch_elementwise_kernel(equal_c32_kernel, a, b, out, a.data<std::complex<float>>(), b.data<std::complex<float>>(), out.data<bool>()); break;
        case DType::C64:   launch_elementwise_kernel(equal_c64_kernel, a, b, out, a.data<std::complex<double>>(), b.data<std::complex<double>>(), out.data<bool>()); break;
        case DType::BOOL:  launch_elementwise_kernel(equal_kernel<bool>, a, b, out, a.data<bool>(), b.data<bool>(), out.data<bool>()); break;
        default: INS_THROW("equal: unsupported dtype ", dtype_name(a.dtype()));
        }
        return { out };
    }

    static OpArgs not_equal_wrapper(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        Array out(a.shape(), DType::BOOL, a.place());

        switch (a.dtype()) {
        case DType::F32:   launch_elementwise_kernel(not_equal_kernel<float>, a, b, out, a.data<float>(), b.data<float>(), out.data<bool>()); break;
        case DType::F64:   launch_elementwise_kernel(not_equal_kernel<double>, a, b, out, a.data<double>(), b.data<double>(), out.data<bool>()); break;
        case DType::I8:    launch_elementwise_kernel(not_equal_kernel<int8_t>, a, b, out, a.data<int8_t>(), b.data<int8_t>(), out.data<bool>()); break;
        case DType::I16:   launch_elementwise_kernel(not_equal_kernel<int16_t>, a, b, out, a.data<int16_t>(), b.data<int16_t>(), out.data<bool>()); break;
        case DType::I32:   launch_elementwise_kernel(not_equal_kernel<int32_t>, a, b, out, a.data<int32_t>(), b.data<int32_t>(), out.data<bool>()); break;
        case DType::I64:   launch_elementwise_kernel(not_equal_kernel<int64_t>, a, b, out, a.data<int64_t>(), b.data<int64_t>(), out.data<bool>()); break;
        case DType::U8:    launch_elementwise_kernel(not_equal_kernel<uint8_t>, a, b, out, a.data<uint8_t>(), b.data<uint8_t>(), out.data<bool>()); break;
        case DType::U16:   launch_elementwise_kernel(not_equal_kernel<uint16_t>, a, b, out, a.data<uint16_t>(), b.data<uint16_t>(), out.data<bool>()); break;
        case DType::U32:   launch_elementwise_kernel(not_equal_kernel<uint32_t>, a, b, out, a.data<uint32_t>(), b.data<uint32_t>(), out.data<bool>()); break;
        case DType::U64:   launch_elementwise_kernel(not_equal_kernel<uint64_t>, a, b, out, a.data<uint64_t>(), b.data<uint64_t>(), out.data<bool>()); break;
        case DType::C32:   launch_elementwise_kernel(not_equal_c32_kernel, a, b, out, a.data<std::complex<float>>(), b.data<std::complex<float>>(), out.data<bool>()); break;
        case DType::C64:   launch_elementwise_kernel(not_equal_c64_kernel, a, b, out, a.data<std::complex<double>>(), b.data<std::complex<double>>(), out.data<bool>()); break;
        case DType::BOOL:  launch_elementwise_kernel(not_equal_kernel<bool>, a, b, out, a.data<bool>(), b.data<bool>(), out.data<bool>()); break;
        default: INS_THROW("not_equal: unsupported dtype ", dtype_name(a.dtype()));
        }
        return { out };
    }

    static OpArgs greater_wrapper(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        Array out(a.shape(), DType::BOOL, a.place());

        switch (a.dtype()) {
        case DType::F32:   launch_elementwise_kernel(greater_kernel<float>, a, b, out, a.data<float>(), b.data<float>(), out.data<bool>()); break;
        case DType::F64:   launch_elementwise_kernel(greater_kernel<double>, a, b, out, a.data<double>(), b.data<double>(), out.data<bool>()); break;
        case DType::I8:    launch_elementwise_kernel(greater_kernel<int8_t>, a, b, out, a.data<int8_t>(), b.data<int8_t>(), out.data<bool>()); break;
        case DType::I16:   launch_elementwise_kernel(greater_kernel<int16_t>, a, b, out, a.data<int16_t>(), b.data<int16_t>(), out.data<bool>()); break;
        case DType::I32:   launch_elementwise_kernel(greater_kernel<int32_t>, a, b, out, a.data<int32_t>(), b.data<int32_t>(), out.data<bool>()); break;
        case DType::I64:   launch_elementwise_kernel(greater_kernel<int64_t>, a, b, out, a.data<int64_t>(), b.data<int64_t>(), out.data<bool>()); break;
        case DType::U8:    launch_elementwise_kernel(greater_kernel<uint8_t>, a, b, out, a.data<uint8_t>(), b.data<uint8_t>(), out.data<bool>()); break;
        case DType::U16:   launch_elementwise_kernel(greater_kernel<uint16_t>, a, b, out, a.data<uint16_t>(), b.data<uint16_t>(), out.data<bool>()); break;
        case DType::U32:   launch_elementwise_kernel(greater_kernel<uint32_t>, a, b, out, a.data<uint32_t>(), b.data<uint32_t>(), out.data<bool>()); break;
        case DType::U64:   launch_elementwise_kernel(greater_kernel<uint64_t>, a, b, out, a.data<uint64_t>(), b.data<uint64_t>(), out.data<bool>()); break;
        case DType::C32:   launch_elementwise_kernel(greater_c32_kernel, a, b, out, a.data<std::complex<float>>(), b.data<std::complex<float>>(), out.data<bool>()); break;
        case DType::C64:   launch_elementwise_kernel(greater_c64_kernel, a, b, out, a.data<std::complex<double>>(), b.data<std::complex<double>>(), out.data<bool>()); break;
        case DType::BOOL:  launch_elementwise_kernel(greater_kernel<bool>, a, b, out, a.data<bool>(), b.data<bool>(), out.data<bool>()); break;
        default: INS_THROW("greater: unsupported dtype ", dtype_name(a.dtype()));
        }
        return { out };
    }

    static OpArgs less_wrapper(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        Array out(a.shape(), DType::BOOL, a.place());

        switch (a.dtype()) {
        case DType::F32:   launch_elementwise_kernel(less_kernel<float>, a, b, out, a.data<float>(), b.data<float>(), out.data<bool>()); break;
        case DType::F64:   launch_elementwise_kernel(less_kernel<double>, a, b, out, a.data<double>(), b.data<double>(), out.data<bool>()); break;
        case DType::I8:    launch_elementwise_kernel(less_kernel<int8_t>, a, b, out, a.data<int8_t>(), b.data<int8_t>(), out.data<bool>()); break;
        case DType::I16:   launch_elementwise_kernel(less_kernel<int16_t>, a, b, out, a.data<int16_t>(), b.data<int16_t>(), out.data<bool>()); break;
        case DType::I32:   launch_elementwise_kernel(less_kernel<int32_t>, a, b, out, a.data<int32_t>(), b.data<int32_t>(), out.data<bool>()); break;
        case DType::I64:   launch_elementwise_kernel(less_kernel<int64_t>, a, b, out, a.data<int64_t>(), b.data<int64_t>(), out.data<bool>()); break;
        case DType::U8:    launch_elementwise_kernel(less_kernel<uint8_t>, a, b, out, a.data<uint8_t>(), b.data<uint8_t>(), out.data<bool>()); break;
        case DType::U16:   launch_elementwise_kernel(less_kernel<uint16_t>, a, b, out, a.data<uint16_t>(), b.data<uint16_t>(), out.data<bool>()); break;
        case DType::U32:   launch_elementwise_kernel(less_kernel<uint32_t>, a, b, out, a.data<uint32_t>(), b.data<uint32_t>(), out.data<bool>()); break;
        case DType::U64:   launch_elementwise_kernel(less_kernel<uint64_t>, a, b, out, a.data<uint64_t>(), b.data<uint64_t>(), out.data<bool>()); break;
        case DType::C32:   launch_elementwise_kernel(less_c32_kernel, a, b, out, a.data<std::complex<float>>(), b.data<std::complex<float>>(), out.data<bool>()); break;
        case DType::C64:   launch_elementwise_kernel(less_c64_kernel, a, b, out, a.data<std::complex<double>>(), b.data<std::complex<double>>(), out.data<bool>()); break;
        case DType::BOOL:  launch_elementwise_kernel(less_kernel<bool>, a, b, out, a.data<bool>(), b.data<bool>(), out.data<bool>()); break;
        default: INS_THROW("less: unsupported dtype ", dtype_name(a.dtype()));
        }
        return { out };
    }

    static OpArgs greater_equal_wrapper(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        Array out(a.shape(), DType::BOOL, a.place());

        switch (a.dtype()) {
        case DType::F32:   launch_elementwise_kernel(greater_equal_kernel<float>, a, b, out, a.data<float>(), b.data<float>(), out.data<bool>()); break;
        case DType::F64:   launch_elementwise_kernel(greater_equal_kernel<double>, a, b, out, a.data<double>(), b.data<double>(), out.data<bool>()); break;
        case DType::I8:    launch_elementwise_kernel(greater_equal_kernel<int8_t>, a, b, out, a.data<int8_t>(), b.data<int8_t>(), out.data<bool>()); break;
        case DType::I16:   launch_elementwise_kernel(greater_equal_kernel<int16_t>, a, b, out, a.data<int16_t>(), b.data<int16_t>(), out.data<bool>()); break;
        case DType::I32:   launch_elementwise_kernel(greater_equal_kernel<int32_t>, a, b, out, a.data<int32_t>(), b.data<int32_t>(), out.data<bool>()); break;
        case DType::I64:   launch_elementwise_kernel(greater_equal_kernel<int64_t>, a, b, out, a.data<int64_t>(), b.data<int64_t>(), out.data<bool>()); break;
        case DType::U8:    launch_elementwise_kernel(greater_equal_kernel<uint8_t>, a, b, out, a.data<uint8_t>(), b.data<uint8_t>(), out.data<bool>()); break;
        case DType::U16:   launch_elementwise_kernel(greater_equal_kernel<uint16_t>, a, b, out, a.data<uint16_t>(), b.data<uint16_t>(), out.data<bool>()); break;
        case DType::U32:   launch_elementwise_kernel(greater_equal_kernel<uint32_t>, a, b, out, a.data<uint32_t>(), b.data<uint32_t>(), out.data<bool>()); break;
        case DType::U64:   launch_elementwise_kernel(greater_equal_kernel<uint64_t>, a, b, out, a.data<uint64_t>(), b.data<uint64_t>(), out.data<bool>()); break;
        case DType::C32:   launch_elementwise_kernel(greater_equal_c32_kernel, a, b, out, a.data<std::complex<float>>(), b.data<std::complex<float>>(), out.data<bool>()); break;
        case DType::C64:   launch_elementwise_kernel(greater_equal_c64_kernel, a, b, out, a.data<std::complex<double>>(), b.data<std::complex<double>>(), out.data<bool>()); break;
        case DType::BOOL:  launch_elementwise_kernel(greater_equal_kernel<bool>, a, b, out, a.data<bool>(), b.data<bool>(), out.data<bool>()); break;
        default: INS_THROW("greater_equal: unsupported dtype ", dtype_name(a.dtype()));
        }
        return { out };
    }

    static OpArgs less_equal_wrapper(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        Array out(a.shape(), DType::BOOL, a.place());

        switch (a.dtype()) {
        case DType::F32:   launch_elementwise_kernel(less_equal_kernel<float>, a, b, out, a.data<float>(), b.data<float>(), out.data<bool>()); break;
        case DType::F64:   launch_elementwise_kernel(less_equal_kernel<double>, a, b, out, a.data<double>(), b.data<double>(), out.data<bool>()); break;
        case DType::I8:    launch_elementwise_kernel(less_equal_kernel<int8_t>, a, b, out, a.data<int8_t>(), b.data<int8_t>(), out.data<bool>()); break;
        case DType::I16:   launch_elementwise_kernel(less_equal_kernel<int16_t>, a, b, out, a.data<int16_t>(), b.data<int16_t>(), out.data<bool>()); break;
        case DType::I32:   launch_elementwise_kernel(less_equal_kernel<int32_t>, a, b, out, a.data<int32_t>(), b.data<int32_t>(), out.data<bool>()); break;
        case DType::I64:   launch_elementwise_kernel(less_equal_kernel<int64_t>, a, b, out, a.data<int64_t>(), b.data<int64_t>(), out.data<bool>()); break;
        case DType::U8:    launch_elementwise_kernel(less_equal_kernel<uint8_t>, a, b, out, a.data<uint8_t>(), b.data<uint8_t>(), out.data<bool>()); break;
        case DType::U16:   launch_elementwise_kernel(less_equal_kernel<uint16_t>, a, b, out, a.data<uint16_t>(), b.data<uint16_t>(), out.data<bool>()); break;
        case DType::U32:   launch_elementwise_kernel(less_equal_kernel<uint32_t>, a, b, out, a.data<uint32_t>(), b.data<uint32_t>(), out.data<bool>()); break;
        case DType::U64:   launch_elementwise_kernel(less_equal_kernel<uint64_t>, a, b, out, a.data<uint64_t>(), b.data<uint64_t>(), out.data<bool>()); break;
        case DType::C32:   launch_elementwise_kernel(less_equal_c32_kernel, a, b, out, a.data<std::complex<float>>(), b.data<std::complex<float>>(), out.data<bool>()); break;
        case DType::C64:   launch_elementwise_kernel(less_equal_c64_kernel, a, b, out, a.data<std::complex<double>>(), b.data<std::complex<double>>(), out.data<bool>()); break;
        case DType::BOOL:  launch_elementwise_kernel(less_equal_kernel<bool>, a, b, out, a.data<bool>(), b.data<bool>(), out.data<bool>()); break;
        default: INS_THROW("less_equal: unsupported dtype ", dtype_name(a.dtype()));
        }
        return { out };
    }

    static OpArgs logical_and_wrapper(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        Array out(a.shape(), DType::BOOL, a.place());

        switch (a.dtype()) {
        case DType::F32:   launch_elementwise_kernel(logical_and_kernel<float>, a, b, out, a.data<float>(), b.data<float>(), out.data<bool>()); break;
        case DType::F64:   launch_elementwise_kernel(logical_and_kernel<double>, a, b, out, a.data<double>(), b.data<double>(), out.data<bool>()); break;
        case DType::I8:    launch_elementwise_kernel(logical_and_kernel<int8_t>, a, b, out, a.data<int8_t>(), b.data<int8_t>(), out.data<bool>()); break;
        case DType::I16:   launch_elementwise_kernel(logical_and_kernel<int16_t>, a, b, out, a.data<int16_t>(), b.data<int16_t>(), out.data<bool>()); break;
        case DType::I32:   launch_elementwise_kernel(logical_and_kernel<int32_t>, a, b, out, a.data<int32_t>(), b.data<int32_t>(), out.data<bool>()); break;
        case DType::I64:   launch_elementwise_kernel(logical_and_kernel<int64_t>, a, b, out, a.data<int64_t>(), b.data<int64_t>(), out.data<bool>()); break;
        case DType::U8:    launch_elementwise_kernel(logical_and_kernel<uint8_t>, a, b, out, a.data<uint8_t>(), b.data<uint8_t>(), out.data<bool>()); break;
        case DType::U16:   launch_elementwise_kernel(logical_and_kernel<uint16_t>, a, b, out, a.data<uint16_t>(), b.data<uint16_t>(), out.data<bool>()); break;
        case DType::U32:   launch_elementwise_kernel(logical_and_kernel<uint32_t>, a, b, out, a.data<uint32_t>(), b.data<uint32_t>(), out.data<bool>()); break;
        case DType::U64:   launch_elementwise_kernel(logical_and_kernel<uint64_t>, a, b, out, a.data<uint64_t>(), b.data<uint64_t>(), out.data<bool>()); break;
        case DType::BOOL:  launch_elementwise_kernel(logical_and_kernel<bool>, a, b, out, a.data<bool>(), b.data<bool>(), out.data<bool>()); break;
        default: INS_THROW("logical_and: unsupported dtype ", dtype_name(a.dtype()));
        }
        return { out };
    }

    static OpArgs logical_or_wrapper(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        Array out(a.shape(), DType::BOOL, a.place());

        switch (a.dtype()) {
        case DType::F32:   launch_elementwise_kernel(logical_or_kernel<float>, a, b, out, a.data<float>(), b.data<float>(), out.data<bool>()); break;
        case DType::F64:   launch_elementwise_kernel(logical_or_kernel<double>, a, b, out, a.data<double>(), b.data<double>(), out.data<bool>()); break;
        case DType::I8:    launch_elementwise_kernel(logical_or_kernel<int8_t>, a, b, out, a.data<int8_t>(), b.data<int8_t>(), out.data<bool>()); break;
        case DType::I16:   launch_elementwise_kernel(logical_or_kernel<int16_t>, a, b, out, a.data<int16_t>(), b.data<int16_t>(), out.data<bool>()); break;
        case DType::I32:   launch_elementwise_kernel(logical_or_kernel<int32_t>, a, b, out, a.data<int32_t>(), b.data<int32_t>(), out.data<bool>()); break;
        case DType::I64:   launch_elementwise_kernel(logical_or_kernel<int64_t>, a, b, out, a.data<int64_t>(), b.data<int64_t>(), out.data<bool>()); break;
        case DType::U8:    launch_elementwise_kernel(logical_or_kernel<uint8_t>, a, b, out, a.data<uint8_t>(), b.data<uint8_t>(), out.data<bool>()); break;
        case DType::U16:   launch_elementwise_kernel(logical_or_kernel<uint16_t>, a, b, out, a.data<uint16_t>(), b.data<uint16_t>(), out.data<bool>()); break;
        case DType::U32:   launch_elementwise_kernel(logical_or_kernel<uint32_t>, a, b, out, a.data<uint32_t>(), b.data<uint32_t>(), out.data<bool>()); break;
        case DType::U64:   launch_elementwise_kernel(logical_or_kernel<uint64_t>, a, b, out, a.data<uint64_t>(), b.data<uint64_t>(), out.data<bool>()); break;
        case DType::BOOL:  launch_elementwise_kernel(logical_or_kernel<bool>, a, b, out, a.data<bool>(), b.data<bool>(), out.data<bool>()); break;
        default: INS_THROW("logical_or: unsupported dtype ", dtype_name(a.dtype()));
        }
        return { out };
    }

    static OpArgs logical_xor_wrapper(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        Array out(a.shape(), DType::BOOL, a.place());

        switch (a.dtype()) {
        case DType::F32:   launch_elementwise_kernel(logical_xor_kernel<float>, a, b, out, a.data<float>(), b.data<float>(), out.data<bool>()); break;
        case DType::F64:   launch_elementwise_kernel(logical_xor_kernel<double>, a, b, out, a.data<double>(), b.data<double>(), out.data<bool>()); break;
        case DType::I8:    launch_elementwise_kernel(logical_xor_kernel<int8_t>, a, b, out, a.data<int8_t>(), b.data<int8_t>(), out.data<bool>()); break;
        case DType::I16:   launch_elementwise_kernel(logical_xor_kernel<int16_t>, a, b, out, a.data<int16_t>(), b.data<int16_t>(), out.data<bool>()); break;
        case DType::I32:   launch_elementwise_kernel(logical_xor_kernel<int32_t>, a, b, out, a.data<int32_t>(), b.data<int32_t>(), out.data<bool>()); break;
        case DType::I64:   launch_elementwise_kernel(logical_xor_kernel<int64_t>, a, b, out, a.data<int64_t>(), b.data<int64_t>(), out.data<bool>()); break;
        case DType::U8:    launch_elementwise_kernel(logical_xor_kernel<uint8_t>, a, b, out, a.data<uint8_t>(), b.data<uint8_t>(), out.data<bool>()); break;
        case DType::U16:   launch_elementwise_kernel(logical_xor_kernel<uint16_t>, a, b, out, a.data<uint16_t>(), b.data<uint16_t>(), out.data<bool>()); break;
        case DType::U32:   launch_elementwise_kernel(logical_xor_kernel<uint32_t>, a, b, out, a.data<uint32_t>(), b.data<uint32_t>(), out.data<bool>()); break;
        case DType::U64:   launch_elementwise_kernel(logical_xor_kernel<uint64_t>, a, b, out, a.data<uint64_t>(), b.data<uint64_t>(), out.data<bool>()); break;
        case DType::BOOL:  launch_elementwise_kernel(logical_xor_kernel<bool>, a, b, out, a.data<bool>(), b.data<bool>(), out.data<bool>()); break;
        default: INS_THROW("logical_xor: unsupported dtype ", dtype_name(a.dtype()));
        }
        return { out };
    }

    static OpArgs maximum_wrapper(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        Array out(a.shape(), a.dtype(), a.place());

        switch (a.dtype()) {
        case DType::F32:   launch_elementwise_kernel(maximum_kernel<float>, a, b, out, a.data<float>(), b.data<float>(), out.data<float>()); break;
        case DType::F64:   launch_elementwise_kernel(maximum_kernel<double>, a, b, out, a.data<double>(), b.data<double>(), out.data<double>()); break;
        case DType::I8:    launch_elementwise_kernel(maximum_kernel<int8_t>, a, b, out, a.data<int8_t>(), b.data<int8_t>(), out.data<int8_t>()); break;
        case DType::I16:   launch_elementwise_kernel(maximum_kernel<int16_t>, a, b, out, a.data<int16_t>(), b.data<int16_t>(), out.data<int16_t>()); break;
        case DType::I32:   launch_elementwise_kernel(maximum_kernel<int32_t>, a, b, out, a.data<int32_t>(), b.data<int32_t>(), out.data<int32_t>()); break;
        case DType::I64:   launch_elementwise_kernel(maximum_kernel<int64_t>, a, b, out, a.data<int64_t>(), b.data<int64_t>(), out.data<int64_t>()); break;
        case DType::U8:    launch_elementwise_kernel(maximum_kernel<uint8_t>, a, b, out, a.data<uint8_t>(), b.data<uint8_t>(), out.data<uint8_t>()); break;
        case DType::U16:   launch_elementwise_kernel(maximum_kernel<uint16_t>, a, b, out, a.data<uint16_t>(), b.data<uint16_t>(), out.data<uint16_t>()); break;
        case DType::U32:   launch_elementwise_kernel(maximum_kernel<uint32_t>, a, b, out, a.data<uint32_t>(), b.data<uint32_t>(), out.data<uint32_t>()); break;
        case DType::U64:   launch_elementwise_kernel(maximum_kernel<uint64_t>, a, b, out, a.data<uint64_t>(), b.data<uint64_t>(), out.data<uint64_t>()); break;
        case DType::C32:   launch_elementwise_kernel(maximum_c32_kernel, a, b, out, a.data<std::complex<float>>(), b.data<std::complex<float>>(), out.data<std::complex<float>>()); break;
        case DType::C64:   launch_elementwise_kernel(maximum_c64_kernel, a, b, out, a.data<std::complex<double>>(), b.data<std::complex<double>>(), out.data<std::complex<double>>()); break;
        case DType::BOOL:  launch_elementwise_kernel(maximum_kernel<bool>, a, b, out, a.data<bool>(), b.data<bool>(), out.data<bool>()); break;
        default: INS_THROW("maximum: unsupported dtype ", dtype_name(a.dtype()));
        }
        return { out };
    }

    static OpArgs minimum_wrapper(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        Array out(a.shape(), a.dtype(), a.place());

        switch (a.dtype()) {
        case DType::F32:   launch_elementwise_kernel(minimum_kernel<float>, a, b, out, a.data<float>(), b.data<float>(), out.data<float>()); break;
        case DType::F64:   launch_elementwise_kernel(minimum_kernel<double>, a, b, out, a.data<double>(), b.data<double>(), out.data<double>()); break;
        case DType::I8:    launch_elementwise_kernel(minimum_kernel<int8_t>, a, b, out, a.data<int8_t>(), b.data<int8_t>(), out.data<int8_t>()); break;
        case DType::I16:   launch_elementwise_kernel(minimum_kernel<int16_t>, a, b, out, a.data<int16_t>(), b.data<int16_t>(), out.data<int16_t>()); break;
        case DType::I32:   launch_elementwise_kernel(minimum_kernel<int32_t>, a, b, out, a.data<int32_t>(), b.data<int32_t>(), out.data<int32_t>()); break;
        case DType::I64:   launch_elementwise_kernel(minimum_kernel<int64_t>, a, b, out, a.data<int64_t>(), b.data<int64_t>(), out.data<int64_t>()); break;
        case DType::U8:    launch_elementwise_kernel(minimum_kernel<uint8_t>, a, b, out, a.data<uint8_t>(), b.data<uint8_t>(), out.data<uint8_t>()); break;
        case DType::U16:   launch_elementwise_kernel(minimum_kernel<uint16_t>, a, b, out, a.data<uint16_t>(), b.data<uint16_t>(), out.data<uint16_t>()); break;
        case DType::U32:   launch_elementwise_kernel(minimum_kernel<uint32_t>, a, b, out, a.data<uint32_t>(), b.data<uint32_t>(), out.data<uint32_t>()); break;
        case DType::U64:   launch_elementwise_kernel(minimum_kernel<uint64_t>, a, b, out, a.data<uint64_t>(), b.data<uint64_t>(), out.data<uint64_t>()); break;
        case DType::C32:   launch_elementwise_kernel(minimum_c32_kernel, a, b, out, a.data<std::complex<float>>(), b.data<std::complex<float>>(), out.data<std::complex<float>>()); break;
        case DType::C64:   launch_elementwise_kernel(minimum_c64_kernel, a, b, out, a.data<std::complex<double>>(), b.data<std::complex<double>>(), out.data<std::complex<double>>()); break;
        case DType::BOOL:  launch_elementwise_kernel(minimum_kernel<bool>, a, b, out, a.data<bool>(), b.data<bool>(), out.data<bool>()); break;
        default: INS_THROW("minimum: unsupported dtype ", dtype_name(a.dtype()));
        }
        return { out };
    }

    static OpArgs power_wrapper(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        Array out(a.shape(), a.dtype(), a.place());

        switch (a.dtype()) {
        case DType::F32:   launch_elementwise_kernel(power_kernel<float>, a, b, out, a.data<float>(), b.data<float>(), out.data<float>()); break;
        case DType::F64:   launch_elementwise_kernel(power_kernel<double>, a, b, out, a.data<double>(), b.data<double>(), out.data<double>()); break;
        case DType::I8:    launch_elementwise_kernel(power_int_kernel<int8_t>, a, b, out, a.data<int8_t>(), b.data<int8_t>(), out.data<int8_t>()); break;
        case DType::I16:   launch_elementwise_kernel(power_int_kernel<int16_t>, a, b, out, a.data<int16_t>(), b.data<int16_t>(), out.data<int16_t>()); break;
        case DType::I32:   launch_elementwise_kernel(power_int_kernel<int32_t>, a, b, out, a.data<int32_t>(), b.data<int32_t>(), out.data<int32_t>()); break;
        case DType::I64:   launch_elementwise_kernel(power_int_kernel<int64_t>, a, b, out, a.data<int64_t>(), b.data<int64_t>(), out.data<int64_t>()); break;
        case DType::U8:    launch_elementwise_kernel(power_int_kernel<uint8_t>, a, b, out, a.data<uint8_t>(), b.data<uint8_t>(), out.data<uint8_t>()); break;
        case DType::U16:   launch_elementwise_kernel(power_int_kernel<uint16_t>, a, b, out, a.data<uint16_t>(), b.data<uint16_t>(), out.data<uint16_t>()); break;
        case DType::U32:   launch_elementwise_kernel(power_int_kernel<uint32_t>, a, b, out, a.data<uint32_t>(), b.data<uint32_t>(), out.data<uint32_t>()); break;
        case DType::U64:   launch_elementwise_kernel(power_int_kernel<uint64_t>, a, b, out, a.data<uint64_t>(), b.data<uint64_t>(), out.data<uint64_t>()); break;
        default: INS_THROW("power: unsupported dtype ", dtype_name(a.dtype()));
        }
        return { out };
    }

    static OpArgs bitwise_and_wrapper(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        Array out(a.shape(), a.dtype(), a.place());

        switch (a.dtype()) {
        case DType::BOOL:  launch_elementwise_kernel(bitwise_and_kernel<bool>, a, b, out, a.data<bool>(), b.data<bool>(), out.data<bool>()); break;
        case DType::U8:    launch_elementwise_kernel(bitwise_and_kernel<uint8_t>, a, b, out, a.data<uint8_t>(), b.data<uint8_t>(), out.data<uint8_t>()); break;
        case DType::I8:    launch_elementwise_kernel(bitwise_and_kernel<int8_t>, a, b, out, a.data<int8_t>(), b.data<int8_t>(), out.data<int8_t>()); break;
        case DType::I16:   launch_elementwise_kernel(bitwise_and_kernel<int16_t>, a, b, out, a.data<int16_t>(), b.data<int16_t>(), out.data<int16_t>()); break;
        case DType::I32:   launch_elementwise_kernel(bitwise_and_kernel<int32_t>, a, b, out, a.data<int32_t>(), b.data<int32_t>(), out.data<int32_t>()); break;
        case DType::I64:   launch_elementwise_kernel(bitwise_and_kernel<int64_t>, a, b, out, a.data<int64_t>(), b.data<int64_t>(), out.data<int64_t>()); break;
        default: INS_THROW("bitwise_and: unsupported dtype ", dtype_name(a.dtype()));
        }
        return { out };
    }

    static OpArgs bitwise_or_wrapper(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        Array out(a.shape(), a.dtype(), a.place());

        switch (a.dtype()) {
        case DType::BOOL:  launch_elementwise_kernel(bitwise_or_kernel<bool>, a, b, out, a.data<bool>(), b.data<bool>(), out.data<bool>()); break;
        case DType::U8:    launch_elementwise_kernel(bitwise_or_kernel<uint8_t>, a, b, out, a.data<uint8_t>(), b.data<uint8_t>(), out.data<uint8_t>()); break;
        case DType::I8:    launch_elementwise_kernel(bitwise_or_kernel<int8_t>, a, b, out, a.data<int8_t>(), b.data<int8_t>(), out.data<int8_t>()); break;
        case DType::I16:   launch_elementwise_kernel(bitwise_or_kernel<int16_t>, a, b, out, a.data<int16_t>(), b.data<int16_t>(), out.data<int16_t>()); break;
        case DType::I32:   launch_elementwise_kernel(bitwise_or_kernel<int32_t>, a, b, out, a.data<int32_t>(), b.data<int32_t>(), out.data<int32_t>()); break;
        case DType::I64:   launch_elementwise_kernel(bitwise_or_kernel<int64_t>, a, b, out, a.data<int64_t>(), b.data<int64_t>(), out.data<int64_t>()); break;
        default: INS_THROW("bitwise_or: unsupported dtype ", dtype_name(a.dtype()));
        }
        return { out };
    }

    static OpArgs bitwise_xor_wrapper(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        Array out(a.shape(), a.dtype(), a.place());

        switch (a.dtype()) {
        case DType::BOOL:  launch_elementwise_kernel(bitwise_xor_kernel<bool>, a, b, out, a.data<bool>(), b.data<bool>(), out.data<bool>()); break;
        case DType::U8:    launch_elementwise_kernel(bitwise_xor_kernel<uint8_t>, a, b, out, a.data<uint8_t>(), b.data<uint8_t>(), out.data<uint8_t>()); break;
        case DType::I8:    launch_elementwise_kernel(bitwise_xor_kernel<int8_t>, a, b, out, a.data<int8_t>(), b.data<int8_t>(), out.data<int8_t>()); break;
        case DType::I16:   launch_elementwise_kernel(bitwise_xor_kernel<int16_t>, a, b, out, a.data<int16_t>(), b.data<int16_t>(), out.data<int16_t>()); break;
        case DType::I32:   launch_elementwise_kernel(bitwise_xor_kernel<int32_t>, a, b, out, a.data<int32_t>(), b.data<int32_t>(), out.data<int32_t>()); break;
        case DType::I64:   launch_elementwise_kernel(bitwise_xor_kernel<int64_t>, a, b, out, a.data<int64_t>(), b.data<int64_t>(), out.data<int64_t>()); break;
        default: INS_THROW("bitwise_xor: unsupported dtype ", dtype_name(a.dtype()));
        }
        return { out };
    }

    static OpArgs left_shift_wrapper(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b_raw = std::any_cast<const Array&>(args[1]);

        INS_CHECK(is_integer(b_raw.dtype()) || b_raw.dtype() == DType::BOOL,
            "bitwise_left_shift: shift amount must be integer type, got ", dtype_name(b_raw.dtype()));

        Array b = (b_raw.dtype() == DType::I64) ? b_raw : b_raw.to(DType::I64);

        Array out(a.shape(), a.dtype(), a.place());

        switch (a.dtype()) {
        case DType::BOOL: launch_elementwise_kernel(left_shift_kernel<bool>, a, b, out, a.data<bool>(), b.data<int64_t>(), out.data<bool>()); break;
        case DType::U8:   launch_elementwise_kernel(left_shift_kernel<uint8_t>, a, b, out, a.data<uint8_t>(), b.data<int64_t>(), out.data<uint8_t>()); break;
        case DType::I8:   launch_elementwise_kernel(left_shift_kernel<int8_t>, a, b, out, a.data<int8_t>(), b.data<int64_t>(), out.data<int8_t>()); break;
        case DType::I16:  launch_elementwise_kernel(left_shift_kernel<int16_t>, a, b, out, a.data<int16_t>(), b.data<int64_t>(), out.data<int16_t>()); break;
        case DType::I32:  launch_elementwise_kernel(left_shift_kernel<int32_t>, a, b, out, a.data<int32_t>(), b.data<int64_t>(), out.data<int32_t>()); break;
        case DType::I64:  launch_elementwise_kernel(left_shift_kernel<int64_t>, a, b, out, a.data<int64_t>(), b.data<int64_t>(), out.data<int64_t>()); break;
        default: INS_THROW("bitwise_left_shift: unsupported dtype ", dtype_name(a.dtype()));
        }
        return { out };
    }

    static OpArgs right_shift_wrapper(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b_raw = std::any_cast<const Array&>(args[1]);

        INS_CHECK(is_integer(b_raw.dtype()) || b_raw.dtype() == DType::BOOL,
            "bitwise_right_shift: shift amount must be integer type, got ", dtype_name(b_raw.dtype()));

        Array b = (b_raw.dtype() == DType::I64) ? b_raw : b_raw.to(DType::I64);

        Array out(a.shape(), a.dtype(), a.place());

        switch (a.dtype()) {
        case DType::BOOL: launch_elementwise_kernel(right_shift_kernel<bool>, a, b, out, a.data<bool>(), b.data<int64_t>(), out.data<bool>()); break;
        case DType::U8:   launch_elementwise_kernel(right_shift_kernel<uint8_t>, a, b, out, a.data<uint8_t>(), b.data<int64_t>(), out.data<uint8_t>()); break;
        case DType::I8:   launch_elementwise_kernel(right_shift_kernel<int8_t>, a, b, out, a.data<int8_t>(), b.data<int64_t>(), out.data<int8_t>()); break;
        case DType::I16:  launch_elementwise_kernel(right_shift_kernel<int16_t>, a, b, out, a.data<int16_t>(), b.data<int64_t>(), out.data<int16_t>()); break;
        case DType::I32:  launch_elementwise_kernel(right_shift_kernel<int32_t>, a, b, out, a.data<int32_t>(), b.data<int64_t>(), out.data<int32_t>()); break;
        case DType::I64:  launch_elementwise_kernel(right_shift_kernel<int64_t>, a, b, out, a.data<int64_t>(), b.data<int64_t>(), out.data<int64_t>()); break;
        default: INS_THROW("bitwise_right_shift: unsupported dtype ", dtype_name(a.dtype()));
        }
        return { out };
    }

    // ============================================================================
    // Kernel Registration
    // ============================================================================

    REGISTER_KERNEL(add, GPU, F32, add_wrapper);
    REGISTER_KERNEL(add, GPU, F64, add_wrapper);
    REGISTER_KERNEL(add, GPU, I8, add_wrapper);
    REGISTER_KERNEL(add, GPU, I16, add_wrapper);
    REGISTER_KERNEL(add, GPU, I32, add_wrapper);
    REGISTER_KERNEL(add, GPU, I64, add_wrapper);
    REGISTER_KERNEL(add, GPU, U8, add_wrapper);
    REGISTER_KERNEL(add, GPU, U16, add_wrapper);
    REGISTER_KERNEL(add, GPU, U32, add_wrapper);
    REGISTER_KERNEL(add, GPU, U64, add_wrapper);
    REGISTER_KERNEL(add, GPU, C32, add_wrapper);
    REGISTER_KERNEL(add, GPU, C64, add_wrapper);
    REGISTER_KERNEL(add, GPU, BOOL, add_wrapper);

    REGISTER_KERNEL(sub, GPU, F32, sub_wrapper);
    REGISTER_KERNEL(sub, GPU, F64, sub_wrapper);
    REGISTER_KERNEL(sub, GPU, I8, sub_wrapper);
    REGISTER_KERNEL(sub, GPU, I16, sub_wrapper);
    REGISTER_KERNEL(sub, GPU, I32, sub_wrapper);
    REGISTER_KERNEL(sub, GPU, I64, sub_wrapper);
    REGISTER_KERNEL(sub, GPU, U8, sub_wrapper);
    REGISTER_KERNEL(sub, GPU, U16, sub_wrapper);
    REGISTER_KERNEL(sub, GPU, U32, sub_wrapper);
    REGISTER_KERNEL(sub, GPU, U64, sub_wrapper);
    REGISTER_KERNEL(sub, GPU, C32, sub_wrapper);
    REGISTER_KERNEL(sub, GPU, C64, sub_wrapper);
    REGISTER_KERNEL(sub, GPU, BOOL, sub_wrapper);

    REGISTER_KERNEL(mul, GPU, F32, mul_wrapper);
    REGISTER_KERNEL(mul, GPU, F64, mul_wrapper);
    REGISTER_KERNEL(mul, GPU, I8, mul_wrapper);
    REGISTER_KERNEL(mul, GPU, I16, mul_wrapper);
    REGISTER_KERNEL(mul, GPU, I32, mul_wrapper);
    REGISTER_KERNEL(mul, GPU, I64, mul_wrapper);
    REGISTER_KERNEL(mul, GPU, U8, mul_wrapper);
    REGISTER_KERNEL(mul, GPU, U16, mul_wrapper);
    REGISTER_KERNEL(mul, GPU, U32, mul_wrapper);
    REGISTER_KERNEL(mul, GPU, U64, mul_wrapper);
    REGISTER_KERNEL(mul, GPU, C32, mul_wrapper);
    REGISTER_KERNEL(mul, GPU, C64, mul_wrapper);
    REGISTER_KERNEL(mul, GPU, BOOL, mul_wrapper);

    REGISTER_KERNEL(div, GPU, F32, div_wrapper);
    REGISTER_KERNEL(div, GPU, F64, div_wrapper);
    REGISTER_KERNEL(div, GPU, I8, div_wrapper);
    REGISTER_KERNEL(div, GPU, I16, div_wrapper);
    REGISTER_KERNEL(div, GPU, I32, div_wrapper);
    REGISTER_KERNEL(div, GPU, I64, div_wrapper);
    REGISTER_KERNEL(div, GPU, U8, div_wrapper);
    REGISTER_KERNEL(div, GPU, U16, div_wrapper);
    REGISTER_KERNEL(div, GPU, U32, div_wrapper);
    REGISTER_KERNEL(div, GPU, U64, div_wrapper);
    REGISTER_KERNEL(div, GPU, C32, div_wrapper);
    REGISTER_KERNEL(div, GPU, C64, div_wrapper);
    REGISTER_KERNEL(div, GPU, BOOL, div_wrapper);

    REGISTER_KERNEL(mod, GPU, I8, mod_wrapper);
    REGISTER_KERNEL(mod, GPU, I16, mod_wrapper);
    REGISTER_KERNEL(mod, GPU, I32, mod_wrapper);
    REGISTER_KERNEL(mod, GPU, I64, mod_wrapper);
    REGISTER_KERNEL(mod, GPU, U8, mod_wrapper);
    REGISTER_KERNEL(mod, GPU, U16, mod_wrapper);
    REGISTER_KERNEL(mod, GPU, U32, mod_wrapper);
    REGISTER_KERNEL(mod, GPU, U64, mod_wrapper);
    REGISTER_KERNEL(mod, GPU, F32, mod_wrapper);
    REGISTER_KERNEL(mod, GPU, F64, mod_wrapper);

    REGISTER_KERNEL(equal, GPU, F32, equal_wrapper);
    REGISTER_KERNEL(equal, GPU, F64, equal_wrapper);
    REGISTER_KERNEL(equal, GPU, I8, equal_wrapper);
    REGISTER_KERNEL(equal, GPU, I16, equal_wrapper);
    REGISTER_KERNEL(equal, GPU, I32, equal_wrapper);
    REGISTER_KERNEL(equal, GPU, I64, equal_wrapper);
    REGISTER_KERNEL(equal, GPU, U8, equal_wrapper);
    REGISTER_KERNEL(equal, GPU, U16, equal_wrapper);
    REGISTER_KERNEL(equal, GPU, U32, equal_wrapper);
    REGISTER_KERNEL(equal, GPU, U64, equal_wrapper);
    REGISTER_KERNEL(equal, GPU, C32, equal_wrapper);
    REGISTER_KERNEL(equal, GPU, C64, equal_wrapper);
    REGISTER_KERNEL(equal, GPU, BOOL, equal_wrapper);

    REGISTER_KERNEL(not_equal, GPU, F32, not_equal_wrapper);
    REGISTER_KERNEL(not_equal, GPU, F64, not_equal_wrapper);
    REGISTER_KERNEL(not_equal, GPU, I8, not_equal_wrapper);
    REGISTER_KERNEL(not_equal, GPU, I16, not_equal_wrapper);
    REGISTER_KERNEL(not_equal, GPU, I32, not_equal_wrapper);
    REGISTER_KERNEL(not_equal, GPU, I64, not_equal_wrapper);
    REGISTER_KERNEL(not_equal, GPU, U8, not_equal_wrapper);
    REGISTER_KERNEL(not_equal, GPU, U16, not_equal_wrapper);
    REGISTER_KERNEL(not_equal, GPU, U32, not_equal_wrapper);
    REGISTER_KERNEL(not_equal, GPU, U64, not_equal_wrapper);
    REGISTER_KERNEL(not_equal, GPU, C32, not_equal_wrapper);
    REGISTER_KERNEL(not_equal, GPU, C64, not_equal_wrapper);
    REGISTER_KERNEL(not_equal, GPU, BOOL, not_equal_wrapper);

    REGISTER_KERNEL(greater, GPU, F32, greater_wrapper);
    REGISTER_KERNEL(greater, GPU, F64, greater_wrapper);
    REGISTER_KERNEL(greater, GPU, I8, greater_wrapper);
    REGISTER_KERNEL(greater, GPU, I16, greater_wrapper);
    REGISTER_KERNEL(greater, GPU, I32, greater_wrapper);
    REGISTER_KERNEL(greater, GPU, I64, greater_wrapper);
    REGISTER_KERNEL(greater, GPU, U8, greater_wrapper);
    REGISTER_KERNEL(greater, GPU, U16, greater_wrapper);
    REGISTER_KERNEL(greater, GPU, U32, greater_wrapper);
    REGISTER_KERNEL(greater, GPU, U64, greater_wrapper);
    REGISTER_KERNEL(greater, GPU, C32, greater_wrapper);
    REGISTER_KERNEL(greater, GPU, C64, greater_wrapper);
    REGISTER_KERNEL(greater, GPU, BOOL, greater_wrapper);

    REGISTER_KERNEL(less, GPU, F32, less_wrapper);
    REGISTER_KERNEL(less, GPU, F64, less_wrapper);
    REGISTER_KERNEL(less, GPU, I8, less_wrapper);
    REGISTER_KERNEL(less, GPU, I16, less_wrapper);
    REGISTER_KERNEL(less, GPU, I32, less_wrapper);
    REGISTER_KERNEL(less, GPU, I64, less_wrapper);
    REGISTER_KERNEL(less, GPU, U8, less_wrapper);
    REGISTER_KERNEL(less, GPU, U16, less_wrapper);
    REGISTER_KERNEL(less, GPU, U32, less_wrapper);
    REGISTER_KERNEL(less, GPU, U64, less_wrapper);
    REGISTER_KERNEL(less, GPU, C32, less_wrapper);
    REGISTER_KERNEL(less, GPU, C64, less_wrapper);
    REGISTER_KERNEL(less, GPU, BOOL, less_wrapper);

    REGISTER_KERNEL(greater_equal, GPU, F32, greater_equal_wrapper);
    REGISTER_KERNEL(greater_equal, GPU, F64, greater_equal_wrapper);
    REGISTER_KERNEL(greater_equal, GPU, I8, greater_equal_wrapper);
    REGISTER_KERNEL(greater_equal, GPU, I16, greater_equal_wrapper);
    REGISTER_KERNEL(greater_equal, GPU, I32, greater_equal_wrapper);
    REGISTER_KERNEL(greater_equal, GPU, I64, greater_equal_wrapper);
    REGISTER_KERNEL(greater_equal, GPU, U8, greater_equal_wrapper);
    REGISTER_KERNEL(greater_equal, GPU, U16, greater_equal_wrapper);
    REGISTER_KERNEL(greater_equal, GPU, U32, greater_equal_wrapper);
    REGISTER_KERNEL(greater_equal, GPU, U64, greater_equal_wrapper);
    REGISTER_KERNEL(greater_equal, GPU, C32, greater_equal_wrapper);
    REGISTER_KERNEL(greater_equal, GPU, C64, greater_equal_wrapper);
    REGISTER_KERNEL(greater_equal, GPU, BOOL, greater_equal_wrapper);

    REGISTER_KERNEL(less_equal, GPU, F32, less_equal_wrapper);
    REGISTER_KERNEL(less_equal, GPU, F64, less_equal_wrapper);
    REGISTER_KERNEL(less_equal, GPU, I8, less_equal_wrapper);
    REGISTER_KERNEL(less_equal, GPU, I16, less_equal_wrapper);
    REGISTER_KERNEL(less_equal, GPU, I32, less_equal_wrapper);
    REGISTER_KERNEL(less_equal, GPU, I64, less_equal_wrapper);
    REGISTER_KERNEL(less_equal, GPU, U8, less_equal_wrapper);
    REGISTER_KERNEL(less_equal, GPU, U16, less_equal_wrapper);
    REGISTER_KERNEL(less_equal, GPU, U32, less_equal_wrapper);
    REGISTER_KERNEL(less_equal, GPU, U64, less_equal_wrapper);
    REGISTER_KERNEL(less_equal, GPU, C32, less_equal_wrapper);
    REGISTER_KERNEL(less_equal, GPU, C64, less_equal_wrapper);
    REGISTER_KERNEL(less_equal, GPU, BOOL, less_equal_wrapper);

    REGISTER_KERNEL(logical_and, GPU, F32, logical_and_wrapper);
    REGISTER_KERNEL(logical_and, GPU, F64, logical_and_wrapper);
    REGISTER_KERNEL(logical_and, GPU, I8, logical_and_wrapper);
    REGISTER_KERNEL(logical_and, GPU, I16, logical_and_wrapper);
    REGISTER_KERNEL(logical_and, GPU, I32, logical_and_wrapper);
    REGISTER_KERNEL(logical_and, GPU, I64, logical_and_wrapper);
    REGISTER_KERNEL(logical_and, GPU, U8, logical_and_wrapper);
    REGISTER_KERNEL(logical_and, GPU, U16, logical_and_wrapper);
    REGISTER_KERNEL(logical_and, GPU, U32, logical_and_wrapper);
    REGISTER_KERNEL(logical_and, GPU, U64, logical_and_wrapper);
    REGISTER_KERNEL(logical_and, GPU, BOOL, logical_and_wrapper);

    REGISTER_KERNEL(logical_or, GPU, F32, logical_or_wrapper);
    REGISTER_KERNEL(logical_or, GPU, F64, logical_or_wrapper);
    REGISTER_KERNEL(logical_or, GPU, I8, logical_or_wrapper);
    REGISTER_KERNEL(logical_or, GPU, I16, logical_or_wrapper);
    REGISTER_KERNEL(logical_or, GPU, I32, logical_or_wrapper);
    REGISTER_KERNEL(logical_or, GPU, I64, logical_or_wrapper);
    REGISTER_KERNEL(logical_or, GPU, U8, logical_or_wrapper);
    REGISTER_KERNEL(logical_or, GPU, U16, logical_or_wrapper);
    REGISTER_KERNEL(logical_or, GPU, U32, logical_or_wrapper);
    REGISTER_KERNEL(logical_or, GPU, U64, logical_or_wrapper);
    REGISTER_KERNEL(logical_or, GPU, BOOL, logical_or_wrapper);

    REGISTER_KERNEL(logical_xor, GPU, F32, logical_xor_wrapper);
    REGISTER_KERNEL(logical_xor, GPU, F64, logical_xor_wrapper);
    REGISTER_KERNEL(logical_xor, GPU, I8, logical_xor_wrapper);
    REGISTER_KERNEL(logical_xor, GPU, I16, logical_xor_wrapper);
    REGISTER_KERNEL(logical_xor, GPU, I32, logical_xor_wrapper);
    REGISTER_KERNEL(logical_xor, GPU, I64, logical_xor_wrapper);
    REGISTER_KERNEL(logical_xor, GPU, U8, logical_xor_wrapper);
    REGISTER_KERNEL(logical_xor, GPU, U16, logical_xor_wrapper);
    REGISTER_KERNEL(logical_xor, GPU, U32, logical_xor_wrapper);
    REGISTER_KERNEL(logical_xor, GPU, U64, logical_xor_wrapper);
    REGISTER_KERNEL(logical_xor, GPU, BOOL, logical_xor_wrapper);

    REGISTER_KERNEL(maximum, GPU, F32, maximum_wrapper);
    REGISTER_KERNEL(maximum, GPU, F64, maximum_wrapper);
    REGISTER_KERNEL(maximum, GPU, I8, maximum_wrapper);
    REGISTER_KERNEL(maximum, GPU, I16, maximum_wrapper);
    REGISTER_KERNEL(maximum, GPU, I32, maximum_wrapper);
    REGISTER_KERNEL(maximum, GPU, I64, maximum_wrapper);
    REGISTER_KERNEL(maximum, GPU, U8, maximum_wrapper);
    REGISTER_KERNEL(maximum, GPU, U16, maximum_wrapper);
    REGISTER_KERNEL(maximum, GPU, U32, maximum_wrapper);
    REGISTER_KERNEL(maximum, GPU, U64, maximum_wrapper);
    REGISTER_KERNEL(maximum, GPU, C32, maximum_wrapper);
    REGISTER_KERNEL(maximum, GPU, C64, maximum_wrapper);
    REGISTER_KERNEL(maximum, GPU, BOOL, maximum_wrapper);

    REGISTER_KERNEL(minimum, GPU, F32, minimum_wrapper);
    REGISTER_KERNEL(minimum, GPU, F64, minimum_wrapper);
    REGISTER_KERNEL(minimum, GPU, I8, minimum_wrapper);
    REGISTER_KERNEL(minimum, GPU, I16, minimum_wrapper);
    REGISTER_KERNEL(minimum, GPU, I32, minimum_wrapper);
    REGISTER_KERNEL(minimum, GPU, I64, minimum_wrapper);
    REGISTER_KERNEL(minimum, GPU, U8, minimum_wrapper);
    REGISTER_KERNEL(minimum, GPU, U16, minimum_wrapper);
    REGISTER_KERNEL(minimum, GPU, U32, minimum_wrapper);
    REGISTER_KERNEL(minimum, GPU, U64, minimum_wrapper);
    REGISTER_KERNEL(minimum, GPU, C32, minimum_wrapper);
    REGISTER_KERNEL(minimum, GPU, C64, minimum_wrapper);
    REGISTER_KERNEL(minimum, GPU, BOOL, minimum_wrapper);

    REGISTER_KERNEL(pow, GPU, F32, power_wrapper);
    REGISTER_KERNEL(pow, GPU, F64, power_wrapper);
    REGISTER_KERNEL(pow, GPU, I8, power_wrapper);
    REGISTER_KERNEL(pow, GPU, I16, power_wrapper);
    REGISTER_KERNEL(pow, GPU, I32, power_wrapper);
    REGISTER_KERNEL(pow, GPU, I64, power_wrapper);
    REGISTER_KERNEL(pow, GPU, U8, power_wrapper);
    REGISTER_KERNEL(pow, GPU, U16, power_wrapper);
    REGISTER_KERNEL(pow, GPU, U32, power_wrapper);
    REGISTER_KERNEL(pow, GPU, U64, power_wrapper);

    REGISTER_KERNEL(bitwise_and, GPU, BOOL, bitwise_and_wrapper);
    REGISTER_KERNEL(bitwise_and, GPU, U8, bitwise_and_wrapper);
    REGISTER_KERNEL(bitwise_and, GPU, I8, bitwise_and_wrapper);
    REGISTER_KERNEL(bitwise_and, GPU, I16, bitwise_and_wrapper);
    REGISTER_KERNEL(bitwise_and, GPU, I32, bitwise_and_wrapper);
    REGISTER_KERNEL(bitwise_and, GPU, I64, bitwise_and_wrapper);

    REGISTER_KERNEL(bitwise_or, GPU, BOOL, bitwise_or_wrapper);
    REGISTER_KERNEL(bitwise_or, GPU, U8, bitwise_or_wrapper);
    REGISTER_KERNEL(bitwise_or, GPU, I8, bitwise_or_wrapper);
    REGISTER_KERNEL(bitwise_or, GPU, I16, bitwise_or_wrapper);
    REGISTER_KERNEL(bitwise_or, GPU, I32, bitwise_or_wrapper);
    REGISTER_KERNEL(bitwise_or, GPU, I64, bitwise_or_wrapper);

    REGISTER_KERNEL(bitwise_xor, GPU, BOOL, bitwise_xor_wrapper);
    REGISTER_KERNEL(bitwise_xor, GPU, U8, bitwise_xor_wrapper);
    REGISTER_KERNEL(bitwise_xor, GPU, I8, bitwise_xor_wrapper);
    REGISTER_KERNEL(bitwise_xor, GPU, I16, bitwise_xor_wrapper);
    REGISTER_KERNEL(bitwise_xor, GPU, I32, bitwise_xor_wrapper);
    REGISTER_KERNEL(bitwise_xor, GPU, I64, bitwise_xor_wrapper);

    REGISTER_KERNEL(bitwise_left_shift, GPU, U8, left_shift_wrapper);
    REGISTER_KERNEL(bitwise_left_shift, GPU, I8, left_shift_wrapper);
    REGISTER_KERNEL(bitwise_left_shift, GPU, I16, left_shift_wrapper);
    REGISTER_KERNEL(bitwise_left_shift, GPU, I32, left_shift_wrapper);
    REGISTER_KERNEL(bitwise_left_shift, GPU, I64, left_shift_wrapper);

    REGISTER_KERNEL(bitwise_right_shift, GPU, U8, right_shift_wrapper);
    REGISTER_KERNEL(bitwise_right_shift, GPU, I8, right_shift_wrapper);
    REGISTER_KERNEL(bitwise_right_shift, GPU, I16, right_shift_wrapper);
    REGISTER_KERNEL(bitwise_right_shift, GPU, I32, right_shift_wrapper);
    REGISTER_KERNEL(bitwise_right_shift, GPU, I64, right_shift_wrapper);

} // namespace ins::gpu

REGISTER_MODULE(elementwise, GPU);