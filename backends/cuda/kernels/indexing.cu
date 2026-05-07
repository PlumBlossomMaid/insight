// backends/cuda/kernels/indexing.cu
#include "insight/plugin/op_registry.h"
#include "insight/core/array.h"
#include "insight/core/exception.h"
#include "insight/core/launch_config.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#ifdef INSIGHT_USE_THRUST
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/partition.h>
#include <thrust/device_ptr.h>
#endif

namespace ins::gpu {

    // ============================================================================
    // Custom complex struct for CUDA kernels (since ComplexPod may not be fully supported in device code)
    // ============================================================================

    template<typename T>
    struct ComplexPod {
        T r, i;
        __host__ __device__ ComplexPod() : r(T(0)), i(T(0)) {}
        __host__ __device__ ComplexPod(T real, T imag) : r(real), i(imag) {}
    };

	// ===========================================================================
	// Bool inverse mapper for partitioning (maps true to 1 if there's at least one false, otherwise maps to 0)
	// This is used to ensure that if there are any false values, they will be placed before true values in the partitioned output.
	// ===========================================================================

    struct BoolInverseMapper {
        bool has_false;
        __host__ __device__ int64_t operator()(bool val) const {
            return val ? (has_false ? 1 : 0) : 0;
        }
    };

    // ============================================================================
    // Device helper functions for complex numbers
    // ============================================================================

    __device__ inline float real(const ComplexPod<float>& z) {
        return reinterpret_cast<const float*>(&z)[0];
    }

    __device__ inline float imag(const ComplexPod<float>& z) {
        return reinterpret_cast<const float*>(&z)[1];
    }

    __device__ inline double real(const ComplexPod<double>& z) {
        return reinterpret_cast<const double*>(&z)[0];
    }

    __device__ inline double imag(const ComplexPod<double>& z) {
        return reinterpret_cast<const double*>(&z)[1];
    }

    __host__ __device__ inline float host_real(const ComplexPod<float>& z) {
        return reinterpret_cast<const float*>(&z)[0];
    }

    __host__ __device__ inline double host_real(const ComplexPod<double>& z) {
        return reinterpret_cast<const double*>(&z)[0];
    }

    // ============================================================================
    // Complex comparator for sort operations (compare real part only, aligns with NumPy)
    // ============================================================================

    struct ComplexRealComparator {
        template<typename T>
        __host__ __device__ bool operator()(const T& a, const T& b) const {
            return host_real(a) < host_real(b);
        }
    };

	// ============================================================================
	// Constant functor for filling arrays with a constant value (used in unique and partition)
	// ============================================================================

    struct ConstantOne {
        __host__ __device__ int64_t operator()(int64_t) const { return 1; }
    };

    // ============================================================================
    // Comparator (thrust functor)
    // ============================================================================

    template<typename T>
    struct ComplexDescendingComparator {
        __host__ __device__ bool operator()(const T& a, const T& b) const {
            return host_real(a) > host_real(b);
        }
    };

    // lexsort comparator functor
    template<typename T>
    struct LexSortComparator {
        const T* src;
        int64_t batch;
        int64_t nkeys;
        int64_t last_dim;

        __host__ __device__ bool operator()(int64_t a, int64_t b) const {
            for (int64_t k = 0; k < nkeys; ++k) {
                T va = src[(batch * nkeys + k) * last_dim + a];
                T vb = src[(batch * nkeys + k) * last_dim + b];
                if (va < vb) return true;
                if (va > vb) return false;
            }
            return false;
        }
    };

    // ============================================================================
    // Helper: compute strides from shape
    // ============================================================================

    __host__ __device__ inline void compute_strides(const int64_t* dims, int64_t* strides, int ndim) {
        if (ndim == 0) return;
        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
    }

    // ============================================================================
    // Helper: linear index from coordinates and strides
    // ============================================================================

    __device__ inline int64_t linear_index(const int64_t* coords, const int64_t* strides, int ndim) {
        int64_t idx = 0;
        for (int i = 0; i < ndim; ++i) {
            idx += coords[i] * strides[i];
        }
        return idx;
    }

    // ============================================================================
    // take kernel
    // ============================================================================

    template<typename T>
    __global__ void take_flat_kernel(
        T* dst, const T* src, const int64_t* idx,
        int64_t n, int64_t x_size
    ) {
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            int64_t pos = idx[i];
            if (pos < 0) pos += x_size;
            dst[i] = src[pos];
        }
    }

    template<typename T>
    __global__ void take_axis_kernel(
        T* dst, const T* src, const int64_t* idx,
        const int64_t* out_shape, const int64_t* x_shape,
        const int64_t* out_strides, const int64_t* x_strides,
        int ndim, int axis, int64_t total
    ) {
        int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
        if (linear >= total) return;

        int64_t coord[4] = { 0 };
        int64_t tmp = linear;
        for (int d = ndim - 1; d >= 0; --d) {
            coord[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }

        int64_t src_idx = 0;
        for (int d = 0; d < ndim; ++d) {
            if (d == axis) {
                int64_t pos = idx[coord[d]];
                int64_t dim_size = x_shape[d];
                if (pos < 0) pos += dim_size;
                src_idx += pos * x_strides[d];
            }
            else {
                src_idx += coord[d] * x_strides[d];
            }
        }
        dst[linear] = src[src_idx];
    }

    template<typename T>
    static void take_impl(const Array& out, const Array& x, const Array& indices,
        int orig_axis, bool has_axis) {
        int64_t n = out.numel();

        if (!has_axis) {
            LaunchConfig config(n);
            take_flat_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
                const_cast<T*>(out.data<T>()), x.data<T>(), indices.data<int64_t>(),
                n, x.numel()
                );
        }
        else {
            int ndim = x.shape().ndim();
            int axis = orig_axis;
            if (axis < 0) axis += ndim;

            int64_t h_out_shape[4] = { 1, 1, 1, 1 };
            int64_t h_x_shape[4] = { 1, 1, 1, 1 };
            int64_t h_out_strides[4] = { 0, 0, 0, 0 };
            int64_t h_x_strides[4] = { 0, 0, 0, 0 };

            for (int i = 0; i < ndim; ++i) {
                h_out_shape[i] = out.shape().dim(i);
                h_x_shape[i] = x.shape().dim(i);
            }
            compute_strides(h_out_shape, h_out_strides, ndim);
            compute_strides(h_x_shape, h_x_strides, ndim);

            int64_t* d_out_shape;
            int64_t* d_x_shape;
            int64_t* d_out_strides;
            int64_t* d_x_strides;
            cudaMalloc(&d_out_shape, 4 * sizeof(int64_t));
            cudaMalloc(&d_x_shape, 4 * sizeof(int64_t));
            cudaMalloc(&d_out_strides, 4 * sizeof(int64_t));
            cudaMalloc(&d_x_strides, 4 * sizeof(int64_t));
            cudaMemcpy(d_out_shape, h_out_shape, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_x_shape, h_x_shape, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_out_strides, h_out_strides, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_x_strides, h_x_strides, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);

            LaunchConfig config(n);
            take_axis_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
                const_cast<T*>(out.data<T>()), x.data<T>(), indices.data<int64_t>(),
                d_out_shape, d_x_shape, d_out_strides, d_x_strides,
                ndim, axis, n
                );

            cudaFree(d_out_shape);
            cudaFree(d_x_shape);
            cudaFree(d_out_strides);
            cudaFree(d_x_strides);
        }
    }

    static OpArgs take_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& x = std::any_cast<const Array&>(args[1]);
        const Array& indices = std::any_cast<const Array&>(args[2]);
        int orig_axis = std::any_cast<int>(args[3]);
        bool has_axis = std::any_cast<bool>(args[4]);

        DType dtype = x.dtype();
        switch (dtype) {
        case DType::F32: take_impl<float>(out, x, indices, orig_axis, has_axis); break;
        case DType::F64: take_impl<double>(out, x, indices, orig_axis, has_axis); break;
        case DType::I32: take_impl<int32_t>(out, x, indices, orig_axis, has_axis); break;
        case DType::I64: take_impl<int64_t>(out, x, indices, orig_axis, has_axis); break;
        case DType::U8:  take_impl<uint8_t>(out, x, indices, orig_axis, has_axis); break;
        case DType::BOOL:take_impl<bool>(out, x, indices, orig_axis, has_axis); break;
        case DType::C32: take_impl<ComplexPod<float>>(out, x, indices, orig_axis, has_axis); break;
        case DType::C64: take_impl<ComplexPod<double>>(out, x, indices, orig_axis, has_axis); break;
        default: INS_THROW("take: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(take, GPU, F32, take_wrapper);
    REGISTER_KERNEL(take, GPU, F64, take_wrapper);
    REGISTER_KERNEL(take, GPU, I32, take_wrapper);
    REGISTER_KERNEL(take, GPU, I64, take_wrapper);
    REGISTER_KERNEL(take, GPU, U8, take_wrapper);
    REGISTER_KERNEL(take, GPU, BOOL, take_wrapper);
    REGISTER_KERNEL(take, GPU, C32, take_wrapper);
    REGISTER_KERNEL(take, GPU, C64, take_wrapper);

    // ============================================================================
    // take_along_axis kernel
    // ============================================================================

    template<typename T>
    __global__ void take_along_axis_kernel(
        T* dst, const T* src, const int64_t* idx,
        const int64_t* out_shape, const int64_t* x_shape,
        const int64_t* idx_shape, const int64_t* out_strides,
        const int64_t* x_strides, const int64_t* idx_strides,
        int ndim, int axis, int64_t total
    ) {
        int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
        if (linear >= total) return;

        int64_t coord[4] = { 0 };
        int64_t tmp = linear;
        for (int d = ndim - 1; d >= 0; --d) {
            coord[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }

        int64_t idx_pos = 0;
        for (int d = 0; d < ndim; ++d) {
            int64_t idx_coord = (coord[d] < idx_shape[d]) ? coord[d] : 0;
            idx_pos += idx_coord * idx_strides[d];
        }

        int64_t pos = idx[idx_pos];
        int64_t dim_size = x_shape[axis];
        if (pos < 0) pos += dim_size;

        int64_t src_idx = 0;
        for (int d = 0; d < ndim; ++d) {
            if (d == axis) {
                src_idx += pos * x_strides[d];
            }
            else {
                src_idx += coord[d] * x_strides[d];
            }
        }
        dst[linear] = src[src_idx];
    }

    template<typename T>
    static void take_along_axis_impl(const Array& out, const Array& x, const Array& indices, int axis) {
        int ndim = x.shape().ndim();
        if (axis < 0) axis += ndim;
        int64_t total = out.numel();

        int64_t h_out_shape[4] = { 1, 1, 1, 1 };
        int64_t h_x_shape[4] = { 1, 1, 1, 1 };
        int64_t h_idx_shape[4] = { 1, 1, 1, 1 };
        int64_t h_out_strides[4];
        int64_t h_x_strides[4];
        int64_t h_idx_strides[4];

        for (int i = 0; i < ndim; ++i) {
            h_out_shape[i] = out.shape().dim(i);
            h_x_shape[i] = x.shape().dim(i);
            h_idx_shape[i] = indices.shape().dim(i);
        }
        compute_strides(h_out_shape, h_out_strides, ndim);
        compute_strides(h_x_shape, h_x_strides, ndim);
        compute_strides(h_idx_shape, h_idx_strides, ndim);

        int64_t* d_out_shape, * d_x_shape, * d_idx_shape;
        int64_t* d_out_strides, * d_x_strides, * d_idx_strides;
        cudaMalloc(&d_out_shape, 4 * sizeof(int64_t));
        cudaMalloc(&d_x_shape, 4 * sizeof(int64_t));
        cudaMalloc(&d_idx_shape, 4 * sizeof(int64_t));
        cudaMalloc(&d_out_strides, 4 * sizeof(int64_t));
        cudaMalloc(&d_x_strides, 4 * sizeof(int64_t));
        cudaMalloc(&d_idx_strides, 4 * sizeof(int64_t));
        cudaMemcpy(d_out_shape, h_out_shape, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_shape, h_x_shape, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_idx_shape, h_idx_shape, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_out_strides, h_out_strides, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_strides, h_x_strides, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_idx_strides, h_idx_strides, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);

        LaunchConfig config(total);
        take_along_axis_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            const_cast<T*>(out.data<T>()), x.data<T>(), indices.data<int64_t>(),
            d_out_shape, d_x_shape, d_idx_shape, d_out_strides, d_x_strides, d_idx_strides,
            ndim, axis, total
            );

        cudaFree(d_out_shape);
        cudaFree(d_x_shape);
        cudaFree(d_idx_shape);
        cudaFree(d_out_strides);
        cudaFree(d_x_strides);
        cudaFree(d_idx_strides);
    }

    static OpArgs take_along_axis_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& x = std::any_cast<const Array&>(args[1]);
        const Array& indices = std::any_cast<const Array&>(args[2]);
        int axis = std::any_cast<int>(args[3]);

        DType dtype = x.dtype();
        switch (dtype) {
        case DType::F32: take_along_axis_impl<float>(out, x, indices, axis); break;
        case DType::F64: take_along_axis_impl<double>(out, x, indices, axis); break;
        case DType::I32: take_along_axis_impl<int32_t>(out, x, indices, axis); break;
        case DType::I64: take_along_axis_impl<int64_t>(out, x, indices, axis); break;
        case DType::U8:  take_along_axis_impl<uint8_t>(out, x, indices, axis); break;
        case DType::BOOL:take_along_axis_impl<bool>(out, x, indices, axis); break;
        case DType::C32: take_along_axis_impl<ComplexPod<float>>(out, x, indices, axis); break;
        case DType::C64: take_along_axis_impl<ComplexPod<double>>(out, x, indices, axis); break;
        default: INS_THROW("take_along_axis: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(take_along_axis, GPU, F32, take_along_axis_wrapper);
    REGISTER_KERNEL(take_along_axis, GPU, F64, take_along_axis_wrapper);
    REGISTER_KERNEL(take_along_axis, GPU, I32, take_along_axis_wrapper);
    REGISTER_KERNEL(take_along_axis, GPU, I64, take_along_axis_wrapper);
    REGISTER_KERNEL(take_along_axis, GPU, U8, take_along_axis_wrapper);
    REGISTER_KERNEL(take_along_axis, GPU, BOOL, take_along_axis_wrapper);
    REGISTER_KERNEL(take_along_axis, GPU, C32, take_along_axis_wrapper);
    REGISTER_KERNEL(take_along_axis, GPU, C64, take_along_axis_wrapper);

    // ============================================================================
    // put kernel
    // ============================================================================

    template<typename T>
    __global__ void put_kernel(
        T* dst, const int64_t* idx, const T* val,
        int64_t n, int64_t out_size, int64_t val_size
    ) {
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            int64_t pos = idx[i];
            if (pos < 0) pos += out_size;
            dst[pos] = val[i % val_size];
        }
    }

    template<typename T>
    static void put_impl(const Array& out, const Array& indices, const Array& values) {
        int64_t n = indices.numel();
        LaunchConfig config(n);
        put_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            const_cast<T*>(out.data<T>()), indices.data<int64_t>(), values.data<T>(),
            n, out.numel(), values.numel()
            );
    }

    static OpArgs put_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& indices = std::any_cast<const Array&>(args[1]);
        const Array& values = std::any_cast<const Array&>(args[2]);

        DType dtype = out.dtype();
        switch (dtype) {
        case DType::F32: put_impl<float>(out, indices, values); break;
        case DType::F64: put_impl<double>(out, indices, values); break;
        case DType::I32: put_impl<int32_t>(out, indices, values); break;
        case DType::I64: put_impl<int64_t>(out, indices, values); break;
        case DType::U8:  put_impl<uint8_t>(out, indices, values); break;
        case DType::BOOL:put_impl<bool>(out, indices, values); break;
        case DType::C32: put_impl<ComplexPod<float>>(out, indices, values); break;
        case DType::C64: put_impl<ComplexPod<double>>(out, indices, values); break;
        default: INS_THROW("put: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(put, GPU, F32, put_wrapper);
    REGISTER_KERNEL(put, GPU, F64, put_wrapper);
    REGISTER_KERNEL(put, GPU, I32, put_wrapper);
    REGISTER_KERNEL(put, GPU, I64, put_wrapper);
    REGISTER_KERNEL(put, GPU, U8, put_wrapper);
    REGISTER_KERNEL(put, GPU, BOOL, put_wrapper);
    REGISTER_KERNEL(put, GPU, C32, put_wrapper);
    REGISTER_KERNEL(put, GPU, C64, put_wrapper);

    // ============================================================================
    // put_along_axis kernel
    // ============================================================================

    template<typename T>
    __global__ void put_along_axis_kernel(
        T* dst, const int64_t* idx, const T* val,
        const int64_t* out_shape, const int64_t* idx_shape,
        const int64_t* out_strides,
        int ndim, int axis, int64_t total, int64_t val_size
    ) {
        int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
        if (linear >= total) return;

        int64_t coord[4] = { 0 };
        int64_t tmp = linear;
        for (int d = ndim - 1; d >= 0; --d) {
            coord[d] = tmp % idx_shape[d];
            tmp /= idx_shape[d];
        }

        int64_t pos = idx[linear];
        if (pos < 0) pos += out_shape[axis];

        int64_t out_idx = 0;
        for (int d = 0; d < ndim; ++d) {
            if (d == axis) {
                out_idx += pos * out_strides[d];
            }
            else {
                out_idx += coord[d] * out_strides[d];
            }
        }
        dst[out_idx] = val[linear % val_size];
    }

    template<typename T>
    static void put_along_axis_impl(const Array& out, const Array& indices,
        const Array& values, int axis) {
        int ndim = out.shape().ndim();
        if (axis < 0) axis += ndim;
        int64_t total = indices.numel();

        int64_t h_out_shape[4] = { 1, 1, 1, 1 };
        int64_t h_idx_shape[4] = { 1, 1, 1, 1 };
        int64_t h_out_strides[4];

        for (int i = 0; i < ndim; ++i) {
            h_out_shape[i] = out.shape().dim(i);
            h_idx_shape[i] = indices.shape().dim(i);
        }
        compute_strides(h_out_shape, h_out_strides, ndim);

        int64_t* d_out_shape, * d_idx_shape, * d_out_strides;
        cudaMalloc(&d_out_shape, 4 * sizeof(int64_t));
        cudaMalloc(&d_idx_shape, 4 * sizeof(int64_t));
        cudaMalloc(&d_out_strides, 4 * sizeof(int64_t));
        cudaMemcpy(d_out_shape, h_out_shape, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_idx_shape, h_idx_shape, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_out_strides, h_out_strides, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);

        LaunchConfig config(total);
        put_along_axis_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            const_cast<T*>(out.data<T>()), indices.data<int64_t>(), values.data<T>(),
            d_out_shape, d_idx_shape, d_out_strides,
            ndim, axis, total, values.numel()
            );

        cudaFree(d_out_shape);
        cudaFree(d_idx_shape);
        cudaFree(d_out_strides);
    }

    static OpArgs put_along_axis_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& indices = std::any_cast<const Array&>(args[1]);
        const Array& values = std::any_cast<const Array&>(args[2]);
        int axis = std::any_cast<int>(args[3]);

        DType dtype = out.dtype();
        switch (dtype) {
        case DType::F32: put_along_axis_impl<float>(out, indices, values, axis); break;
        case DType::F64: put_along_axis_impl<double>(out, indices, values, axis); break;
        case DType::I32: put_along_axis_impl<int32_t>(out, indices, values, axis); break;
        case DType::I64: put_along_axis_impl<int64_t>(out, indices, values, axis); break;
        case DType::U8:  put_along_axis_impl<uint8_t>(out, indices, values, axis); break;
        case DType::BOOL:put_along_axis_impl<bool>(out, indices, values, axis); break;
        case DType::C32: put_along_axis_impl<ComplexPod<float>>(out, indices, values, axis); break;
        case DType::C64: put_along_axis_impl<ComplexPod<double>>(out, indices, values, axis); break;
        default: INS_THROW("put_along_axis: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(put_along_axis, GPU, F32, put_along_axis_wrapper);
    REGISTER_KERNEL(put_along_axis, GPU, F64, put_along_axis_wrapper);
    REGISTER_KERNEL(put_along_axis, GPU, I32, put_along_axis_wrapper);
    REGISTER_KERNEL(put_along_axis, GPU, I64, put_along_axis_wrapper);
    REGISTER_KERNEL(put_along_axis, GPU, U8, put_along_axis_wrapper);
    REGISTER_KERNEL(put_along_axis, GPU, BOOL, put_along_axis_wrapper);
    REGISTER_KERNEL(put_along_axis, GPU, C32, put_along_axis_wrapper);
    REGISTER_KERNEL(put_along_axis, GPU, C64, put_along_axis_wrapper);

    // ============================================================================
    // scatter_reduce kernel
    // ============================================================================

    template<typename T>
    __global__ void scatter_reduce_kernel(
        T* dst, const int64_t* idx, const T* val,
        const int64_t* out_shape, const int64_t* idx_shape,
        const int64_t* out_strides,
        int ndim, int dim, int64_t total, int64_t val_size, int reduce_mode
    ) {
        int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
        if (linear >= total) return;

        int64_t coord[4] = { 0 };
        int64_t tmp = linear;
        for (int d = ndim - 1; d >= 0; --d) {
            coord[d] = tmp % idx_shape[d];
            tmp /= idx_shape[d];
        }

        int64_t pos = idx[linear];
        if (pos < 0) pos += out_shape[dim];

        int64_t out_idx = 0;
        for (int d = 0; d < ndim; ++d) {
            if (d == dim) {
                out_idx += pos * out_strides[d];
            }
            else {
                out_idx += coord[d] * out_strides[d];
            }
        }

        T val_cur = val[linear % val_size];
        switch (reduce_mode) {
        case 0: // replace
            dst[out_idx] = val_cur;
            break;
        case 1: // add
            if constexpr (std::is_same_v<T, float>) {
                atomicAdd(&dst[out_idx], val_cur);
            }
            else if constexpr (std::is_same_v<T, double>) {
                unsigned long long* addr = reinterpret_cast<unsigned long long*>(&dst[out_idx]);
                unsigned long long old = *addr;
                unsigned long long assumed;
                do {
                    assumed = old;
                    double new_val = *reinterpret_cast<const double*>(&assumed) + val_cur;
                    old = atomicCAS(addr, assumed, *reinterpret_cast<unsigned long long*>(&new_val));
                } while (assumed != old);
            }
            else if constexpr (std::is_same_v<T, int32_t>) {
                atomicAdd(reinterpret_cast<int*>(&dst[out_idx]), static_cast<int>(val_cur));
            }
            else if constexpr (std::is_same_v<T, int64_t>) {
                unsigned long long* addr = reinterpret_cast<unsigned long long*>(&dst[out_idx]);
                unsigned long long old = *addr;
                unsigned long long assumed;
                do {
                    assumed = old;
                    unsigned long long new_val = assumed + static_cast<unsigned long long>(val_cur);
                    old = atomicCAS(addr, assumed, new_val);
                } while (assumed != old);
            }
            else if constexpr (std::is_same_v<T, uint8_t>) {
                unsigned int* ptr = reinterpret_cast<unsigned int*>(&dst[out_idx]);
                atomicAdd(ptr, static_cast<unsigned int>(val_cur));
            }
            else if constexpr (std::is_same_v<T, bool>) {
                dst[out_idx] = true;
            }
            else if constexpr (std::is_same_v<T, ComplexPod<float>>) {
                float* ptr = reinterpret_cast<float*>(&dst[out_idx]);
                const float* vp = reinterpret_cast<const float*>(&val_cur);
                atomicAdd(&ptr[0], vp[0]);
                atomicAdd(&ptr[1], vp[1]);
            }
            else if constexpr (std::is_same_v<T, ComplexPod<double>>) {
                unsigned long long* ptr = reinterpret_cast<unsigned long long*>(&dst[out_idx]);
                const double* vp = reinterpret_cast<const double*>(&val_cur);

                // real part
                unsigned long long old = ptr[0];
                unsigned long long assumed;
                do {
                    assumed = old;
                    double new_val = *reinterpret_cast<const double*>(&assumed) + vp[0];
                    old = atomicCAS(&ptr[0], assumed, *reinterpret_cast<unsigned long long*>(&new_val));
                } while (assumed != old);

                // imag part
                old = ptr[1];
                do {
                    assumed = old;
                    double new_val = *reinterpret_cast<const double*>(&assumed) + vp[1];
                    old = atomicCAS(&ptr[1], assumed, *reinterpret_cast<unsigned long long*>(&new_val));
                } while (assumed != old);
            }
            else {
                dst[out_idx] = val_cur;
            }
            break;
        default:
            dst[out_idx] = val_cur;
            break;
        }
    }

    template<typename T>
    static void scatter_reduce_impl(const Array& out, const Array& indices,
        const Array& src, int dim, const std::string& reduce) {
        int ndim = out.shape().ndim();
        if (dim < 0) dim += ndim;
        int64_t total = indices.numel();

        int64_t h_out_shape[4] = { 1, 1, 1, 1 };
        int64_t h_idx_shape[4] = { 1, 1, 1, 1 };
        int64_t h_out_strides[4];

        for (int i = 0; i < ndim; ++i) {
            h_out_shape[i] = out.shape().dim(i);
            h_idx_shape[i] = indices.shape().dim(i);
        }
        compute_strides(h_out_shape, h_out_strides, ndim);

        int64_t* d_out_shape, * d_idx_shape, * d_out_strides;
        cudaMalloc(&d_out_shape, 4 * sizeof(int64_t));
        cudaMalloc(&d_idx_shape, 4 * sizeof(int64_t));
        cudaMalloc(&d_out_strides, 4 * sizeof(int64_t));
        cudaMemcpy(d_out_shape, h_out_shape, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_idx_shape, h_idx_shape, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_out_strides, h_out_strides, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);

        int reduce_mode = 0;
        if (reduce == "replace") reduce_mode = 0;
        else if (reduce == "add") reduce_mode = 1;
        else reduce_mode = 0;

        LaunchConfig config(total);
        scatter_reduce_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            const_cast<T*>(out.data<T>()), indices.data<int64_t>(), src.data<T>(),
            d_out_shape, d_idx_shape, d_out_strides,
            ndim, dim, total, src.numel(), reduce_mode
            );

        cudaFree(d_out_shape);
        cudaFree(d_idx_shape);
        cudaFree(d_out_strides);
    }

    static OpArgs scatter_reduce_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& indices = std::any_cast<const Array&>(args[1]);
        const Array& src = std::any_cast<const Array&>(args[2]);
        int dim = std::any_cast<int>(args[3]);
        std::string reduce = std::any_cast<std::string>(args[4]);

        DType dtype = out.dtype();
        switch (dtype) {
        case DType::F32: scatter_reduce_impl<float>(out, indices, src, dim, reduce); break;
        case DType::F64: scatter_reduce_impl<double>(out, indices, src, dim, reduce); break;
        case DType::I32: scatter_reduce_impl<int32_t>(out, indices, src, dim, reduce); break;
        case DType::I64: scatter_reduce_impl<int64_t>(out, indices, src, dim, reduce); break;
        case DType::U8:  scatter_reduce_impl<uint8_t>(out, indices, src, dim, reduce); break;
        case DType::BOOL:scatter_reduce_impl<bool>(out, indices, src, dim, reduce); break;
        case DType::C32: scatter_reduce_impl<ComplexPod<float>>(out, indices, src, dim, reduce); break;
        case DType::C64: scatter_reduce_impl<ComplexPod<double>>(out, indices, src, dim, reduce); break;
        default: INS_THROW("scatter_reduce: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(scatter_reduce, GPU, F32, scatter_reduce_wrapper);
    REGISTER_KERNEL(scatter_reduce, GPU, F64, scatter_reduce_wrapper);
    REGISTER_KERNEL(scatter_reduce, GPU, I32, scatter_reduce_wrapper);
    REGISTER_KERNEL(scatter_reduce, GPU, I64, scatter_reduce_wrapper);
    REGISTER_KERNEL(scatter_reduce, GPU, U8, scatter_reduce_wrapper);
    REGISTER_KERNEL(scatter_reduce, GPU, BOOL, scatter_reduce_wrapper);
    REGISTER_KERNEL(scatter_reduce, GPU, C32, scatter_reduce_wrapper);
    REGISTER_KERNEL(scatter_reduce, GPU, C64, scatter_reduce_wrapper);

    // ============================================================================
    // masked_select kernel
    // ============================================================================

    template<typename T>
    __global__ void masked_select_flat_kernel(
        T* dst, const T* src, const bool* mask, int64_t total
    ) {
        int64_t out_idx = 0;
        for (int64_t i = 0; i < total; ++i) {
            if (mask[i]) {
                dst[out_idx++] = src[i];
            }
        }
    }

    template<typename T>
    static void masked_select_impl(const Array& out, const Array& x, const Array& mask) {
        int64_t total = x.numel();
        LaunchConfig config(1);
        masked_select_flat_kernel<T> << <1, 1 >> > (
            const_cast<T*>(out.data<T>()), x.data<T>(), mask.data<bool>(), total
            );
    }

    static OpArgs masked_select_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& x = std::any_cast<const Array&>(args[1]);
        const Array& mask = std::any_cast<const Array&>(args[2]);

        DType dtype = x.dtype();
        switch (dtype) {
        case DType::F32: masked_select_impl<float>(out, x, mask); break;
        case DType::F64: masked_select_impl<double>(out, x, mask); break;
        case DType::I32: masked_select_impl<int32_t>(out, x, mask); break;
        case DType::I64: masked_select_impl<int64_t>(out, x, mask); break;
        case DType::U8:  masked_select_impl<uint8_t>(out, x, mask); break;
        case DType::BOOL:masked_select_impl<bool>(out, x, mask); break;
        case DType::C32: masked_select_impl<ComplexPod<float>>(out, x, mask); break;
        case DType::C64: masked_select_impl<ComplexPod<double>>(out, x, mask); break;
        default: INS_THROW("masked_select: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(masked_select, GPU, F32, masked_select_wrapper);
    REGISTER_KERNEL(masked_select, GPU, F64, masked_select_wrapper);
    REGISTER_KERNEL(masked_select, GPU, I32, masked_select_wrapper);
    REGISTER_KERNEL(masked_select, GPU, I64, masked_select_wrapper);
    REGISTER_KERNEL(masked_select, GPU, U8, masked_select_wrapper);
    REGISTER_KERNEL(masked_select, GPU, BOOL, masked_select_wrapper);
    REGISTER_KERNEL(masked_select, GPU, C32, masked_select_wrapper);
    REGISTER_KERNEL(masked_select, GPU, C64, masked_select_wrapper);

    // ============================================================================
    // compress kernel
    // ============================================================================

    template<typename T>
    __global__ void compress_kernel(
        T* dst, const T* src, const bool* cond,
        const int64_t* out_shape, const int64_t* x_shape,
        const int64_t* out_strides, const int64_t* x_strides,
        int ndim, int axis, int64_t total
    ) {
        int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
        if (linear >= total) return;

        int64_t coord[4] = { 0 };
        int64_t tmp = linear;
        for (int d = ndim - 1; d >= 0; --d) {
            coord[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }

        int64_t axis_dim = x_shape[axis];
        int64_t cond_idx = 0;
        int64_t orig_axis_idx = 0;
        for (int64_t i = 0; i < axis_dim; ++i) {
            if (cond[i]) {
                if (cond_idx == coord[axis]) {
                    orig_axis_idx = i;
                    break;
                }
                cond_idx++;
            }
        }

        int64_t src_idx = 0;
        for (int d = 0; d < ndim; ++d) {
            if (d == axis) {
                src_idx += orig_axis_idx * x_strides[d];
            }
            else {
                src_idx += coord[d] * x_strides[d];
            }
        }
        dst[linear] = src[src_idx];
    }

    template<typename T>
    static void compress_impl(const Array& out, const Array& x,
        const Array& condition, int axis) {
        int ndim = x.shape().ndim();
        if (axis < 0) axis += ndim;
        int64_t total = out.numel();

        int64_t h_out_shape[4] = { 1, 1, 1, 1 };
        int64_t h_x_shape[4] = { 1, 1, 1, 1 };
        int64_t h_out_strides[4];
        int64_t h_x_strides[4];

        for (int i = 0; i < ndim; ++i) {
            h_out_shape[i] = out.shape().dim(i);
            h_x_shape[i] = x.shape().dim(i);
        }
        compute_strides(h_out_shape, h_out_strides, ndim);
        compute_strides(h_x_shape, h_x_strides, ndim);

        int64_t* d_out_shape, * d_x_shape, * d_out_strides, * d_x_strides;
        cudaMalloc(&d_out_shape, 4 * sizeof(int64_t));
        cudaMalloc(&d_x_shape, 4 * sizeof(int64_t));
        cudaMalloc(&d_out_strides, 4 * sizeof(int64_t));
        cudaMalloc(&d_x_strides, 4 * sizeof(int64_t));
        cudaMemcpy(d_out_shape, h_out_shape, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_shape, h_x_shape, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_out_strides, h_out_strides, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_strides, h_x_strides, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);

        LaunchConfig config(total);
        compress_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            const_cast<T*>(out.data<T>()), x.data<T>(), condition.data<bool>(),
            d_out_shape, d_x_shape, d_out_strides, d_x_strides,
            ndim, axis, total
            );

        cudaFree(d_out_shape);
        cudaFree(d_x_shape);
        cudaFree(d_out_strides);
        cudaFree(d_x_strides);
    }

    static OpArgs compress_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& x = std::any_cast<const Array&>(args[1]);
        const Array& condition = std::any_cast<const Array&>(args[2]);
        int axis = std::any_cast<int>(args[3]);

        DType dtype = x.dtype();
        switch (dtype) {
        case DType::F32: compress_impl<float>(out, x, condition, axis); break;
        case DType::F64: compress_impl<double>(out, x, condition, axis); break;
        case DType::I32: compress_impl<int32_t>(out, x, condition, axis); break;
        case DType::I64: compress_impl<int64_t>(out, x, condition, axis); break;
        case DType::U8:  compress_impl<uint8_t>(out, x, condition, axis); break;
        case DType::BOOL:compress_impl<bool>(out, x, condition, axis); break;
        case DType::C32: compress_impl<ComplexPod<float>>(out, x, condition, axis); break;
        case DType::C64: compress_impl<ComplexPod<double>>(out, x, condition, axis); break;
        default: INS_THROW("compress: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(compress, GPU, F32, compress_wrapper);
    REGISTER_KERNEL(compress, GPU, F64, compress_wrapper);
    REGISTER_KERNEL(compress, GPU, I32, compress_wrapper);
    REGISTER_KERNEL(compress, GPU, I64, compress_wrapper);
    REGISTER_KERNEL(compress, GPU, U8, compress_wrapper);
    REGISTER_KERNEL(compress, GPU, BOOL, compress_wrapper);
    REGISTER_KERNEL(compress, GPU, C32, compress_wrapper);
    REGISTER_KERNEL(compress, GPU, C64, compress_wrapper);

    // ============================================================================
    // where kernel
    // ============================================================================

    template<typename T>
    __global__ void where_kernel(
        T* dst, const bool* cond, const T* x, const T* y, int64_t n
    ) {
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            dst[i] = cond[i] ? x[i] : y[i];
        }
    }

    template<typename T>
    static void where_impl(const Array& out, const Array& condition,
        const Array& x, const Array& y) {
        int64_t n = out.numel();
        LaunchConfig config(n);
        where_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            const_cast<T*>(out.data<T>()), condition.data<bool>(),
            x.data<T>(), y.data<T>(), n
            );
    }

    static OpArgs where_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& condition = std::any_cast<const Array&>(args[1]);
        const Array& x = std::any_cast<const Array&>(args[2]);
        const Array& y = std::any_cast<const Array&>(args[3]);

        DType dtype = x.dtype();
        switch (dtype) {
        case DType::F32: where_impl<float>(out, condition, x, y); break;
        case DType::F64: where_impl<double>(out, condition, x, y); break;
        case DType::I32: where_impl<int32_t>(out, condition, x, y); break;
        case DType::I64: where_impl<int64_t>(out, condition, x, y); break;
        case DType::U8:  where_impl<uint8_t>(out, condition, x, y); break;
        case DType::BOOL:where_impl<bool>(out, condition, x, y); break;
        case DType::C32: where_impl<ComplexPod<float>>(out, condition, x, y); break;
        case DType::C64: where_impl<ComplexPod<double>>(out, condition, x, y); break;
        default: INS_THROW("where: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(where, GPU, F32, where_wrapper);
    REGISTER_KERNEL(where, GPU, F64, where_wrapper);
    REGISTER_KERNEL(where, GPU, I32, where_wrapper);
    REGISTER_KERNEL(where, GPU, I64, where_wrapper);
    REGISTER_KERNEL(where, GPU, U8, where_wrapper);
    REGISTER_KERNEL(where, GPU, BOOL, where_wrapper);
    REGISTER_KERNEL(where, GPU, C32, where_wrapper);
    REGISTER_KERNEL(where, GPU, C64, where_wrapper);

    // ============================================================================
    // nonzero kernel
    // ============================================================================

    template<typename T>
    __global__ void nonzero_count_kernel(
        const T* data, int64_t* block_counts, int64_t total
    ) {
        extern __shared__ int64_t shared_count[];
        int64_t tid = threadIdx.x;
        int64_t idx = blockIdx.x * blockDim.x + tid;

        shared_count[tid] = 0;
        __syncthreads();

        if (idx < total) {
            bool is_nonzero = false;
            if constexpr (std::is_same_v<T, ComplexPod<float>> ||
                std::is_same_v<T, ComplexPod<double>>) {
                is_nonzero = (real(data[idx]) != 0) || (imag(data[idx]) != 0);
            }
            else {
                is_nonzero = (data[idx] != T(0));
            }
            if (is_nonzero) {
                shared_count[tid] = 1;
            }
        }
        __syncthreads();

        // Block-level reduction
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_count[tid] += shared_count[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            block_counts[blockIdx.x] = shared_count[0];
        }
    }

    template<typename T>
    __global__ void nonzero_compact_and_fill_kernel(
        int64_t* result, const T* data,
        const int64_t* block_offsets, const int64_t* shape,
        int ndim, int64_t total, int64_t nz_count
    ) {
        extern __shared__ int64_t shared_scan[];
        int64_t tid = threadIdx.x;
        int64_t idx = blockIdx.x * blockDim.x + tid;
        int64_t block_id = blockIdx.x;

        int64_t has_value = 0;
        int64_t global_offset = 0;

        if (idx < total) {
            bool is_nonzero = false;
            if constexpr (std::is_same_v<T, ComplexPod<float>> ||
                std::is_same_v<T, ComplexPod<double>>) {
                is_nonzero = (real(data[idx]) != 0) || (imag(data[idx]) != 0);
            }
            else {
                is_nonzero = (data[idx] != T(0));
            }
            has_value = is_nonzero ? 1 : 0;
        }

        shared_scan[tid] = has_value;
        __syncthreads();

        // Block-level inclusive scan
        for (int stride = 1; stride < blockDim.x; stride <<= 1) {
            int64_t val = 0;
            if (tid >= stride) {
                val = shared_scan[tid - stride];
            }
            __syncthreads();
            shared_scan[tid] += val;
            __syncthreads();
        }

        if (has_value) {
            int64_t out_idx = block_offsets[block_id] + shared_scan[tid] - 1;
            int64_t tmp = idx;
            for (int d = ndim - 1; d >= 0; --d) {
                result[d * nz_count + out_idx] = tmp % shape[d];
                tmp /= shape[d];
            }
        }
    }

    template<typename T>
    static void nonzero_impl(Array& result, const Array& x) {
        int64_t total = x.numel();
        int ndim = x.shape().ndim();

        LaunchConfig config(total);
        int64_t num_blocks = config.blocks.x;

        // Allocate block counts
        int64_t* d_block_counts, * d_block_offsets;
        cudaMalloc(&d_block_counts, num_blocks * sizeof(int64_t));
        cudaMalloc(&d_block_offsets, num_blocks * sizeof(int64_t));

        // Step 1: Count nonzeros per block
        size_t shared_mem_size = config.threads.x * sizeof(int64_t);
        nonzero_count_kernel<T> << <num_blocks, config.threads, shared_mem_size, config.stream >> > (
            x.data<T>(), d_block_counts, total
            );

        // Step 2: Compute offsets on host
        std::vector<int64_t> h_block_counts(num_blocks);
        cudaMemcpy(h_block_counts.data(), d_block_counts, num_blocks * sizeof(int64_t), cudaMemcpyDeviceToHost);

        std::vector<int64_t> h_block_offsets(num_blocks);
        int64_t nz_count = 0;
        for (int64_t i = 0; i < num_blocks; ++i) {
            h_block_offsets[i] = nz_count;
            nz_count += h_block_counts[i];
        }

        if (nz_count == 0) {
            cudaFree(d_block_counts);
            cudaFree(d_block_offsets);
            return;
        }

        cudaMemcpy(d_block_offsets, h_block_offsets.data(), num_blocks * sizeof(int64_t), cudaMemcpyHostToDevice);

        // Step 3: Compact and fill multidimensional indices
        int64_t h_shape[4] = { 1, 1, 1, 1 };
        for (int i = 0; i < ndim; ++i) {
            h_shape[i] = x.shape().dim(i);
        }

        int64_t* d_shape;
        cudaMalloc(&d_shape, 4 * sizeof(int64_t));
        cudaMemcpy(d_shape, h_shape, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);

        nonzero_compact_and_fill_kernel<T> << <num_blocks, config.threads, shared_mem_size * 2, config.stream >> > (
            result.data<int64_t>(), x.data<T>(), d_block_offsets, d_shape,
            ndim, total, nz_count
            );

        cudaFree(d_block_counts);
        cudaFree(d_block_offsets);
        cudaFree(d_shape);
    }

    static OpArgs nonzero_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);

        int64_t total = x.numel();
        int ndim = x.shape().ndim();
        int64_t nz_count = 0;

        // Count nonzeros using GPU kernel
        LaunchConfig config(total);
        int64_t num_blocks = config.blocks.x;

        int64_t* d_block_counts;
        cudaMalloc(&d_block_counts, num_blocks * sizeof(int64_t));

        DType dtype = x.dtype();
        switch (dtype) {
        case DType::F32: {
            nonzero_count_kernel<float> << <num_blocks, config.threads, config.threads.x * sizeof(int64_t) >> > (
                x.data<float>(), d_block_counts, total
                );
            break;
        }
        case DType::F64: {
            nonzero_count_kernel<double> << <num_blocks, config.threads, config.threads.x * sizeof(int64_t) >> > (
                x.data<double>(), d_block_counts, total
                );
            break;
        }
        case DType::I32: {
            nonzero_count_kernel<int32_t> << <num_blocks, config.threads, config.threads.x * sizeof(int64_t) >> > (
                x.data<int32_t>(), d_block_counts, total
                );
            break;
        }
        case DType::I64: {
            nonzero_count_kernel<int64_t> << <num_blocks, config.threads, config.threads.x * sizeof(int64_t) >> > (
                x.data<int64_t>(), d_block_counts, total
                );
            break;
        }
        case DType::U8: {
            nonzero_count_kernel<uint8_t> << <num_blocks, config.threads, config.threads.x * sizeof(int64_t) >> > (
                x.data<uint8_t>(), d_block_counts, total
                );
            break;
        }
        case DType::BOOL: {
            nonzero_count_kernel<bool> << <num_blocks, config.threads, config.threads.x * sizeof(int64_t) >> > (
                x.data<bool>(), d_block_counts, total
                );
            break;
        }
        case DType::C32: {
            nonzero_count_kernel<ComplexPod<float>> << <num_blocks, config.threads, config.threads.x * sizeof(int64_t) >> > (
                x.data<ComplexPod<float>>(), d_block_counts, total
                );
            break;
        }
        case DType::C64: {
            nonzero_count_kernel<ComplexPod<double>> << <num_blocks, config.threads, config.threads.x * sizeof(int64_t) >> > (
                x.data<ComplexPod<double>>(), d_block_counts, total
                );
            break;
        }
        default: {
            cudaFree(d_block_counts);
            INS_THROW("nonzero: unsupported dtype");
        }
        }

        // Read back total count
        std::vector<int64_t> h_block_counts(num_blocks);
        cudaMemcpy(h_block_counts.data(), d_block_counts, num_blocks * sizeof(int64_t), cudaMemcpyDeviceToHost);
        for (int64_t i = 0; i < num_blocks; ++i) {
            nz_count += h_block_counts[i];
        }
        cudaFree(d_block_counts);

        if (nz_count == 0) {
            return { Array(Shape({ ndim, 0 }), DType::I64, x.place()) };
        }

        // Allocate and fill result
        Array result(Shape({ ndim, nz_count }), DType::I64, x.place());

        switch (dtype) {
        case DType::F32: nonzero_impl<float>(result, x); break;
        case DType::F64: nonzero_impl<double>(result, x); break;
        case DType::I32: nonzero_impl<int32_t>(result, x); break;
        case DType::I64: nonzero_impl<int64_t>(result, x); break;
        case DType::U8:  nonzero_impl<uint8_t>(result, x); break;
        case DType::BOOL:nonzero_impl<bool>(result, x); break;
        case DType::C32: nonzero_impl<ComplexPod<float>>(result, x); break;
        case DType::C64: nonzero_impl<ComplexPod<double>>(result, x); break;
        default: INS_THROW("nonzero: unsupported dtype");
        }

        return { result };
    }

    REGISTER_KERNEL(nonzero, GPU, F32, nonzero_wrapper);
    REGISTER_KERNEL(nonzero, GPU, F64, nonzero_wrapper);
    REGISTER_KERNEL(nonzero, GPU, I32, nonzero_wrapper);
    REGISTER_KERNEL(nonzero, GPU, I64, nonzero_wrapper);
    REGISTER_KERNEL(nonzero, GPU, U8, nonzero_wrapper);
    REGISTER_KERNEL(nonzero, GPU, BOOL, nonzero_wrapper);
    REGISTER_KERNEL(nonzero, GPU, C32, nonzero_wrapper);
    REGISTER_KERNEL(nonzero, GPU, C64, nonzero_wrapper);

    // ============================================================================
    // indices kernel
    // ============================================================================

    __global__ void indices_kernel(
        int64_t* dst, const int64_t* dims, const int64_t* strides,
        int ndim, int64_t total
    ) {
        int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
        if (linear >= total) return;

        int64_t tmp = linear;
        for (int d = 0; d < ndim; ++d) {
            int64_t coord = tmp % dims[d];
            tmp /= dims[d];
            dst[d * total + linear] = coord;
        }
    }

    static void indices_impl(const Array& out, const Shape& shape) {
        int ndim = shape.ndim();
        int64_t total = shape.numel();

        int64_t h_dims[4] = { 1, 1, 1, 1 };
        int64_t h_strides[4] = { 0, 0, 0, 0 };
        for (int i = 0; i < ndim; ++i) {
            h_dims[i] = shape.dim(i);
        }
        compute_strides(h_dims, h_strides, ndim);

        int64_t* d_dims, * d_strides;
        cudaMalloc(&d_dims, 4 * sizeof(int64_t));
        cudaMalloc(&d_strides, 4 * sizeof(int64_t));
        cudaMemcpy(d_dims, h_dims, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_strides, h_strides, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);

        LaunchConfig config(total);
        indices_kernel << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            const_cast<int64_t*>(out.data<int64_t>()), d_dims, d_strides, ndim, total
            );

        cudaFree(d_dims);
        cudaFree(d_strides);
    }

    static OpArgs indices_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Shape& shape = std::any_cast<const Shape&>(args[1]);

        indices_impl(out, shape);
        return { out };
    }

    REGISTER_KERNEL(indices, GPU, I64, indices_wrapper);

#ifdef INSIGHT_USE_THRUST

    // ============================================================================
    // argsort (thrust path)
    // ============================================================================

    template<typename T>
    static void argsort_impl(const Array& out, const Array& x, bool descending) {
        int64_t total = x.numel();
        const Shape& shape = x.shape();
        int ndim = shape.ndim();
        int64_t last_dim = shape.dim(ndim - 1);
        int64_t batch_size = total / last_dim;

        int64_t* dst = const_cast<int64_t*>(out.data<int64_t>());

        // Allocate work buffer and copy input to avoid modifying original
        T* work;
        cudaMalloc(&work, total * sizeof(T));
        cudaMemcpy(work, x.data<T>(), total * sizeof(T), cudaMemcpyDeviceToDevice);

        for (int64_t batch = 0; batch < batch_size; ++batch) {
            thrust::device_ptr<int64_t> idx_ptr(dst + batch * last_dim);
            thrust::sequence(idx_ptr, idx_ptr + last_dim);

            if (descending) {
                thrust::sort_by_key(
                    thrust::device_ptr<T>(work + batch * last_dim),
                    thrust::device_ptr<T>(work + batch * last_dim + last_dim),
                    idx_ptr,
                    thrust::greater<T>()
                );
            }
            else {
                thrust::sort_by_key(
                    thrust::device_ptr<T>(work + batch * last_dim),
                    thrust::device_ptr<T>(work + batch * last_dim + last_dim),
                    idx_ptr
                );
            }
        }

        cudaFree(work);
    }

    template<typename T>
    static void argsort_complex_impl(const Array& out, const Array& x, bool descending) {
        int64_t total = x.numel();
        const Shape& shape = x.shape();
        int ndim = shape.ndim();
        int64_t last_dim = shape.dim(ndim - 1);
        int64_t batch_size = total / last_dim;

        int64_t* dst = const_cast<int64_t*>(out.data<int64_t>());

        // Allocate work buffer and copy input to avoid modifying original
        T* work;
        cudaMalloc(&work, total * sizeof(T));
        cudaMemcpy(work, x.data<T>(), total * sizeof(T), cudaMemcpyDeviceToDevice);

        for (int64_t batch = 0; batch < batch_size; ++batch) {
            thrust::device_ptr<int64_t> idx_ptr(dst + batch * last_dim);
            thrust::sequence(idx_ptr, idx_ptr + last_dim);

            if (descending) {
                thrust::sort_by_key(
                    thrust::device_ptr<T>(work + batch * last_dim),
                    thrust::device_ptr<T>(work + batch * last_dim + last_dim),
                    idx_ptr,
                    ComplexDescendingComparator<T>()
                );
            }
            else {
                thrust::sort_by_key(
                    thrust::device_ptr<T>(work + batch * last_dim),
                    thrust::device_ptr<T>(work + batch * last_dim + last_dim),
                    idx_ptr,
                    ComplexRealComparator()
                );
            }
        }

        cudaFree(work);
    }

    static OpArgs argsort_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& x = std::any_cast<const Array&>(args[1]);
        bool descending = std::any_cast<bool>(args[2]);

        DType dtype = x.dtype();
        switch (dtype) {
        case DType::F32: argsort_impl<float>(out, x, descending); break;
        case DType::F64: argsort_impl<double>(out, x, descending); break;
        case DType::I32: argsort_impl<int32_t>(out, x, descending); break;
        case DType::I64: argsort_impl<int64_t>(out, x, descending); break;
        case DType::U8:  argsort_impl<uint8_t>(out, x, descending); break;
        case DType::BOOL:argsort_impl<bool>(out, x, descending); break;
        case DType::C32: argsort_complex_impl<ComplexPod<float>>(out, x, descending); break;
        case DType::C64: argsort_complex_impl<ComplexPod<double>>(out, x, descending); break;
        default: INS_THROW("argsort: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(argsort, GPU, F32, argsort_wrapper);
    REGISTER_KERNEL(argsort, GPU, F64, argsort_wrapper);
    REGISTER_KERNEL(argsort, GPU, I32, argsort_wrapper);
    REGISTER_KERNEL(argsort, GPU, I64, argsort_wrapper);
    REGISTER_KERNEL(argsort, GPU, U8, argsort_wrapper);
    REGISTER_KERNEL(argsort, GPU, BOOL, argsort_wrapper);
    REGISTER_KERNEL(argsort, GPU, C32, argsort_wrapper);
    REGISTER_KERNEL(argsort, GPU, C64, argsort_wrapper);

    // ============================================================================
    // topk (thrust path)
    // ============================================================================

    template<typename T>
    __global__ void topk_gather_kernel(
        T* dst, const T* src, int64_t batch_size, int64_t last_dim, int64_t k
    ) {
        int64_t batch = blockIdx.x * blockDim.x + threadIdx.x;
        if (batch >= batch_size) return;

        for (int64_t i = 0; i < k; ++i) {
            dst[batch * k + i] = src[batch * last_dim + i];
        }
    }

    template<typename T>
    static void topk_values_impl(const Array& out, const Array& x, int64_t k, bool largest) {
        int64_t total = x.numel();
        const Shape& shape = x.shape();
        int ndim = shape.ndim();
        int64_t last_dim = shape.dim(ndim - 1);
        int64_t batch_size = total / last_dim;

        T* dst = const_cast<T*>(out.data<T>());

        // Allocate work buffer and copy input to avoid modifying original
        T* work;
        cudaMalloc(&work, total * sizeof(T));
        cudaMemcpy(work, x.data<T>(), total * sizeof(T), cudaMemcpyDeviceToDevice);

        for (int64_t batch = 0; batch < batch_size; ++batch) {
            T* batch_ptr = work + batch * last_dim;
            if (largest) {
                thrust::sort(
                    thrust::device_ptr<T>(batch_ptr),
                    thrust::device_ptr<T>(batch_ptr + last_dim),
                    thrust::greater<T>()
                );
            }
            else {
                thrust::sort(
                    thrust::device_ptr<T>(batch_ptr),
                    thrust::device_ptr<T>(batch_ptr + last_dim)
                );
            }
        }

        LaunchConfig config(batch_size);
        topk_gather_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            dst, work, batch_size, last_dim, k
            );

        cudaFree(work);
    }

    template<typename T>
    __global__ void topk_indices_gather_kernel(
        int64_t* dst, const int64_t* src, int64_t batch_size, int64_t last_dim, int64_t k
    ) {
        int64_t batch = blockIdx.x * blockDim.x + threadIdx.x;
        if (batch >= batch_size) return;

        for (int64_t i = 0; i < k; ++i) {
            dst[batch * k + i] = src[batch * last_dim + i];
        }
    }

    template<typename T>
    static void topk_indices_impl(const Array& out, const Array& x, int64_t k, bool largest) {
        int64_t total = x.numel();
        const Shape& shape = x.shape();
        int ndim = shape.ndim();
        int64_t last_dim = shape.dim(ndim - 1);
        int64_t batch_size = total / last_dim;

        int64_t* dst = const_cast<int64_t*>(out.data<int64_t>());

        // Allocate work buffer and copy input to avoid modifying original
        T* work;
        cudaMalloc(&work, total * sizeof(T));
        cudaMemcpy(work, x.data<T>(), total * sizeof(T), cudaMemcpyDeviceToDevice);

        int64_t* temp_indices;
        cudaMalloc(&temp_indices, total * sizeof(int64_t));

        for (int64_t batch = 0; batch < batch_size; ++batch) {
            thrust::device_ptr<int64_t> idx_ptr(temp_indices + batch * last_dim);
            thrust::sequence(idx_ptr, idx_ptr + last_dim);

            if (largest) {
                thrust::sort_by_key(
                    thrust::device_ptr<T>(work + batch * last_dim),
                    thrust::device_ptr<T>(work + batch * last_dim + last_dim),
                    idx_ptr,
                    thrust::greater<T>()
                );
            }
            else {
                thrust::sort_by_key(
                    thrust::device_ptr<T>(work + batch * last_dim),
                    thrust::device_ptr<T>(work + batch * last_dim + last_dim),
                    idx_ptr
                );
            }
        }

        LaunchConfig config(batch_size);
        topk_indices_gather_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            dst, temp_indices, batch_size, last_dim, k
            );

        cudaFree(temp_indices);
        cudaFree(work);
    }

    static OpArgs topk_wrapper(const OpArgs& args) {
        const Array& values = std::any_cast<const Array&>(args[0]);
        const Array& indices = std::any_cast<const Array&>(args[1]);
        const Array& x = std::any_cast<const Array&>(args[2]);
        int64_t k = std::any_cast<int64_t>(args[3]);
        bool largest = std::any_cast<bool>(args[4]);
        bool sorted = std::any_cast<bool>(args[5]);

        DType dtype = x.dtype();
        switch (dtype) {
        case DType::F32: topk_values_impl<float>(values, x, k, largest); topk_indices_impl<float>(indices, x, k, largest); break;
        case DType::F64: topk_values_impl<double>(values, x, k, largest); topk_indices_impl<double>(indices, x, k, largest); break;
        case DType::I32: topk_values_impl<int32_t>(values, x, k, largest); topk_indices_impl<int32_t>(indices, x, k, largest); break;
        case DType::I64: topk_values_impl<int64_t>(values, x, k, largest); topk_indices_impl<int64_t>(indices, x, k, largest); break;
        case DType::U8:  topk_values_impl<uint8_t>(values, x, k, largest); topk_indices_impl<uint8_t>(indices, x, k, largest); break;
        case DType::BOOL:topk_values_impl<bool>(values, x, k, largest); topk_indices_impl<bool>(indices, x, k, largest); break;
        default: INS_THROW("topk: unsupported dtype");
        }
        return { values, indices };
    }

    REGISTER_KERNEL(topk, GPU, F32, topk_wrapper);
    REGISTER_KERNEL(topk, GPU, F64, topk_wrapper);
    REGISTER_KERNEL(topk, GPU, I32, topk_wrapper);
    REGISTER_KERNEL(topk, GPU, I64, topk_wrapper);
    REGISTER_KERNEL(topk, GPU, U8, topk_wrapper);
    REGISTER_KERNEL(topk, GPU, BOOL, topk_wrapper);

    // ============================================================================
    // searchsorted (thrust path - binary search per element)
    // ============================================================================

    template<typename T>
    __global__ void searchsorted_kernel(
        int64_t* dst, const T* src, const T* vals,
        int64_t n, int64_t m, int side
    ) {
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            T val = vals[i];
            int64_t lo = 0, hi = m;
            while (lo < hi) {
                int64_t mid = (lo + hi) / 2;
                if (side == 0) {  // left
                    if (src[mid] < val) {
                        lo = mid + 1;
                    }
                    else {
                        hi = mid;
                    }
                }
                else {  // right
                    if (src[mid] <= val) {
                        lo = mid + 1;
                    }
                    else {
                        hi = mid;
                    }
                }
            }
            dst[i] = lo;
        }
    }

    template<typename T>
    static void searchsorted_impl(const Array& out, const Array& x,
        const Array& v, const std::string& side) {
        int64_t n = v.numel();
        int64_t m = x.numel();
        int side_int = (side == "left") ? 0 : 1;

        LaunchConfig config(n);
        searchsorted_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            const_cast<int64_t*>(out.data<int64_t>()), x.data<T>(), v.data<T>(),
            n, m, side_int
            );
    }

    static OpArgs searchsorted_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& x = std::any_cast<const Array&>(args[1]);
        const Array& v = std::any_cast<const Array&>(args[2]);
        std::string side = std::any_cast<std::string>(args[3]);

        DType dtype = x.dtype();
        switch (dtype) {
        case DType::F32: searchsorted_impl<float>(out, x, v, side); break;
        case DType::F64: searchsorted_impl<double>(out, x, v, side); break;
        case DType::I32: searchsorted_impl<int32_t>(out, x, v, side); break;
        case DType::I64: searchsorted_impl<int64_t>(out, x, v, side); break;
        case DType::U8:  searchsorted_impl<uint8_t>(out, x, v, side); break;
        default: INS_THROW("searchsorted: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(searchsorted, GPU, F32, searchsorted_wrapper);
    REGISTER_KERNEL(searchsorted, GPU, F64, searchsorted_wrapper);
    REGISTER_KERNEL(searchsorted, GPU, I32, searchsorted_wrapper);
    REGISTER_KERNEL(searchsorted, GPU, I64, searchsorted_wrapper);
    REGISTER_KERNEL(searchsorted, GPU, U8, searchsorted_wrapper);

    // ============================================================================
    // unique (thrust path)
    // ============================================================================

    template<typename T>
    __global__ void inverse_mapping_kernel(
        int64_t* inv_out, const T* original, const T* sorted_unique,
        int64_t n, int64_t nu
    ) {
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            T val = original[i];
            int64_t lo = 0, hi = nu;
            while (lo < hi) {
                int64_t mid = (lo + hi) / 2;
                if constexpr (std::is_same_v<T, bool>) {
                    // bool: 0 < 1, simple comparison
                    if (static_cast<int>(sorted_unique[mid]) < static_cast<int>(val)) {
                        lo = mid + 1;
                    }
                    else {
                        hi = mid;
                    }
                }
                else if constexpr (std::is_same_v<T, std::complex<float>> ||
                    std::is_same_v<T, std::complex<double>>) {
                    if (host_real(sorted_unique[mid]) < host_real(val)) {
                        lo = mid + 1;
                    }
                    else if (host_real(val) < host_real(sorted_unique[mid])) {
                        hi = mid;
                    }
                    else {
                        hi = mid;
                    }
                }
                else {
                    if (sorted_unique[mid] < val) {
                        lo = mid + 1;
                    }
                    else {
                        hi = mid;
                    }
                }
            }
            inv_out[i] = lo;
        }
    }

    template<typename T>
    static int64_t unique_impl(const Array& flattened, Array& unique_out, Array& indices_out,
        Array& inverse_out, Array& counts_out,
        bool return_indices, bool return_inverse, bool return_counts) {
        int64_t n = flattened.numel();

        T* work;
        cudaMalloc(&work, n * sizeof(T));
        cudaMemcpy(work, flattened.data<T>(), n * sizeof(T), cudaMemcpyDeviceToDevice);

        int64_t* orig_indices = nullptr;
        if (return_indices || return_inverse) {
            cudaMalloc(&orig_indices, n * sizeof(int64_t));
            thrust::sequence(thrust::device_ptr<int64_t>(orig_indices),
                thrust::device_ptr<int64_t>(orig_indices + n));
        }

        T* unique_ptr = unique_out.data<T>();
        int64_t nu = 0;

        if (return_indices || return_inverse) {
            thrust::sort_by_key(
                thrust::device_ptr<T>(work),
                thrust::device_ptr<T>(work + n),
                thrust::device_ptr<int64_t>(orig_indices)
            );
        }
        else {
            thrust::sort(thrust::device_ptr<T>(work), thrust::device_ptr<T>(work + n));
        }

        if (return_indices) {
            auto end = thrust::reduce_by_key(
                thrust::device_ptr<T>(work),
                thrust::device_ptr<T>(work + n),
                thrust::device_ptr<int64_t>(orig_indices),
                thrust::device_ptr<T>(unique_ptr),
                thrust::device_ptr<int64_t>(indices_out.data<int64_t>()),
                thrust::equal_to<T>(),
                thrust::minimum<int64_t>()
            );
            nu = end.first - thrust::device_ptr<T>(unique_ptr);

            if (return_inverse) {
                LaunchConfig config(n);
                inverse_mapping_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
                    inverse_out.data<int64_t>(), work, unique_ptr, n, nu
                    );
            }

            if (return_counts) {
                thrust::reduce_by_key(
                    thrust::device_ptr<T>(work),
                    thrust::device_ptr<T>(work + n),
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator<int64_t>(0),
                        ConstantOne()
                    ),
                    thrust::device_ptr<T>(unique_ptr),
                    thrust::device_ptr<int64_t>(counts_out.data<int64_t>())
                );
            }
        }
        else {
            auto new_end = thrust::unique(thrust::device_ptr<T>(work), thrust::device_ptr<T>(work + n));
            nu = new_end - thrust::device_ptr<T>(work);

            cudaMemcpy(unique_ptr, work, nu * sizeof(T), cudaMemcpyDeviceToDevice);

            if (return_inverse) {
                LaunchConfig config(n);
                inverse_mapping_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
                    inverse_out.data<int64_t>(), flattened.data<T>(), work, n, nu
                    );
            }

            if (return_counts) {
                thrust::reduce_by_key(
                    thrust::device_ptr<T>(work),
                    thrust::device_ptr<T>(work + n),
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator<int64_t>(0),
                        ConstantOne()
                    ),
                    thrust::device_ptr<T>(unique_ptr),
                    thrust::device_ptr<int64_t>(counts_out.data<int64_t>())
                );
            }
        }

        cudaFree(work);
        if (orig_indices) cudaFree(orig_indices);

        return nu;
    }

    static OpArgs unique_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        bool return_indices = std::any_cast<bool>(args[1]);
        bool return_inverse = std::any_cast<bool>(args[2]);
        bool return_counts = std::any_cast<bool>(args[3]);

        DType dtype = x.dtype();
        OpArgs output;

        switch (dtype) {
        case DType::F32: {
            int64_t n = x.numel();
            Array unique_arr(Shape({ n }), DType::F32, x.place());
            Array indices_arr(Shape({ n }), DType::I64, x.place());
            Array inverse_arr(Shape({ n }), DType::I64, x.place());
            Array counts_arr(Shape({ n }), DType::I64, x.place());

            int64_t nu = unique_impl<float>(x, unique_arr, indices_arr, inverse_arr, counts_arr,
                return_indices, return_inverse, return_counts);

            unique_arr = unique_arr.slice(0, 0, nu);
            output.push_back(unique_arr);
            if (return_indices) {
                indices_arr = indices_arr.slice(0, 0, nu);
                output.push_back(indices_arr);
            }
            if (return_inverse) {
                output.push_back(inverse_arr);
            }
            if (return_counts) {
                counts_arr = counts_arr.slice(0, 0, nu);
                output.push_back(counts_arr);
            }
            break;
        }
        case DType::F64: {
            int64_t n = x.numel();
            Array unique_arr(Shape({ n }), DType::F64, x.place());
            Array indices_arr(Shape({ n }), DType::I64, x.place());
            Array inverse_arr(Shape({ n }), DType::I64, x.place());
            Array counts_arr(Shape({ n }), DType::I64, x.place());

            int64_t nu = unique_impl<double>(x, unique_arr, indices_arr, inverse_arr, counts_arr,
                return_indices, return_inverse, return_counts);

            unique_arr = unique_arr.slice(0, 0, nu);
            output.push_back(unique_arr);
            if (return_indices) {
                indices_arr = indices_arr.slice(0, 0, nu);
                output.push_back(indices_arr);
            }
            if (return_inverse) {
                output.push_back(inverse_arr);
            }
            if (return_counts) {
                counts_arr = counts_arr.slice(0, 0, nu);
                output.push_back(counts_arr);
            }
            break;
        }
        case DType::I32: {
            int64_t n = x.numel();
            Array unique_arr(Shape({ n }), DType::I32, x.place());
            Array indices_arr(Shape({ n }), DType::I64, x.place());
            Array inverse_arr(Shape({ n }), DType::I64, x.place());
            Array counts_arr(Shape({ n }), DType::I64, x.place());

            int64_t nu = unique_impl<int32_t>(x, unique_arr, indices_arr, inverse_arr, counts_arr,
                return_indices, return_inverse, return_counts);

            unique_arr = unique_arr.slice(0, 0, nu);
            output.push_back(unique_arr);
            if (return_indices) {
                indices_arr = indices_arr.slice(0, 0, nu);
                output.push_back(indices_arr);
            }
            if (return_inverse) {
                output.push_back(inverse_arr);
            }
            if (return_counts) {
                counts_arr = counts_arr.slice(0, 0, nu);
                output.push_back(counts_arr);
            }
            break;
        }
        case DType::I64: {
            int64_t n = x.numel();
            Array unique_arr(Shape({ n }), DType::I64, x.place());
            Array indices_arr(Shape({ n }), DType::I64, x.place());
            Array inverse_arr(Shape({ n }), DType::I64, x.place());
            Array counts_arr(Shape({ n }), DType::I64, x.place());

            int64_t nu = unique_impl<int64_t>(x, unique_arr, indices_arr, inverse_arr, counts_arr,
                return_indices, return_inverse, return_counts);

            unique_arr = unique_arr.slice(0, 0, nu);
            output.push_back(unique_arr);
            if (return_indices) {
                indices_arr = indices_arr.slice(0, 0, nu);
                output.push_back(indices_arr);
            }
            if (return_inverse) {
                output.push_back(inverse_arr);
            }
            if (return_counts) {
                counts_arr = counts_arr.slice(0, 0, nu);
                output.push_back(counts_arr);
            }
            break;
        }
        case DType::U8: {
            int64_t n = x.numel();
            Array unique_arr(Shape({ n }), DType::U8, x.place());
            Array indices_arr(Shape({ n }), DType::I64, x.place());
            Array inverse_arr(Shape({ n }), DType::I64, x.place());
            Array counts_arr(Shape({ n }), DType::I64, x.place());

            int64_t nu = unique_impl<uint8_t>(x, unique_arr, indices_arr, inverse_arr, counts_arr,
                return_indices, return_inverse, return_counts);

            unique_arr = unique_arr.slice(0, 0, nu);
            output.push_back(unique_arr);
            if (return_indices) {
                indices_arr = indices_arr.slice(0, 0, nu);
                output.push_back(indices_arr);
            }
            if (return_inverse) {
                output.push_back(inverse_arr);
            }
            if (return_counts) {
                counts_arr = counts_arr.slice(0, 0, nu);
                output.push_back(counts_arr);
            }
            break;
        }
        case DType::BOOL: {
            int64_t n = x.numel();

            // Step 1: Count true occurrences on GPU
            int64_t true_count = thrust::count(
                thrust::device_ptr<const bool>(x.data<bool>()),
                thrust::device_ptr<const bool>(x.data<bool>() + n),
                true
            );
            int64_t false_count = n - true_count;
            bool has_true = (true_count > 0);
            bool has_false = (false_count > 0);
            int64_t nu = (has_true ? 1 : 0) + (has_false ? 1 : 0);

            // Step 2: Find first occurrences
            int64_t first_true = -1, first_false = -1;
            if (has_true) {
                auto it = thrust::find(
                    thrust::device_ptr<const bool>(x.data<bool>()),
                    thrust::device_ptr<const bool>(x.data<bool>() + n),
                    true
                );
                first_true = it - thrust::device_ptr<const bool>(x.data<bool>());
            }
            if (has_false) {
                auto it = thrust::find(
                    thrust::device_ptr<const bool>(x.data<bool>()),
                    thrust::device_ptr<const bool>(x.data<bool>() + n),
                    false
                );
                first_false = it - thrust::device_ptr<const bool>(x.data<bool>());
            }

            // Step 3: Allocate output arrays
            Array unique_arr(Shape({ nu }), DType::BOOL, x.place());
            output.push_back(unique_arr);

            if (nu > 0) {
                bool host_unique[2];
                if (has_false) host_unique[0] = false;
                if (has_true) host_unique[has_false ? 1 : 0] = true;
                cudaMemcpy(unique_arr.data<bool>(), host_unique, nu * sizeof(bool), cudaMemcpyHostToDevice);
            }

            // Step 4: Indices (first occurrence)
            if (return_indices) {
                Array indices_arr(Shape({ nu }), DType::I64, x.place());
                if (nu > 0) {
                    int64_t host_indices[2];
                    if (has_false) host_indices[0] = first_false;
                    if (has_true) host_indices[has_false ? 1 : 0] = first_true;
                    cudaMemcpy(indices_arr.data<int64_t>(), host_indices, nu * sizeof(int64_t), cudaMemcpyHostToDevice);
                }
                output.push_back(indices_arr);
            }

            // Step 5: Inverse
            if (return_inverse) {
                Array inverse_arr(Shape({ n }), DType::I64, x.place());
                BoolInverseMapper mapper{ has_false };
                thrust::transform(
                    thrust::device_ptr<const bool>(x.data<bool>()),
                    thrust::device_ptr<const bool>(x.data<bool>() + n),
                    thrust::device_ptr<int64_t>(inverse_arr.data<int64_t>()),
                    mapper
                );
                output.push_back(inverse_arr);
            }

            // Step 6: Counts
            if (return_counts) {
                Array counts_arr(Shape({ nu }), DType::I64, x.place());
                if (nu > 0) {
                    int64_t host_counts[2];
                    if (has_false) host_counts[0] = false_count;
                    if (has_true) host_counts[has_false ? 1 : 0] = true_count;
                    cudaMemcpy(counts_arr.data<int64_t>(), host_counts, nu * sizeof(int64_t), cudaMemcpyHostToDevice);
                }
                output.push_back(counts_arr);
            }
            break;
        }
        default: INS_THROW("unique: unsupported dtype");
        }

        return output;
    }

    REGISTER_KERNEL(unique, GPU, F32, unique_wrapper);
    REGISTER_KERNEL(unique, GPU, F64, unique_wrapper);
    REGISTER_KERNEL(unique, GPU, I32, unique_wrapper);
    REGISTER_KERNEL(unique, GPU, I64, unique_wrapper);
    REGISTER_KERNEL(unique, GPU, U8, unique_wrapper);
    REGISTER_KERNEL(unique, GPU, BOOL, unique_wrapper);

    // ============================================================================
    // lexsort (thrust path)
    // ============================================================================

    template<typename T>
    static void lexsort_impl(const Array& out, const Array& transposed,
        int64_t batch_size, int64_t last_dim, int64_t nkeys) {
        int64_t* dst = const_cast<int64_t*>(out.data<int64_t>());
        T* src = const_cast<T*>(transposed.data<T>());

        for (int64_t batch = 0; batch < batch_size; ++batch) {
            thrust::device_ptr<int64_t> idx_ptr(dst + batch * last_dim);
            thrust::sequence(idx_ptr, idx_ptr + last_dim);

            LexSortComparator<T> comp{ src, batch, nkeys, last_dim };
            thrust::sort(idx_ptr, idx_ptr + last_dim, comp);
        }
    }

    static OpArgs lexsort_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& transposed = std::any_cast<const Array&>(args[1]);
        int64_t batch_size = std::any_cast<int64_t>(args[2]);
        int64_t last_dim = std::any_cast<int64_t>(args[3]);
        int64_t nkeys = std::any_cast<int64_t>(args[4]);

        DType dtype = transposed.dtype();
        switch (dtype) {
        case DType::F32: lexsort_impl<float>(out, transposed, batch_size, last_dim, nkeys); break;
        case DType::F64: lexsort_impl<double>(out, transposed, batch_size, last_dim, nkeys); break;
        case DType::I32: lexsort_impl<int32_t>(out, transposed, batch_size, last_dim, nkeys); break;
        case DType::I64: lexsort_impl<int64_t>(out, transposed, batch_size, last_dim, nkeys); break;
        case DType::U8:  lexsort_impl<uint8_t>(out, transposed, batch_size, last_dim, nkeys); break;
        default: INS_THROW("lexsort: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(lexsort, GPU, F32, lexsort_wrapper);
    REGISTER_KERNEL(lexsort, GPU, F64, lexsort_wrapper);
    REGISTER_KERNEL(lexsort, GPU, I32, lexsort_wrapper);
    REGISTER_KERNEL(lexsort, GPU, I64, lexsort_wrapper);
    REGISTER_KERNEL(lexsort, GPU, U8, lexsort_wrapper);

    // ============================================================================
    // partition (thrust path)
    // ============================================================================

    template<typename T>
    static void partition_impl(const Array& out, const Array& x, int64_t kth, int axis) {
        int64_t total = x.numel();
        const Shape& shape = x.shape();
        int ndim = shape.ndim();
        if (axis < 0) axis += ndim;

        int64_t last_dim = shape.dim(ndim - 1);
        int64_t batch_size = total / last_dim;

        // Copy input to output
        cudaMemcpy(const_cast<T*>(out.data<T>()), x.data<T>(), total * sizeof(T), cudaMemcpyDeviceToDevice);

        T* work = const_cast<T*>(out.data<T>());
        for (int64_t batch = 0; batch < batch_size; ++batch) {
            T* row = work + batch * last_dim;
            thrust::sort(
                thrust::device_ptr<T>(row),
                thrust::device_ptr<T>(row + last_dim)
            );
        }
    }

    static OpArgs partition_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& x = std::any_cast<const Array&>(args[1]);
        int64_t kth = std::any_cast<int64_t>(args[2]);
        int axis = std::any_cast<int>(args[3]);

        DType dtype = x.dtype();
        switch (dtype) {
        case DType::F32: partition_impl<float>(out, x, kth, axis); break;
        case DType::F64: partition_impl<double>(out, x, kth, axis); break;
        case DType::I32: partition_impl<int32_t>(out, x, kth, axis); break;
        case DType::I64: partition_impl<int64_t>(out, x, kth, axis); break;
        case DType::U8:  partition_impl<uint8_t>(out, x, kth, axis); break;
        default: INS_THROW("partition: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(partition, GPU, F32, partition_wrapper);
    REGISTER_KERNEL(partition, GPU, F64, partition_wrapper);
    REGISTER_KERNEL(partition, GPU, I32, partition_wrapper);
    REGISTER_KERNEL(partition, GPU, I64, partition_wrapper);
    REGISTER_KERNEL(partition, GPU, U8, partition_wrapper);

    // ============================================================================
    // argpartition (thrust path)
    // ============================================================================

    template<typename T>
    static void argpartition_impl(const Array& out, const Array& x, int64_t kth, int axis) {
        int64_t total = x.numel();
        const Shape& shape = x.shape();
        int ndim = shape.ndim();
        if (axis < 0) axis += ndim;

        int64_t last_dim = shape.dim(ndim - 1);
        int64_t batch_size = total / last_dim;

        int64_t* dst = const_cast<int64_t*>(out.data<int64_t>());

        // Allocate work buffer and copy input to avoid modifying original
        T* work;
        cudaMalloc(&work, total * sizeof(T));
        cudaMemcpy(work, x.data<T>(), total * sizeof(T), cudaMemcpyDeviceToDevice);

        for (int64_t batch = 0; batch < batch_size; ++batch) {
            thrust::device_ptr<int64_t> idx_ptr(dst + batch * last_dim);
            thrust::sequence(idx_ptr, idx_ptr + last_dim);

            thrust::sort_by_key(
                thrust::device_ptr<T>(work + batch * last_dim),
                thrust::device_ptr<T>(work + batch * last_dim + last_dim),
                idx_ptr
            );
        }

        cudaFree(work);
    }

    static OpArgs argpartition_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& x = std::any_cast<const Array&>(args[1]);
        int64_t kth = std::any_cast<int64_t>(args[2]);
        int axis = std::any_cast<int>(args[3]);

        DType dtype = x.dtype();
        switch (dtype) {
        case DType::F32: argpartition_impl<float>(out, x, kth, axis); break;
        case DType::F64: argpartition_impl<double>(out, x, kth, axis); break;
        case DType::I32: argpartition_impl<int32_t>(out, x, kth, axis); break;
        case DType::I64: argpartition_impl<int64_t>(out, x, kth, axis); break;
        case DType::U8:  argpartition_impl<uint8_t>(out, x, kth, axis); break;
        default: INS_THROW("argpartition: unsupported dtype, got: ", dtype_name(dtype));
        }
        return { out };
    }

    REGISTER_KERNEL(argpartition, GPU, F32 , argpartition_wrapper);
    REGISTER_KERNEL(argpartition, GPU, F64 , argpartition_wrapper);
    REGISTER_KERNEL(argpartition, GPU, I8  , argpartition_wrapper);
    REGISTER_KERNEL(argpartition, GPU, I16 , argpartition_wrapper);
    REGISTER_KERNEL(argpartition, GPU, I32 , argpartition_wrapper);
    REGISTER_KERNEL(argpartition, GPU, I64 , argpartition_wrapper);
    REGISTER_KERNEL(argpartition, GPU, U8  , argpartition_wrapper);
    REGISTER_KERNEL(argpartition, GPU, U16 , argpartition_wrapper);
    REGISTER_KERNEL(argpartition, GPU, U32 , argpartition_wrapper);
    REGISTER_KERNEL(argpartition, GPU, U64 , argpartition_wrapper);
    REGISTER_KERNEL(argpartition, GPU, BOOL, argpartition_wrapper);

#endif // INSIGHT_USE_THRUST

} // namespace ins::gpu

// ============================================================================
// Module Registration
// ============================================================================

REGISTER_MODULE(indexing, GPU);