// backends/cuda/kernels/reduction.cu
#include "insight/plugin/op_registry.h"
#include "insight/core/array.h"
#include "insight/core/exception.h"
#include "insight/core/launch_config.h"
#include <cmath>

#ifdef INSIGHT_USE_THRUST
#include <thrust/sort.h>
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

    // ============================================================================
    // Device helper: check NaN
    // ============================================================================

    template<typename T>
    __device__ inline bool is_nan_device(T val) {
        return false;
    }

    template<>
    __device__ inline bool is_nan_device<float>(float val) {
        return isnan(val);
    }

    template<>
    __device__ inline bool is_nan_device<double>(double val) {
        return isnan(val);
    }

    // ============================================================================
    // Device helper: generate NaN constant
    // ============================================================================

    template<typename T>
    __device__ T device_nan();

    template<>
    __device__ float device_nan<float>() {
        return __int_as_float(0x7fc00000);
    }

    template<>
    __device__ double device_nan<double>() {
        return __longlong_as_double(0x7ff8000000000000ULL);
    }

    // ============================================================================
    // sum kernel
    // ============================================================================

    template<typename T>
    __global__ void sum_kernel(T* dst, const T* src, int64_t total_out, int64_t reduce_size) {
        extern __shared__ char __sdata_raw[];
        T* sdata = reinterpret_cast<T*>(__sdata_raw);

        int tid = threadIdx.x;
        int idx = blockIdx.x;

        T sum = T(0);
        for (int64_t j = tid; j < reduce_size; j += blockDim.x) {
            sum += src[idx * reduce_size + j];
        }

        sdata[tid] = sum;
        __syncthreads();

        // Tree reduction in shared memory
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            dst[idx] = sdata[0];
        }
    }

    template<typename T>
    static void sum_impl(const Array& out, const Array& prepared) {
        int64_t total_out = out.numel();
        int64_t reduce_size = prepared.numel() / total_out;

        int threads = 256;
        dim3 blocks(total_out);
        sum_kernel<T> << <blocks, threads, threads * sizeof(T) >> > (
            const_cast<T*>(out.data<T>()), prepared.data<T>(), total_out, reduce_size
            );
    }

    static OpArgs sum_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);

        switch (out.dtype()) {
        case DType::F32: sum_impl<float>(out, prepared); break;
        case DType::F64: sum_impl<double>(out, prepared); break;
        case DType::I32: sum_impl<int32_t>(out, prepared); break;
        case DType::I64: sum_impl<int64_t>(out, prepared); break;
        case DType::U8:  sum_impl<uint8_t>(out, prepared); break;
        default: INS_THROW("sum: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(sum, GPU, F32, sum_wrapper);
    REGISTER_KERNEL(sum, GPU, F64, sum_wrapper);
    REGISTER_KERNEL(sum, GPU, I32, sum_wrapper);
    REGISTER_KERNEL(sum, GPU, I64, sum_wrapper);
    REGISTER_KERNEL(sum, GPU, U8, sum_wrapper);

    // ============================================================================
    // max kernel
    // ============================================================================

    template<typename T>
    __global__ void max_kernel(T* dst, const T* src, int64_t total_out, int64_t reduce_size) {
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < total_out) {
            T max_val = src[i * reduce_size];
            for (int64_t j = 1; j < reduce_size; ++j) {
                T val = src[i * reduce_size + j];
                if (val > max_val) max_val = val;
            }
            dst[i] = max_val;
        }
    }

    template<typename T>
    static void max_impl(const Array& out, const Array& prepared) {
        int64_t total_out = out.numel();
        int64_t reduce_size = prepared.numel() / total_out;
        LaunchConfig config(total_out);
        max_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            const_cast<T*>(out.data<T>()), prepared.data<T>(), total_out, reduce_size
            );
    }

    static OpArgs max_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);

        switch (out.dtype()) {
        case DType::F32: max_impl<float>(out, prepared); break;
        case DType::F64: max_impl<double>(out, prepared); break;
        case DType::I32: max_impl<int32_t>(out, prepared); break;
        case DType::I64: max_impl<int64_t>(out, prepared); break;
        case DType::U8:  max_impl<uint8_t>(out, prepared); break;
        default: INS_THROW("max: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(max, GPU, F32, max_wrapper);
    REGISTER_KERNEL(max, GPU, F64, max_wrapper);
    REGISTER_KERNEL(max, GPU, I32, max_wrapper);
    REGISTER_KERNEL(max, GPU, I64, max_wrapper);
    REGISTER_KERNEL(max, GPU, U8, max_wrapper);

    // ============================================================================
    // min kernel
    // ============================================================================

    template<typename T>
    __global__ void min_kernel(T* dst, const T* src, int64_t total_out, int64_t reduce_size) {
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < total_out) {
            T min_val = src[i * reduce_size];
            for (int64_t j = 1; j < reduce_size; ++j) {
                T val = src[i * reduce_size + j];
                if (val < min_val) min_val = val;
            }
            dst[i] = min_val;
        }
    }

    template<typename T>
    static void min_impl(const Array& out, const Array& prepared) {
        int64_t total_out = out.numel();
        int64_t reduce_size = prepared.numel() / total_out;
        LaunchConfig config(total_out);
        min_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            const_cast<T*>(out.data<T>()), prepared.data<T>(), total_out, reduce_size
            );
    }

    static OpArgs min_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);

        switch (out.dtype()) {
        case DType::F32: min_impl<float>(out, prepared); break;
        case DType::F64: min_impl<double>(out, prepared); break;
        case DType::I32: min_impl<int32_t>(out, prepared); break;
        case DType::I64: min_impl<int64_t>(out, prepared); break;
        case DType::U8:  min_impl<uint8_t>(out, prepared); break;
        default: INS_THROW("min: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(min, GPU, F32, min_wrapper);
    REGISTER_KERNEL(min, GPU, F64, min_wrapper);
    REGISTER_KERNEL(min, GPU, I32, min_wrapper);
    REGISTER_KERNEL(min, GPU, I64, min_wrapper);
    REGISTER_KERNEL(min, GPU, U8, min_wrapper);

    // ============================================================================
    // prod kernel
    // ============================================================================

    template<typename T>
    __global__ void prod_kernel(T* dst, const T* src, int64_t total_out, int64_t reduce_size) {
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < total_out) {
            T prod = T(1);
            for (int64_t j = 0; j < reduce_size; ++j) {
                prod *= src[i * reduce_size + j];
            }
            dst[i] = prod;
        }
    }

    template<typename T>
    static void prod_impl(const Array& out, const Array& prepared) {
        int64_t total_out = out.numel();
        int64_t reduce_size = prepared.numel() / total_out;
        LaunchConfig config(total_out);
        prod_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            const_cast<T*>(out.data<T>()), prepared.data<T>(), total_out, reduce_size
            );
    }

    static OpArgs prod_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);

        switch (out.dtype()) {
        case DType::F32: prod_impl<float>(out, prepared); break;
        case DType::F64: prod_impl<double>(out, prepared); break;
        case DType::I32: prod_impl<int32_t>(out, prepared); break;
        case DType::I64: prod_impl<int64_t>(out, prepared); break;
        case DType::U8:  prod_impl<uint8_t>(out, prepared); break;
        default: INS_THROW("prod: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(prod, GPU, F32, prod_wrapper);
    REGISTER_KERNEL(prod, GPU, F64, prod_wrapper);
    REGISTER_KERNEL(prod, GPU, I32, prod_wrapper);
    REGISTER_KERNEL(prod, GPU, I64, prod_wrapper);
    REGISTER_KERNEL(prod, GPU, U8, prod_wrapper);

    // ============================================================================
    // any kernel
    // ============================================================================

    template<typename T>
    __global__ void any_kernel(bool* dst, const T* src, int64_t total_out, int64_t reduce_size) {
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < total_out) {
            bool any = false;
            for (int64_t j = 0; j < reduce_size; ++j) {
                if (src[i * reduce_size + j] != T(0)) {
                    any = true;
                    break;
                }
            }
            dst[i] = any;
        }
    }

    template<typename T>
    static void any_impl(const Array& out, const Array& prepared) {
        int64_t total_out = out.numel();
        int64_t reduce_size = prepared.numel() / total_out;
        LaunchConfig config(total_out);
        any_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            const_cast<bool*>(out.data<bool>()), prepared.data<T>(), total_out, reduce_size
            );
    }

    static OpArgs any_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);

        switch (prepared.dtype()) {
        case DType::F32: any_impl<float>(out, prepared); break;
        case DType::F64: any_impl<double>(out, prepared); break;
        case DType::I8: any_impl<int8_t>(out, prepared); break;
        case DType::I16: any_impl<int8_t>(out, prepared); break;
        case DType::I32: any_impl<int32_t>(out, prepared); break;
        case DType::I64: any_impl<int64_t>(out, prepared); break;
        case DType::U8:  any_impl<uint8_t>(out, prepared); break;
        case DType::U16:  any_impl<uint16_t>(out, prepared); break;
        case DType::U32:  any_impl<uint32_t>(out, prepared); break;
        case DType::U64:  any_impl<uint64_t>(out, prepared); break;
        case DType::BOOL:  any_impl<bool>(out, prepared); break;
        default: INS_THROW("any: unsupported dtype, got dtype: ", dtype_name(prepared.dtype()));
        }
        return { out };
    }

    REGISTER_KERNEL(any, GPU, F32, any_wrapper);
    REGISTER_KERNEL(any, GPU, F64, any_wrapper);
    REGISTER_KERNEL(any, GPU, I8, any_wrapper);
    REGISTER_KERNEL(any, GPU, I16, any_wrapper);
    REGISTER_KERNEL(any, GPU, I32, any_wrapper);
    REGISTER_KERNEL(any, GPU, I64, any_wrapper);
    REGISTER_KERNEL(any, GPU, U8, any_wrapper);
    REGISTER_KERNEL(any, GPU, U16, any_wrapper);
    REGISTER_KERNEL(any, GPU, U32, any_wrapper);
    REGISTER_KERNEL(any, GPU, U64, any_wrapper);
    REGISTER_KERNEL(any, GPU, BOOL, any_wrapper);

    // ============================================================================
    // all kernel
    // ============================================================================

    template<typename T>
    __global__ void all_kernel(bool* dst, const T* src, int64_t total_out, int64_t reduce_size) {
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < total_out) {
            bool all_true = true;
            for (int64_t j = 0; j < reduce_size; ++j) {
                if (src[i * reduce_size + j] == T(0)) {
                    all_true = false;
                    break;
                }
            }
            dst[i] = all_true;
        }
    }

    template<typename T>
    static void all_impl(const Array& out, const Array& prepared) {
        int64_t total_out = out.numel();
        int64_t reduce_size = prepared.numel() / total_out;
        LaunchConfig config(total_out);
        all_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            const_cast<bool*>(out.data<bool>()), prepared.data<T>(), total_out, reduce_size
            );
    }

    static OpArgs all_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);

        switch (prepared.dtype()) {
        case DType::F32 : all_impl<float   >(out, prepared); break;
        case DType::F64 : all_impl<double  >(out, prepared); break;
        case DType::I8  : all_impl<int8_t  >(out, prepared); break;
        case DType::I16 : all_impl<int8_t  >(out, prepared); break;
        case DType::I32 : all_impl<int32_t >(out, prepared); break;
        case DType::I64 : all_impl<int64_t >(out, prepared); break;
        case DType::U8  : all_impl<uint8_t >(out, prepared); break;
        case DType::U16 : all_impl<uint16_t>(out, prepared); break;
        case DType::U32 : all_impl<uint32_t>(out, prepared); break;
        case DType::U64 : all_impl<uint64_t>(out, prepared); break;
        case DType::BOOL: all_impl<bool    >(out, prepared); break;
        default: INS_THROW("all: unsupported dtype, got dtype: ", dtype_name(prepared.dtype()));
        }
        return { out };
    }

    REGISTER_KERNEL(all, GPU, F32 , all_wrapper);
    REGISTER_KERNEL(all, GPU, F64 , all_wrapper);
    REGISTER_KERNEL(all, GPU, I8  , all_wrapper);
    REGISTER_KERNEL(all, GPU, I16 , all_wrapper);
    REGISTER_KERNEL(all, GPU, I32 , all_wrapper);
    REGISTER_KERNEL(all, GPU, I64 , all_wrapper);
    REGISTER_KERNEL(all, GPU, U8  , all_wrapper);
    REGISTER_KERNEL(all, GPU, U16 , all_wrapper);
    REGISTER_KERNEL(all, GPU, U32 , all_wrapper);
    REGISTER_KERNEL(all, GPU, U64 , all_wrapper);
    REGISTER_KERNEL(all, GPU, BOOL, all_wrapper);

    // ============================================================================
    // argmax kernel
    // ============================================================================

    template<typename T>
    __global__ void argmax_kernel(int64_t* dst, const T* src, int64_t total_out, int64_t reduce_size) {
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < total_out) {
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
    }

    template<typename T>
    static void argmax_impl(const Array& out, const Array& prepared) {
        int64_t total_out = out.numel();
        int64_t reduce_size = prepared.numel() / total_out;
        LaunchConfig config(total_out);
        argmax_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            const_cast<int64_t*>(out.data<int64_t>()), prepared.data<T>(), total_out, reduce_size
            );
    }

    static OpArgs argmax_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);

        switch (prepared.dtype()) {
        case DType::F32: argmax_impl<float>(out, prepared); break;
        case DType::F64: argmax_impl<double>(out, prepared); break;
        case DType::I32: argmax_impl<int32_t>(out, prepared); break;
        case DType::I64: argmax_impl<int64_t>(out, prepared); break;
        case DType::U8:  argmax_impl<uint8_t>(out, prepared); break;
        default: INS_THROW("argmax: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(argmax, GPU, F32, argmax_wrapper);
    REGISTER_KERNEL(argmax, GPU, F64, argmax_wrapper);
    REGISTER_KERNEL(argmax, GPU, I32, argmax_wrapper);
    REGISTER_KERNEL(argmax, GPU, I64, argmax_wrapper);
    REGISTER_KERNEL(argmax, GPU, U8, argmax_wrapper);

    // ============================================================================
    // argmin kernel
    // ============================================================================

    template<typename T>
    __global__ void argmin_kernel(int64_t* dst, const T* src, int64_t total_out, int64_t reduce_size) {
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < total_out) {
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
    }

    template<typename T>
    static void argmin_impl(const Array& out, const Array& prepared) {
        int64_t total_out = out.numel();
        int64_t reduce_size = prepared.numel() / total_out;
        LaunchConfig config(total_out);
        argmin_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            const_cast<int64_t*>(out.data<int64_t>()), prepared.data<T>(), total_out, reduce_size
            );
    }

    static OpArgs argmin_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);

        switch (prepared.dtype()) {
        case DType::F32: argmin_impl<float>(out, prepared); break;
        case DType::F64: argmin_impl<double>(out, prepared); break;
        case DType::I32: argmin_impl<int32_t>(out, prepared); break;
        case DType::I64: argmin_impl<int64_t>(out, prepared); break;
        case DType::U8:  argmin_impl<uint8_t>(out, prepared); break;
        default: INS_THROW("argmin: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(argmin, GPU, F32, argmin_wrapper);
    REGISTER_KERNEL(argmin, GPU, F64, argmin_wrapper);
    REGISTER_KERNEL(argmin, GPU, I32, argmin_wrapper);
    REGISTER_KERNEL(argmin, GPU, I64, argmin_wrapper);
    REGISTER_KERNEL(argmin, GPU, U8, argmin_wrapper);

    // ============================================================================
    // count_nonzero kernel
    // ============================================================================

    template<typename T>
    __global__ void count_nonzero_kernel(int64_t* dst, const T* src, int64_t total_out, int64_t reduce_size) {
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < total_out) {
            int64_t count = 0;
            for (int64_t j = 0; j < reduce_size; ++j) {
                T val = src[i * reduce_size + j];
                bool is_nonzero;
                if constexpr (std::is_same_v<T, ComplexPod<float>> || std::is_same_v<T, ComplexPod<double>>) {
                    is_nonzero = (val.r != 0) || (val.i != 0);
                }
                else {
                    is_nonzero = (val != T(0));
                }
                if (is_nonzero) ++count;
            }
            dst[i] = count;
        }
    }

    template<typename T>
    static void count_nonzero_impl(const Array& out, const Array& prepared) {
        int64_t total_out = out.numel();
        int64_t reduce_size = prepared.numel() / total_out;
        LaunchConfig config(total_out);
        count_nonzero_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            const_cast<int64_t*>(out.data<int64_t>()), prepared.data<T>(), total_out, reduce_size
            );
    }

    static OpArgs count_nonzero_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);

        switch (prepared.dtype()) {
        case DType::BOOL: count_nonzero_impl<bool>(out, prepared); break;
        case DType::U8:   count_nonzero_impl<uint8_t>(out, prepared); break;
        case DType::I8:   count_nonzero_impl<int8_t>(out, prepared); break;
        case DType::I16:  count_nonzero_impl<int16_t>(out, prepared); break;
        case DType::I32:  count_nonzero_impl<int32_t>(out, prepared); break;
        case DType::I64:  count_nonzero_impl<int64_t>(out, prepared); break;
        case DType::U16:  count_nonzero_impl<uint16_t>(out, prepared); break;
        case DType::U32:  count_nonzero_impl<uint32_t>(out, prepared); break;
        case DType::U64:  count_nonzero_impl<uint64_t>(out, prepared); break;
        case DType::F32:  count_nonzero_impl<float>(out, prepared); break;
        case DType::F64:  count_nonzero_impl<double>(out, prepared); break;
        case DType::C32:  count_nonzero_impl<ComplexPod<float>>(out, prepared); break;
        case DType::C64:  count_nonzero_impl<ComplexPod<double>>(out, prepared); break;
        default: INS_THROW("count_nonzero: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(count_nonzero, GPU, BOOL, count_nonzero_wrapper);
    REGISTER_KERNEL(count_nonzero, GPU, U8, count_nonzero_wrapper);
    REGISTER_KERNEL(count_nonzero, GPU, I8, count_nonzero_wrapper);
    REGISTER_KERNEL(count_nonzero, GPU, I16, count_nonzero_wrapper);
    REGISTER_KERNEL(count_nonzero, GPU, I32, count_nonzero_wrapper);
    REGISTER_KERNEL(count_nonzero, GPU, I64, count_nonzero_wrapper);
    REGISTER_KERNEL(count_nonzero, GPU, U16, count_nonzero_wrapper);
    REGISTER_KERNEL(count_nonzero, GPU, U32, count_nonzero_wrapper);
    REGISTER_KERNEL(count_nonzero, GPU, U64, count_nonzero_wrapper);
    REGISTER_KERNEL(count_nonzero, GPU, F32, count_nonzero_wrapper);
    REGISTER_KERNEL(count_nonzero, GPU, F64, count_nonzero_wrapper);
    REGISTER_KERNEL(count_nonzero, GPU, C32, count_nonzero_wrapper);
    REGISTER_KERNEL(count_nonzero, GPU, C64, count_nonzero_wrapper);

    // ============================================================================
    // cumsum kernel
    // ============================================================================

    template<typename InT, typename OutT>
    __global__ void cumsum_kernel(
        OutT* dst, const InT* src,
        int64_t total, const int64_t* __restrict__ dims,
        const int64_t* __restrict__ strides, int ndim, int axis
    ) {
        int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
        if (linear >= total) return;

        int64_t coords[4] = { 0 };
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

        int64_t axis_stride = strides[axis];
        OutT sum = OutT(0);
        for (int64_t k = 0; k <= coords[axis]; ++k) {
            int64_t idx = base_idx + k * axis_stride;
            sum += static_cast<OutT>(src[idx]);
        }
        dst[linear] = sum;
    }

    template<typename InT, typename OutT>
    static void cumsum_impl(const Array& out, const Array& x, int axis) {
        int64_t total = x.numel();
        int ndim = x.shape().ndim();

        int64_t h_dims[4] = { 1, 1, 1, 1 };
        int64_t h_strides[4] = { 0, 0, 0, 0 };
        for (int i = 0; i < ndim; ++i) {
            h_dims[i] = x.shape().dim(i);
        }
        h_strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            h_strides[i] = h_strides[i + 1] * h_dims[i + 1];
        }

        int64_t* d_dims;
        int64_t* d_strides;
        cudaMalloc(&d_dims, 4 * sizeof(int64_t));
        cudaMalloc(&d_strides, 4 * sizeof(int64_t));
        cudaMemcpy(d_dims, h_dims, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_strides, h_strides, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);

        LaunchConfig config(total);
        cumsum_kernel<InT, OutT> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            const_cast<OutT*>(out.data<OutT>()), x.data<InT>(),
            total, d_dims, d_strides, ndim, axis
            );

        cudaFree(d_dims);
        cudaFree(d_strides);
    }

    static OpArgs cumsum_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& x = std::any_cast<const Array&>(args[1]);
        int axis = std::any_cast<int>(args[2]);

        DType in_dtype = x.dtype();
        DType out_dtype = out.dtype();

        if (in_dtype == DType::F32 && out_dtype == DType::F64) {
            cumsum_impl<float, double>(out, x, axis);
        }
        else if (in_dtype == DType::F32 && out_dtype == DType::F32) {
            cumsum_impl<float, float>(out, x, axis);
        }
        else if (in_dtype == DType::F64 && out_dtype == DType::F64) {
            cumsum_impl<double, double>(out, x, axis);
        }
        else if (in_dtype == DType::I32 && out_dtype == DType::I32) {
            cumsum_impl<int32_t, int32_t>(out, x, axis);
        }
        else if (in_dtype == DType::I64 && out_dtype == DType::I64) {
            cumsum_impl<int64_t, int64_t>(out, x, axis);
        }
        else if (in_dtype == DType::I32 && out_dtype == DType::F64) {
            cumsum_impl<int32_t, double>(out, x, axis);
        }
        else if (in_dtype == DType::I64 && out_dtype == DType::F64) {
            cumsum_impl<int64_t, double>(out, x, axis);
        }
        else {
            INS_THROW("cumsum: unsupported dtype combination");
        }
        return { out };
    }

    REGISTER_KERNEL(cumsum, GPU, F32, cumsum_wrapper);
    REGISTER_KERNEL(cumsum, GPU, F64, cumsum_wrapper);
    REGISTER_KERNEL(cumsum, GPU, I32, cumsum_wrapper);
    REGISTER_KERNEL(cumsum, GPU, I64, cumsum_wrapper);

    // ============================================================================
    // cumprod kernel
    // ============================================================================

    template<typename InT, typename OutT>
    __global__ void cumprod_kernel(
        OutT* dst, const InT* src,
        int64_t total, const int64_t* __restrict__ dims,
        const int64_t* __restrict__ strides, int ndim, int axis
    ) {
        int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
        if (linear >= total) return;

        int64_t coords[4] = { 0 };
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

        int64_t axis_stride = strides[axis];
        OutT prod = OutT(1);
        for (int64_t k = 0; k <= coords[axis]; ++k) {
            int64_t idx = base_idx + k * axis_stride;
            prod *= static_cast<OutT>(src[idx]);
        }
        dst[linear] = prod;
    }

    template<typename InT, typename OutT>
    static void cumprod_impl(const Array& out, const Array& x, int axis) {
        int64_t total = x.numel();
        int ndim = x.shape().ndim();

        int64_t h_dims[4] = { 1, 1, 1, 1 };
        int64_t h_strides[4] = { 0, 0, 0, 0 };
        for (int i = 0; i < ndim; ++i) {
            h_dims[i] = x.shape().dim(i);
        }
        h_strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            h_strides[i] = h_strides[i + 1] * h_dims[i + 1];
        }

        int64_t* d_dims;
        int64_t* d_strides;
        cudaMalloc(&d_dims, 4 * sizeof(int64_t));
        cudaMalloc(&d_strides, 4 * sizeof(int64_t));
        cudaMemcpy(d_dims, h_dims, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_strides, h_strides, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);

        LaunchConfig config(total);
        cumprod_kernel<InT, OutT> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            const_cast<OutT*>(out.data<OutT>()), x.data<InT>(),
            total, d_dims, d_strides, ndim, axis
            );

        cudaFree(d_dims);
        cudaFree(d_strides);
    }

    static OpArgs cumprod_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& x = std::any_cast<const Array&>(args[1]);
        int axis = std::any_cast<int>(args[2]);

        DType in_dtype = x.dtype();
        DType out_dtype = out.dtype();

        if (in_dtype == DType::F32 && out_dtype == DType::F64) {
            cumprod_impl<float, double>(out, x, axis);
        }
        else if (in_dtype == DType::F32 && out_dtype == DType::F32) {
            cumprod_impl<float, float>(out, x, axis);
        }
        else if (in_dtype == DType::F64 && out_dtype == DType::F64) {
            cumprod_impl<double, double>(out, x, axis);
        }
        else if (in_dtype == DType::I32 && out_dtype == DType::I32) {
            cumprod_impl<int32_t, int32_t>(out, x, axis);
        }
        else if (in_dtype == DType::I64 && out_dtype == DType::I64) {
            cumprod_impl<int64_t, int64_t>(out, x, axis);
        }
        else if (in_dtype == DType::I32 && out_dtype == DType::F64) {
            cumprod_impl<int32_t, double>(out, x, axis);
        }
        else if (in_dtype == DType::I64 && out_dtype == DType::F64) {
            cumprod_impl<int64_t, double>(out, x, axis);
        }
        else {
            INS_THROW("cumprod: unsupported dtype combination");
        }
        return { out };
    }

    REGISTER_KERNEL(cumprod, GPU, F32, cumprod_wrapper);
    REGISTER_KERNEL(cumprod, GPU, F64, cumprod_wrapper);
    REGISTER_KERNEL(cumprod, GPU, I32, cumprod_wrapper);
    REGISTER_KERNEL(cumprod, GPU, I64, cumprod_wrapper);

    // ============================================================================
    // cummax kernel
    // ============================================================================

    template<typename T>
    __global__ void cummax_kernel(
        T* dst, const T* src,
        int64_t total, const int64_t* __restrict__ dims,
        const int64_t* __restrict__ strides, int ndim, int axis
    ) {
        int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
        if (linear >= total) return;

        int64_t coords[4] = { 0 };
        int64_t tmp = linear;
        for (int d = ndim - 1; d >= 0; --d) {
            coords[d] = tmp % dims[d];
            tmp /= dims[d];
        }

        T max_val = src[linear];
        for (int64_t k = 0; k <= coords[axis]; ++k) {
            int64_t idx = 0;
            for (int d = 0; d < ndim; ++d) {
                int64_t coord = (d == axis) ? k : coords[d];
                idx += coord * strides[d];
            }
            if (src[idx] > max_val) max_val = src[idx];
        }
        dst[linear] = max_val;
    }

    template<typename T>
    static void cummax_impl(const Array& out, const Array& x, int axis) {
        int64_t total = x.numel();
        int ndim = x.shape().ndim();

        int64_t h_dims[4] = { 1, 1, 1, 1 };
        int64_t h_strides[4] = { 0, 0, 0, 0 };
        for (int i = 0; i < ndim; ++i) {
            h_dims[i] = x.shape().dim(i);
        }
        h_strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            h_strides[i] = h_strides[i + 1] * h_dims[i + 1];
        }

        int64_t* d_dims;
        int64_t* d_strides;
        cudaMalloc(&d_dims, 4 * sizeof(int64_t));
        cudaMalloc(&d_strides, 4 * sizeof(int64_t));
        cudaMemcpy(d_dims, h_dims, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_strides, h_strides, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);

        LaunchConfig config(total);
        cummax_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            const_cast<T*>(out.data<T>()), x.data<T>(),
            total, d_dims, d_strides, ndim, axis);

        cudaFree(d_dims);
        cudaFree(d_strides);
    }

    static OpArgs cummax_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& x = std::any_cast<const Array&>(args[1]);
        int axis = std::any_cast<int>(args[2]);

        switch (out.dtype()) {
        case DType::F32: cummax_impl<float>(out, x, axis); break;
        case DType::F64: cummax_impl<double>(out, x, axis); break;
        case DType::I32: cummax_impl<int32_t>(out, x, axis); break;
        case DType::I64: cummax_impl<int64_t>(out, x, axis); break;
        default: INS_THROW("cummax: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(cummax, GPU, F32, cummax_wrapper);
    REGISTER_KERNEL(cummax, GPU, F64, cummax_wrapper);
    REGISTER_KERNEL(cummax, GPU, I32, cummax_wrapper);
    REGISTER_KERNEL(cummax, GPU, I64, cummax_wrapper);

    // ============================================================================
    // cummin kernel
    // ============================================================================

    template<typename T>
    __global__ void cummin_kernel(
        T* dst, const T* src,
        int64_t total, const int64_t* __restrict__ dims,
        const int64_t* __restrict__ strides, int ndim, int axis
    ) {
        int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
        if (linear >= total) return;

        int64_t coords[4] = { 0 };
        int64_t tmp = linear;
        for (int d = ndim - 1; d >= 0; --d) {
            coords[d] = tmp % dims[d];
            tmp /= dims[d];
        }

        T min_val = src[linear];
        for (int64_t k = 0; k <= coords[axis]; ++k) {
            int64_t idx = 0;
            for (int d = 0; d < ndim; ++d) {
                int64_t coord = (d == axis) ? k : coords[d];
                idx += coord * strides[d];
            }
            if (src[idx] < min_val) min_val = src[idx];
        }
        dst[linear] = min_val;
    }

    template<typename T>
    static void cummin_impl(const Array& out, const Array& x, int axis) {
        int64_t total = x.numel();
        int ndim = x.shape().ndim();

        int64_t h_dims[4] = { 1, 1, 1, 1 };
        int64_t h_strides[4] = { 0, 0, 0, 0 };
        for (int i = 0; i < ndim; ++i) {
            h_dims[i] = x.shape().dim(i);
        }
        h_strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            h_strides[i] = h_strides[i + 1] * h_dims[i + 1];
        }

        int64_t* d_dims;
        int64_t* d_strides;
        cudaMalloc(&d_dims, 4 * sizeof(int64_t));
        cudaMalloc(&d_strides, 4 * sizeof(int64_t));
        cudaMemcpy(d_dims, h_dims, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_strides, h_strides, 4 * sizeof(int64_t), cudaMemcpyHostToDevice);

        LaunchConfig config(total);
        cummin_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            const_cast<T*>(out.data<T>()), x.data<T>(),
            total, d_dims, d_strides, ndim, axis
            );

        cudaFree(d_dims);
        cudaFree(d_strides);
    }

    static OpArgs cummin_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& x = std::any_cast<const Array&>(args[1]);
        int axis = std::any_cast<int>(args[2]);

        switch (out.dtype()) {
        case DType::F32: cummin_impl<float>(out, x, axis); break;
        case DType::F64: cummin_impl<double>(out, x, axis); break;
        case DType::I32: cummin_impl<int32_t>(out, x, axis); break;
        case DType::I64: cummin_impl<int64_t>(out, x, axis); break;
        default: INS_THROW("cummin: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(cummin, GPU, F32, cummin_wrapper);
    REGISTER_KERNEL(cummin, GPU, F64, cummin_wrapper);
    REGISTER_KERNEL(cummin, GPU, I32, cummin_wrapper);
    REGISTER_KERNEL(cummin, GPU, I64, cummin_wrapper);

    // ============================================================================
    // nansum kernel
    // ============================================================================

    template<typename T>
    __global__ void nansum_kernel(T* sum_dst, int64_t* count_dst, const T* src, int64_t total_out, int64_t reduce_size) {
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < total_out) {
            T sum = T(0);
            int64_t cnt = 0;
            for (int64_t j = 0; j < reduce_size; ++j) {
                T val = src[i * reduce_size + j];
                if (!is_nan_device(val)) {
                    sum += val;
                    ++cnt;
                }
            }
            sum_dst[i] = sum;
            count_dst[i] = cnt;
        }
    }

    template<typename T>
    static void nansum_impl(const Array& sum_out, const Array& count_out, const Array& prepared) {
        int64_t total_out = sum_out.numel();
        int64_t reduce_size = prepared.numel() / total_out;
        LaunchConfig config(total_out);
        nansum_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            const_cast<T*>(sum_out.data<T>()), const_cast<int64_t*>(count_out.data<int64_t>()),
            prepared.data<T>(), total_out, reduce_size
            );
    }

    static OpArgs nansum_wrapper(const OpArgs& args) {
        const Array& sum_out = std::any_cast<const Array&>(args[0]);
        const Array& count_out = std::any_cast<const Array&>(args[1]);
        const Array& prepared = std::any_cast<const Array&>(args[2]);

        switch (prepared.dtype()) {
        case DType::F32: nansum_impl<float>(sum_out, count_out, prepared); break;
        case DType::F64: nansum_impl<double>(sum_out, count_out, prepared); break;
        default: INS_THROW("nansum: only float32/64 supported");
        }
        return { sum_out, count_out };
    }

    REGISTER_KERNEL(nansum, GPU, F32, nansum_wrapper);
    REGISTER_KERNEL(nansum, GPU, F64, nansum_wrapper);

    // ============================================================================
    // nanmax kernel
    // ============================================================================

    template<typename T>
    __global__ void nanmax_kernel(T* dst, const T* src, int64_t total_out, int64_t reduce_size) {
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < total_out) {
            T max_val = T(0);
            bool found = false;
            for (int64_t j = 0; j < reduce_size; ++j) {
                T val = src[i * reduce_size + j];
                if (!is_nan_device(val)) {
                    if (!found) {
                        max_val = val;
                        found = true;
                    }
                    else if (val > max_val) {
                        max_val = val;
                    }
                }
            }
            dst[i] = found ? max_val : device_nan<T>();
        }
    }

    template<typename T>
    static void nanmax_impl(const Array& out, const Array& prepared) {
        int64_t total_out = out.numel();
        int64_t reduce_size = prepared.numel() / total_out;
        LaunchConfig config(total_out);
        nanmax_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            const_cast<T*>(out.data<T>()), prepared.data<T>(), total_out, reduce_size
            );
    }

    static OpArgs nanmax_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);

        switch (prepared.dtype()) {
        case DType::F32: nanmax_impl<float>(out, prepared); break;
        case DType::F64: nanmax_impl<double>(out, prepared); break;
        default: INS_THROW("nanmax: only float32/64 supported");
        }
        return { out };
    }

    REGISTER_KERNEL(nanmax, GPU, F32, nanmax_wrapper);
    REGISTER_KERNEL(nanmax, GPU, F64, nanmax_wrapper);

    // ============================================================================
    // nanmin kernel
    // ============================================================================

    template<typename T>
    __global__ void nanmin_kernel(T* dst, const T* src, int64_t total_out, int64_t reduce_size) {
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < total_out) {
            T min_val = T(0);
            bool found = false;
            for (int64_t j = 0; j < reduce_size; ++j) {
                T val = src[i * reduce_size + j];
                if (!is_nan_device(val)) {
                    if (!found) {
                        min_val = val;
                        found = true;
                    }
                    else if (val < min_val) {
                        min_val = val;
                    }
                }
            }
            dst[i] = found ? min_val : device_nan<T>();
        }
    }

    template<typename T>
    static void nanmin_impl(const Array& out, const Array& prepared) {
        int64_t total_out = out.numel();
        int64_t reduce_size = prepared.numel() / total_out;
        LaunchConfig config(total_out);
        nanmin_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            const_cast<T*>(out.data<T>()), prepared.data<T>(), total_out, reduce_size
            );
    }

    static OpArgs nanmin_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);

        switch (prepared.dtype()) {
        case DType::F32: nanmin_impl<float>(out, prepared); break;
        case DType::F64: nanmin_impl<double>(out, prepared); break;
        default: INS_THROW("nanmin: only float32/64 supported");
        }
        return { out };
    }

    REGISTER_KERNEL(nanmin, GPU, F32, nanmin_wrapper);
    REGISTER_KERNEL(nanmin, GPU, F64, nanmin_wrapper);

    // ============================================================================
    // nanvar kernel
    // ============================================================================

    template<typename T>
    __global__ void nanvar_kernel(T* dst, const T* src, int64_t total_out, int64_t reduce_size, int ddof) {
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < total_out) {
            const T* row = src + i * reduce_size;
            int64_t count = 0;
            T sum = T(0);
            for (int64_t j = 0; j < reduce_size; ++j) {
                T val = row[j];
                if (!is_nan_device(val)) {
                    sum += val;
                    ++count;
                }
            }
            if (count == 0) {
                dst[i] = device_nan<T>();
                return;
            }
            T mean = sum / static_cast<T>(count);
            T sq_sum = T(0);
            for (int64_t j = 0; j < reduce_size; ++j) {
                T val = row[j];
                if (!is_nan_device(val)) {
                    T diff = val - mean;
                    sq_sum += diff * diff;
                }
            }
            T divisor = static_cast<T>(count - ddof);
            if (divisor <= T(0)) divisor = T(1);
            dst[i] = sq_sum / divisor;
        }
    }

    template<typename T>
    static void nanvar_impl(const Array& out, const Array& prepared, int ddof) {
        int64_t total_out = out.numel();
        int64_t reduce_size = prepared.numel() / total_out;
        LaunchConfig config(total_out);
        nanvar_kernel<T> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            const_cast<T*>(out.data<T>()), prepared.data<T>(), total_out, reduce_size, ddof
            );
    }

    static OpArgs nanvar_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);
        int ddof = std::any_cast<int>(args[4]);

        switch (prepared.dtype()) {
        case DType::F32: nanvar_impl<float>(out, prepared, ddof); break;
        case DType::F64: nanvar_impl<double>(out, prepared, ddof); break;
        default: INS_THROW("nanvar: only float32/64 supported");
        }
        return { out };
    }

    REGISTER_KERNEL(nanvar, GPU, F32, nanvar_wrapper);
    REGISTER_KERNEL(nanvar, GPU, F64, nanvar_wrapper);

#ifdef INSIGHT_USE_THRUST

    // ============================================================================
    // quantile (thrust path)
    // ============================================================================

    template<typename InT, typename OutT>
    __global__ void quantile_extract_kernel(
        OutT* dst, const InT* src, int64_t batch_size, int64_t reduce_size, double q
    ) {
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= batch_size) return;

        const InT* row = src + i * reduce_size;

        if (q == 0.5) {
            if (reduce_size % 2 == 1) {
                dst[i] = static_cast<OutT>(row[reduce_size / 2]);
            }
            else {
                InT a = row[reduce_size / 2 - 1];
                InT b = row[reduce_size / 2];
                dst[i] = static_cast<OutT>((a + b) / static_cast<InT>(2));
            }
        }
        else {
            double idx = q * (reduce_size - 1);
            int64_t lo = static_cast<int64_t>(idx);
            int64_t hi = lo + 1;
            if (hi >= reduce_size) {
                dst[i] = static_cast<OutT>(row[lo]);
            }
            else {
                double frac = idx - lo;
                InT a = row[lo];
                InT b = row[hi];
                dst[i] = static_cast<OutT>(a * (1 - frac) + b * frac);
            }
        }
    }

    template<typename InT, typename OutT>
    static void quantile_impl(const Array& out, const Array& prepared,
        int64_t batch_size, int64_t reduce_size, double q) {
        InT* src = const_cast<InT*>(prepared.data<InT>());

        // 每行排序
        for (int64_t i = 0; i < batch_size; ++i) {
            thrust::device_ptr<InT> row(src + i * reduce_size);
            thrust::sort(row, row + reduce_size);
        }

        // 在 GPU 上提取分位数结果
        LaunchConfig config(batch_size);
        quantile_extract_kernel<InT, OutT> << <config.blocks, config.threads, config.shmSize, config.stream >> > (
            const_cast<OutT*>(out.data<OutT>()), src, batch_size, reduce_size, q
            );
    }

    static OpArgs quantile_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);
        int64_t batch_size = std::any_cast<int64_t>(args[2]);
        int64_t reduce_size = std::any_cast<int64_t>(args[3]);
        double q = std::any_cast<double>(args[4]);

        DType in_dtype = prepared.dtype();
        DType out_dtype = out.dtype();

        if (in_dtype == DType::F32 && out_dtype == DType::F64) {
            quantile_impl<float, double>(out, prepared, batch_size, reduce_size, q);
        }
        else if (in_dtype == DType::F32 && out_dtype == DType::F32) {
            quantile_impl<float, float>(out, prepared, batch_size, reduce_size, q);
        }
        else if (in_dtype == DType::F64 && out_dtype == DType::F64) {
            quantile_impl<double, double>(out, prepared, batch_size, reduce_size, q);
        }
        else {
            INS_THROW("quantile: unsupported dtype combination");
        }
        return { out };
    }

    REGISTER_KERNEL(quantile, GPU, F32, quantile_wrapper);
    REGISTER_KERNEL(quantile, GPU, F64, quantile_wrapper);

    // ============================================================================
    // nanquantile (thrust path)
    // ============================================================================

    template<typename T>
    __global__ void nanquantile_compact_kernel(
        T* data, int64_t* valid_counts, int64_t batch_size, int64_t reduce_size
    ) {
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= batch_size) return;

        T* row = data + i * reduce_size;
        int64_t write_pos = 0;
        for (int64_t j = 0; j < reduce_size; ++j) {
            T val = row[j];
            bool is_nan = (val != val);
            if (!is_nan) {
                if (write_pos != j) {
                    row[write_pos] = val;
                }
                ++write_pos;
            }
        }
        // Fill tail with NaN
        for (int64_t j = write_pos; j < reduce_size; ++j) {
            row[j] = device_nan<T>();
        }
        valid_counts[i] = write_pos;
    }

    template<typename T>
    __global__ void nanquantile_extract_kernel(
        T* dst, const T* src, const int64_t* valid_counts,
        int64_t batch_size, int64_t reduce_size, double q
    ) {
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= batch_size) return;

        int64_t valid_count = valid_counts[i];
        if (valid_count == 0) {
            dst[i] = device_nan<T>();
            return;
        }

        const T* row = src + i * reduce_size;

        if (q == 0.5) {
            if (valid_count % 2 == 1) {
                dst[i] = row[valid_count / 2];
            }
            else {
                T a = row[valid_count / 2 - 1];
                T b = row[valid_count / 2];
                dst[i] = (a + b) / static_cast<T>(2);
            }
        }
        else {
            double idx = q * (valid_count - 1);
            int64_t lo = static_cast<int64_t>(idx);
            int64_t hi = lo + 1;
            if (hi >= valid_count) {
                dst[i] = row[lo];
            }
            else {
                double frac = idx - lo;
                dst[i] = static_cast<T>(row[lo] * (1.0 - frac) + row[hi] * frac);
            }
        }
    }

    template<typename T>
    static void nanquantile_impl(const Array& out, const Array& prepared,
        int64_t batch_size, int64_t reduce_size, double q) {
        T* src = const_cast<T*>(prepared.data<T>());

        // Step 1: Compact NaN values to front of each row
        int64_t* d_valid_counts;
        cudaMalloc(&d_valid_counts, batch_size * sizeof(int64_t));

        {
            LaunchConfig config_compact(batch_size);
            nanquantile_compact_kernel<T> << <config_compact.blocks, config_compact.threads,
                config_compact.shmSize, config_compact.stream >> > (
                    src, d_valid_counts, batch_size, reduce_size
                    );
        }
        cudaDeviceSynchronize();

        // Step 2: Sort each row (only valid portion)
        std::vector<int64_t> h_valid_counts(batch_size);
        cudaMemcpy(h_valid_counts.data(), d_valid_counts, batch_size * sizeof(int64_t), cudaMemcpyDeviceToHost);

        for (int64_t i = 0; i < batch_size; ++i) {
            if (h_valid_counts[i] > 0) {
                thrust::device_ptr<T> row(src + i * reduce_size);
                thrust::sort(row, row + h_valid_counts[i]);
            }
        }

        // Step 3: Extract quantile values on GPU
        {
            LaunchConfig config_extract(batch_size);
            nanquantile_extract_kernel<T> << <config_extract.blocks, config_extract.threads,
                config_extract.shmSize, config_extract.stream >> > (
                    const_cast<T*>(out.data<T>()), src, d_valid_counts,
                    batch_size, reduce_size, q
                    );
        }
        cudaDeviceSynchronize();

        cudaFree(d_valid_counts);
    }

    static OpArgs nanquantile_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& prepared = std::any_cast<const Array&>(args[1]);
        int64_t batch_size = std::any_cast<int64_t>(args[2]);
        int64_t reduce_size = std::any_cast<int64_t>(args[3]);
        double q = std::any_cast<double>(args[4]);

        switch (prepared.dtype()) {
        case DType::F32: nanquantile_impl<float>(out, prepared, batch_size, reduce_size, q); break;
        case DType::F64: nanquantile_impl<double>(out, prepared, batch_size, reduce_size, q); break;
        default: INS_THROW("nanquantile: only float32/64 supported");
        }
        return { out };
    }

    REGISTER_KERNEL(nanquantile, GPU, F32, nanquantile_wrapper);
    REGISTER_KERNEL(nanquantile, GPU, F64, nanquantile_wrapper);

#endif // INSIGHT_USE_THRUST

} // namespace ins::gpu

// ============================================================================
// Module Registration
// ============================================================================

REGISTER_MODULE(reduction, GPU);