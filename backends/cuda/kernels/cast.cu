// backends/cuda/kernels/cast.cu
#include "insight/plugin/op_registry.h"
#include "insight/core/array.h"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>

namespace ins::gpu {

    // ========== Helper: Kernel Launch Config ==========

    static inline void launch_config(int64_t n, int& blocks, int& threads) {
        threads = 256;
        blocks = static_cast<int>((n + threads - 1) / threads);
    }

    // ========== Generic Cast Kernel ==========

    template<typename SrcT, typename DstT>
    __global__ void cast_kernel(const SrcT* src, DstT* dst, int64_t n) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = static_cast<DstT>(src[idx]);
        }
    }

    // ========== Bool Specializations ==========

    template<typename DstT>
    __global__ void cast_bool_to_kernel(const bool* src, DstT* dst, int64_t n) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = src[idx] ? static_cast<DstT>(1) : static_cast<DstT>(0);
        }
    }

    template<typename SrcT>
    __global__ void cast_to_bool_kernel(const SrcT* src, bool* dst, int64_t n) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = (src[idx] != SrcT(0));
        }
    }

    // ========== Float/Complex Conversions ==========

    __global__ void cast_f32_to_c32_kernel(const float* src, cuFloatComplex* dst, int64_t n) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = make_cuFloatComplex(src[idx], 0.0f);
        }
    }

    __global__ void cast_f64_to_c64_kernel(const double* src, cuDoubleComplex* dst, int64_t n) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = make_cuDoubleComplex(src[idx], 0.0);
        }
    }

    __global__ void cast_c32_to_f32_kernel(const cuFloatComplex* src, float* dst, int64_t n) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = cuCrealf(src[idx]);
        }
    }

    __global__ void cast_c32_to_f64_kernel(const cuFloatComplex* src, double* dst, int64_t n) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = static_cast<double>(cuCrealf(src[idx]));
        }
    }

    __global__ void cast_c32_to_bool_kernel(const cuFloatComplex* src, bool* dst, int64_t n) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = (cuCrealf(src[idx]) != 0.0f || cuCimagf(src[idx]) != 0.0f);
        }
    }

    __global__ void cast_c32_to_c64_kernel(const cuFloatComplex* src, cuDoubleComplex* dst, int64_t n) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = make_cuDoubleComplex(
                static_cast<double>(cuCrealf(src[idx])),
                static_cast<double>(cuCimagf(src[idx]))
            );
        }
    }

    __global__ void cast_c64_to_f64_kernel(const cuDoubleComplex* src, double* dst, int64_t n) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = cuCreal(src[idx]);
        }
    }

    __global__ void cast_c64_to_f32_kernel(const cuDoubleComplex* src, float* dst, int64_t n) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = static_cast<float>(cuCreal(src[idx]));
        }
    }

    __global__ void cast_c64_to_bool_kernel(const cuDoubleComplex* src, bool* dst, int64_t n) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = (cuCreal(src[idx]) != 0.0 || cuCimag(src[idx]) != 0.0);
        }
    }

    __global__ void cast_c64_to_c32_kernel(const cuDoubleComplex* src, cuFloatComplex* dst, int64_t n) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = make_cuFloatComplex(
                static_cast<float>(cuCreal(src[idx])),
                static_cast<float>(cuCimag(src[idx]))
            );
        }
    }

    // ========== Cast Implementation ==========

    static Array cast_impl(const Array& x, DType target_dtype) {
        Array out(x.shape(), target_dtype, x.place());
        int64_t n = x.numel();
        int blocks, threads;
        launch_config(n, blocks, threads);

        DType src_dtype = x.dtype();

        if (src_dtype == target_dtype) {
            return x.copy();
        }

        switch (src_dtype) {
            // ========== From BOOL ==========
        case DType::BOOL: {
            switch (target_dtype) {
            case DType::U8:
                cast_bool_to_kernel<uint8_t> << <blocks, threads >> > (x.data<bool>(), out.data<uint8_t>(), n);
                break;
            case DType::I8:
                cast_bool_to_kernel<int8_t> << <blocks, threads >> > (x.data<bool>(), out.data<int8_t>(), n);
                break;
            case DType::I16:
                cast_bool_to_kernel<int16_t> << <blocks, threads >> > (x.data<bool>(), out.data<int16_t>(), n);
                break;
            case DType::I32:
                cast_bool_to_kernel<int32_t> << <blocks, threads >> > (x.data<bool>(), out.data<int32_t>(), n);
                break;
            case DType::I64:
                cast_bool_to_kernel<int64_t> << <blocks, threads >> > (x.data<bool>(), out.data<int64_t>(), n);
                break;
            case DType::U16:
                cast_bool_to_kernel<uint16_t> << <blocks, threads >> > (x.data<bool>(), out.data<uint16_t>(), n);
                break;
            case DType::U32:
                cast_bool_to_kernel<uint32_t> << <blocks, threads >> > (x.data<bool>(), out.data<uint32_t>(), n);
                break;
            case DType::U64:
                cast_bool_to_kernel<uint64_t> << <blocks, threads >> > (x.data<bool>(), out.data<uint64_t>(), n);
                break;
            case DType::F16:
                cast_bool_to_kernel<uint16_t> << <blocks, threads >> > (x.data<bool>(), out.data<uint16_t>(), n);
                break;
            case DType::BF16:
                cast_bool_to_kernel<uint16_t> << <blocks, threads >> > (x.data<bool>(), out.data<uint16_t>(), n);
                break;
            case DType::F32:
                cast_bool_to_kernel<float> << <blocks, threads >> > (x.data<bool>(), out.data<float>(), n);
                break;
            case DType::F64:
                cast_bool_to_kernel<double> << <blocks, threads >> > (x.data<bool>(), out.data<double>(), n);
                break;
            case DType::C32:
                cast_bool_to_kernel<cuFloatComplex> << <blocks, threads >> > (x.data<bool>(), out.data<cuFloatComplex>(), n);
                break;
            case DType::C64:
                cast_bool_to_kernel<cuDoubleComplex> << <blocks, threads >> > (x.data<bool>(), out.data<cuDoubleComplex>(), n);
                break;
            default:
                INS_THROW("cast: unsupported target type from bool");
            }
            cudaDeviceSynchronize();
            return out;
        }

                        // ========== From U8 ==========
        case DType::U8: {
            using SrcT = uint8_t;
            switch (target_dtype) {
            case DType::BOOL:
                cast_to_bool_kernel<SrcT> << <blocks, threads >> > (x.data<SrcT>(), out.data<bool>(), n);
                break;
            case DType::I8:
                cast_kernel<SrcT, int8_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int8_t>(), n);
                break;
            case DType::I16:
                cast_kernel<SrcT, int16_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int16_t>(), n);
                break;
            case DType::I32:
                cast_kernel<SrcT, int32_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int32_t>(), n);
                break;
            case DType::I64:
                cast_kernel<SrcT, int64_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int64_t>(), n);
                break;
            case DType::U16:
                cast_kernel<SrcT, uint16_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint16_t>(), n);
                break;
            case DType::U32:
                cast_kernel<SrcT, uint32_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint32_t>(), n);
                break;
            case DType::U64:
                cast_kernel<SrcT, uint64_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint64_t>(), n);
                break;
            case DType::F32:
                cast_kernel<SrcT, float> << <blocks, threads >> > (x.data<SrcT>(), out.data<float>(), n);
                break;
            case DType::F64:
                cast_kernel<SrcT, double> << <blocks, threads >> > (x.data<SrcT>(), out.data<double>(), n);
                break;
            case DType::C32:
                cast_kernel<SrcT, cuFloatComplex> << <blocks, threads >> > (x.data<SrcT>(), out.data<cuFloatComplex>(), n);
                break;
            case DType::C64:
                cast_kernel<SrcT, cuDoubleComplex> << <blocks, threads >> > (x.data<SrcT>(), out.data<cuDoubleComplex>(), n);
                break;
            default:
                INS_THROW("cast: unsupported target type from uint8");
            }
            cudaDeviceSynchronize();
            return out;
        }

                      // ========== From I8 ==========
        case DType::I8: {
            using SrcT = int8_t;
            switch (target_dtype) {
            case DType::BOOL:
                cast_to_bool_kernel<SrcT> << <blocks, threads >> > (x.data<SrcT>(), out.data<bool>(), n);
                break;
            case DType::U8:
                cast_kernel<SrcT, uint8_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint8_t>(), n);
                break;
            case DType::I16:
                cast_kernel<SrcT, int16_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int16_t>(), n);
                break;
            case DType::I32:
                cast_kernel<SrcT, int32_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int32_t>(), n);
                break;
            case DType::I64:
                cast_kernel<SrcT, int64_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int64_t>(), n);
                break;
            case DType::U16:
                cast_kernel<SrcT, uint16_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint16_t>(), n);
                break;
            case DType::U32:
                cast_kernel<SrcT, uint32_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint32_t>(), n);
                break;
            case DType::U64:
                cast_kernel<SrcT, uint64_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint64_t>(), n);
                break;
            case DType::F32:
                cast_kernel<SrcT, float> << <blocks, threads >> > (x.data<SrcT>(), out.data<float>(), n);
                break;
            case DType::F64:
                cast_kernel<SrcT, double> << <blocks, threads >> > (x.data<SrcT>(), out.data<double>(), n);
                break;
            case DType::C32:
                cast_kernel<SrcT, cuFloatComplex> << <blocks, threads >> > (x.data<SrcT>(), out.data<cuFloatComplex>(), n);
                break;
            case DType::C64:
                cast_kernel<SrcT, cuDoubleComplex> << <blocks, threads >> > (x.data<SrcT>(), out.data<cuDoubleComplex>(), n);
                break;
            default:
                INS_THROW("cast: unsupported target type from int8");
            }
            cudaDeviceSynchronize();
            return out;
        }

                      // ========== From I16 ==========
        case DType::I16: {
            using SrcT = int16_t;
            switch (target_dtype) {
            case DType::BOOL:
                cast_to_bool_kernel<SrcT> << <blocks, threads >> > (x.data<SrcT>(), out.data<bool>(), n);
                break;
            case DType::U8:
                cast_kernel<SrcT, uint8_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint8_t>(), n);
                break;
            case DType::I8:
                cast_kernel<SrcT, int8_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int8_t>(), n);
                break;
            case DType::I32:
                cast_kernel<SrcT, int32_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int32_t>(), n);
                break;
            case DType::I64:
                cast_kernel<SrcT, int64_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int64_t>(), n);
                break;
            case DType::U16:
                cast_kernel<SrcT, uint16_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint16_t>(), n);
                break;
            case DType::U32:
                cast_kernel<SrcT, uint32_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint32_t>(), n);
                break;
            case DType::U64:
                cast_kernel<SrcT, uint64_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint64_t>(), n);
                break;
            case DType::F32:
                cast_kernel<SrcT, float> << <blocks, threads >> > (x.data<SrcT>(), out.data<float>(), n);
                break;
            case DType::F64:
                cast_kernel<SrcT, double> << <blocks, threads >> > (x.data<SrcT>(), out.data<double>(), n);
                break;
            case DType::C32:
                cast_kernel<SrcT, cuFloatComplex> << <blocks, threads >> > (x.data<SrcT>(), out.data<cuFloatComplex>(), n);
                break;
            case DType::C64:
                cast_kernel<SrcT, cuDoubleComplex> << <blocks, threads >> > (x.data<SrcT>(), out.data<cuDoubleComplex>(), n);
                break;
            default:
                INS_THROW("cast: unsupported target type from int16");
            }
            cudaDeviceSynchronize();
            return out;
        }

                       // ========== From I32 ==========
        case DType::I32: {
            using SrcT = int32_t;
            switch (target_dtype) {
            case DType::BOOL:
                cast_to_bool_kernel<SrcT> << <blocks, threads >> > (x.data<SrcT>(), out.data<bool>(), n);
                break;
            case DType::U8:
                cast_kernel<SrcT, uint8_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint8_t>(), n);
                break;
            case DType::I8:
                cast_kernel<SrcT, int8_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int8_t>(), n);
                break;
            case DType::I16:
                cast_kernel<SrcT, int16_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int16_t>(), n);
                break;
            case DType::I64:
                cast_kernel<SrcT, int64_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int64_t>(), n);
                break;
            case DType::U16:
                cast_kernel<SrcT, uint16_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint16_t>(), n);
                break;
            case DType::U32:
                cast_kernel<SrcT, uint32_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint32_t>(), n);
                break;
            case DType::U64:
                cast_kernel<SrcT, uint64_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint64_t>(), n);
                break;
            case DType::F32:
                cast_kernel<SrcT, float> << <blocks, threads >> > (x.data<SrcT>(), out.data<float>(), n);
                break;
            case DType::F64:
                cast_kernel<SrcT, double> << <blocks, threads >> > (x.data<SrcT>(), out.data<double>(), n);
                break;
            case DType::C32:
                cast_kernel<SrcT, cuFloatComplex> << <blocks, threads >> > (x.data<SrcT>(), out.data<cuFloatComplex>(), n);
                break;
            case DType::C64:
                cast_kernel<SrcT, cuDoubleComplex> << <blocks, threads >> > (x.data<SrcT>(), out.data<cuDoubleComplex>(), n);
                break;
            default:
                INS_THROW("cast: unsupported target type from int32");
            }
            cudaDeviceSynchronize();
            return out;
        }

                       // ========== From I64 ==========
        case DType::I64: {
            using SrcT = int64_t;
            switch (target_dtype) {
            case DType::BOOL:
                cast_to_bool_kernel<SrcT> << <blocks, threads >> > (x.data<SrcT>(), out.data<bool>(), n);
                break;
            case DType::U8:
                cast_kernel<SrcT, uint8_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint8_t>(), n);
                break;
            case DType::I8:
                cast_kernel<SrcT, int8_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int8_t>(), n);
                break;
            case DType::I16:
                cast_kernel<SrcT, int16_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int16_t>(), n);
                break;
            case DType::I32:
                cast_kernel<SrcT, int32_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int32_t>(), n);
                break;
            case DType::U16:
                cast_kernel<SrcT, uint16_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint16_t>(), n);
                break;
            case DType::U32:
                cast_kernel<SrcT, uint32_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint32_t>(), n);
                break;
            case DType::U64:
                cast_kernel<SrcT, uint64_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint64_t>(), n);
                break;
            case DType::F32:
                cast_kernel<SrcT, float> << <blocks, threads >> > (x.data<SrcT>(), out.data<float>(), n);
                break;
            case DType::F64:
                cast_kernel<SrcT, double> << <blocks, threads >> > (x.data<SrcT>(), out.data<double>(), n);
                break;
            case DType::C32:
                cast_kernel<SrcT, cuFloatComplex> << <blocks, threads >> > (x.data<SrcT>(), out.data<cuFloatComplex>(), n);
                break;
            case DType::C64:
                cast_kernel<SrcT, cuDoubleComplex> << <blocks, threads >> > (x.data<SrcT>(), out.data<cuDoubleComplex>(), n);
                break;
            default:
                INS_THROW("cast: unsupported target type from int64");
            }
            cudaDeviceSynchronize();
            return out;
        }

                       // ========== From U16 ==========
        case DType::U16: {
            using SrcT = uint16_t;
            switch (target_dtype) {
            case DType::BOOL:
                cast_to_bool_kernel<SrcT> << <blocks, threads >> > (x.data<SrcT>(), out.data<bool>(), n);
                break;
            case DType::U8:
                cast_kernel<SrcT, uint8_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint8_t>(), n);
                break;
            case DType::I8:
                cast_kernel<SrcT, int8_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int8_t>(), n);
                break;
            case DType::I16:
                cast_kernel<SrcT, int16_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int16_t>(), n);
                break;
            case DType::I32:
                cast_kernel<SrcT, int32_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int32_t>(), n);
                break;
            case DType::I64:
                cast_kernel<SrcT, int64_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int64_t>(), n);
                break;
            case DType::U32:
                cast_kernel<SrcT, uint32_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint32_t>(), n);
                break;
            case DType::U64:
                cast_kernel<SrcT, uint64_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint64_t>(), n);
                break;
            case DType::F32:
                cast_kernel<SrcT, float> << <blocks, threads >> > (x.data<SrcT>(), out.data<float>(), n);
                break;
            case DType::F64:
                cast_kernel<SrcT, double> << <blocks, threads >> > (x.data<SrcT>(), out.data<double>(), n);
                break;
            case DType::C32:
                cast_kernel<SrcT, cuFloatComplex> << <blocks, threads >> > (x.data<SrcT>(), out.data<cuFloatComplex>(), n);
                break;
            case DType::C64:
                cast_kernel<SrcT, cuDoubleComplex> << <blocks, threads >> > (x.data<SrcT>(), out.data<cuDoubleComplex>(), n);
                break;
            default:
                INS_THROW("cast: unsupported target type from uint16");
            }
            cudaDeviceSynchronize();
            return out;
        }

                       // ========== From U32 ==========
        case DType::U32: {
            using SrcT = uint32_t;
            switch (target_dtype) {
            case DType::BOOL:
                cast_to_bool_kernel<SrcT> << <blocks, threads >> > (x.data<SrcT>(), out.data<bool>(), n);
                break;
            case DType::U8:
                cast_kernel<SrcT, uint8_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint8_t>(), n);
                break;
            case DType::I8:
                cast_kernel<SrcT, int8_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int8_t>(), n);
                break;
            case DType::I16:
                cast_kernel<SrcT, int16_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int16_t>(), n);
                break;
            case DType::I32:
                cast_kernel<SrcT, int32_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int32_t>(), n);
                break;
            case DType::I64:
                cast_kernel<SrcT, int64_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int64_t>(), n);
                break;
            case DType::U16:
                cast_kernel<SrcT, uint16_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint16_t>(), n);
                break;
            case DType::U64:
                cast_kernel<SrcT, uint64_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint64_t>(), n);
                break;
            case DType::F32:
                cast_kernel<SrcT, float> << <blocks, threads >> > (x.data<SrcT>(), out.data<float>(), n);
                break;
            case DType::F64:
                cast_kernel<SrcT, double> << <blocks, threads >> > (x.data<SrcT>(), out.data<double>(), n);
                break;
            case DType::C32:
                cast_kernel<SrcT, cuFloatComplex> << <blocks, threads >> > (x.data<SrcT>(), out.data<cuFloatComplex>(), n);
                break;
            case DType::C64:
                cast_kernel<SrcT, cuDoubleComplex> << <blocks, threads >> > (x.data<SrcT>(), out.data<cuDoubleComplex>(), n);
                break;
            default:
                INS_THROW("cast: unsupported target type from uint32");
            }
            cudaDeviceSynchronize();
            return out;
        }

                       // ========== From U64 ==========
        case DType::U64: {
            using SrcT = uint64_t;
            switch (target_dtype) {
            case DType::BOOL:
                cast_to_bool_kernel<SrcT> << <blocks, threads >> > (x.data<SrcT>(), out.data<bool>(), n);
                break;
            case DType::U8:
                cast_kernel<SrcT, uint8_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint8_t>(), n);
                break;
            case DType::I8:
                cast_kernel<SrcT, int8_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int8_t>(), n);
                break;
            case DType::I16:
                cast_kernel<SrcT, int16_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int16_t>(), n);
                break;
            case DType::I32:
                cast_kernel<SrcT, int32_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int32_t>(), n);
                break;
            case DType::I64:
                cast_kernel<SrcT, int64_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int64_t>(), n);
                break;
            case DType::U16:
                cast_kernel<SrcT, uint16_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint16_t>(), n);
                break;
            case DType::U32:
                cast_kernel<SrcT, uint32_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint32_t>(), n);
                break;
            case DType::F32:
                cast_kernel<SrcT, float> << <blocks, threads >> > (x.data<SrcT>(), out.data<float>(), n);
                break;
            case DType::F64:
                cast_kernel<SrcT, double> << <blocks, threads >> > (x.data<SrcT>(), out.data<double>(), n);
                break;
            case DType::C32:
                cast_kernel<SrcT, cuFloatComplex> << <blocks, threads >> > (x.data<SrcT>(), out.data<cuFloatComplex>(), n);
                break;
            case DType::C64:
                cast_kernel<SrcT, cuDoubleComplex> << <blocks, threads >> > (x.data<SrcT>(), out.data<cuDoubleComplex>(), n);
                break;
            default:
                INS_THROW("cast: unsupported target type from uint64");
            }
            cudaDeviceSynchronize();
            return out;
        }

                       // ========== From F32 ==========
        case DType::F32: {
            using SrcT = float;
            switch (target_dtype) {
            case DType::BOOL:
                cast_to_bool_kernel<SrcT> << <blocks, threads >> > (x.data<SrcT>(), out.data<bool>(), n);
                break;
            case DType::U8:
                cast_kernel<SrcT, uint8_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint8_t>(), n);
                break;
            case DType::I8:
                cast_kernel<SrcT, int8_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int8_t>(), n);
                break;
            case DType::I16:
                cast_kernel<SrcT, int16_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int16_t>(), n);
                break;
            case DType::I32:
                cast_kernel<SrcT, int32_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int32_t>(), n);
                break;
            case DType::I64:
                cast_kernel<SrcT, int64_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int64_t>(), n);
                break;
            case DType::U16:
                cast_kernel<SrcT, uint16_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint16_t>(), n);
                break;
            case DType::U32:
                cast_kernel<SrcT, uint32_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint32_t>(), n);
                break;
            case DType::U64:
                cast_kernel<SrcT, uint64_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint64_t>(), n);
                break;
            case DType::F64:
                cast_kernel<SrcT, double> << <blocks, threads >> > (x.data<SrcT>(), out.data<double>(), n);
                break;
            case DType::C32:
                cast_f32_to_c32_kernel << <blocks, threads >> > (x.data<SrcT>(), out.data<cuFloatComplex>(), n);
                break;
            case DType::C64:
                cast_kernel<SrcT, cuDoubleComplex> << <blocks, threads >> > (x.data<SrcT>(), out.data<cuDoubleComplex>(), n);
                break;
            default:
                INS_THROW("cast: unsupported target type from float32");
            }
            cudaDeviceSynchronize();
            return out;
        }

                       // ========== From F64 ==========
        case DType::F64: {
            using SrcT = double;
            switch (target_dtype) {
            case DType::BOOL:
                cast_to_bool_kernel<SrcT> << <blocks, threads >> > (x.data<SrcT>(), out.data<bool>(), n);
                break;
            case DType::U8:
                cast_kernel<SrcT, uint8_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint8_t>(), n);
                break;
            case DType::I8:
                cast_kernel<SrcT, int8_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int8_t>(), n);
                break;
            case DType::I16:
                cast_kernel<SrcT, int16_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int16_t>(), n);
                break;
            case DType::I32:
                cast_kernel<SrcT, int32_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int32_t>(), n);
                break;
            case DType::I64:
                cast_kernel<SrcT, int64_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<int64_t>(), n);
                break;
            case DType::U16:
                cast_kernel<SrcT, uint16_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint16_t>(), n);
                break;
            case DType::U32:
                cast_kernel<SrcT, uint32_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint32_t>(), n);
                break;
            case DType::U64:
                cast_kernel<SrcT, uint64_t> << <blocks, threads >> > (x.data<SrcT>(), out.data<uint64_t>(), n);
                break;
            case DType::F32:
                cast_kernel<SrcT, float> << <blocks, threads >> > (x.data<SrcT>(), out.data<float>(), n);
                break;
            case DType::C32:
                cast_kernel<SrcT, cuFloatComplex> << <blocks, threads >> > (x.data<SrcT>(), out.data<cuFloatComplex>(), n);
                break;
            case DType::C64:
                cast_f64_to_c64_kernel << <blocks, threads >> > (x.data<SrcT>(), out.data<cuDoubleComplex>(), n);
                break;
            default:
                INS_THROW("cast: unsupported target type from float64");
            }
            cudaDeviceSynchronize();
            return out;
        }

                       // ========== From C32 ==========
        case DType::C32: {
            switch (target_dtype) {
            case DType::BOOL:
                cast_c32_to_bool_kernel << <blocks, threads >> > (x.data<cuFloatComplex>(), out.data<bool>(), n);
                break;
            case DType::F32:
                cast_c32_to_f32_kernel << <blocks, threads >> > (x.data<cuFloatComplex>(), out.data<float>(), n);
                break;
            case DType::F64:
                cast_c32_to_f64_kernel << <blocks, threads >> > (x.data<cuFloatComplex>(), out.data<double>(), n);
                break;
            case DType::C64:
                cast_c32_to_c64_kernel << <blocks, threads >> > (x.data<cuFloatComplex>(), out.data<cuDoubleComplex>(), n);
                break;
            default:
                INS_THROW("cast: unsupported target type from complex64");
            }
            cudaDeviceSynchronize();
            return out;
        }

                       // ========== From C64 ==========
        case DType::C64: {
            switch (target_dtype) {
            case DType::BOOL:
                cast_c64_to_bool_kernel << <blocks, threads >> > (x.data<cuDoubleComplex>(), out.data<bool>(), n);
                break;
            case DType::F32:
                cast_c64_to_f32_kernel << <blocks, threads >> > (x.data<cuDoubleComplex>(), out.data<float>(), n);
                break;
            case DType::F64:
                cast_c64_to_f64_kernel << <blocks, threads >> > (x.data<cuDoubleComplex>(), out.data<double>(), n);
                break;
            case DType::C32:
                cast_c64_to_c32_kernel << <blocks, threads >> > (x.data<cuDoubleComplex>(), out.data<cuFloatComplex>(), n);
                break;
            default:
                INS_THROW("cast: unsupported target type from complex128");
            }
            cudaDeviceSynchronize();
            return out;
        }

        default:
            INS_THROW("cast: unsupported source type");
        }

        return out;
    }

    // ========== Wrapper Function ==========

    static OpArgs cast_gpu_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        DType target_dtype = std::any_cast<DType>(args[1]);

        if (x.dtype() == target_dtype) {
            return { x };
        }

        Array result = cast_impl(x, target_dtype);
        return { result };
    }

    // ========== Registration ==========

    REGISTER_KERNEL(cast, GPU, BOOL, cast_gpu_kernel);
    REGISTER_KERNEL(cast, GPU, U8, cast_gpu_kernel);
    REGISTER_KERNEL(cast, GPU, I8, cast_gpu_kernel);
    REGISTER_KERNEL(cast, GPU, I16, cast_gpu_kernel);
    REGISTER_KERNEL(cast, GPU, I32, cast_gpu_kernel);
    REGISTER_KERNEL(cast, GPU, I64, cast_gpu_kernel);
    REGISTER_KERNEL(cast, GPU, U16, cast_gpu_kernel);
    REGISTER_KERNEL(cast, GPU, U32, cast_gpu_kernel);
    REGISTER_KERNEL(cast, GPU, U64, cast_gpu_kernel);
    REGISTER_KERNEL(cast, GPU, F32, cast_gpu_kernel);
    REGISTER_KERNEL(cast, GPU, F64, cast_gpu_kernel);
    REGISTER_KERNEL(cast, GPU, C32, cast_gpu_kernel);
    REGISTER_KERNEL(cast, GPU, C64, cast_gpu_kernel);
} // namespace ins::gpu

REGISTER_MODULE(cast, GPU);