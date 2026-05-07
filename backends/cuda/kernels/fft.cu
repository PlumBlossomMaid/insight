// backends/cuda/kernels/fft.cu
#include "insight/plugin/op_registry.h"
#include "insight/core/array.h"
#include "insight/core/exception.h"
#include "insight/core/launch_config.h"
#include <cufft.h>
#include <cmath>
#include <complex>
#include <map>

namespace inner {
    static const char* cufft_error_string(cufftResult result) {
        switch (result) {
        case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
        case CUFFT_INCOMPLETE_PARAMETER_LIST: return "CUFFT_INCOMPLETE_PARAMETER_LIST";
        case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
        case CUFFT_PARSE_ERROR: return "CUFFT_PARSE_ERROR";
        case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
        case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
        case CUFFT_LICENSE_ERROR: return "CUFFT_LICENSE_ERROR";
        case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
        default: return "UNKNOWN_CUFFT_ERROR";
        }
    }
}

#define CUFFT_CHECK(call) do { \
    cufftResult result = (call); \
    if (result != CUFFT_SUCCESS) { \
        INS_THROW("cuFFT error at ", __FILE__, ":", __LINE__, ": ", \
                  inner::cufft_error_string(result)); \
    } \
} while(0)

namespace ins::gpu {

    // ============================================================================
    // cuFFT Plan Cache (per-device)
    // ============================================================================

    struct CUFFTPlanKey {
        int64_t n;
        int64_t batch;
        int device_id;
        cufftType type;

        bool operator<(const CUFFTPlanKey& other) const {
            if (n != other.n) return n < other.n;
            if (batch != other.batch) return batch < other.batch;
            if (device_id != other.device_id) return device_id < other.device_id;
            return type < other.type;
        }
    };

    static std::map<CUFFTPlanKey, cufftHandle> g_plan_cache;

    static cufftHandle get_or_create_plan(int64_t n, int64_t batch, cufftType type) {
        int device_id;
        cudaGetDevice(&device_id);

        CUFFTPlanKey key{ n, batch, device_id, type };

        auto it = g_plan_cache.find(key);
        if (it != g_plan_cache.end()) {
            return it->second;
        }

        cufftHandle plan;
        if (type == CUFFT_R2C || type == CUFFT_D2Z) {
            CUFFT_CHECK(cufftPlan1d(&plan, static_cast<int>(n), type, static_cast<int>(batch)));
        }
        else if (type == CUFFT_C2R || type == CUFFT_Z2D) {
            CUFFT_CHECK(cufftPlan1d(&plan, static_cast<int>(n), type, static_cast<int>(batch)));
        }
        else {
            CUFFT_CHECK(cufftPlan1d(&plan, static_cast<int>(n), type, static_cast<int>(batch)));
        }

        g_plan_cache.insert({ key, plan });
        return plan;
    }

    // ============================================================================
    // FFT C2C (Complex to Complex)
    // ============================================================================

    static void fft_c2c_impl(const Array& out, const Array& input,
        int64_t fft_len, int64_t batch_size, bool inverse, const std::string& norm) {

        int direction = inverse ? CUFFT_INVERSE : CUFFT_FORWARD;

        int ndim = input.shape().ndim();
        const Array* work_input = &input;
        Array* work_out = const_cast<Array*>(&out);
        Array temp_in, temp_out;
        std::vector<int> rev_perm;

        // 检测：如果最后一维 stride 不是 1，需要手动转置
        bool need_reorder = false;
        if (ndim >= 2 && input.strides()[ndim - 1] != 1) {
            need_reorder = true;

            // 找 stride=1 的维度
            int fft_axis = -1;
            for (int i = 0; i < ndim; ++i) {
                if (input.strides()[i] == 1) {
                    fft_axis = i;
                    break;
                }
            }

            // 构造转置排列：把 fft_axis 移到最后一维
            std::vector<int> perm(ndim);
            int idx = 0;
            for (int i = 0; i < ndim; ++i) {
                if (i != fft_axis) perm[idx++] = i;
            }
            perm[ndim - 1] = fft_axis;

            // 构造逆转置排列
            rev_perm.resize(ndim);
            for (int i = 0; i < ndim; ++i) {
                rev_perm[perm[i]] = i;
            }

            temp_in = input.transpose(perm).contiguous();
            temp_out = Array(temp_in.shape(), temp_in.dtype(), temp_in.place());
            work_input = &temp_in;
            work_out = &temp_out;
        }

        // Copy input to output
        size_t elem_size = (input.dtype() == DType::F64 || input.dtype() == DType::C64)
            ? sizeof(cufftDoubleComplex) : sizeof(cufftComplex);
        size_t total_bytes = work_input->numel() * elem_size;
        cudaMemcpy(const_cast<void*>(static_cast<const void*>(work_out->data())),
            work_input->data(), total_bytes, cudaMemcpyDeviceToDevice);

        cufftType type;
        if (input.dtype() == DType::F64 || input.dtype() == DType::C64) {
            type = CUFFT_Z2Z;
        }
        else {
            type = CUFFT_C2C;
        }

        cufftHandle plan = get_or_create_plan(fft_len, batch_size, type);

        if (input.dtype() == DType::F64 || input.dtype() == DType::C64) {
            CUFFT_CHECK(cufftExecZ2Z(plan,
                const_cast<cufftDoubleComplex*>(reinterpret_cast<const cufftDoubleComplex*>(work_out->data())),
                reinterpret_cast<cufftDoubleComplex*>(const_cast<void*>(static_cast<const void*>(work_out->data()))),
                direction));
        }
        else {
            CUFFT_CHECK(cufftExecC2C(plan,
                const_cast<cufftComplex*>(reinterpret_cast<const cufftComplex*>(work_out->data())),
                reinterpret_cast<cufftComplex*>(const_cast<void*>(static_cast<const void*>(work_out->data()))),
                direction));
        }

        // 逆转置
        if (need_reorder) {
            Array result = work_out->transpose(rev_perm).contiguous();
            cudaMemcpy(const_cast<void*>(static_cast<const void*>(out.data())),
                result.data(),
                out.numel() * elem_size,
                cudaMemcpyDeviceToDevice);
        }
    }

    // ============================================================================
    // FFT R2C (Real to Complex)
    // ============================================================================

    static void fft_r2c_impl(const Array& out, const Array& input,
        int64_t fft_len, int64_t batch_size, const std::string& norm) {

        cufftType type = (input.dtype() == DType::F64) ? CUFFT_D2Z : CUFFT_R2C;
        cufftHandle plan = get_or_create_plan(fft_len, batch_size, type);

        if (input.dtype() == DType::F64) {
            CUFFT_CHECK(cufftExecD2Z(plan,
                const_cast<cufftDoubleReal*>(reinterpret_cast<const cufftDoubleReal*>(input.data())),
                reinterpret_cast<cufftDoubleComplex*>(const_cast<void*>(static_cast<const void*>(out.data())))));
        }
        else {
            CUFFT_CHECK(cufftExecR2C(plan,
                const_cast<cufftReal*>(reinterpret_cast<const cufftReal*>(input.data())),
                reinterpret_cast<cufftComplex*>(const_cast<void*>(static_cast<const void*>(out.data())))));
        }
    }

    // ============================================================================
    // FFT C2R (Complex to Real)
    // ============================================================================

    static void fft_c2r_impl(const Array& out, const Array& input,
        int64_t fft_len, int64_t batch_size, const std::string& norm) {

        cufftType type = (input.dtype() == DType::F64 || input.dtype() == DType::C64) ? CUFFT_Z2D : CUFFT_C2R;
        cufftHandle plan = get_or_create_plan(fft_len, batch_size, type);

        if (input.dtype() == DType::F64 || input.dtype() == DType::C64) {
            CUFFT_CHECK(cufftExecZ2D(plan,
                const_cast<cufftDoubleComplex*>(reinterpret_cast<const cufftDoubleComplex*>(input.data())),
                reinterpret_cast<cufftDoubleReal*>(const_cast<void*>(static_cast<const void*>(out.data())))));
        }
        else {
            CUFFT_CHECK(cufftExecC2R(plan,
                const_cast<cufftComplex*>(reinterpret_cast<const cufftComplex*>(input.data())),
                reinterpret_cast<cufftReal*>(const_cast<void*>(static_cast<const void*>(out.data())))));
        }
    }

    // ============================================================================
    // Wrapper functions
    // ============================================================================

    static OpArgs fft_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& input = std::any_cast<const Array&>(args[1]);
        int64_t fft_len = std::any_cast<int64_t>(args[2]);
        int64_t batch_size = std::any_cast<int64_t>(args[3]);
        bool inverse = std::any_cast<bool>(args[4]);
        bool real_input = std::any_cast<bool>(args[5]);
        std::string norm = std::any_cast<std::string>(args[6]);

        if (real_input) {
            fft_r2c_impl(out, input, fft_len, batch_size, norm);
        }
        else {
            fft_c2c_impl(out, input, fft_len, batch_size, inverse, norm);
        }

        return { out };
    }

    static OpArgs rfft_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& input = std::any_cast<const Array&>(args[1]);
        int64_t fft_len = std::any_cast<int64_t>(args[2]);
        int64_t batch_size = std::any_cast<int64_t>(args[3]);
        std::string norm = std::any_cast<std::string>(args[6]);

        fft_r2c_impl(out, input, fft_len, batch_size, norm);

        return { out };
    }

    static OpArgs irfft_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& input = std::any_cast<const Array&>(args[1]);
        int64_t fft_len = std::any_cast<int64_t>(args[2]);
        int64_t batch_size = std::any_cast<int64_t>(args[3]);
        std::string norm = std::any_cast<std::string>(args[6]);

        fft_c2r_impl(out, input, fft_len, batch_size, norm);

        return { out };
    }

    // ============================================================================
    // Kernel Registration
    // ============================================================================

    REGISTER_KERNEL(fft, GPU, F32, fft_wrapper);
    REGISTER_KERNEL(fft, GPU, F64, fft_wrapper);
    REGISTER_KERNEL(fft, GPU, C32, fft_wrapper);
    REGISTER_KERNEL(fft, GPU, C64, fft_wrapper);

    REGISTER_KERNEL(rfft, GPU, F32, rfft_wrapper);
    REGISTER_KERNEL(rfft, GPU, F64, rfft_wrapper);

    REGISTER_KERNEL(irfft, GPU, C32, irfft_wrapper);
    REGISTER_KERNEL(irfft, GPU, C64, irfft_wrapper);

} // namespace ins::gpu

// ============================================================================
// Module Registration
// ============================================================================

REGISTER_MODULE(fft, GPU);