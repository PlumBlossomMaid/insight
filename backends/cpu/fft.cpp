// backends/cpu/fft.cpp
#include "insight/plugin/op_registry.h"
#include "insight/core/array.h"
#include <fftw3.h>
#include <cmath>
#include <complex>
#include <vector>
#include <cstring>

namespace ins::cpu {

    // ============================================================================
    // Helper: Convert between Insight complex format [real, imag] and FFTW complex
    // ============================================================================

    // Insight complex format: interleaved real, imag (same as FFTW)
    // So no conversion needed for data layout, just pointer casting.
    // But we need to handle shape: Insight uses last dimension = 2,
    // while FFTW expects (..., n) complex numbers.

    static fftw_complex* to_fftw_complex(double* data) {
        return reinterpret_cast<fftw_complex*>(data);
    }

    static fftwf_complex* to_fftwf_complex(float* data) {
        return reinterpret_cast<fftwf_complex*>(data);
    }

    // ============================================================================
    // FFTW Plan Manager (thread-local, reuse plans)
    // ============================================================================

    struct FFTWPlanCache {
        fftw_plan plan_f64 = nullptr;
        fftwf_plan plan_f32 = nullptr;
        int64_t n = 0;
        int64_t batch = 0;
        int direction = FFTW_FORWARD;
        bool is_r2c = false;
        bool is_c2r = false;

        ~FFTWPlanCache() {
            if (plan_f64) fftw_destroy_plan(plan_f64);
            if (plan_f32) fftwf_destroy_plan(plan_f32);
        }
    };

    static thread_local FFTWPlanCache tls_plan_cache;

    static void ensure_plan_f64(int32_t n, int32_t batch, int direction, bool is_r2c, bool is_c2r) {

        if (tls_plan_cache.plan_f64 &&
            tls_plan_cache.n == n &&
            tls_plan_cache.batch == batch &&
            tls_plan_cache.direction == direction &&
            tls_plan_cache.is_r2c == is_r2c &&
            tls_plan_cache.is_c2r == is_c2r) {
            return;
        }

        if (tls_plan_cache.plan_f64) {
            fftw_destroy_plan(tls_plan_cache.plan_f64);
            tls_plan_cache.plan_f64 = nullptr;
        }

        tls_plan_cache.n = n;
        tls_plan_cache.batch = batch;
        tls_plan_cache.direction = direction;
        tls_plan_cache.is_r2c = is_r2c;
        tls_plan_cache.is_c2r = is_c2r;

        if (is_r2c) {
            // Real to complex
            double* dummy_in = static_cast<double*>(fftw_malloc(n * batch * sizeof(double)));
            fftw_complex* dummy_out = static_cast<fftw_complex*>(fftw_malloc((n / 2 + 1) * batch * sizeof(fftw_complex)));
            tls_plan_cache.plan_f64 = fftw_plan_many_dft_r2c(
                1, &n, batch,
                dummy_in, nullptr, 1, n,
                dummy_out, nullptr, 1, n / 2 + 1,
                FFTW_ESTIMATE);
            fftw_free(dummy_in);
            fftw_free(dummy_out);
        }
        else if (is_c2r) {
            // Complex to real
            fftw_complex* dummy_in = static_cast<fftw_complex*>(fftw_malloc((n / 2 + 1) * batch * sizeof(fftw_complex)));
            double* dummy_out = static_cast<double*>(fftw_malloc(n * batch * sizeof(double)));
            tls_plan_cache.plan_f64 = fftw_plan_many_dft_c2r(
                1, &n, batch,
                dummy_in, nullptr, 1, n / 2 + 1,
                dummy_out, nullptr, 1, n,
                FFTW_ESTIMATE);
            fftw_free(dummy_in);
            fftw_free(dummy_out);
        }
        else {
            // Complex to complex
            fftw_complex* dummy_in = static_cast<fftw_complex*>(fftw_malloc(n * batch * sizeof(fftw_complex)));
            fftw_complex* dummy_out = static_cast<fftw_complex*>(fftw_malloc(n * batch * sizeof(fftw_complex)));
            tls_plan_cache.plan_f64 = fftw_plan_many_dft(
                1, &n, batch,
                dummy_in, nullptr, 1, n,
                dummy_out, nullptr, 1, n,
                direction, FFTW_ESTIMATE);
            fftw_free(dummy_in);
            fftw_free(dummy_out);
        }
    }

    static void ensure_plan_f32(int32_t n, int32_t batch, int direction, bool is_r2c, bool is_c2r) {

        if (tls_plan_cache.plan_f32 &&
            tls_plan_cache.n == n &&
            tls_plan_cache.batch == batch &&
            tls_plan_cache.direction == direction &&
            tls_plan_cache.is_r2c == is_r2c &&
            tls_plan_cache.is_c2r == is_c2r) {
            return;
        }

        if (tls_plan_cache.plan_f32) {
            fftwf_destroy_plan(tls_plan_cache.plan_f32);
            tls_plan_cache.plan_f32 = nullptr;
        }

        tls_plan_cache.n = n;
        tls_plan_cache.batch = batch;
        tls_plan_cache.direction = direction;
        tls_plan_cache.is_r2c = is_r2c;
        tls_plan_cache.is_c2r = is_c2r;

        if (is_r2c) {
            // Real to complex
            float* dummy_in = static_cast<float*>(fftwf_malloc(n * batch * sizeof(float)));
            fftwf_complex* dummy_out = static_cast<fftwf_complex*>(fftwf_malloc((n / 2 + 1) * batch * sizeof(fftwf_complex)));
            tls_plan_cache.plan_f32 = fftwf_plan_many_dft_r2c(
                1, &n, batch,
                dummy_in, nullptr, 1, n,
                dummy_out, nullptr, 1, n / 2 + 1,
                FFTW_ESTIMATE);
            fftwf_free(dummy_in);
            fftwf_free(dummy_out);
        }
        else if (is_c2r) {
            // Complex to real
            fftwf_complex* dummy_in = static_cast<fftwf_complex*>(fftwf_malloc((n / 2 + 1) * batch * sizeof(fftwf_complex)));
            float* dummy_out = static_cast<float*>(fftwf_malloc(n * batch * sizeof(float)));
            tls_plan_cache.plan_f32 = fftwf_plan_many_dft_c2r(
                1, &n, batch,
                dummy_in, nullptr, 1, n / 2 + 1,
                dummy_out, nullptr, 1, n,
                FFTW_ESTIMATE);
            fftwf_free(dummy_in);
            fftwf_free(dummy_out);
        }
        else {
            // Complex to complex
            fftwf_complex* dummy_in = static_cast<fftwf_complex*>(fftwf_malloc(n * batch * sizeof(fftwf_complex)));
            fftwf_complex* dummy_out = static_cast<fftwf_complex*>(fftwf_malloc(n * batch * sizeof(fftwf_complex)));
            tls_plan_cache.plan_f32 = fftwf_plan_many_dft(
                1, &n, batch,
                dummy_in, nullptr, 1, n,
                dummy_out, nullptr, 1, n,
                direction, FFTW_ESTIMATE);
            fftwf_free(dummy_in);
            fftwf_free(dummy_out);
        }
    }

    // ============================================================================
    // Standard FFT (Complex to Complex)
    // ============================================================================
    static Array fft_c2c_impl(const Array& out, const Array& input,
        int64_t fft_len, int64_t batch_size,
        bool inverse, const std::string& norm) {
        int direction = inverse ? FFTW_BACKWARD : FFTW_FORWARD;
        int64_t total_complex = fft_len * batch_size;

        int64_t input_complex = input.numel();
        int64_t output_complex = out.numel();
        INS_CHECK(input_complex == total_complex,
            "fft_c2c_impl: input size mismatch. Expected ", total_complex,
            ", got ", input_complex);
        INS_CHECK(output_complex == total_complex,
            "fft_c2c_impl: output size mismatch. Expected ", total_complex,
            ", got ", output_complex);

        if (input.dtype() == DType::F64 || input.dtype() == DType::C64) {
            const double* src = input.data<double>();
            double* dst = const_cast<double*>(out.data<double>());

            std::memcpy(dst, src, total_complex * 2 * sizeof(double));

            ensure_plan_f64(fft_len, batch_size, direction, false, false);
            fftw_execute_dft(tls_plan_cache.plan_f64,
                to_fftw_complex(dst),
                to_fftw_complex(dst));
        }
        else if (input.dtype() == DType::F32 || input.dtype() == DType::C32) {
            const float* src = input.data<float>();
            float* dst = const_cast<float*>(out.data<float>());

            std::memcpy(dst, src, total_complex * 2 * sizeof(float));

            ensure_plan_f32(fft_len, batch_size, direction, false, false);
            fftwf_execute_dft(tls_plan_cache.plan_f32,
                to_fftwf_complex(dst),
                to_fftwf_complex(dst));
        }
        else {
            INS_THROW("fft_c2c_impl: unsupported dtype: ", dtype_name(input.dtype()));
        }

        return out;
    }

    // ============================================================================
    // Real FFT (Real to Complex)
    // ============================================================================

    static Array fft_r2c_impl(const Array& out, const Array& input,
        int64_t fft_len, int64_t batch_size,
        const std::string& norm) {
        int64_t out_len = fft_len / 2 + 1;

        if (input.dtype() == DType::F64) {
            const double* src = input.data<double>();
            double* dst = const_cast<double*>(out.data<double>());

            ensure_plan_f64(fft_len, batch_size, FFTW_FORWARD, true, false);
            fftw_execute_dft_r2c(tls_plan_cache.plan_f64,
                const_cast<double*>(src),
                to_fftw_complex(dst));
        }
        else {
            const float* src = input.data<float>();
            float* dst = const_cast<float*>(out.data<float>());

            ensure_plan_f32(fft_len, batch_size, FFTW_FORWARD, true, false);
            fftwf_execute_dft_r2c(tls_plan_cache.plan_f32,
                const_cast<float*>(src),
                to_fftwf_complex(dst));
        }

        return out;
    }

    // ============================================================================
    // Inverse Real FFT (Complex to Real)
    // ============================================================================

    static Array fft_c2r_impl(const Array& out, const Array& input,
        int64_t fft_len, int64_t batch_size,
        const std::string& norm) {
        int64_t in_len = fft_len / 2 + 1;  
        int64_t actual_in_len = input.shape().dim(input.shape().ndim() - 1);

        if (actual_in_len < in_len) {
            INS_CHECK(actual_in_len == in_len,
                "fft_c2r_impl: input length mismatch after prepare_input. Expected ", in_len,
                ", got ", actual_in_len);
        }

        if (input.dtype() == DType::F64 || input.dtype() == DType::C64) {
            const double* src = input.data<double>();
            double* dst = const_cast<double*>(out.data<double>());

            ensure_plan_f64(fft_len, batch_size, FFTW_BACKWARD, false, true);
            fftw_execute_dft_c2r(tls_plan_cache.plan_f64,
                to_fftw_complex(const_cast<double*>(src)),
                dst);
        }
        else if (input.dtype() == DType::F32 || input.dtype() == DType::C32) {
            const float* src = input.data<float>();
            float* dst = const_cast<float*>(out.data<float>());

            ensure_plan_f32(fft_len, batch_size, FFTW_BACKWARD, false, true);
            fftwf_execute_dft_c2r(tls_plan_cache.plan_f32,
                to_fftwf_complex(const_cast<float*>(src)),
                dst);
        }
        else {
            INS_THROW("fft_c2r_impl: unsupported dtype");
        }

        return out;
    }

    // ============================================================================
    // Kernel registration (C2C)
    // ============================================================================

    static OpArgs fft_c2c_kernel(const OpArgs& args) {

        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& input = std::any_cast<const Array&>(args[1]);
        int64_t fft_len = std::any_cast<int64_t>(args[2]);
        int64_t batch_size = std::any_cast<int64_t>(args[3]);
        bool inverse = std::any_cast<bool>(args[4]);
        bool real_input = std::any_cast<bool>(args[5]);
        std::string norm = std::any_cast<std::string>(args[6]);

        Array& mutable_out = const_cast<Array&>(out);
        Array result = fft_c2c_impl(mutable_out, input, fft_len, batch_size, inverse, norm);

        return { result };
    }

    REGISTER_KERNEL(fft, CPU, F32, fft_c2c_kernel);
    REGISTER_KERNEL(fft, CPU, F64, fft_c2c_kernel);
    REGISTER_KERNEL(fft, CPU, C32, fft_c2c_kernel);
    REGISTER_KERNEL(fft, CPU, C64, fft_c2c_kernel);

    // ============================================================================
    // Kernel registration (R2C - rfft)
    // ============================================================================

    static OpArgs fft_r2c_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& input = std::any_cast<const Array&>(args[1]);
        int64_t fft_len = std::any_cast<int64_t>(args[2]);
        int64_t batch_size = std::any_cast<int64_t>(args[3]);
        bool inverse = std::any_cast<bool>(args[4]);  // should be false
        bool real_input = std::any_cast<bool>(args[5]);
        std::string norm = std::any_cast<std::string>(args[6]);

        Array& mutable_out = const_cast<Array&>(out);
        return { fft_r2c_impl(mutable_out, input, fft_len, batch_size, norm) };
    }

    REGISTER_KERNEL(rfft, CPU, F32, fft_r2c_kernel);
    REGISTER_KERNEL(rfft, CPU, F64, fft_r2c_kernel);

    // ============================================================================
    // Kernel registration (C2R - irfft)
    // ============================================================================

    static OpArgs fft_c2r_kernel(const OpArgs& args) {

        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& input = std::any_cast<const Array&>(args[1]);
        int64_t fft_len = std::any_cast<int64_t>(args[2]);
        int64_t batch_size = std::any_cast<int64_t>(args[3]);
        bool inverse = std::any_cast<bool>(args[4]);
        bool real_input = std::any_cast<bool>(args[5]);
        std::string norm = std::any_cast<std::string>(args[6]);

        Array& mutable_out = const_cast<Array&>(out);
        Array result = fft_c2r_impl(mutable_out, input, fft_len, batch_size, norm);

        return { result };
    }

    REGISTER_KERNEL(irfft, CPU, C32, fft_c2r_kernel);
    REGISTER_KERNEL(irfft, CPU, C64, fft_c2r_kernel);

    // ============================================================================
    // Module registration
    // ============================================================================

    REGISTER_MODULE(fft, CPU);

} // namespace ins::cpu