// insight/ops/fft.h
#pragma once
#include <vector>
#include <string>
#include "insight/core/array.h"

namespace ins {
    namespace fft {

        // ============================================================================
        // Standard FFT (Complex <-> Complex)
        // ============================================================================

        /**
         * @brief 1D discrete Fourier transform.
         *
         * Computes the FFT along the specified axis.
         *
         * @param x Input array (real or complex). Complex input must have last dimension size 2.
         * @param n FFT length. If n > input length, zero-padding; if n < input length, truncation.
         *        If n == -1, use input length.
         * @param axis Axis to perform FFT on. Negative values wrap around.
         * @param norm Normalization mode: "backward", "forward", or "ortho".
         * @return Complex output array with shape determined by n.
         */
        Array fft(const Array& x, int n = -1, int axis = -1, const std::string& norm = "backward");

        /**
         * @brief 1D inverse discrete Fourier transform.
         */
        Array ifft(const Array& x, int n = -1, int axis = -1, const std::string& norm = "backward");

        /**
         * @brief 2D discrete Fourier transform.
         */
        Array fft2(const Array& x, const std::vector<int64_t>& s = {},
            const std::vector<int>& axes = { -2, -1 }, const std::string& norm = "backward");

        /**
         * @brief 2D inverse discrete Fourier transform.
         */
        Array ifft2(const Array& x, const std::vector<int64_t>& s = {},
            const std::vector<int>& axes = { -2, -1 }, const std::string& norm = "backward");

        /**
         * @brief N-dimensional discrete Fourier transform.
         */
        Array fftn(const Array& x, const std::vector<int64_t>& s = {},
            const std::vector<int>& axes = {}, const std::string& norm = "backward");

        /**
         * @brief N-dimensional inverse discrete Fourier transform.
         */
        Array ifftn(const Array& x, const std::vector<int64_t>& s = {},
            const std::vector<int>& axes = {}, const std::string& norm = "backward");

        // ============================================================================
        // Real FFT (Real <-> Complex)
        // ============================================================================

        /**
         * @brief 1D real FFT.
         */
        Array rfft(const Array& x, int n = -1, int axis = -1, const std::string& norm = "backward");

        /**
         * @brief 1D inverse real FFT.
         */
        Array irfft(const Array& x, int n = -1, int axis = -1, const std::string& norm = "backward");

        /**
         * @brief 2D real FFT.
         */
        Array rfft2(const Array& x, const std::vector<int64_t>& s = {},
            const std::vector<int>& axes = { -2, -1 }, const std::string& norm = "backward");

        /**
         * @brief 2D inverse real FFT.
         */
        Array irfft2(const Array& x, const std::vector<int64_t>& s = {},
            const std::vector<int>& axes = { -2, -1 }, const std::string& norm = "backward");

        /**
         * @brief N-dimensional real FFT.
         */
        Array rfftn(const Array& x, const std::vector<int64_t>& s = {},
            const std::vector<int>& axes = {}, const std::string& norm = "backward");

        /**
         * @brief N-dimensional inverse real FFT.
         */
        Array irfftn(const Array& x, const std::vector<int64_t>& s = {},
            const std::vector<int>& axes = {}, const std::string& norm = "backward");

        // ============================================================================
        // Hermitian FFT (Hermitian symmetric complex <-> Real)
        // ============================================================================

        /**
         * @brief 1D Hermitian FFT (complex Hermitian input -> real output).
         */
        Array hfft(const Array& x, int n = -1, int axis = -1, const std::string& norm = "backward");

        /**
         * @brief 1D inverse Hermitian FFT (real input -> complex Hermitian output).
         */
        Array ihfft(const Array& x, int n = -1, int axis = -1, const std::string& norm = "backward");

        /**
         * @brief 2D Hermitian FFT.
         */
        Array hfft2(const Array& x, const std::vector<int64_t>& s = {},
            const std::vector<int>& axes = { -2, -1 }, const std::string& norm = "backward");

        /**
         * @brief 2D inverse Hermitian FFT.
         */
        Array ihfft2(const Array& x, const std::vector<int64_t>& s = {},
            const std::vector<int>& axes = { -2, -1 }, const std::string& norm = "backward");

        /**
         * @brief N-dimensional Hermitian FFT.
         */
        Array hfftn(const Array& x, const std::vector<int64_t>& s = {},
            const std::vector<int>& axes = {}, const std::string& norm = "backward");

        /**
         * @brief N-dimensional inverse Hermitian FFT.
         */
        Array ihfftn(const Array& x, const std::vector<int64_t>& s = {},
            const std::vector<int>& axes = {}, const std::string& norm = "backward");

        // ============================================================================
        // Helper functions
        // ============================================================================

        /**
         * @brief Compute FFT sample frequencies.
         */
        Array fftfreq(int64_t n, double d = 1.0);

        /**
         * @brief Compute sample frequencies for rfft.
         */
        Array rfftfreq(int64_t n, double d = 1.0);

        /**
         * @brief Shift zero-frequency component to center of spectrum.
         */
        Array fftshift(const Array& x, int axis = -1);

        /**
         * @brief Inverse of fftshift.
         */
        Array ifftshift(const Array& x, int axis = -1);

        /**
         * @brief Find the smallest fast FFT length >= target.
         */
        int next_fast_len(int target);

    } // namespace fft
} // namespace ins