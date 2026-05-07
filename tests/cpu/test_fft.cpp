// tests/cpu/test_fft.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include "insight/insight.h"
#include "insight/ops/fft.h"

using namespace ins;

class FFTTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ins::init();
        set_device(CPUPlace());
        seed(42);
    }
};

// ============================================================================
// Helper functions
// ============================================================================

static Array create_test_data(const Shape& shape, DType dtype) {
    if (shape.numel() == 0) return Array();
    if (dtype == DType::F32) {
        return rand(shape, DType::F32);
    }
    else {
        return randn(shape, DType::F64);
    }
}

static void compare_arrays(const Array& a, const Array& b, double rtol, double atol) {
    int64_t n = a.numel();
    const double* a_data = a.data<double>();
    const double* b_data = b.data<double>();

    double max_diff = 0.0;
    double max_val = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double diff = std::abs(a_data[i] - b_data[i]);
        if (diff > max_diff) max_diff = diff;
        double val = std::max(std::abs(a_data[i]), std::abs(b_data[i]));
        if (val > max_val) max_val = val;
    }

    EXPECT_LE(max_diff, atol + rtol * max_val);
}

static void expect_close(const Array& a, const Array& b, double rtol = 1e-5, double atol = 1e-7) {
    Array a_cpu = a.to(CPUPlace());
    Array b_cpu = b.to(CPUPlace());

    if (is_complex(a_cpu) && is_complex(b_cpu)) {
        if (a_cpu.dtype() != DType::F32) a_cpu = a_cpu.to(DType::F32);
        if (b_cpu.dtype() != DType::F32) b_cpu = b_cpu.to(DType::F32);
        a_cpu = a_cpu.to(DType::F64);
        b_cpu = b_cpu.to(DType::F64);
        compare_arrays(a_cpu, b_cpu, rtol, atol);
        return;
    }

    if (is_complex(a_cpu) || is_complex(b_cpu)) {
        INS_THROW("expect_close: cannot compare complex with non-complex");
    }

    if (a_cpu.dtype() != DType::F64) a_cpu = a_cpu.to(DType::F64);
    if (b_cpu.dtype() != DType::F64) b_cpu = b_cpu.to(DType::F64);
    compare_arrays(a_cpu, b_cpu, rtol, atol);
}

// ============================================================================
// Test cases
// ============================================================================

TEST_F(FFTTest, RFFT_IRFFT_Reconstruction) {
    int n = 8;
    Array x = create_test_data({ n }, DType::F32);
    Array X = fft::rfft(x);
    Array x_recon = fft::irfft(X, n);
    expect_close(x, x_recon);
}

TEST_F(FFTTest, FFT_IFFT_Reconstruction) {
    int n = 8;
    Array x_real = create_test_data({ n }, DType::F32);
    Array x = to_complex(x_real);
    Array X = fft::fft(x);
    Array x_recon = fft::ifft(X);
    expect_close(x, x_recon);
}

TEST_F(FFTTest, FFT2_IFFT2_Reconstruction) {
    Array x_real = create_test_data({ 8, 8 }, DType::F32);
    Array x = to_complex(x_real);
    Array X = fft::fft2(x);
    Array x_recon = fft::ifft2(X);
    Array x_recon_real = real(x_recon);
    expect_close(x_real, x_recon_real);
}

TEST_F(FFTTest, FFTN_IFFTN_Reconstruction) {
    Array x = create_test_data({ 4, 4, 4 }, DType::F32);
    Array X = fft::fftn(x);
    Array x_recon = fft::ifftn(X);
    x_recon = real(x_recon);
    expect_close(x, x_recon);
}

TEST_F(FFTTest, RFFTN_IRFFTN_Reconstruction) {
    Array x = create_test_data({ 4, 5, 6 }, DType::F32);
    Array X = fft::rfftn(x);
    Array x_recon = fft::irfftn(X, { 4, 5, 6 });
    expect_close(x, x_recon);
}

TEST_F(FFTTest, FFT_Linearity) {
    int n = 8;
    Array a = create_test_data({ n }, DType::F32);
    Array b = create_test_data({ n }, DType::F32);

    Array a_c = to_complex(a);
    Array b_c = to_complex(b);

    Array sum_fft = fft::fft(ins::add(a_c, b_c));
    Array fft_sum = ins::add(fft::fft(a_c), fft::fft(b_c));

    expect_close(sum_fft, fft_sum);
}

TEST_F(FFTTest, RFFT_WithNParameter) {
    int n = 8;
    int n_pad = 16;
    Array x = create_test_data({ n }, DType::F32);
    Array X = fft::rfft(x, n_pad);

    int64_t expected_len = n_pad / 2 + 1;
    EXPECT_EQ(X.numel(), expected_len);
    EXPECT_TRUE(is_complex(X));
}

TEST_F(FFTTest, FFT_AlongAxis) {
    Array x = create_test_data({ 8, 4 }, DType::F32);
    Array x_c = to_complex(x);

    Array X0 = fft::fft(x_c, -1, 0);
    EXPECT_EQ(X0.shape().ndim(), 2);
    EXPECT_EQ(X0.shape().dim(0), 8);
    EXPECT_EQ(X0.shape().dim(1), 4);

    Array X1 = fft::fft(x_c, -1, 1);
    EXPECT_EQ(X1.shape().ndim(), 2);
    EXPECT_EQ(X1.shape().dim(0), 8);
    EXPECT_EQ(X1.shape().dim(1), 4);
}

TEST_F(FFTTest, FFTFreq) {
    int n = 8;
    double d = 0.5;
    Array f = fft::fftfreq(n, d);

    EXPECT_EQ(f.numel(), n);
    double inv = 1.0 / (d * n);
    EXPECT_NEAR(f.at(0).item<double>(), 0.0, 1e-10);
    EXPECT_NEAR(f.at(1).item<double>(), inv, 1e-10);
}

TEST_F(FFTTest, RFFTFreq) {
    int n = 8;
    double d = 0.5;
    Array f = fft::rfftfreq(n, d);

    int64_t expected_len = n / 2 + 1;
    EXPECT_EQ(f.numel(), expected_len);
    double inv = 1.0 / (d * n);
    EXPECT_NEAR(f.at(0).item<double>(), 0.0, 1e-10);
    EXPECT_NEAR(f.at(1).item<double>(), inv, 1e-10);
}

TEST_F(FFTTest, NextFastLen) {
    EXPECT_EQ(fft::next_fast_len(1), 1);
    EXPECT_EQ(fft::next_fast_len(11), 12);
    EXPECT_EQ(fft::next_fast_len(31), 32);
    EXPECT_EQ(fft::next_fast_len(1009), 1024);
}

TEST_F(FFTTest, FFTShift) {
    int n = 8;
    Array x = arange(0.0, static_cast<double>(n), 1.0).to(DType::F64);
    Array y = fft::fftshift(x, 0);

    EXPECT_NEAR(y.at(0).item<double>(), 4.0, 1e-10);
    EXPECT_NEAR(y.at(4).item<double>(), 0.0, 1e-10);
}

TEST_F(FFTTest, IFFTShift) {
    int n = 8;
    Array x = arange(0.0, static_cast<double>(n), 1.0).to(DType::F64);
    Array y = fft::ifftshift(x, 0);

    EXPECT_NEAR(y.at(0).item<double>(), 4.0, 1e-10);
    EXPECT_NEAR(y.at(4).item<double>(), 0.0, 1e-10);
}

TEST_F(FFTTest, HFFT_IFFT_Hermitian) {
    int n = 8;
    Array x = create_test_data({ n }, DType::F32);

    Array X = fft::ihfft(x);
    Array x_recon = fft::hfft(X, n);

    expect_close(x, x_recon);
}

TEST_F(FFTTest, NormBackward) {
    int n = 8;
    Array x = create_test_data({ n }, DType::F32);
    Array X = fft::rfft(x, -1, -1, "backward");
    Array x_recon = fft::irfft(X, n, -1, "backward");
    expect_close(x, x_recon);
}

TEST_F(FFTTest, NormForward) {
    int n = 8;
    Array x = create_test_data({ n }, DType::F32);
    Array X = fft::rfft(x, -1, -1, "forward");
    Array x_recon = fft::irfft(X, n, -1, "forward");
    expect_close(x, x_recon);
}

TEST_F(FFTTest, NormOrtho) {
    int n = 8;
    Array x = create_test_data({ n }, DType::F32);
    Array X = fft::rfft(x, -1, -1, "ortho");
    Array x_recon = fft::irfft(X, n, -1, "ortho");
    expect_close(x, x_recon);
}

TEST_F(FFTTest, ErrorRFFTComplexInput) {
    Array x = to_complex(rand({ 8 }, DType::F32));
    EXPECT_THROW(fft::rfft(x), Exception);
}

TEST_F(FFTTest, ErrorIRFFTRealInput) {
    Array x = rand({ 8 }, DType::F32);
    EXPECT_THROW(fft::irfft(x), Exception);
}