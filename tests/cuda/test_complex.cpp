// tests/cuda/test_complex.cu
#include <gtest/gtest.h>
#include <complex>
#include "insight/insight.h"
#include "insight/ops/complex.h"

using namespace ins;

class ComplexTestGPU : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ins::init();
        set_device(GPUPlace(0));
    }
};

// ============================================================================
// is_complex tests (using dtype, not shape)
// ============================================================================

TEST_F(ComplexTestGPU, IsComplexReturnsFalseForReal) {
    Array real = ones({ 3, 4 }, DType::F32, GPUPlace(0));
    EXPECT_FALSE(is_complex(real));
}

TEST_F(ComplexTestGPU, IsComplexReturnsTrueForComplex) {
    Array real = ones({ 3, 4 }, DType::F32, GPUPlace(0));
    Array complex = to_complex(real);
    EXPECT_TRUE(is_complex(complex));
}

TEST_F(ComplexTestGPU, IsComplexReturnsFalseForScalar) {
    Array scalar = full({}, 1.0f, DType::F32, GPUPlace(0));
    EXPECT_FALSE(is_complex(scalar));
}

// ============================================================================
// has_complex_shape tests (checks last dimension = 2)
// ============================================================================

TEST_F(ComplexTestGPU, HasComplexShapeReturnsFalseForReal) {
    Array real = ones({ 3, 4 }, DType::F32, GPUPlace(0));
    EXPECT_FALSE(has_complex_shape(real));
}

TEST_F(ComplexTestGPU, HasComplexShapeReturnsFalseForNativeComplex) {
    Array real = ones({ 3, 4 }, DType::F32, GPUPlace(0));
    Array complex = to_complex(real);
    EXPECT_FALSE(has_complex_shape(complex));
}

// ============================================================================
// to_complex (single argument) tests
// ============================================================================

TEST_F(ComplexTestGPU, ToComplexSingleArgumentShape) {
    Array real = to_array({ 1.0f, 2.0f, 3.0f }, Shape({ 3 }), DType::F32, GPUPlace(0));
    Array z = to_complex(real);

    EXPECT_EQ(z.shape().ndim(), 1);
    EXPECT_EQ(z.shape().dim(0), 3);
    EXPECT_EQ(z.dtype(), DType::C32);
}

TEST_F(ComplexTestGPU, ToComplexSingleArgumentValues) {
    Array real = to_array({ 1.0f, 2.0f, 3.0f }, Shape({ 3 }), DType::F32, GPUPlace(0));
    Array z = to_complex(real);

    Array z_cpu = z.to(CPUPlace());
    const std::complex<float>* data = z_cpu.data<std::complex<float>>();
    EXPECT_FLOAT_EQ(data[0].real(), 1.0f);
    EXPECT_FLOAT_EQ(data[0].imag(), 0.0f);
    EXPECT_FLOAT_EQ(data[1].real(), 2.0f);
    EXPECT_FLOAT_EQ(data[1].imag(), 0.0f);
    EXPECT_FLOAT_EQ(data[2].real(), 3.0f);
    EXPECT_FLOAT_EQ(data[2].imag(), 0.0f);
}

TEST_F(ComplexTestGPU, ToComplexSingleArgument2D) {
    Array real = arange(0.0, 6.0, 1.0, DType::F32, GPUPlace(0)).reshape({ 2, 3 });
    Array z = to_complex(real);

    EXPECT_EQ(z.shape().ndim(), 2);
    EXPECT_EQ(z.shape().dim(0), 2);
    EXPECT_EQ(z.shape().dim(1), 3);
    EXPECT_EQ(z.dtype(), DType::C32);
}

// ============================================================================
// to_complex (two arguments) tests
// ============================================================================

TEST_F(ComplexTestGPU, ToComplexTwoArgumentsShape) {
    Array real = to_array({ 1.0f, 2.0f, 3.0f }, Shape({ 3 }), DType::F32, GPUPlace(0));
    Array imag = to_array({ 4.0f, 5.0f, 6.0f }, Shape({ 3 }), DType::F32, GPUPlace(0));
    Array z = to_complex(real, imag);

    EXPECT_EQ(z.shape().ndim(), 1);
    EXPECT_EQ(z.shape().dim(0), 3);
    EXPECT_EQ(z.dtype(), DType::C32);
}

TEST_F(ComplexTestGPU, ToComplexTwoArgumentsValues) {
    Array real = to_array({ 1.0f, 2.0f, 3.0f }, Shape({ 3 }), DType::F32, GPUPlace(0));
    Array imag = to_array({ 4.0f, 5.0f, 6.0f }, Shape({ 3 }), DType::F32, GPUPlace(0));
    Array z = to_complex(real, imag);

    Array z_cpu = z.to(CPUPlace());
    const std::complex<float>* data = z_cpu.data<std::complex<float>>();
    EXPECT_FLOAT_EQ(data[0].real(), 1.0f);
    EXPECT_FLOAT_EQ(data[0].imag(), 4.0f);
    EXPECT_FLOAT_EQ(data[1].real(), 2.0f);
    EXPECT_FLOAT_EQ(data[1].imag(), 5.0f);
    EXPECT_FLOAT_EQ(data[2].real(), 3.0f);
    EXPECT_FLOAT_EQ(data[2].imag(), 6.0f);
}

TEST_F(ComplexTestGPU, ToComplexTwoArgumentsShapeMismatchThrows) {
    Array real = to_array({ 1.0f, 2.0f, 3.0f }, Shape({ 3 }), DType::F32, GPUPlace(0));
    Array imag = to_array({ 4.0f, 5.0f }, Shape({ 2 }), DType::F32, GPUPlace(0));

    EXPECT_THROW(to_complex(real, imag), Exception);
}

// ============================================================================
// real() and imag() tests
// ============================================================================

TEST_F(ComplexTestGPU, RealExtractsRealPart) {
    Array real = to_array({ 1.0f, 2.0f, 3.0f }, Shape({ 3 }), DType::F32, GPUPlace(0));
    Array imag = to_array({ 4.0f, 5.0f, 6.0f }, Shape({ 3 }), DType::F32, GPUPlace(0));
    Array z = to_complex(real, imag);

    Array r = ins::real(z);

    EXPECT_EQ(r.shape().ndim(), 1);
    EXPECT_EQ(r.shape().dim(0), 3);
    Array r_cpu = r.to(CPUPlace());
    EXPECT_FLOAT_EQ(r_cpu.at(0).item<float>(), 1.0f);
    EXPECT_FLOAT_EQ(r_cpu.at(1).item<float>(), 2.0f);
    EXPECT_FLOAT_EQ(r_cpu.at(2).item<float>(), 3.0f);
}

TEST_F(ComplexTestGPU, ImagExtractsImagPart) {
    Array real = to_array({ 1.0f, 2.0f, 3.0f }, Shape({ 3 }), DType::F32, GPUPlace(0));
    Array imag = to_array({ 4.0f, 5.0f, 6.0f }, Shape({ 3 }), DType::F32, GPUPlace(0));
    Array z = to_complex(real, imag);

    Array i = ins::imag(z);

    EXPECT_EQ(i.shape().ndim(), 1);
    EXPECT_EQ(i.shape().dim(0), 3);
    Array i_cpu = i.to(CPUPlace());
    EXPECT_FLOAT_EQ(i_cpu.at(0).item<float>(), 4.0f);
    EXPECT_FLOAT_EQ(i_cpu.at(1).item<float>(), 5.0f);
    EXPECT_FLOAT_EQ(i_cpu.at(2).item<float>(), 6.0f);
}

TEST_F(ComplexTestGPU, RealAndImagValues) {
    Array real_arr = to_array({ 1.0f, 2.0f, 3.0f }, Shape({ 3 }), DType::F32, GPUPlace(0));
    Array imag_arr = to_array({ 4.0f, 5.0f, 6.0f }, Shape({ 3 }), DType::F32, GPUPlace(0));

    Array z = to_complex(real_arr, imag_arr);
    Array r = real(z);
    Array i = imag(z);

    Array r_cpu = r.to(CPUPlace());
    Array i_cpu = i.to(CPUPlace());
    EXPECT_FLOAT_EQ(r_cpu.at(0).item<float>(), 1.0f);
    EXPECT_FLOAT_EQ(r_cpu.at(1).item<float>(), 2.0f);
    EXPECT_FLOAT_EQ(r_cpu.at(2).item<float>(), 3.0f);
    EXPECT_FLOAT_EQ(i_cpu.at(0).item<float>(), 4.0f);
    EXPECT_FLOAT_EQ(i_cpu.at(1).item<float>(), 5.0f);
    EXPECT_FLOAT_EQ(i_cpu.at(2).item<float>(), 6.0f);
}

// ============================================================================
// as_complex tests
// ============================================================================

TEST_F(ComplexTestGPU, AsComplexShape) {
    std::vector<float> data = {
        1, 4,   2, 5,   3, 6,
        7, 10,  8, 11,  9, 12
    };
    Array x = to_array(data, Shape({ 2, 3, 2 }), DType::F32, GPUPlace(0));

    Array z = as_complex(x);

    EXPECT_EQ(z.shape().ndim(), 2);
    EXPECT_EQ(z.shape().dim(0), 2);
    EXPECT_EQ(z.shape().dim(1), 3);
    EXPECT_EQ(z.dtype(), DType::C32);
}

TEST_F(ComplexTestGPU, AsComplexDtype) {
    std::vector<double> data = {
        1, 4,   2, 5,   3, 6,
        7, 10,  8, 11,  9, 12
    };
    Array x = to_array(data, Shape({ 2, 3, 2 }), DType::F64, GPUPlace(0));

    Array z = as_complex(x);

    EXPECT_EQ(z.dtype(), DType::C64);
}

TEST_F(ComplexTestGPU, AsComplexInvalidInputThrows) {
    Array x = ones({ 3, 4 }, DType::F32, GPUPlace(0));
    EXPECT_THROW(as_complex(x), Exception);
}

// ============================================================================
// as_real tests
// ============================================================================

TEST_F(ComplexTestGPU, AsRealShape) {
    std::vector<float> data = {
        1, 4,   2, 5,   3, 6,
        7, 10,  8, 11,  9, 12
    };
    Array c = to_array(data, Shape({ 2, 3, 2 }), DType::F32, GPUPlace(0));
    Array z = as_complex(c);

    Array x = as_real(z);

    EXPECT_EQ(x.shape().ndim(), 3);
    EXPECT_EQ(x.shape().dim(0), 2);
    EXPECT_EQ(x.shape().dim(1), 3);
    EXPECT_EQ(x.shape().dim(2), 2);
    EXPECT_EQ(x.dtype(), DType::F32);
}

TEST_F(ComplexTestGPU, AsRealRoundTrip) {
    std::vector<float> original_data = {
        1, 4,   2, 5,   3, 6,
        7, 10,  8, 11,  9, 12
    };
    Array x = to_array(original_data, Shape({ 2, 3, 2 }), DType::F32, GPUPlace(0));

    Array z = as_complex(x);
    Array x2 = as_real(z);

    Array x_cpu = x.to(CPUPlace());
    Array x2_cpu = x2.to(CPUPlace());
    const float* x_data = x_cpu.data<float>();
    const float* x2_data = x2_cpu.data<float>();
    for (int i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(x_data[i], x2_data[i]);
    }
}

// ============================================================================
// Memory sharing tests
// ============================================================================

TEST_F(ComplexTestGPU, AsComplexSharesMemory) {
    std::vector<float> data = { 1, 4, 2, 5, 3, 6 };
    Array x = to_array(data, Shape({ 3, 2 }), DType::F32, GPUPlace(0));

    Array z = as_complex(x);
    Array x2 = as_real(z);

    // Compare GPU pointers as integers
    EXPECT_EQ(reinterpret_cast<uintptr_t>(x.data<float>()),
        reinterpret_cast<uintptr_t>(x2.data<float>()));

    // Verify values match
    Array x_cpu = x.to(CPUPlace());
    Array x2_cpu = x2.to(CPUPlace());
    const float* x_ptr = x_cpu.data<float>();
    const float* x2_ptr = x2_cpu.data<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(x_ptr[i], x2_ptr[i]);
    }
}

// ============================================================================
// Edge cases
// ============================================================================

TEST_F(ComplexTestGPU, ToComplexScalar) {
    Array scalar = full({}, 3.14f, DType::F32, GPUPlace(0));
    Array z = to_complex(scalar);

    EXPECT_EQ(z.shape().ndim(), 0);
    EXPECT_EQ(z.dtype(), DType::C32);

    Array z_cpu = z.to(CPUPlace());
    const std::complex<float>* data = z_cpu.data<std::complex<float>>();
    EXPECT_FLOAT_EQ(data->real(), 3.14f);
    EXPECT_FLOAT_EQ(data->imag(), 0.0f);
}

TEST_F(ComplexTestGPU, AsComplexScalar) {
    Array x = to_array({ 3.14f, 2.71f }, Shape({ 2 }), DType::F32, GPUPlace(0));
    Array z = as_complex(x);

    EXPECT_EQ(z.shape().ndim(), 0);
    EXPECT_EQ(z.dtype(), DType::C32);
}