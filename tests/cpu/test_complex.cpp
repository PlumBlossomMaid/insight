// tests/cpu/test_complex.cpp
#include <gtest/gtest.h>
#include <complex>
#include "insight/insight.h"
#include "insight/ops/complex.h"

using namespace ins;

class ComplexTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ins::init();
        set_device(CPUPlace());
    }
};

// ============================================================================
// is_complex tests (using dtype, not shape)
// ============================================================================

TEST_F(ComplexTest, IsComplexReturnsFalseForReal) {
    Array real = ones({ 3, 4 });
    EXPECT_FALSE(is_complex(real));
}

TEST_F(ComplexTest, IsComplexReturnsTrueForComplex) {
    Array real = ones({ 3, 4 });
    Array complex = to_complex(real);
    EXPECT_TRUE(is_complex(complex));
}

TEST_F(ComplexTest, IsComplexReturnsFalseForScalar) {
    Array scalar = full({}, 1.0f);
    EXPECT_FALSE(is_complex(scalar));
}

// ============================================================================
// has_complex_shape tests (checks last dimension = 2)
// ============================================================================

TEST_F(ComplexTest, HasComplexShapeReturnsFalseForReal) {
    Array real = ones({ 3, 4 });
    EXPECT_FALSE(has_complex_shape(real));
}

TEST_F(ComplexTest, HasComplexShapeReturnsFalseForNativeComplex) {
    Array real = ones({ 3, 4 });
    Array complex = to_complex(real);
    // Native complex type does NOT have last dimension = 2
    EXPECT_FALSE(has_complex_shape(complex));
}

// ============================================================================
// to_complex (single argument) tests
// ============================================================================

TEST_F(ComplexTest, ToComplexSingleArgumentShape) {
    Array real = to_array({ 1.0f, 2.0f, 3.0f }, Shape({ 3 }));
    Array z = to_complex(real);

    // Native complex: shape is [3], dtype is C32
    EXPECT_EQ(z.shape().ndim(), 1);
    EXPECT_EQ(z.shape().dim(0), 3);
    EXPECT_EQ(z.dtype(), DType::C32);
}

TEST_F(ComplexTest, ToComplexSingleArgumentValues) {
    Array real = to_array({ 1.0f, 2.0f, 3.0f }, Shape({ 3 }));
    Array z = to_complex(real);

    // Access as complex
    const std::complex<float>* data = z.data<std::complex<float>>();
    EXPECT_FLOAT_EQ(data[0].real(), 1.0f);
    EXPECT_FLOAT_EQ(data[0].imag(), 0.0f);
    EXPECT_FLOAT_EQ(data[1].real(), 2.0f);
    EXPECT_FLOAT_EQ(data[1].imag(), 0.0f);
    EXPECT_FLOAT_EQ(data[2].real(), 3.0f);
    EXPECT_FLOAT_EQ(data[2].imag(), 0.0f);
}

TEST_F(ComplexTest, ToComplexSingleArgument2D) {
    Array real = arange(0.0, 6.0, 1.0).to(DType::F32).reshape({ 2, 3 });
    Array z = to_complex(real);

    EXPECT_EQ(z.shape().ndim(), 2);
    EXPECT_EQ(z.shape().dim(0), 2);
    EXPECT_EQ(z.shape().dim(1), 3);
    EXPECT_EQ(z.dtype(), DType::C32);
}

// ============================================================================
// to_complex (two arguments) tests
// ============================================================================

TEST_F(ComplexTest, ToComplexTwoArgumentsShape) {
    Array real = to_array({ 1.0f, 2.0f, 3.0f }, Shape({ 3 }));
    Array imag = to_array({ 4.0f, 5.0f, 6.0f }, Shape({ 3 }));
    Array z = to_complex(real, imag);

    EXPECT_EQ(z.shape().ndim(), 1);
    EXPECT_EQ(z.shape().dim(0), 3);
    EXPECT_EQ(z.dtype(), DType::C32);
}

TEST_F(ComplexTest, ToComplexTwoArgumentsValues) {
    Array real = to_array({ 1.0f, 2.0f, 3.0f }, Shape({ 3 }));
    Array imag = to_array({ 4.0f, 5.0f, 6.0f }, Shape({ 3 }));
    Array z = to_complex(real, imag);

    const std::complex<float>* data = z.data<std::complex<float>>();
    EXPECT_FLOAT_EQ(data[0].real(), 1.0f);
    EXPECT_FLOAT_EQ(data[0].imag(), 4.0f);
    EXPECT_FLOAT_EQ(data[1].real(), 2.0f);
    EXPECT_FLOAT_EQ(data[1].imag(), 5.0f);
    EXPECT_FLOAT_EQ(data[2].real(), 3.0f);
    EXPECT_FLOAT_EQ(data[2].imag(), 6.0f);
}

TEST_F(ComplexTest, ToComplexTwoArgumentsShapeMismatchThrows) {
    Array real = to_array({ 1.0f, 2.0f, 3.0f }, Shape({ 3 }));
    Array imag = to_array({ 4.0f, 5.0f }, Shape({ 2 }));

    EXPECT_THROW(to_complex(real, imag), Exception);
}

// ============================================================================
// real() and imag() tests
// ============================================================================

TEST_F(ComplexTest, RealExtractsRealPart) {
    Array real = to_array({ 1.0f, 2.0f, 3.0f }, Shape({ 3 }));
    Array imag = to_array({ 4.0f, 5.0f, 6.0f }, Shape({ 3 }));
    Array z = to_complex(real, imag);

    Array r = ins::real(z);

    EXPECT_EQ(r.shape().ndim(), 1);
    EXPECT_EQ(r.shape().dim(0), 3);
    EXPECT_FLOAT_EQ(r.at(0).item<float>(), 1.0f);
    EXPECT_FLOAT_EQ(r.at(1).item<float>(), 2.0f);
    EXPECT_FLOAT_EQ(r.at(2).item<float>(), 3.0f);
}

TEST_F(ComplexTest, ImagExtractsImagPart) {
    Array real = to_array({ 1.0f, 2.0f, 3.0f }, Shape({ 3 }));
    Array imag = to_array({ 4.0f, 5.0f, 6.0f }, Shape({ 3 }));
    Array z = to_complex(real, imag);

    Array i = ins::imag(z);

    EXPECT_EQ(i.shape().ndim(), 1);
    EXPECT_EQ(i.shape().dim(0), 3);
    EXPECT_FLOAT_EQ(i.at(0).item<float>(), 4.0f);
    EXPECT_FLOAT_EQ(i.at(1).item<float>(), 5.0f);
    EXPECT_FLOAT_EQ(i.at(2).item<float>(), 6.0f);
}

TEST_F(ComplexTest, RealAndImagValues) {
    Array real_arr = to_array({ 1.0f, 2.0f, 3.0f }, Shape({ 3 }));
    Array imag_arr = to_array({ 4.0f, 5.0f, 6.0f }, Shape({ 3 }));

    Array z = to_complex(real_arr, imag_arr);
    Array r = real(z);
    Array i = imag(z);

    EXPECT_FLOAT_EQ(r.at(0).item<float>(), 1.0f);
    EXPECT_FLOAT_EQ(r.at(1).item<float>(), 2.0f);
    EXPECT_FLOAT_EQ(r.at(2).item<float>(), 3.0f);
    EXPECT_FLOAT_EQ(i.at(0).item<float>(), 4.0f);
    EXPECT_FLOAT_EQ(i.at(1).item<float>(), 5.0f);
    EXPECT_FLOAT_EQ(i.at(2).item<float>(), 6.0f);
}

// ============================================================================
// as_complex tests
// ============================================================================

TEST_F(ComplexTest, AsComplexShape) {
    // Create [2, 3, 2] array with interleaved real/imag
    std::vector<float> data = {
        1, 4,   2, 5,   3, 6,
        7, 10,  8, 11,  9, 12
    };
    Array x = to_array(data, Shape({ 2, 3, 2 }));

    Array z = as_complex(x);

    EXPECT_EQ(z.shape().ndim(), 2);
    EXPECT_EQ(z.shape().dim(0), 2);
    EXPECT_EQ(z.shape().dim(1), 3);
    EXPECT_EQ(z.dtype(), DType::C32);
}

TEST_F(ComplexTest, AsComplexDtype) {
    std::vector<double> data = {
        1, 4,   2, 5,   3, 6,
        7, 10,  8, 11,  9, 12
    };
    Array x = to_array(data, Shape({ 2, 3, 2 }));

    Array z = as_complex(x);

    EXPECT_EQ(z.dtype(), DType::C64);
}

TEST_F(ComplexTest, AsComplexInvalidInputThrows) {
    // Input missing last dimension (not a complex storage layout)
    Array x = ones({ 3, 4 });
    EXPECT_THROW(as_complex(x), Exception);
}

// ============================================================================
// as_real tests
// ============================================================================

TEST_F(ComplexTest, AsRealShape) {
    std::vector<float> data = {
        1, 4,   2, 5,   3, 6,
        7, 10,  8, 11,  9, 12
    };
    Array c = to_array(data, Shape({ 2, 3, 2 }));
    Array z = as_complex(c);

    Array x = as_real(z);

    EXPECT_EQ(x.shape().ndim(), 3);
    EXPECT_EQ(x.shape().dim(0), 2);
    EXPECT_EQ(x.shape().dim(1), 3);
    EXPECT_EQ(x.shape().dim(2), 2);
    EXPECT_EQ(x.dtype(), DType::F32);
}

TEST_F(ComplexTest, AsRealRoundTrip) {
    std::vector<float> original_data = {
        1, 4,   2, 5,   3, 6,
        7, 10,  8, 11,  9, 12
    };
    Array x = to_array(original_data, Shape({ 2, 3, 2 }));

    Array z = as_complex(x);
    Array x2 = as_real(z);

    const float* x_data = x.data<float>();
    const float* x2_data = x2.data<float>();
    for (int i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(x_data[i], x2_data[i]);
    }
}

// ============================================================================
// Memory sharing tests
// ============================================================================

TEST_F(ComplexTest, AsComplexSharesMemory) {
    std::vector<float> data = { 1, 4, 2, 5, 3, 6 };
    Array x = to_array(data, Shape({ 3, 2 }));

    Array z = as_complex(x);
    Array x2 = as_real(z);

    EXPECT_EQ(x.data<float>(), x2.data<float>());

    float* x_ptr = x.data<float>();
    x_ptr[0] = 100.0f;

    const float* z_ptr = z.data<float>();
    EXPECT_FLOAT_EQ(z_ptr[0], 100.0f);
}

// ============================================================================
// Edge cases
// ============================================================================

TEST_F(ComplexTest, ToComplexScalar) {
    Array scalar = full({}, 3.14f);
    Array z = to_complex(scalar);

    EXPECT_EQ(z.shape().ndim(), 0);
    EXPECT_EQ(z.dtype(), DType::C32);

    const std::complex<float>* data = z.data<std::complex<float>>();
    EXPECT_FLOAT_EQ(data->real(), 3.14f);
    EXPECT_FLOAT_EQ(data->imag(), 0.0f);
}

TEST_F(ComplexTest, AsComplexScalar) {
    // Scalar complex represented as [real, imag] with shape [2]
    Array x = to_array({ 3.14f, 2.71f }, Shape({ 2 }));
    Array z = as_complex(x);

    EXPECT_EQ(z.shape().ndim(), 0);
    EXPECT_EQ(z.dtype(), DType::C32);
}