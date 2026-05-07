// tests/cuda/test_elementwise.cu
#include <gtest/gtest.h>
#include "insight/insight.h"
#include <complex>
#include <cmath>

using namespace ins;

class ElementwiseTestGPU : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ins::init();
        set_device(ins::GPUPlace(0));
        seed(42);
    }
};

// ============================================================================
// Helper functions
// ============================================================================
namespace {
    template<typename T>
    void fill_sequential_cpu(Array& cpu_arr) {
        T* data = cpu_arr.data<T>();
        int64_t n = cpu_arr.numel();
        for (int64_t i = 0; i < n; ++i) {
            data[i] = static_cast<T>(i);
        }
    }

    template<typename T>
    void fill_sequential_gpu(Array& gpu_arr) {
        Array cpu_arr(gpu_arr.shape(), gpu_arr.dtype(), CPUPlace());
        fill_sequential_cpu<T>(cpu_arr);
        gpu_arr = cpu_arr.to(GPUPlace(0));
    }

    template<typename T>
    void fill_constant_gpu(Array& gpu_arr, T val) {
        Array cpu_arr(gpu_arr.shape(), gpu_arr.dtype(), CPUPlace());
        T* data = cpu_arr.data<T>();
        int64_t n = cpu_arr.numel();
        for (int64_t i = 0; i < n; ++i) {
            data[i] = val;
        }
        gpu_arr = cpu_arr.to(GPUPlace(0));
    }

    template<typename T>
    void expect_equal_gpu(const Array& gpu_arr, const std::vector<T>& expected) {
        Array cpu_arr = gpu_arr.to(CPUPlace());
        ASSERT_EQ(cpu_arr.numel(), static_cast<int64_t>(expected.size()));
        const T* data = cpu_arr.data<T>();
        for (int64_t i = 0; i < cpu_arr.numel(); ++i) {
            EXPECT_EQ(data[i], expected[i]);
        }
    }

    template<typename T>
    void expect_float_equal_gpu(const Array& gpu_arr, const std::vector<T>& expected, T tol = 1e-6) {
        Array cpu_arr = gpu_arr.to(CPUPlace());
        ASSERT_EQ(cpu_arr.numel(), static_cast<int64_t>(expected.size()));
        const T* data = cpu_arr.data<T>();
        for (int64_t i = 0; i < cpu_arr.numel(); ++i) {
            EXPECT_NEAR(data[i], expected[i], tol);
        }
    }

    void expect_bool_equal_gpu(const Array& gpu_arr, const std::vector<bool>& expected) {
        Array cpu_arr = gpu_arr.to(CPUPlace());
        ASSERT_EQ(cpu_arr.numel(), static_cast<int64_t>(expected.size()));
        const bool* data = cpu_arr.data<bool>();
        for (int64_t i = 0; i < cpu_arr.numel(); ++i) {
            EXPECT_EQ(data[i], expected[i]);
        }
    }

    template<typename T>
    void expect_complex_equal_gpu(const Array& gpu_arr, const std::vector<std::complex<T>>& expected, T tol = 1e-6) {
        Array cpu_arr = gpu_arr.to(CPUPlace());
        ASSERT_EQ(cpu_arr.numel(), static_cast<int64_t>(expected.size()));
        const std::complex<T>* data = cpu_arr.data<std::complex<T>>();
        for (int64_t i = 0; i < cpu_arr.numel(); ++i) {
            EXPECT_NEAR(data[i].real(), expected[i].real(), tol);
            EXPECT_NEAR(data[i].imag(), expected[i].imag(), tol);
        }
    }
}

// ============================================================================
// Subtraction Tests (sub)
// ============================================================================

TEST_F(ElementwiseTestGPU, SubSameShape) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    Array b({ 2, 3 }, DType::F32, GPUPlace(0));
    fill_sequential_gpu<float>(a);
    fill_sequential_gpu<float>(b);

    Array c = sub(a, b);

    std::vector<float> expected = { 0, 0, 0, 0, 0, 0 };
    expect_float_equal_gpu<float>(c, expected);
}

TEST_F(ElementwiseTestGPU, SubBroadcast) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    Array b({ 3 }, DType::F32, GPUPlace(0));
    fill_sequential_gpu<float>(a);
    fill_sequential_gpu<float>(b);

    Array c = sub(a, b);

    // a: [0,1,2,3,4,5], b: [0,1,2] -> broadcast to [0,1,2,0,1,2]
    // result: [0-0,1-1,2-2,3-0,4-1,5-2] = [0,0,0,3,3,3]
    std::vector<float> expected = { 0, 0, 0, 3, 3, 3 };
    expect_float_equal_gpu<float>(c, expected);
}

// ============================================================================
// Multiplication Tests (mul)
// ============================================================================

TEST_F(ElementwiseTestGPU, MulSameShape) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    Array b({ 2, 3 }, DType::F32, GPUPlace(0));
    fill_sequential_gpu<float>(a);
    fill_sequential_gpu<float>(b);

    Array c = mul(a, b);

    // 0*0=0, 1*1=1, 2*2=4, 3*3=9, 4*4=16, 5*5=25
    std::vector<float> expected = { 0, 1, 4, 9, 16, 25 };
    expect_float_equal_gpu<float>(c, expected);
}

// ============================================================================
// Division Tests (div)
// ============================================================================

TEST_F(ElementwiseTestGPU, DivSameShape) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    Array b({ 2, 3 }, DType::F32, GPUPlace(0));
    fill_sequential_gpu<float>(a);

    // Fill b with 1..6 on GPU
    Array b_cpu({ 2, 3 }, DType::F32, CPUPlace());
    float* b_cpu_data = b_cpu.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        b_cpu_data[i] = static_cast<float>(i + 1);
    }
    b = b_cpu.to(GPUPlace(0));

    Array c = div(a, b);

    Array cpu_c = c.to(CPUPlace());
    const float* data = cpu_c.data<float>();
    EXPECT_FLOAT_EQ(data[0], 0.0f);
    EXPECT_FLOAT_EQ(data[1], 1.0f / 2.0f);
    EXPECT_NEAR(data[2], 2.0f / 3.0f, 1e-6);
    EXPECT_FLOAT_EQ(data[3], 3.0f / 4.0f);
    EXPECT_FLOAT_EQ(data[4], 4.0f / 5.0f);
    EXPECT_NEAR(data[5], 5.0f / 6.0f, 1e-6);
}

// ============================================================================
// Power Tests (pow)
// ============================================================================

TEST_F(ElementwiseTestGPU, PowSameShape) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    Array b({ 2, 3 }, DType::F32, GPUPlace(0));
    fill_sequential_gpu<float>(a);
    fill_sequential_gpu<float>(b);

    Array c = pow(a, b);

    Array cpu_c = c.to(CPUPlace());
    const float* data = cpu_c.data<float>();
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 1.0f);
    EXPECT_FLOAT_EQ(data[2], 4.0f);
    EXPECT_FLOAT_EQ(data[3], 27.0f);
    EXPECT_FLOAT_EQ(data[4], 256.0f);
    EXPECT_FLOAT_EQ(data[5], 3125.0f);
}

// ============================================================================
// Modulo Tests (mod)
// ============================================================================

TEST_F(ElementwiseTestGPU, ModSameShape) {
    Array a({ 2, 3 }, DType::I32, GPUPlace(0));
    Array b({ 2, 3 }, DType::I32, GPUPlace(0));

    Array a_cpu({ 2, 3 }, DType::I32, CPUPlace());
    Array b_cpu({ 2, 3 }, DType::I32, CPUPlace());
    int32_t* a_cpu_data = a_cpu.data<int32_t>();
    int32_t* b_cpu_data = b_cpu.data<int32_t>();
    for (int64_t i = 0; i < 6; ++i) {
        a_cpu_data[i] = static_cast<int32_t>(i + 10);
        b_cpu_data[i] = static_cast<int32_t>(i + 3);
    }
    a = a_cpu.to(GPUPlace(0));
    b = b_cpu.to(GPUPlace(0));

    Array c = mod(a, b);

    std::vector<int32_t> expected = { 10 % 3, 11 % 4, 12 % 5, 13 % 6, 14 % 7, 15 % 8 };
    expect_equal_gpu<int32_t>(c, expected);
}

TEST_F(ElementwiseTestGPU, ModFloat) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    Array b({ 2, 3 }, DType::F32, GPUPlace(0));

    Array a_cpu({ 2, 3 }, DType::F32, CPUPlace());
    Array b_cpu({ 2, 3 }, DType::F32, CPUPlace());
    float* a_cpu_data = a_cpu.data<float>();
    float* b_cpu_data = b_cpu.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        a_cpu_data[i] = static_cast<float>(i + 10);
        b_cpu_data[i] = static_cast<float>(i + 3);
    }
    a = a_cpu.to(GPUPlace(0));
    b = b_cpu.to(GPUPlace(0));

    Array c = mod(a, b);

    Array cpu_c = c.to(CPUPlace());
    const float* data = cpu_c.data<float>();
    EXPECT_FLOAT_EQ(data[0], fmodf(10, 3));
    EXPECT_FLOAT_EQ(data[1], fmodf(11, 4));
    EXPECT_FLOAT_EQ(data[2], fmodf(12, 5));
    EXPECT_FLOAT_EQ(data[3], fmodf(13, 6));
    EXPECT_FLOAT_EQ(data[4], fmodf(14, 7));
    EXPECT_FLOAT_EQ(data[5], fmodf(15, 8));
}

// ============================================================================
// Comparison Tests
// ============================================================================

TEST_F(ElementwiseTestGPU, Equal) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    Array b({ 2, 3 }, DType::F32, GPUPlace(0));

    Array a_cpu({ 2, 3 }, DType::F32, CPUPlace());
    Array b_cpu({ 2, 3 }, DType::F32, CPUPlace());
    float* a_cpu_data = a_cpu.data<float>();
    float* b_cpu_data = b_cpu.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        a_cpu_data[i] = static_cast<float>(i);
        b_cpu_data[i] = static_cast<float>(i);
    }
    a = a_cpu.to(GPUPlace(0));
    b = b_cpu.to(GPUPlace(0));

    Array c = equal(a, b);

    std::vector<bool> expected = { true, true, true, true, true, true };
    expect_bool_equal_gpu(c, expected);
}

TEST_F(ElementwiseTestGPU, NotEqual) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    Array b({ 2, 3 }, DType::F32, GPUPlace(0));

    Array a_cpu({ 2, 3 }, DType::F32, CPUPlace());
    Array b_cpu({ 2, 3 }, DType::F32, CPUPlace());
    float* a_cpu_data = a_cpu.data<float>();
    float* b_cpu_data = b_cpu.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        a_cpu_data[i] = static_cast<float>(i);
        b_cpu_data[i] = static_cast<float>(i + 1);
    }
    a = a_cpu.to(GPUPlace(0));
    b = b_cpu.to(GPUPlace(0));

    Array c = not_equal(a, b);

    std::vector<bool> expected = { true, true, true, true, true, true };
    expect_bool_equal_gpu(c, expected);
}

TEST_F(ElementwiseTestGPU, Greater) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    Array b({ 2, 3 }, DType::F32, GPUPlace(0));
    fill_sequential_gpu<float>(a);
    fill_sequential_gpu<float>(b);

    Array c = greater(a, b);

    std::vector<bool> expected = { false, false, false, false, false, false };
    expect_bool_equal_gpu(c, expected);
}

TEST_F(ElementwiseTestGPU, Less) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    Array b({ 2, 3 }, DType::F32, GPUPlace(0));
    fill_sequential_gpu<float>(a);
    fill_sequential_gpu<float>(b);

    Array c = less(a, b);

    std::vector<bool> expected = { false, false, false, false, false, false };
    expect_bool_equal_gpu(c, expected);
}

TEST_F(ElementwiseTestGPU, GreaterEqual) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    Array b({ 2, 3 }, DType::F32, GPUPlace(0));

    Array a_cpu({ 2, 3 }, DType::F32, CPUPlace());
    Array b_cpu({ 2, 3 }, DType::F32, CPUPlace());
    float* a_cpu_data = a_cpu.data<float>();
    float* b_cpu_data = b_cpu.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        a_cpu_data[i] = static_cast<float>(i);
        b_cpu_data[i] = static_cast<float>(i);
    }
    a = a_cpu.to(GPUPlace(0));
    b = b_cpu.to(GPUPlace(0));

    Array c = greater_equal(a, b);

    std::vector<bool> expected = { true, true, true, true, true, true };
    expect_bool_equal_gpu(c, expected);
}

TEST_F(ElementwiseTestGPU, LessEqual) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    Array b({ 2, 3 }, DType::F32, GPUPlace(0));

    Array a_cpu({ 2, 3 }, DType::F32, CPUPlace());
    Array b_cpu({ 2, 3 }, DType::F32, CPUPlace());
    float* a_cpu_data = a_cpu.data<float>();
    float* b_cpu_data = b_cpu.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        a_cpu_data[i] = static_cast<float>(i);
        b_cpu_data[i] = static_cast<float>(i);
    }
    a = a_cpu.to(GPUPlace(0));
    b = b_cpu.to(GPUPlace(0));

    Array c = less_equal(a, b);

    std::vector<bool> expected = { true, true, true, true, true, true };
    expect_bool_equal_gpu(c, expected);
}

// ============================================================================
// Logical Operations Tests
// ============================================================================

TEST_F(ElementwiseTestGPU, LogicalAnd) {
    Array a({ 2, 3 }, DType::BOOL, GPUPlace(0));
    Array b({ 2, 3 }, DType::BOOL, GPUPlace(0));

    Array a_cpu({ 2, 3 }, DType::BOOL, CPUPlace());
    Array b_cpu({ 2, 3 }, DType::BOOL, CPUPlace());
    bool* a_cpu_data = a_cpu.data<bool>();
    bool* b_cpu_data = b_cpu.data<bool>();

    for (int64_t i = 0; i < 6; ++i) {
        a_cpu_data[i] = (i % 2 == 0);
        b_cpu_data[i] = (i < 3);
    }
    a = a_cpu.to(GPUPlace(0));
    b = b_cpu.to(GPUPlace(0));

    Array c = logical_and(a, b);

    std::vector<bool> expected = { true, false, true, false, false, false };
    expect_bool_equal_gpu(c, expected);
}

TEST_F(ElementwiseTestGPU, LogicalOr) {
    Array a({ 2, 3 }, DType::BOOL, GPUPlace(0));
    Array b({ 2, 3 }, DType::BOOL, GPUPlace(0));

    Array a_cpu({ 2, 3 }, DType::BOOL, CPUPlace());
    Array b_cpu({ 2, 3 }, DType::BOOL, CPUPlace());
    bool* a_cpu_data = a_cpu.data<bool>();
    bool* b_cpu_data = b_cpu.data<bool>();

    for (int64_t i = 0; i < 6; ++i) {
        a_cpu_data[i] = (i % 2 == 0);
        b_cpu_data[i] = (i % 3 != 0);
    }
    a = a_cpu.to(GPUPlace(0));
    b = b_cpu.to(GPUPlace(0));

    Array c = logical_or(a, b);

    std::vector<bool> expected = { true, true, true, false, true, true };
    expect_bool_equal_gpu(c, expected);
}

TEST_F(ElementwiseTestGPU, LogicalXor) {
    Array a({ 2, 3 }, DType::BOOL, GPUPlace(0));
    Array b({ 2, 3 }, DType::BOOL, GPUPlace(0));

    Array a_cpu({ 2, 3 }, DType::BOOL, CPUPlace());
    Array b_cpu({ 2, 3 }, DType::BOOL, CPUPlace());
    bool* a_cpu_data = a_cpu.data<bool>();
    bool* b_cpu_data = b_cpu.data<bool>();

    for (int64_t i = 0; i < 6; ++i) {
        a_cpu_data[i] = (i % 2 == 0);
        b_cpu_data[i] = (i < 3);
    }
    a = a_cpu.to(GPUPlace(0));
    b = b_cpu.to(GPUPlace(0));

    Array c = logical_xor(a, b);

    std::vector<bool> expected = { false, true, false, false, true, false };
    expect_bool_equal_gpu(c, expected);
}

// ============================================================================
// Bitwise Operations Tests
// ============================================================================

TEST_F(ElementwiseTestGPU, BitwiseAnd) {
    Array a({ 2, 3 }, DType::I32, GPUPlace(0));
    Array b({ 2, 3 }, DType::I32, GPUPlace(0));

    Array a_cpu({ 2, 3 }, DType::I32, CPUPlace());
    Array b_cpu({ 2, 3 }, DType::I32, CPUPlace());
    int32_t* a_cpu_data = a_cpu.data<int32_t>();
    int32_t* b_cpu_data = b_cpu.data<int32_t>();
    for (int64_t i = 0; i < 6; ++i) {
        a_cpu_data[i] = static_cast<int32_t>(i + 1);
        b_cpu_data[i] = static_cast<int32_t>(i + 5);
    }
    a = a_cpu.to(GPUPlace(0));
    b = b_cpu.to(GPUPlace(0));

    Array c = bitwise_and(a, b);

    Array cpu_c = c.to(CPUPlace());
    const int32_t* data = cpu_c.data<int32_t>();
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_EQ(data[i], (i + 1) & (i + 5));
    }
}

TEST_F(ElementwiseTestGPU, BitwiseOr) {
    Array a({ 2, 3 }, DType::I32, GPUPlace(0));
    Array b({ 2, 3 }, DType::I32, GPUPlace(0));

    Array a_cpu({ 2, 3 }, DType::I32, CPUPlace());
    Array b_cpu({ 2, 3 }, DType::I32, CPUPlace());
    int32_t* a_cpu_data = a_cpu.data<int32_t>();
    int32_t* b_cpu_data = b_cpu.data<int32_t>();
    for (int64_t i = 0; i < 6; ++i) {
        a_cpu_data[i] = static_cast<int32_t>(i + 1);
        b_cpu_data[i] = static_cast<int32_t>(i + 5);
    }
    a = a_cpu.to(GPUPlace(0));
    b = b_cpu.to(GPUPlace(0));

    Array c = bitwise_or(a, b);

    Array cpu_c = c.to(CPUPlace());
    const int32_t* data = cpu_c.data<int32_t>();
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_EQ(data[i], (i + 1) | (i + 5));
    }
}

TEST_F(ElementwiseTestGPU, BitwiseXor) {
    Array a({ 2, 3 }, DType::I32, GPUPlace(0));
    Array b({ 2, 3 }, DType::I32, GPUPlace(0));

    Array a_cpu({ 2, 3 }, DType::I32, CPUPlace());
    Array b_cpu({ 2, 3 }, DType::I32, CPUPlace());
    int32_t* a_cpu_data = a_cpu.data<int32_t>();
    int32_t* b_cpu_data = b_cpu.data<int32_t>();
    for (int64_t i = 0; i < 6; ++i) {
        a_cpu_data[i] = static_cast<int32_t>(i + 1);
        b_cpu_data[i] = static_cast<int32_t>(i + 5);
    }
    a = a_cpu.to(GPUPlace(0));
    b = b_cpu.to(GPUPlace(0));

    Array c = bitwise_xor(a, b);

    Array cpu_c = c.to(CPUPlace());
    const int32_t* data = cpu_c.data<int32_t>();
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_EQ(data[i], (i + 1) ^ (i + 5));
    }
}

TEST_F(ElementwiseTestGPU, BitwiseLeftShift) {
    Array a({ 2, 3 }, DType::I32, GPUPlace(0));
    Array b({ 2, 3 }, DType::I32, GPUPlace(0));

    Array a_cpu({ 2, 3 }, DType::I32, CPUPlace());
    Array b_cpu({ 2, 3 }, DType::I32, CPUPlace());
    int32_t* a_cpu_data = a_cpu.data<int32_t>();
    int32_t* b_cpu_data = b_cpu.data<int32_t>();
    for (int64_t i = 0; i < 6; ++i) {
        a_cpu_data[i] = static_cast<int32_t>(i + 1);
        b_cpu_data[i] = static_cast<int32_t>(i % 3);
    }
    a = a_cpu.to(GPUPlace(0));
    b = b_cpu.to(GPUPlace(0));

    Array c = bitwise_left_shift(a, b);

    Array cpu_c = c.to(CPUPlace());
    const int32_t* data = cpu_c.data<int32_t>();
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_EQ(data[i], (i + 1) << (i % 3));
    }
}

TEST_F(ElementwiseTestGPU, BitwiseRightShift) {
    Array a({ 2, 3 }, DType::I32, GPUPlace(0));
    Array b({ 2, 3 }, DType::I32, GPUPlace(0));

    Array a_cpu({ 2, 3 }, DType::I32, CPUPlace());
    Array b_cpu({ 2, 3 }, DType::I32, CPUPlace());
    int32_t* a_cpu_data = a_cpu.data<int32_t>();
    int32_t* b_cpu_data = b_cpu.data<int32_t>();
    for (int64_t i = 0; i < 6; ++i) {
        a_cpu_data[i] = static_cast<int32_t>((i + 1) * 8);
        b_cpu_data[i] = static_cast<int32_t>(i % 4);
    }
    a = a_cpu.to(GPUPlace(0));
    b = b_cpu.to(GPUPlace(0));

    Array c = bitwise_right_shift(a, b);

    Array cpu_c = c.to(CPUPlace());
    const int32_t* data = cpu_c.data<int32_t>();
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_EQ(data[i], (i + 1) * 8 >> (i % 4));
    }
}

// ============================================================================
// Maximum / Minimum Tests
// ============================================================================

TEST_F(ElementwiseTestGPU, Maximum) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    Array b({ 2, 3 }, DType::F32, GPUPlace(0));

    Array a_cpu({ 2, 3 }, DType::F32, CPUPlace());
    Array b_cpu({ 2, 3 }, DType::F32, CPUPlace());
    float* a_cpu_data = a_cpu.data<float>();
    float* b_cpu_data = b_cpu.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        a_cpu_data[i] = static_cast<float>(i);
        b_cpu_data[i] = static_cast<float>(5 - i);
    }
    a = a_cpu.to(GPUPlace(0));
    b = b_cpu.to(GPUPlace(0));

    Array c = maximum(a, b);

    std::vector<float> expected = { 5, 4, 3, 3, 4, 5 };
    expect_float_equal_gpu<float>(c, expected);
}

TEST_F(ElementwiseTestGPU, Minimum) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    Array b({ 2, 3 }, DType::F32, GPUPlace(0));

    Array a_cpu({ 2, 3 }, DType::F32, CPUPlace());
    Array b_cpu({ 2, 3 }, DType::F32, CPUPlace());
    float* a_cpu_data = a_cpu.data<float>();
    float* b_cpu_data = b_cpu.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        a_cpu_data[i] = static_cast<float>(i);
        b_cpu_data[i] = static_cast<float>(5 - i);
    }
    a = a_cpu.to(GPUPlace(0));
    b = b_cpu.to(GPUPlace(0));

    Array c = minimum(a, b);

    std::vector<float> expected = { 0, 1, 2, 2, 1, 0 };
    expect_float_equal_gpu<float>(c, expected);
}

// ============================================================================
// Broadcasting Tests for all ops
// ============================================================================

TEST_F(ElementwiseTestGPU, Broadcast2D1D) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    Array b({ 3 }, DType::F32, GPUPlace(0));
    fill_sequential_gpu<float>(a);
    fill_sequential_gpu<float>(b);

    Array add_c = add(a, b);
    Array sub_c = sub(a, b);
    Array mul_c = mul(a, b);

    // Expected: a: [0,1,2,3,4,5], b: [0,1,2] broadcast -> [0,1,2,0,1,2]
    // add: [0,2,4,3,5,7]
    std::vector<float> add_expected = { 0, 2, 4, 3, 5, 7 };
    // sub: [0,0,0,3,3,3]
    std::vector<float> sub_expected = { 0, 0, 0, 3, 3, 3 };
    // mul: [0,1,4,0,4,10]
    std::vector<float> mul_expected = { 0, 1, 4, 0, 4, 10 };

    expect_float_equal_gpu<float>(add_c, add_expected);
    expect_float_equal_gpu<float>(sub_c, sub_expected);
    expect_float_equal_gpu<float>(mul_c, mul_expected);
}

// ============================================================================
// Complex Number Tests
// ============================================================================

TEST_F(ElementwiseTestGPU, ComplexAdd) {
    Array a({ 2, 3 }, DType::C32, GPUPlace(0));
    Array b({ 2, 3 }, DType::C32, GPUPlace(0));

    Array a_cpu({ 2, 3 }, DType::C32, CPUPlace());
    Array b_cpu({ 2, 3 }, DType::C32, CPUPlace());
    std::complex<float>* a_cpu_data = a_cpu.data<std::complex<float>>();
    std::complex<float>* b_cpu_data = b_cpu.data<std::complex<float>>();
    for (int64_t i = 0; i < 6; ++i) {
        a_cpu_data[i] = std::complex<float>(static_cast<float>(i), static_cast<float>(i * 2));
        b_cpu_data[i] = std::complex<float>(static_cast<float>(i), static_cast<float>(i * 3));
    }
    a = a_cpu.to(GPUPlace(0));
    b = b_cpu.to(GPUPlace(0));

    Array c = add(a, b);

    std::vector<std::complex<float>> expected(6);
    for (int64_t i = 0; i < 6; ++i) {
        expected[i] = std::complex<float>(
            static_cast<float>(i + i),
            static_cast<float>(i * 2 + i * 3)
        );
    }
    expect_complex_equal_gpu<float>(c, expected);
}

// ============================================================================
// Scalar Broadcast Tests
// ============================================================================

TEST_F(ElementwiseTestGPU, ScalarAdd) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    fill_sequential_gpu<float>(a);
    Array b(5.0f);

    Array c = add(a, b);

    std::vector<float> expected = { 5, 6, 7, 8, 9, 10 };
    expect_float_equal_gpu<float>(c, expected);
}

// ============================================================================
// View (Non-Contiguous) Tests
// ============================================================================

TEST_F(ElementwiseTestGPU, ViewAdd) {
    Array a({ 3, 4 }, DType::F32, GPUPlace(0));
    fill_sequential_gpu<float>(a);

    Array view = a.slice(0, 0, 3, 2);  // 取行0和行2，跳过了行1
    EXPECT_FALSE(view.is_contiguous());

    Array b({ 2, 4 }, DType::F32, GPUPlace(0));
    fill_constant_gpu<float>(b, 1.0f);

    Array c = add(view, b);

    Array cpu_c = c.to(CPUPlace());
    const float* data = cpu_c.data<float>();
    EXPECT_FLOAT_EQ(data[0], 1);
    EXPECT_FLOAT_EQ(data[1], 2);
    EXPECT_FLOAT_EQ(data[2], 3);
    EXPECT_FLOAT_EQ(data[3], 4);
    EXPECT_FLOAT_EQ(data[4], 9);
    EXPECT_FLOAT_EQ(data[5], 10);
    EXPECT_FLOAT_EQ(data[6], 11);
    EXPECT_FLOAT_EQ(data[7], 12);
}

// ============================================================================
// Type Promotion Tests
// ============================================================================

TEST_F(ElementwiseTestGPU, TypePromotionIntToFloat) {
    Array a({ 2, 3 }, DType::I32, GPUPlace(0));
    Array b({ 2, 3 }, DType::F32, GPUPlace(0));

    Array a_cpu({ 2, 3 }, DType::I32, CPUPlace());
    Array b_cpu({ 2, 3 }, DType::F32, CPUPlace());
    int32_t* a_cpu_data = a_cpu.data<int32_t>();
    float* b_cpu_data = b_cpu.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        a_cpu_data[i] = static_cast<int32_t>(i);
        b_cpu_data[i] = static_cast<float>(i) * 1.5f;
    }
    a = a_cpu.to(GPUPlace(0));
    b = b_cpu.to(GPUPlace(0));

    Array c = add(a, b);

    EXPECT_EQ(c.dtype(), DType::F32);
    Array cpu_c = c.to(CPUPlace());
    const float* data = cpu_c.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], static_cast<float>(i) + i * 1.5f);
    }
}