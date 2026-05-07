// tests/cuda/test_unary.cu
#include <gtest/gtest.h>
#include "insight/insight.h"
#include <cmath>
#include <complex>

using namespace ins;

class UnaryTestGPU : public ::testing::Test {
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
    void fill_sequential_gpu(Array& gpu_arr) {
        Array cpu_arr(gpu_arr.shape(), gpu_arr.dtype(), CPUPlace());
        T* data = cpu_arr.data<T>();
        int64_t n = cpu_arr.numel();
        for (int64_t i = 0; i < n; ++i) {
            data[i] = static_cast<T>(i);
        }
        gpu_arr = cpu_arr.to(GPUPlace(0));
    }

    template<typename T>
    void fill_float_range_gpu(Array& gpu_arr, T start, T step) {
        Array cpu_arr(gpu_arr.shape(), gpu_arr.dtype(), CPUPlace());
        T* data = cpu_arr.data<T>();
        int64_t n = cpu_arr.numel();
        for (int64_t i = 0; i < n; ++i) {
            data[i] = start + static_cast<T>(i) * step;
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
}

// ============================================================================
// Basic math tests
// ============================================================================

TEST_F(UnaryTestGPU, Abs) {
    Array a({ 2, 3 }, DType::I32, GPUPlace(0));
    Array a_cpu({ 2, 3 }, DType::I32, CPUPlace());
    int32_t* a_cpu_data = a_cpu.data<int32_t>();
    for (int64_t i = 0; i < 6; ++i) {
        a_cpu_data[i] = static_cast<int32_t>(i - 3);
    }
    a = a_cpu.to(GPUPlace(0));

    Array c = abs(a);

    std::vector<int32_t> expected = { 3, 2, 1, 0, 1, 2 };
    expect_equal_gpu<int32_t>(c, expected);
}

TEST_F(UnaryTestGPU, Negative) {
    Array a({ 2, 3 }, DType::I32, GPUPlace(0));
    fill_sequential_gpu<int32_t>(a);

    Array c = negative(a);

    std::vector<int32_t> expected = { 0, -1, -2, -3, -4, -5 };
    expect_equal_gpu<int32_t>(c, expected);
}

TEST_F(UnaryTestGPU, Square) {
    Array a({ 2, 3 }, DType::I32, GPUPlace(0));
    fill_sequential_gpu<int32_t>(a);

    Array c = square(a);

    std::vector<int32_t> expected = { 0, 1, 4, 9, 16, 25 };
    expect_equal_gpu<int32_t>(c, expected);
}

// ============================================================================
// Exponential and logarithmic tests
// ============================================================================

TEST_F(UnaryTestGPU, Exp) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    fill_float_range_gpu<float>(a, 0.0f, 0.5f);

    Array c = exp(a);

    Array cpu_c = c.to(CPUPlace());
    const float* data = cpu_c.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(data[i], std::exp(i * 0.5f), 1e-6);
    }
}

TEST_F(UnaryTestGPU, Log) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    fill_float_range_gpu<float>(a, 1.0f, 1.0f);

    Array c = log(a);

    Array cpu_c = c.to(CPUPlace());
    const float* data = cpu_c.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(data[i], std::log(1.0f + i), 1e-6);
    }
}

// ============================================================================
// Trigonometric tests
// ============================================================================

TEST_F(UnaryTestGPU, Sin) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    fill_float_range_gpu<float>(a, 0.0f, 0.5f);

    Array c = sin(a);

    Array cpu_c = c.to(CPUPlace());
    const float* data = cpu_c.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(data[i], std::sin(i * 0.5f), 1e-6);
    }
}

TEST_F(UnaryTestGPU, Cos) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    fill_float_range_gpu<float>(a, 0.0f, 0.5f);

    Array c = cos(a);

    Array cpu_c = c.to(CPUPlace());
    const float* data = cpu_c.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(data[i], std::cos(i * 0.5f), 1e-6);
    }
}

TEST_F(UnaryTestGPU, Tan) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    fill_float_range_gpu<float>(a, 0.1f, 0.2f);

    Array c = tan(a);

    Array cpu_c = c.to(CPUPlace());
    const float* data = cpu_c.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(data[i], std::tan(0.1f + i * 0.2f), 1e-5);
    }
}

TEST_F(UnaryTestGPU, Tanh) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    fill_float_range_gpu<float>(a, -2.0f, 1.0f);

    Array c = tanh(a);

    Array cpu_c = c.to(CPUPlace());
    const float* data = cpu_c.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(data[i], std::tanh(-2.0f + i), 1e-6);
    }
}

// ============================================================================
// Rounding tests
// ============================================================================

TEST_F(UnaryTestGPU, Floor) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    Array a_cpu({ 2, 3 }, DType::F32, CPUPlace());
    float* a_cpu_data = a_cpu.data<float>();
    a_cpu_data[0] = 1.2f; a_cpu_data[1] = 1.8f; a_cpu_data[2] = -1.2f;
    a_cpu_data[3] = -1.8f; a_cpu_data[4] = 2.0f; a_cpu_data[5] = -2.0f;
    a = a_cpu.to(GPUPlace(0));

    Array c = floor(a);

    std::vector<float> expected = { 1.0f, 1.0f, -2.0f, -2.0f, 2.0f, -2.0f };
    expect_float_equal_gpu<float>(c, expected);
}

TEST_F(UnaryTestGPU, Ceil) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    Array a_cpu({ 2, 3 }, DType::F32, CPUPlace());
    float* a_cpu_data = a_cpu.data<float>();
    a_cpu_data[0] = 1.2f; a_cpu_data[1] = 1.8f; a_cpu_data[2] = -1.2f;
    a_cpu_data[3] = -1.8f; a_cpu_data[4] = 2.0f; a_cpu_data[5] = -2.0f;
    a = a_cpu.to(GPUPlace(0));

    Array c = ceil(a);

    std::vector<float> expected = { 2.0f, 2.0f, -1.0f, -1.0f, 2.0f, -2.0f };
    expect_float_equal_gpu<float>(c, expected);
}

TEST_F(UnaryTestGPU, Trunc) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    Array a_cpu({ 2, 3 }, DType::F32, CPUPlace());
    float* a_cpu_data = a_cpu.data<float>();
    a_cpu_data[0] = 1.2f; a_cpu_data[1] = 1.8f; a_cpu_data[2] = -1.2f;
    a_cpu_data[3] = -1.8f; a_cpu_data[4] = 2.0f; a_cpu_data[5] = -2.0f;
    a = a_cpu.to(GPUPlace(0));

    Array c = trunc(a);

    std::vector<float> expected = { 1.0f, 1.0f, -1.0f, -1.0f, 2.0f, -2.0f };
    expect_float_equal_gpu<float>(c, expected);
}

TEST_F(UnaryTestGPU, Rint) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    Array a_cpu({ 2, 3 }, DType::F32, CPUPlace());
    float* a_cpu_data = a_cpu.data<float>();
    a_cpu_data[0] = 1.2f; a_cpu_data[1] = 1.8f; a_cpu_data[2] = -1.2f;
    a_cpu_data[3] = -1.8f; a_cpu_data[4] = 2.0f; a_cpu_data[5] = -2.0f;
    a = a_cpu.to(GPUPlace(0));

    Array c = rint(a);

    Array cpu_c = c.to(CPUPlace());
    const float* data = cpu_c.data<float>();
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    EXPECT_FLOAT_EQ(data[2], -1.0f);
    EXPECT_FLOAT_EQ(data[3], -2.0f);
    EXPECT_FLOAT_EQ(data[4], 2.0f);
    EXPECT_FLOAT_EQ(data[5], -2.0f);
}

// ============================================================================
// Sign test
// ============================================================================

TEST_F(UnaryTestGPU, Sign) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    Array a_cpu({ 2, 3 }, DType::F32, CPUPlace());
    float* a_cpu_data = a_cpu.data<float>();
    a_cpu_data[0] = -5.0f; a_cpu_data[1] = 0.0f; a_cpu_data[2] = 3.0f;
    a_cpu_data[3] = -0.0f; a_cpu_data[4] = 0.5f; a_cpu_data[5] = -0.5f;
    a = a_cpu.to(GPUPlace(0));

    Array c = sign(a);

    std::vector<float> expected = { -1.0f, 0.0f, 1.0f, 0.0f, 1.0f, -1.0f };
    expect_float_equal_gpu<float>(c, expected);
}

// ============================================================================
// Complex conjugate test
// ============================================================================

TEST_F(UnaryTestGPU, Conj) {
    Array a({ 2, 3 }, DType::C32, GPUPlace(0));
    Array a_cpu({ 2, 3 }, DType::C32, CPUPlace());
    std::complex<float>* a_cpu_data = a_cpu.data<std::complex<float>>();
    for (int64_t i = 0; i < 6; ++i) {
        a_cpu_data[i] = std::complex<float>(static_cast<float>(i), static_cast<float>(i + 1));
    }
    a = a_cpu.to(GPUPlace(0));

    Array c = conj(a);

    Array cpu_c = c.to(CPUPlace());
    const std::complex<float>* data = cpu_c.data<std::complex<float>>();
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i].real(), static_cast<float>(i));
        EXPECT_FLOAT_EQ(data[i].imag(), -static_cast<float>(i + 1));
    }
}

TEST_F(UnaryTestGPU, ConjReal) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    fill_sequential_gpu<float>(a);

    Array c = conj(a);

    // Conjugate of real numbers should be identity
    Array cpu_c = c.to(CPUPlace());
    const float* data = cpu_c.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], static_cast<float>(i));
    }
}

// ============================================================================
// Degree/radian conversion tests
// ============================================================================

TEST_F(UnaryTestGPU, Deg2Rad) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    Array a_cpu({ 2, 3 }, DType::F32, CPUPlace());
    float* a_cpu_data = a_cpu.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        a_cpu_data[i] = static_cast<float>(i * 45);
    }
    a = a_cpu.to(GPUPlace(0));

    Array c = deg2rad(a);

    Array cpu_c = c.to(CPUPlace());
    const float* data = cpu_c.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(data[i], i * 45 * 3.1415926535f / 180.0f, 1e-6);
    }
}

TEST_F(UnaryTestGPU, Rad2Deg) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    fill_float_range_gpu<float>(a, 0.0f, 0.5f);

    Array c = rad2deg(a);

    Array cpu_c = c.to(CPUPlace());
    const float* data = cpu_c.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(data[i], i * 0.5f * 180.0f / 3.1415926535f, 1e-6);
    }
}

// ============================================================================
// Logical not test
// ============================================================================

TEST_F(UnaryTestGPU, LogicalNot) {
    Array a({ 2, 3 }, DType::F32, GPUPlace(0));
    Array a_cpu({ 2, 3 }, DType::F32, CPUPlace());
    float* a_cpu_data = a_cpu.data<float>();
    a_cpu_data[0] = 0.0f; a_cpu_data[1] = 1.0f; a_cpu_data[2] = -1.0f;
    a_cpu_data[3] = 0.0f; a_cpu_data[4] = 0.5f; a_cpu_data[5] = 0.0f;
    a = a_cpu.to(GPUPlace(0));

    Array c = logical_not(a);

    std::vector<bool> expected = { true, false, false, true, false, true };
    expect_bool_equal_gpu(c, expected);
}

// ============================================================================
// Bitwise not test
// ============================================================================

TEST_F(UnaryTestGPU, BitwiseNot) {
    Array a({ 2, 3 }, DType::I32, GPUPlace(0));
    fill_sequential_gpu<int32_t>(a);

    Array c = bitwise_not(a);

    const Array cpu_c = c.to(CPUPlace());
    const int32_t* data = cpu_c.data<int32_t>();
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_EQ(data[i], ~static_cast<int32_t>(i));
    }
}

// ============================================================================
// View (non-contiguous) test
// ============================================================================

TEST_F(UnaryTestGPU, ViewAbs) {
    Array a({ 3, 4 }, DType::F32, GPUPlace(0));
    fill_float_range_gpu<float>(a, -5.0f, 1.0f);

    // Create non-contiguous view via slice (rows 0 and 2)
    Array view = a.slice(0, 0, 3, 2);
    EXPECT_FALSE(view.is_contiguous());

    Array c = abs(view);

    // Expected: view row 0: [-5,-4,-3,-2] -> [5,4,3,2]
    //           view row 1: [3,4,5,6] -> [3,4,5,6]
    Array cpu_c = c.to(CPUPlace());
    const float* data = cpu_c.data<float>();
    EXPECT_FLOAT_EQ(data[0], 5.0f);
    EXPECT_FLOAT_EQ(data[1], 4.0f);
    EXPECT_FLOAT_EQ(data[2], 3.0f);
    EXPECT_FLOAT_EQ(data[3], 2.0f);
    EXPECT_FLOAT_EQ(data[4], 3.0f);
    EXPECT_FLOAT_EQ(data[5], 4.0f);
    EXPECT_FLOAT_EQ(data[6], 5.0f);
    EXPECT_FLOAT_EQ(data[7], 6.0f);
}