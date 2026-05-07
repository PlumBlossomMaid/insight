// tests/cuda/test_creation.cu
#include <gtest/gtest.h>
#include <complex>
#include "insight/insight.h"
#include "insight/ops/creation.h"

using namespace ins;

class CreationTestGPU : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ins::init();
        set_device(ins::GPUPlace(0));
    }
};

// ============================================================================
// Helper functions
// ============================================================================
namespace {
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

    void expect_eye_2d(const Array& gpu_arr, int rows, int cols, int offset) {
        Array cpu_arr = gpu_arr.to(CPUPlace());
        const float* data = cpu_arr.data<float>();
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (j - i == offset) {
                    EXPECT_FLOAT_EQ(data[i * cols + j], 1.0f);
                }
                else {
                    EXPECT_FLOAT_EQ(data[i * cols + j], 0.0f);
                }
            }
        }
    }
}

// ========== zeros / ones / full ==========

TEST_F(CreationTestGPU, Zeros) {
    Array a = zeros({ 2, 3 }, DType::F32, GPUPlace(0));
    EXPECT_EQ(a.shape(), Shape({ 2, 3 }));
    EXPECT_EQ(a.dtype(), DType::F32);
    EXPECT_EQ(a.place(), GPUPlace(0));

    Array cpu_a = a.to(CPUPlace());
    const float* data = cpu_a.data<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], 0.0f);
    }
}

TEST_F(CreationTestGPU, Ones) {
    Array a = ones({ 2, 3 }, DType::F32, GPUPlace(0));
    EXPECT_EQ(a.place(), GPUPlace(0));

    Array cpu_a = a.to(CPUPlace());
    const float* data = cpu_a.data<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}

TEST_F(CreationTestGPU, Full) {
    Array a = full({ 2, 3 }, 3.14, DType::F32, GPUPlace(0));
    EXPECT_EQ(a.place(), GPUPlace(0));

    Array cpu_a = a.to(CPUPlace());
    const float* data = cpu_a.data<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], 3.14f);
    }
}

TEST_F(CreationTestGPU, FullInt) {
    Array a = full({ 2, 3 }, 42, DType::I32, GPUPlace(0));
    EXPECT_EQ(a.place(), GPUPlace(0));

    Array cpu_a = a.to(CPUPlace());
    const int32_t* data = cpu_a.data<int32_t>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(data[i], 42);
    }
}

// ========== eye ==========

TEST_F(CreationTestGPU, Eye) {
    Array a = eye(3, 3, 0, DType::F32, GPUPlace(0));
    EXPECT_EQ(a.shape(), Shape({ 3, 3 }));
    EXPECT_EQ(a.place(), GPUPlace(0));

    expect_eye_2d(a, 3, 3, 0);
}

TEST_F(CreationTestGPU, EyeWithOffset) {
    Array a = eye(3, 3, 1, DType::F32, GPUPlace(0));
    EXPECT_EQ(a.place(), GPUPlace(0));

    Array cpu_a = a.to(CPUPlace());
    const float* data = cpu_a.data<float>();
    // Expected: [[0,1,0],[0,0,1],[0,0,0]]
    EXPECT_FLOAT_EQ(data[1], 1.0f);   // (0,1)
    EXPECT_FLOAT_EQ(data[5], 1.0f);   // (1,2)
    EXPECT_FLOAT_EQ(data[0], 0.0f);
    EXPECT_FLOAT_EQ(data[4], 0.0f);
    EXPECT_FLOAT_EQ(data[8], 0.0f);
}

TEST_F(CreationTestGPU, EyeRectangular) {
    Array a = eye(2, 3, 0, DType::F32, GPUPlace(0));
    EXPECT_EQ(a.place(), GPUPlace(0));

    expect_eye_2d(a, 2, 3, 0);
}

// ========== arange ==========

TEST_F(CreationTestGPU, ArangeSingle) {
    Array a = arange(5, DType::I64, GPUPlace(0));
    EXPECT_EQ(a.shape(), Shape({ 5 }));
    EXPECT_EQ(a.place(), GPUPlace(0));

    Array cpu_a = a.to(CPUPlace());
    const int64_t* data = cpu_a.data<int64_t>();
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(data[i], i);
    }
}

TEST_F(CreationTestGPU, ArangeStartStop) {
    Array a = arange(2, 7, 1, DType::I64, GPUPlace(0));
    EXPECT_EQ(a.place(), GPUPlace(0));

    Array cpu_a = a.to(CPUPlace());
    const int64_t* data = cpu_a.data<int64_t>();
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(data[i], 2 + i);
    }
}

TEST_F(CreationTestGPU, ArangeWithStep) {
    Array a = arange(1, 10, 2, DType::I64, GPUPlace(0));
    EXPECT_EQ(a.place(), GPUPlace(0));

    Array cpu_a = a.to(CPUPlace());
    const int64_t* data = cpu_a.data<int64_t>();
    EXPECT_EQ(a.numel(), 5);
    EXPECT_EQ(data[0], 1);
    EXPECT_EQ(data[1], 3);
    EXPECT_EQ(data[2], 5);
    EXPECT_EQ(data[3], 7);
    EXPECT_EQ(data[4], 9);
}

TEST_F(CreationTestGPU, ArangeFloat) {
    Array a = arange(0.0, 1.0, 0.2, DType::F32, GPUPlace(0));
    EXPECT_EQ(a.place(), GPUPlace(0));

    Array cpu_a = a.to(CPUPlace());
    const float* data = cpu_a.data<float>();
    EXPECT_EQ(a.numel(), 5);
    EXPECT_NEAR(data[0], 0.0f, 1e-6);
    EXPECT_NEAR(data[1], 0.2f, 1e-6);
    EXPECT_NEAR(data[2], 0.4f, 1e-6);
    EXPECT_NEAR(data[3], 0.6f, 1e-6);
    EXPECT_NEAR(data[4], 0.8f, 1e-6);
}

// ========== linspace ==========

TEST_F(CreationTestGPU, Linspace) {
    Array a = linspace(0, 1, 5, DType::F32, GPUPlace(0));
    EXPECT_EQ(a.place(), GPUPlace(0));

    Array cpu_a = a.to(CPUPlace());
    const float* data = cpu_a.data<float>();
    EXPECT_EQ(a.numel(), 5);
    EXPECT_NEAR(data[0], 0.0f, 1e-6);
    EXPECT_NEAR(data[1], 0.25f, 1e-6);
    EXPECT_NEAR(data[2], 0.5f, 1e-6);
    EXPECT_NEAR(data[3], 0.75f, 1e-6);
    EXPECT_NEAR(data[4], 1.0f, 1e-6);
}

TEST_F(CreationTestGPU, LinspaceSingle) {
    Array a = linspace(0, 1, 1, DType::F32, GPUPlace(0));
    EXPECT_EQ(a.place(), GPUPlace(0));

    Array cpu_a = a.to(CPUPlace());
    const float* data = cpu_a.data<float>();
    EXPECT_EQ(a.numel(), 1);
    EXPECT_NEAR(data[0], 0.0f, 1e-6);
}

TEST_F(CreationTestGPU, LinspaceFloat64) {
    Array a = linspace(0, 1, 5, DType::F64, GPUPlace(0));
    EXPECT_EQ(a.place(), GPUPlace(0));

    Array cpu_a = a.to(CPUPlace());
    const double* data = cpu_a.data<double>();
    EXPECT_NEAR(data[0], 0.0, 1e-12);
    EXPECT_NEAR(data[2], 0.5, 1e-12);
    EXPECT_NEAR(data[4], 1.0, 1e-12);
}

// ========== logspace ==========

TEST_F(CreationTestGPU, Logspace) {
    Array a = logspace(0, 2, 3, 10.0, DType::F32, GPUPlace(0));
    EXPECT_EQ(a.place(), GPUPlace(0));

    Array cpu_a = a.to(CPUPlace());
    const float* data = cpu_a.data<float>();
    EXPECT_EQ(a.numel(), 3);
    // base=10, start=0, stop=2: [10^0, 10^1, 10^2] = [1, 10, 100]
    EXPECT_NEAR(data[0], 1.0f, 1e-6);
    EXPECT_NEAR(data[1], 10.0f, 1e-6);
    EXPECT_NEAR(data[2], 100.0f, 1e-6);
}

TEST_F(CreationTestGPU, LogspaceBase2) {
    Array a = logspace(0, 3, 4, 2.0, DType::F32, GPUPlace(0));
    EXPECT_EQ(a.place(), GPUPlace(0));

    Array cpu_a = a.to(CPUPlace());
    const float* data = cpu_a.data<float>();
    // base=2, start=0, stop=3: [2^0, 2^1, 2^2, 2^3] = [1, 2, 4, 8]
    EXPECT_NEAR(data[0], 1.0f, 1e-6);
    EXPECT_NEAR(data[1], 2.0f, 1e-6);
    EXPECT_NEAR(data[2], 4.0f, 1e-6);
    EXPECT_NEAR(data[3], 8.0f, 1e-6);
}

// ========== to_array ==========

TEST_F(CreationTestGPU, ToArrayFromVector) {
    std::vector<float> data = { 1.0f, 2.0f, 3.0f, 4.0f };
    Array a = to_array(data, DType::F32, GPUPlace(0));
    EXPECT_EQ(a.shape(), Shape({ 4 }));
    EXPECT_EQ(a.place(), GPUPlace(0));

    Array cpu_a = a.to(CPUPlace());
    const float* ptr = cpu_a.data<float>();
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_FLOAT_EQ(ptr[i], data[i]);
    }
}

TEST_F(CreationTestGPU, ToArrayFromInitList) {
    Array a = to_array({ 1.0f, 2.0f, 3.0f }, Shape({ 3 }), DType::F32, GPUPlace(0));
    EXPECT_EQ(a.shape(), Shape({ 3 }));
    EXPECT_EQ(a.place(), GPUPlace(0));

    Array cpu_a = a.to(CPUPlace());
    const float* ptr = cpu_a.data<float>();
    EXPECT_FLOAT_EQ(ptr[0], 1.0f);
    EXPECT_FLOAT_EQ(ptr[1], 2.0f);
    EXPECT_FLOAT_EQ(ptr[2], 3.0f);
}

TEST_F(CreationTestGPU, ToArrayIntToFloat) {
    std::vector<int> data = { 1, 2, 3, 4 };
    Array a = to_array(data, DType::F32, GPUPlace(0));
    EXPECT_EQ(a.dtype(), DType::F32);
    EXPECT_EQ(a.place(), GPUPlace(0));

    Array cpu_a = a.to(CPUPlace());
    const float* ptr = cpu_a.data<float>();
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_FLOAT_EQ(ptr[i], static_cast<float>(data[i]));
    }
}

// ========== zeros_like / ones_like ==========

TEST_F(CreationTestGPU, ZerosLike) {
    Array original({ 2, 3 }, DType::F32, GPUPlace(0));
    Array copy = zeros_like(original);
    EXPECT_EQ(copy.shape(), original.shape());
    EXPECT_EQ(copy.dtype(), original.dtype());
    EXPECT_EQ(copy.place(), original.place());

    Array cpu_copy = copy.to(CPUPlace());
    const float* data = cpu_copy.data<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], 0.0f);
    }
}

TEST_F(CreationTestGPU, OnesLike) {
    Array original({ 2, 3 }, DType::F32, GPUPlace(0));
    Array copy = ones_like(original);
    EXPECT_EQ(copy.place(), GPUPlace(0));

    Array cpu_copy = copy.to(CPUPlace());
    const float* data = cpu_copy.data<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}