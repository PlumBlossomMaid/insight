// tests/cuda/test_cast.cu
#include <gtest/gtest.h>
#include "insight/insight.h"
#include <complex>

using namespace ins;

class CastTestGPU : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ins::init();
        set_device(ins::GPUPlace(0));
        seed(42);
    }
};

// Helper: fill array with sequential values
template<typename T>
void fill_sequential_gpu(Array& arr) {
    Array cpu_arr(arr.shape(), arr.dtype(), CPUPlace());
    T* cpu_data = cpu_arr.data<T>();
    int64_t n = arr.numel();
    for (int64_t i = 0; i < n; ++i) {
        cpu_data[i] = static_cast<T>(i);
    }
    arr = cpu_arr.to(arr.place());
}

// Helper: fill bool array with alternating values
void fill_bool_alternating_gpu(Array& arr) {
    Array cpu_arr(arr.shape(), arr.dtype(), CPUPlace());
    bool* cpu_data = cpu_arr.data<bool>();
    int64_t n = arr.numel();
    for (int64_t i = 0; i < n; ++i) {
        cpu_data[i] = (i % 2 == 0);
    }
    arr = cpu_arr.to(arr.place());
}

// Helper: fill complex array with sequential values
template<typename T>
void fill_complex_sequential_gpu(Array& arr) {
    Array cpu_arr(arr.shape(), arr.dtype(), CPUPlace());
    std::complex<T>* cpu_data = cpu_arr.data<std::complex<T>>();
    int64_t n = arr.numel();
    for (int64_t i = 0; i < n; ++i) {
        cpu_data[i] = std::complex<T>(static_cast<T>(i), static_cast<T>(i * 2));
    }
    arr = cpu_arr.to(arr.place());
}

// Helper: verify float values within tolerance (download from GPU)
template<typename T>
void expect_float_values_gpu(const Array& arr, const std::vector<T>& expected, T tol = 1e-6) {
    Array cpu_arr = arr.to(CPUPlace());
    ASSERT_EQ(cpu_arr.numel(), static_cast<int64_t>(expected.size()));
    const T* data = cpu_arr.data<T>();
    for (int64_t i = 0; i < cpu_arr.numel(); ++i) {
        EXPECT_NEAR(data[i], expected[i], tol);
    }
}

// Helper: verify integer values (download from GPU)
template<typename T>
void expect_int_values_gpu(const Array& arr, const std::vector<T>& expected) {
    Array cpu_arr = arr.to(CPUPlace());
    ASSERT_EQ(cpu_arr.numel(), static_cast<int64_t>(expected.size()));
    const T* data = cpu_arr.data<T>();
    for (int64_t i = 0; i < cpu_arr.numel(); ++i) {
        EXPECT_EQ(data[i], expected[i]);
    }
}

// Helper: verify bool values (download from GPU)
void expect_bool_values_gpu(const Array& arr, const std::vector<bool>& expected) {
    Array cpu_arr = arr.to(CPUPlace());
    ASSERT_EQ(cpu_arr.numel(), static_cast<int64_t>(expected.size()));
    const bool* data = cpu_arr.data<bool>();
    for (int64_t i = 0; i < cpu_arr.numel(); ++i) {
        EXPECT_EQ(data[i], expected[i]);
    }
}

// Helper: verify complex values (download from GPU)
template<typename T>
void expect_complex_values_gpu(const Array& arr, const std::vector<std::complex<T>>& expected, T tol = 1e-6) {
    Array cpu_arr = arr.to(CPUPlace());
    ASSERT_EQ(cpu_arr.numel(), static_cast<int64_t>(expected.size()));
    const std::complex<T>* data = cpu_arr.data<std::complex<T>>();
    for (int64_t i = 0; i < cpu_arr.numel(); ++i) {
        EXPECT_NEAR(data[i].real(), expected[i].real(), tol);
        EXPECT_NEAR(data[i].imag(), expected[i].imag(), tol);
    }
}

// ========== Cast Tests ==========

// Test 1: BOOL to all types
TEST_F(CastTestGPU, BoolToAll) {
    Array src({ 2, 3 }, DType::BOOL, GPUPlace(0));
    fill_bool_alternating_gpu(src);  // [true, false, true, false, true, false]

    // BOOL -> U8
    Array u8 = src.to(DType::U8);
    expect_int_values_gpu<uint8_t>(u8, { 1, 0, 1, 0, 1, 0 });

    // BOOL -> I32
    Array i32 = src.to(DType::I32);
    expect_int_values_gpu<int32_t>(i32, { 1, 0, 1, 0, 1, 0 });

    // BOOL -> I64
    Array i64 = src.to(DType::I64);
    expect_int_values_gpu<int64_t>(i64, { 1, 0, 1, 0, 1, 0 });

    // BOOL -> F32
    Array f32 = src.to(DType::F32);
    expect_float_values_gpu<float>(f32, { 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f });

    // BOOL -> F64
    Array f64 = src.to(DType::F64);
    expect_float_values_gpu<double>(f64, { 1.0, 0.0, 1.0, 0.0, 1.0, 0.0 });

    // BOOL -> C32
    Array c32 = src.to(DType::C32);
    expect_complex_values_gpu<float>(c32, {
        {1.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f},
        {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 0.0f}
        });
}

// Test 2: Integer to all types
TEST_F(CastTestGPU, I32ToAll) {
    Array src({ 2, 3 }, DType::I32, GPUPlace(0));
    fill_sequential_gpu<int32_t>(src);  // 0,1,2,3,4,5

    // I32 -> BOOL (non-zero = true)
    Array bool_arr = src.to(DType::BOOL);
    expect_bool_values_gpu(bool_arr, { false, true, true, true, true, true });

    // I32 -> F32
    Array f32 = src.to(DType::F32);
    expect_float_values_gpu<float>(f32, { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });

    // I32 -> F64
    Array f64 = src.to(DType::F64);
    expect_float_values_gpu<double>(f64, { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 });

    // I32 -> C32
    Array c32 = src.to(DType::C32);
    expect_complex_values_gpu<float>(c32, {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {2.0f, 0.0f},
        {3.0f, 0.0f}, {4.0f, 0.0f}, {5.0f, 0.0f}
        });
}

// Test 3: U8 to all types
TEST_F(CastTestGPU, U8ToAll) {
    Array src({ 2, 3 }, DType::U8, GPUPlace(0));
    fill_sequential_gpu<uint8_t>(src);  // 0,1,2,3,4,5

    // U8 -> I32
    Array i32 = src.to(DType::I32);
    expect_int_values_gpu<int32_t>(i32, { 0, 1, 2, 3, 4, 5 });

    // U8 -> F32
    Array f32 = src.to(DType::F32);
    expect_float_values_gpu<float>(f32, { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });

    // U8 -> C32
    Array c32 = src.to(DType::C32);
    expect_complex_values_gpu<float>(c32, {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {2.0f, 0.0f},
        {3.0f, 0.0f}, {4.0f, 0.0f}, {5.0f, 0.0f}
        });
}

// Test 4: F32 to all types
TEST_F(CastTestGPU, F32ToAll) {
    Array src({ 2, 3 }, DType::F32, GPUPlace(0));
    fill_sequential_gpu<float>(src);  // 0,1,2,3,4,5

    // F32 -> BOOL (non-zero = true)
    Array bool_arr = src.to(DType::BOOL);
    expect_bool_values_gpu(bool_arr, { false, true, true, true, true, true });

    // F32 -> I32 (truncation)
    Array i32 = src.to(DType::I32);
    expect_int_values_gpu<int32_t>(i32, { 0, 1, 2, 3, 4, 5 });

    // F32 -> F64
    Array f64 = src.to(DType::F64);
    expect_float_values_gpu<double>(f64, { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 });

    // F32 -> C32
    Array c32 = src.to(DType::C32);
    expect_complex_values_gpu<float>(c32, {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {2.0f, 0.0f},
        {3.0f, 0.0f}, {4.0f, 0.0f}, {5.0f, 0.0f}
        });
}

// Test 5: F64 to all types
TEST_F(CastTestGPU, F64ToAll) {
    Array src({ 2, 3 }, DType::F64, GPUPlace(0));
    fill_sequential_gpu<double>(src);  // 0,1,2,3,4,5

    // F64 -> F32 (precision loss)
    Array f32 = src.to(DType::F32);
    expect_float_values_gpu<float>(f32, { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });

    // F64 -> I32
    Array i32 = src.to(DType::I32);
    expect_int_values_gpu<int32_t>(i32, { 0, 1, 2, 3, 4, 5 });

    // F64 -> C64
    Array c64 = src.to(DType::C64);
    expect_complex_values_gpu<double>(c64, {
        {0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0},
        {3.0, 0.0}, {4.0, 0.0}, {5.0, 0.0}
        });
}

// Test 6: C32 to all types
TEST_F(CastTestGPU, C32ToAll) {
    Array src({ 2, 3 }, DType::C32, GPUPlace(0));
    fill_complex_sequential_gpu<float>(src);  // (0,0), (1,2), (2,4), (3,6), (4,8), (5,10)

    // C32 -> F32 (取实部)
    Array f32 = src.to(DType::F32);
    expect_float_values_gpu<float>(f32, { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });

    // C32 -> BOOL (非零实部或虚部为true)
    Array bool_arr = src.to(DType::BOOL);
    expect_bool_values_gpu(bool_arr, { false, true, true, true, true, true });

    // C32 -> C64
    Array c64 = src.to(DType::C64);
    expect_complex_values_gpu<double>(c64, {
        {0.0, 0.0}, {1.0, 2.0}, {2.0, 4.0},
        {3.0, 6.0}, {4.0, 8.0}, {5.0, 10.0}
        });
}

// Test 7: C64 to all types
TEST_F(CastTestGPU, C64ToAll) {
    Array src({ 2, 3 }, DType::C64, GPUPlace(0));
    Array cpu_src({ 2, 3 }, DType::C64, CPUPlace());
    std::complex<double>* cpu_data = cpu_src.data<std::complex<double>>();
    for (int64_t i = 0; i < 6; ++i) {
        cpu_data[i] = std::complex<double>(static_cast<double>(i), static_cast<double>(i * 2));
    }
    src = cpu_src.to(GPUPlace(0));

    // C64 -> F64 (取实部)
    Array f64 = src.to(DType::F64);
    expect_float_values_gpu<double>(f64, { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 });

    // C64 -> BOOL
    Array bool_arr = src.to(DType::BOOL);
    expect_bool_values_gpu(bool_arr, { false, true, true, true, true, true });

    // C64 -> C32 (precision loss)
    Array c32 = src.to(DType::C32);
    expect_complex_values_gpu<float>(c32, {
        {0.0f, 0.0f}, {1.0f, 2.0f}, {2.0f, 4.0f},
        {3.0f, 6.0f}, {4.0f, 8.0f}, {5.0f, 10.0f}
        });
}

// Test 8: Type promotion in add requires cast
TEST_F(CastTestGPU, AddTypePromotionUsesCast) {
    Array a({ 2, 3 }, DType::I32, GPUPlace(0));
    Array b({ 2, 3 }, DType::F32, GPUPlace(0));

    // Fill on CPU then copy to GPU
    Array cpu_a({ 2, 3 }, DType::I32, CPUPlace());
    Array cpu_b({ 2, 3 }, DType::F32, CPUPlace());

    int32_t* a_data = cpu_a.data<int32_t>();
    float* b_data = cpu_b.data<float>();

    for (int64_t i = 0; i < 6; ++i) {
        a_data[i] = static_cast<int32_t>(i);
        b_data[i] = static_cast<float>(i) * 1.5f;
    }

    a = cpu_a.to(GPUPlace(0));
    b = cpu_b.to(GPUPlace(0));

    // This should internally use cast to convert I32 to F32
    Array c = ins::add(a, b);

    EXPECT_EQ(c.dtype(), DType::F32);

    Array cpu_c = c.to(CPUPlace());
    const float* c_data = cpu_c.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(c_data[i], static_cast<float>(i) + i * 1.5f);
    }
}

// Test 9: Identity cast (no conversion)
TEST_F(CastTestGPU, IdentityCast) {
    Array src({ 2, 3 }, DType::F32, GPUPlace(0));
    fill_sequential_gpu<float>(src);

    Array dst = src.to(DType::F32);

    // Note: On GPU, pointers may be different even for identity cast on different devices
    // but data should be the same
    EXPECT_EQ(dst.dtype(), src.dtype());
    EXPECT_EQ(dst.shape(), src.shape());
    EXPECT_EQ(dst.numel(), src.numel());

    Array cpu_src = src.to(CPUPlace());
    Array cpu_dst = dst.to(CPUPlace());
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(cpu_dst.data<float>()[i], cpu_src.data<float>()[i]);
    }
}

// Test 10: GPU to GPU cast (same device)
TEST_F(CastTestGPU, GPUToGPUCast) {
    Array src({ 2, 3 }, DType::F32, GPUPlace(0));
    fill_sequential_gpu<float>(src);

    Array dst = src.to(DType::F64);  // F32 -> F64 on GPU

    EXPECT_EQ(dst.dtype(), DType::F64);
    EXPECT_EQ(dst.numel(), src.numel());
    EXPECT_TRUE(dst.place().is_gpu());

    Array cpu_dst = dst.to(CPUPlace());
    Array cpu_src = src.to(CPUPlace());
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ(cpu_dst.data<double>()[i], static_cast<double>(cpu_src.data<float>()[i]));
    }
}