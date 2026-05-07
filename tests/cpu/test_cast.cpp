// tests/cpu/test_cast.cpp
#include <gtest/gtest.h>
#include "insight/core/array.h"
#include "insight/ops/elementwise.h"
#include <complex>

using namespace ins;

// Helper: fill array with sequential values
template<typename T>
void fill_sequential(Array& arr) {
    T* data = arr.data<T>();
    int64_t n = arr.numel();
    for (int64_t i = 0; i < n; ++i) {
        data[i] = static_cast<T>(i);
    }
}

// Helper: fill bool array with alternating values
void fill_bool_alternating(Array& arr) {
    bool* data = arr.data<bool>();
    int64_t n = arr.numel();
    for (int64_t i = 0; i < n; ++i) {
        data[i] = (i % 2 == 0);
    }
}

// Helper: fill complex array with sequential values
template<typename T>
void fill_complex_sequential(Array& arr) {
    std::complex<T>* data = arr.data<std::complex<T>>();
    int64_t n = arr.numel();
    for (int64_t i = 0; i < n; ++i) {
        data[i] = std::complex<T>(static_cast<T>(i), static_cast<T>(i * 2));
    }
}

// Helper: verify float values within tolerance
template<typename T>
void expect_float_values(const Array& arr, const std::vector<T>& expected, T tol = 1e-6) {
    ASSERT_EQ(arr.numel(), static_cast<int64_t>(expected.size()));
    const T* data = arr.data<T>();
    for (int64_t i = 0; i < arr.numel(); ++i) {
        EXPECT_NEAR(data[i], expected[i], tol);
    }
}

// Helper: verify integer values
template<typename T>
void expect_int_values(const Array& arr, const std::vector<T>& expected) {
    ASSERT_EQ(arr.numel(), static_cast<int64_t>(expected.size()));
    const T* data = arr.data<T>();
    for (int64_t i = 0; i < arr.numel(); ++i) {
        EXPECT_EQ(data[i], expected[i]);
    }
}

// Helper: verify bool values
void expect_bool_values(const Array& arr, const std::vector<bool>& expected) {
    ASSERT_EQ(arr.numel(), static_cast<int64_t>(expected.size()));
    const bool* data = arr.data<bool>();
    for (int64_t i = 0; i < arr.numel(); ++i) {
        EXPECT_EQ(data[i], expected[i]);
    }
}

// Helper: verify complex values
template<typename T>
void expect_complex_values(const Array& arr, const std::vector<std::complex<T>>& expected, T tol = 1e-6) {
    ASSERT_EQ(arr.numel(), static_cast<int64_t>(expected.size()));
    const std::complex<T>* data = arr.data<std::complex<T>>();
    for (int64_t i = 0; i < arr.numel(); ++i) {
        EXPECT_NEAR(data[i].real(), expected[i].real(), tol);
        EXPECT_NEAR(data[i].imag(), expected[i].imag(), tol);
    }
}

// ========== Cast Tests ==========

// Test 1: BOOL to all types
TEST(CastTest, BoolToAll) {
    Array src({ 2, 3 }, DType::BOOL);
    fill_bool_alternating(src);  // [true, false, true, false, true, false]

    // BOOL -> U8
    Array u8 = src.to(DType::U8);
    expect_int_values<uint8_t>(u8, { 1, 0, 1, 0, 1, 0 });

    // BOOL -> I32
    Array i32 = src.to(DType::I32);
    expect_int_values<int32_t>(i32, { 1, 0, 1, 0, 1, 0 });

    // BOOL -> I64
    Array i64 = src.to(DType::I64);
    expect_int_values<int64_t>(i64, { 1, 0, 1, 0, 1, 0 });

    // BOOL -> F32
    Array f32 = src.to(DType::F32);
    expect_float_values<float>(f32, { 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f });

    // BOOL -> F64
    Array f64 = src.to(DType::F64);
    expect_float_values<double>(f64, { 1.0, 0.0, 1.0, 0.0, 1.0, 0.0 });

    // BOOL -> C32
    Array c32 = src.to(DType::C32);
    expect_complex_values<float>(c32, {
        {1.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f},
        {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 0.0f}
        });
}

// Test 2: Integer to all types
TEST(CastTest, I32ToAll) {
    Array src({ 2, 3 }, DType::I32);
    fill_sequential<int32_t>(src);  // 0,1,2,3,4,5

    // I32 -> BOOL (non-zero = true)
    Array bool_arr = src.to(DType::BOOL);
    expect_bool_values(bool_arr, { false, true, true, true, true, true });

    // I32 -> F32
    Array f32 = src.to(DType::F32);
    expect_float_values<float>(f32, { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });

    // I32 -> F64
    Array f64 = src.to(DType::F64);
    expect_float_values<double>(f64, { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 });

    // I32 -> C32
    Array c32 = src.to(DType::C32);
    expect_complex_values<float>(c32, {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {2.0f, 0.0f},
        {3.0f, 0.0f}, {4.0f, 0.0f}, {5.0f, 0.0f}
        });
}

// Test 3: U8 to all types
TEST(CastTest, U8ToAll) {
    Array src({ 2, 3 }, DType::U8);
    fill_sequential<uint8_t>(src);  // 0,1,2,3,4,5

    // U8 -> I32
    Array i32 = src.to(DType::I32);
    expect_int_values<int32_t>(i32, { 0, 1, 2, 3, 4, 5 });

    // U8 -> F32
    Array f32 = src.to(DType::F32);
    expect_float_values<float>(f32, { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });

    // U8 -> C32
    Array c32 = src.to(DType::C32);
    expect_complex_values<float>(c32, {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {2.0f, 0.0f},
        {3.0f, 0.0f}, {4.0f, 0.0f}, {5.0f, 0.0f}
        });
}

// Test 4: F32 to all types
TEST(CastTest, F32ToAll) {
    Array src({ 2, 3 }, DType::F32);
    fill_sequential<float>(src);  // 0,1,2,3,4,5

    // F32 -> BOOL (non-zero = true)
    Array bool_arr = src.to(DType::BOOL);
    expect_bool_values(bool_arr, { false, true, true, true, true, true });

    // F32 -> I32 (truncation)
    Array i32 = src.to(DType::I32);
    expect_int_values<int32_t>(i32, { 0, 1, 2, 3, 4, 5 });

    // F32 -> F64
    Array f64 = src.to(DType::F64);
    expect_float_values<double>(f64, { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 });

    // F32 -> C32
    Array c32 = src.to(DType::C32);
    expect_complex_values<float>(c32, {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {2.0f, 0.0f},
        {3.0f, 0.0f}, {4.0f, 0.0f}, {5.0f, 0.0f}
        });
}

// Test 5: F64 to all types
TEST(CastTest, F64ToAll) {
    Array src({ 2, 3 }, DType::F64);
    fill_sequential<double>(src);  // 0,1,2,3,4,5

    // F64 -> F32 (precision loss)
    Array f32 = src.to(DType::F32);
    expect_float_values<float>(f32, { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });

    // F64 -> I32
    Array i32 = src.to(DType::I32);
    expect_int_values<int32_t>(i32, { 0, 1, 2, 3, 4, 5 });

    // F64 -> C64
    Array c64 = src.to(DType::C64);
    expect_complex_values<double>(c64, {
        {0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0},
        {3.0, 0.0}, {4.0, 0.0}, {5.0, 0.0}
        });
}

// Test 6: C32 to all types
TEST(CastTest, C32ToAll) {
    Array src({ 2, 3 }, DType::C32);
    fill_complex_sequential<float>(src);  // (0,0), (1,2), (2,4), (3,6), (4,8), (5,10)

    // C32 -> F32 (取实部)
    Array f32 = src.to(DType::F32);
    expect_float_values<float>(f32, { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });

    // C32 -> BOOL (非零实部或虚部为true)
    Array bool_arr = src.to(DType::BOOL);
    expect_bool_values(bool_arr, { false, true, true, true, true, true });

    // C32 -> C64
    Array c64 = src.to(DType::C64);
    expect_complex_values<double>(c64, {
        {0.0, 0.0}, {1.0, 2.0}, {2.0, 4.0},
        {3.0, 6.0}, {4.0, 8.0}, {5.0, 10.0}
        });
}

// Test 7: C64 to all types
TEST(CastTest, C64ToAll) {
    Array src({ 2, 3 }, DType::C64);
    std::complex<double>* data = src.data<std::complex<double>>();
    for (int64_t i = 0; i < 6; ++i) {
        data[i] = std::complex<double>(static_cast<double>(i), static_cast<double>(i * 2));
    }

    // C64 -> F64 (取实部)
    Array f64 = src.to(DType::F64);
    expect_float_values<double>(f64, { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 });

    // C64 -> BOOL
    Array bool_arr = src.to(DType::BOOL);
    expect_bool_values(bool_arr, { false, true, true, true, true, true });

    // C64 -> C32 (precision loss)
    Array c32 = src.to(DType::C32);
    expect_complex_values<float>(c32, {
        {0.0f, 0.0f}, {1.0f, 2.0f}, {2.0f, 4.0f},
        {3.0f, 6.0f}, {4.0f, 8.0f}, {5.0f, 10.0f}
        });
}

// Test 8: Type promotion in add requires cast
TEST(CastTest, AddTypePromotionUsesCast) {
    Array a({ 2, 3 }, DType::I32);
    Array b({ 2, 3 }, DType::F32);

    int32_t* a_data = a.data<int32_t>();
    float* b_data = b.data<float>();

    for (int64_t i = 0; i < 6; ++i) {
        a_data[i] = static_cast<int32_t>(i);
        b_data[i] = static_cast<float>(i) * 1.5f;
    }

    // This should internally use cast to convert I32 to F32
    Array c = ins::add(a, b);

    EXPECT_EQ(c.dtype(), DType::F32);
    const float* c_data = c.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(c_data[i], static_cast<float>(i) + i * 1.5f);
    }
}

// Test 9: Identity cast (no conversion)
TEST(CastTest, IdentityCast) {
    set_device(ins::CPUPlace());
    Array src({ 2, 3 }, DType::F32);
    fill_sequential<float>(src);

    Array dst = src.to(DType::F32);

    EXPECT_EQ(dst.data<float>(), src.data<float>());  // Same pointer
    EXPECT_EQ(dst.numel(), src.numel());
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(dst.data<float>()[i], src.data<float>()[i]);
    }
}