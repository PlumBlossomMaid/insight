// tests/cuda/test_operator.cu
#include <gtest/gtest.h>
#include "insight/insight.h"
#include "insight/ops/operator.h"

using namespace ins;

class OperatorTestGPU : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ins::init();
        set_device(GPUPlace(0));
    }
};

// ========== Helper Functions ==========

static Array create_test_array_1d() {
    Array cpu = to_array({ 1.0, 2.0, 3.0, 4.0, 5.0 });
    return cpu.to(GPUPlace(0));
}

static Array create_test_array_2d() {
    Array cpu = to_array({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, Shape({ 2, 3 }));
    return cpu.to(GPUPlace(0));
}

static bool approx_equal(double a, double b, double rtol = 1e-6, double atol = 1e-8) {
    return std::abs(a - b) <= atol + rtol * std::abs(b);
}

static void expect_array_eq(const Array& a, const std::vector<double>& expected) {
    Array a_cpu = a.to(CPUPlace());
    ASSERT_EQ(a_cpu.numel(), static_cast<int64_t>(expected.size()));
    for (int64_t i = 0; i < a_cpu.numel(); ++i) {
        EXPECT_TRUE(approx_equal(a_cpu.at(i).item<double>(), expected[i]));
    }
}

// ========== Unary Operator Tests ==========

TEST_F(OperatorTestGPU, UnaryMinus) {
    Array a = create_test_array_1d();
    Array b = -a;

    expect_array_eq(b, { -1.0, -2.0, -3.0, -4.0, -5.0 });
}

TEST_F(OperatorTestGPU, UnaryPlus) {
    Array a = create_test_array_1d();
    Array b = +a;

    expect_array_eq(b, { 1.0, 2.0, 3.0, 4.0, 5.0 });
    // Unary plus returns a view (shallow copy), shares memory with original
    EXPECT_EQ(reinterpret_cast<uintptr_t>(a.data()),
        reinterpret_cast<uintptr_t>(b.data()));
}

TEST_F(OperatorTestGPU, BitwiseNot) {
    Array cpu = to_array<bool>({ 1, 0, 1, 0 });
    Array a = cpu.to(GPUPlace(0));
    Array b = ~a;
    Array b_cpu = b.to(CPUPlace());
    const bool* data = b_cpu.data<bool>();
    EXPECT_FALSE(data[0]);
    EXPECT_TRUE(data[1]);
    EXPECT_FALSE(data[2]);
    EXPECT_TRUE(data[3]);
}

TEST_F(OperatorTestGPU, LogicalNot) {
    Array cpu = to_array({ 1.0, 0.0, 3.0, 0.0, 5.0 });
    Array a = cpu.to(GPUPlace(0));
    Array b = !a;

    Array b_cpu = b.to(CPUPlace());
    const bool* data = b_cpu.data<bool>();
    EXPECT_FALSE(data[0]);
    EXPECT_TRUE(data[1]);
    EXPECT_FALSE(data[2]);
    EXPECT_TRUE(data[3]);
    EXPECT_FALSE(data[4]);
}

// ========== Arithmetic Operators (Array-Array) ==========

TEST_F(OperatorTestGPU, AddArrayArray) {
    Array a = create_test_array_1d();
    Array b = create_test_array_1d();
    Array c = a + b;

    expect_array_eq(c, { 2.0, 4.0, 6.0, 8.0, 10.0 });
}

TEST_F(OperatorTestGPU, SubArrayArray) {
    Array a = create_test_array_1d();
    Array b = create_test_array_1d();
    Array c = a - b;

    expect_array_eq(c, { 0.0, 0.0, 0.0, 0.0, 0.0 });
}

TEST_F(OperatorTestGPU, MulArrayArray) {
    Array a = create_test_array_1d();
    Array b = create_test_array_1d();
    Array c = a * b;

    expect_array_eq(c, { 1.0, 4.0, 9.0, 16.0, 25.0 });
}

TEST_F(OperatorTestGPU, DivArrayArray) {
    Array a = create_test_array_1d();
    Array cpu = to_array({ 1.0, 2.0, 3.0, 4.0, 5.0 });
    Array b = cpu.to(GPUPlace(0));
    Array c = a / b;

    expect_array_eq(c, { 1.0, 1.0, 1.0, 1.0, 1.0 });
}

TEST_F(OperatorTestGPU, ModArrayArray) {
    Array cpu_a = to_array<int64_t>({ 5, 7, 9, 11 });
    Array cpu_b = to_array<int64_t>({ 2, 3, 4, 5 });
    Array a = cpu_a.to(GPUPlace(0));
    Array b = cpu_b.to(GPUPlace(0));
    Array c = a % b;

    Array c_cpu = c.to(CPUPlace());
    const int64_t* data = c_cpu.data<int64_t>();
    EXPECT_EQ(data[0], 1);
    EXPECT_EQ(data[1], 1);
    EXPECT_EQ(data[2], 1);
    EXPECT_EQ(data[3], 1);
}

// ========== Arithmetic Operators (Array-Scalar) ==========

TEST_F(OperatorTestGPU, AddArrayScalar) {
    Array a = create_test_array_1d();
    Array b = a + 10.0;

    expect_array_eq(b, { 11.0, 12.0, 13.0, 14.0, 15.0 });
}

TEST_F(OperatorTestGPU, AddScalarArray) {
    Array a = create_test_array_1d();
    Array b = 10.0 + a;

    expect_array_eq(b, { 11.0, 12.0, 13.0, 14.0, 15.0 });
}

TEST_F(OperatorTestGPU, SubArrayScalar) {
    Array a = create_test_array_1d();
    Array b = a - 1.0;

    expect_array_eq(b, { 0.0, 1.0, 2.0, 3.0, 4.0 });
}

TEST_F(OperatorTestGPU, SubScalarArray) {
    Array a = create_test_array_1d();
    Array b = 10.0 - a;

    expect_array_eq(b, { 9.0, 8.0, 7.0, 6.0, 5.0 });
}

TEST_F(OperatorTestGPU, MulArrayScalar) {
    Array a = create_test_array_1d();
    Array b = a * 2.0;

    expect_array_eq(b, { 2.0, 4.0, 6.0, 8.0, 10.0 });
}

TEST_F(OperatorTestGPU, MulScalarArray) {
    Array a = create_test_array_1d();
    Array b = 2.0 * a;

    expect_array_eq(b, { 2.0, 4.0, 6.0, 8.0, 10.0 });
}

TEST_F(OperatorTestGPU, DivArrayScalar) {
    Array a = create_test_array_1d();
    Array b = a / 2.0;

    expect_array_eq(b, { 0.5, 1.0, 1.5, 2.0, 2.5 });
}

TEST_F(OperatorTestGPU, DivScalarArray) {
    Array a = create_test_array_1d();
    Array b = 10.0 / a;

    expect_array_eq(b, { 10.0, 5.0, 10.0 / 3.0, 2.5, 2.0 });
}

// ========== Comparison Operators (Array-Array) ==========

TEST_F(OperatorTestGPU, EqualArrayArray) {
    Array a = create_test_array_1d();
    Array b = create_test_array_1d();
    Array c = a == b;

    Array c_cpu = c.to(CPUPlace());
    const bool* data = c_cpu.data<bool>();
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(data[i]);
    }
}

TEST_F(OperatorTestGPU, NotEqualArrayArray) {
    Array a = create_test_array_1d();
    Array b = a + 1.0;
    Array c = a != b;

    Array c_cpu = c.to(CPUPlace());
    const bool* data = c_cpu.data<bool>();
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(data[i]);
    }
}

TEST_F(OperatorTestGPU, LessArrayArray) {
    Array a = create_test_array_1d();
    Array b = a + 1.0;
    Array c = a < b;

    Array c_cpu = c.to(CPUPlace());
    const bool* data = c_cpu.data<bool>();
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(data[i]);
    }
}

TEST_F(OperatorTestGPU, GreaterArrayArray) {
    Array a = create_test_array_1d();
    Array b = a + 1.0;
    Array c = a > b;

    Array c_cpu = c.to(CPUPlace());
    const bool* data = c_cpu.data<bool>();
    for (int i = 0; i < 5; ++i) {
        EXPECT_FALSE(data[i]);
    }
}

TEST_F(OperatorTestGPU, LessEqualArrayArray) {
    Array a = create_test_array_1d();
    Array b = a.copy();
    Array c = a <= b;

    Array c_cpu = c.to(CPUPlace());
    const bool* data = c_cpu.data<bool>();
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(data[i]);
    }
}

TEST_F(OperatorTestGPU, GreaterEqualArrayArray) {
    Array a = create_test_array_1d();
    Array b = a.copy();
    Array c = a >= b;

    Array c_cpu = c.to(CPUPlace());
    const bool* data = c_cpu.data<bool>();
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(data[i]);
    }
}

// ========== Comparison Operators (Array-Scalar) ==========

TEST_F(OperatorTestGPU, EqualArrayScalar) {
    Array cpu = to_array({ 3.0, 4.0, 5.0, 4.0, 3.0 });
    Array a = cpu.to(GPUPlace(0));
    Array c = a == 4.0;

    Array c_cpu = c.to(CPUPlace());
    const bool* data = c_cpu.data<bool>();
    EXPECT_FALSE(data[0]);
    EXPECT_TRUE(data[1]);
    EXPECT_FALSE(data[2]);
    EXPECT_TRUE(data[3]);
    EXPECT_FALSE(data[4]);
}

TEST_F(OperatorTestGPU, EqualScalarArray) {
    Array cpu = to_array({ 3.0, 4.0, 5.0, 4.0, 3.0 });
    Array a = cpu.to(GPUPlace(0));
    Array c = 4.0 == a;

    Array c_cpu = c.to(CPUPlace());
    const bool* data = c_cpu.data<bool>();
    EXPECT_FALSE(data[0]);
    EXPECT_TRUE(data[1]);
    EXPECT_FALSE(data[2]);
    EXPECT_TRUE(data[3]);
    EXPECT_FALSE(data[4]);
}

TEST_F(OperatorTestGPU, LessArrayScalar) {
    Array a = create_test_array_1d();
    Array c = a < 3.0;

    Array c_cpu = c.to(CPUPlace());
    const bool* data = c_cpu.data<bool>();
    EXPECT_TRUE(data[0]);
    EXPECT_TRUE(data[1]);
    EXPECT_FALSE(data[2]);
    EXPECT_FALSE(data[3]);
    EXPECT_FALSE(data[4]);
}

TEST_F(OperatorTestGPU, LessScalarArray) {
    Array a = create_test_array_1d();
    Array c = 3.0 < a;

    Array c_cpu = c.to(CPUPlace());
    const bool* data = c_cpu.data<bool>();
    EXPECT_FALSE(data[0]);
    EXPECT_FALSE(data[1]);
    EXPECT_FALSE(data[2]);
    EXPECT_TRUE(data[3]);
    EXPECT_TRUE(data[4]);
}

// ========== Compound Assignment Operators ==========

TEST_F(OperatorTestGPU, AddAssignArray) {
    Array a = create_test_array_1d();
    Array b = create_test_array_1d();

    a += b;
    expect_array_eq(a, { 2.0, 4.0, 6.0, 8.0, 10.0 });
}

TEST_F(OperatorTestGPU, SubAssignArray) {
    Array a = create_test_array_1d();
    Array b = create_test_array_1d();

    a -= b;
    expect_array_eq(a, { 0.0, 0.0, 0.0, 0.0, 0.0 });
}

TEST_F(OperatorTestGPU, MulAssignArray) {
    Array a = create_test_array_1d();
    Array b = create_test_array_1d();

    a *= b;
    expect_array_eq(a, { 1.0, 4.0, 9.0, 16.0, 25.0 });
}

TEST_F(OperatorTestGPU, DivAssignArray) {
    Array a = create_test_array_1d();
    Array b = create_test_array_1d();

    a /= b;
    expect_array_eq(a, { 1.0, 1.0, 1.0, 1.0, 1.0 });
}

TEST_F(OperatorTestGPU, AddAssignScalar) {
    Array a = create_test_array_1d();

    a += 10.0;
    expect_array_eq(a, { 11.0, 12.0, 13.0, 14.0, 15.0 });
}

TEST_F(OperatorTestGPU, SubAssignScalar) {
    Array a = create_test_array_1d();

    a -= 1.0;
    expect_array_eq(a, { 0.0, 1.0, 2.0, 3.0, 4.0 });
}

TEST_F(OperatorTestGPU, MulAssignScalar) {
    Array a = create_test_array_1d();

    a *= 2.0;
    expect_array_eq(a, { 2.0, 4.0, 6.0, 8.0, 10.0 });
}

TEST_F(OperatorTestGPU, DivAssignScalar) {
    Array a = create_test_array_1d();

    a /= 2.0;
    expect_array_eq(a, { 0.5, 1.0, 1.5, 2.0, 2.5 });
}

// ========== Increment/Decrement Operators ==========

TEST_F(OperatorTestGPU, PrefixIncrement) {
    Array a = create_test_array_1d();
    Array& ref = ++a;

    expect_array_eq(a, { 2.0, 3.0, 4.0, 5.0, 6.0 });
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ref.data()),
        reinterpret_cast<uintptr_t>(a.data()));
}

TEST_F(OperatorTestGPU, PostfixIncrement) {
    Array a = create_test_array_1d();
    Array b = a++;

    expect_array_eq(b, { 1.0, 2.0, 3.0, 4.0, 5.0 });
    expect_array_eq(a, { 2.0, 3.0, 4.0, 5.0, 6.0 });
}

TEST_F(OperatorTestGPU, PrefixDecrement) {
    Array a = create_test_array_1d();
    Array& ref = --a;

    expect_array_eq(a, { 0.0, 1.0, 2.0, 3.0, 4.0 });
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ref.data()),
        reinterpret_cast<uintptr_t>(a.data()));
}

TEST_F(OperatorTestGPU, PostfixDecrement) {
    Array a = create_test_array_1d();
    Array b = a--;

    expect_array_eq(b, { 1.0, 2.0, 3.0, 4.0, 5.0 });
    expect_array_eq(a, { 0.0, 1.0, 2.0, 3.0, 4.0 });
}

// ========== Chain Operations ==========

TEST_F(OperatorTestGPU, ChainArithmetic) {
    Array a = create_test_array_1d();
    Array b = create_test_array_1d();
    Array c = a + b * 2.0 - 3.0;

    expect_array_eq(c, { 0.0, 3.0, 6.0, 9.0, 12.0 });
}

TEST_F(OperatorTestGPU, ChainComparison) {
    Array cpu = to_array<double>({ 1.0, 2.0, 3.0, 4.0, 5.0 });
    Array a = cpu.to(GPUPlace(0));
    Array mask = (a > 1) & (a < 5);

    Array mask_cpu = mask.to(CPUPlace());
    const bool* data = mask_cpu.data<bool>();
    EXPECT_FALSE(data[0]);
    EXPECT_TRUE(data[1]);
    EXPECT_TRUE(data[2]);
    EXPECT_TRUE(data[3]);
    EXPECT_FALSE(data[4]);
}

// ========== Array::bool() Tests ==========

TEST_F(OperatorTestGPU, ScalarIntTrue) {
    Array a(42);
    EXPECT_TRUE(static_cast<bool>(a));
}

TEST_F(OperatorTestGPU, ScalarIntFalse) {
    Array a(0);
    EXPECT_FALSE(static_cast<bool>(a));
}

TEST_F(OperatorTestGPU, ScalarBoolTrue) {
    Array a(3.14);
    EXPECT_TRUE(static_cast<bool>(a));
    EXPECT_TRUE(a.any());
    EXPECT_TRUE(a.all());
}

TEST_F(OperatorTestGPU, ScalarBoolFalse) {
    Array a(0.0);
    EXPECT_FALSE(static_cast<bool>(a));
    EXPECT_FALSE(a.any());
    EXPECT_FALSE(a.all());
}

TEST_F(OperatorTestGPU, MultiElementBoolThrows) {
    Array a = create_test_array_1d();
    EXPECT_ANY_THROW(static_cast<bool>(a));
}

TEST_F(OperatorTestGPU, AnyAllOnMultiElement) {
    Array a = create_test_array_1d();
    EXPECT_TRUE(a.any());
    EXPECT_TRUE(a.all());

    Array cpu = to_array({ 0.0, 1.0, 0.0, 2.0, 0.0 });
    Array b = cpu.to(GPUPlace(0));
    EXPECT_TRUE(b.any());
    EXPECT_FALSE(b.all());

    Array c = zeros({ 3, 3 }, DType::F32, GPUPlace(0));
    EXPECT_FALSE(c.any());
    EXPECT_FALSE(c.all());
}

// ========== Edge Cases ==========

TEST_F(OperatorTestGPU, BroadcastingInOperations) {
    Array a = create_test_array_2d();
    Array cpu = to_array({ 10.0, 20.0, 30.0 });
    Array b = cpu.to(GPUPlace(0));
    Array c = a + b;

    Array c_cpu = c.to(CPUPlace());
    const double* data = c_cpu.data<double>();
    EXPECT_NEAR(data[0], 1 + 10, 1e-6);
    EXPECT_NEAR(data[1], 2 + 20, 1e-6);
    EXPECT_NEAR(data[2], 3 + 30, 1e-6);
    EXPECT_NEAR(data[3], 4 + 10, 1e-6);
    EXPECT_NEAR(data[4], 5 + 20, 1e-6);
    EXPECT_NEAR(data[5], 6 + 30, 1e-6);
}

TEST_F(OperatorTestGPU, DivisionByScalarZero) {
    Array a = create_test_array_1d();
    Array b = a / 0.0;
    Array b_cpu = b.to(CPUPlace());
    const double* data = b_cpu.data<double>();
    EXPECT_TRUE(std::isinf(data[0]));
}