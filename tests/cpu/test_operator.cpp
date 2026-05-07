// tests/cpu/test_operator.cpp
#include <gtest/gtest.h>
#include "insight/insight.h"
#include "insight/ops/operator.h"

using namespace ins;

class OperatorTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ins::init();
        set_device(CPUPlace());
    }
};

// ========== Helper Functions ==========

static Array create_test_array_1d() {
    return to_array({ 1.0, 2.0, 3.0, 4.0, 5.0 });
}

static Array create_test_array_2d() {
    return to_array({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, Shape({ 2, 3 }));
}

static bool approx_equal(double a, double b, double rtol = 1e-6, double atol = 1e-8) {
    return std::abs(a - b) <= atol + rtol * std::abs(b);
}

static void expect_array_eq(const Array& a, const std::vector<double>& expected) {
    ASSERT_EQ(a.numel(), static_cast<int64_t>(expected.size()));
    for (int64_t i = 0; i < a.numel(); ++i) {
        EXPECT_TRUE(approx_equal(a.at(i).item<double>(), expected[i]));
    }
}

// ========== Unary Operator Tests ==========

TEST_F(OperatorTest, UnaryMinus) {
    Array a = create_test_array_1d();
    Array b = -a;

    expect_array_eq(b, { -1.0, -2.0, -3.0, -4.0, -5.0 });
}

TEST_F(OperatorTest, UnaryPlus) {
    Array a = create_test_array_1d();
    Array b = +a;

    expect_array_eq(b, { 1.0, 2.0, 3.0, 4.0, 5.0 });
    EXPECT_EQ(a.data(), b.data());

    b.data<double>()[0] = 99.0;
    EXPECT_EQ(a.at(0).item<double>(), 99.0);
}

TEST_F(OperatorTest, BitwiseNot) {
    Array a = to_array<bool>({ 1, 0, 1, 0 });
    Array b = ~a;
    const bool* data = b.data<bool>();
    EXPECT_FALSE(data[0]);
    EXPECT_TRUE(data[1]);
    EXPECT_FALSE(data[2]);
    EXPECT_TRUE(data[3]);
}

TEST_F(OperatorTest, LogicalNot) {
    Array a = to_array({ 1.0, 0.0, 3.0, 0.0, 5.0 });
    Array b = !a;

    const bool* data = b.data<bool>();
    EXPECT_FALSE(data[0]);  // 1.0 != 0 -> false (!true)
    EXPECT_TRUE(data[1]);   // 0.0 == 0 -> true (!false)
    EXPECT_FALSE(data[2]);  // 3.0 != 0 -> false
    EXPECT_TRUE(data[3]);   // 0.0 == 0 -> true
    EXPECT_FALSE(data[4]);  // 5.0 != 0 -> false
}

// ========== Arithmetic Operators (Array-Array) ==========

TEST_F(OperatorTest, AddArrayArray) {
    Array a = create_test_array_1d();
    Array b = create_test_array_1d();
    Array c = a + b;

    expect_array_eq(c, { 2.0, 4.0, 6.0, 8.0, 10.0 });
}

TEST_F(OperatorTest, SubArrayArray) {
    Array a = create_test_array_1d();
    Array b = create_test_array_1d();
    Array c = a - b;

    expect_array_eq(c, { 0.0, 0.0, 0.0, 0.0, 0.0 });
}

TEST_F(OperatorTest, MulArrayArray) {
    Array a = create_test_array_1d();
    Array b = create_test_array_1d();
    Array c = a * b;

    expect_array_eq(c, { 1.0, 4.0, 9.0, 16.0, 25.0 });
}

TEST_F(OperatorTest, DivArrayArray) {
    Array a = create_test_array_1d();
    Array b = to_array({ 1.0, 2.0, 3.0, 4.0, 5.0 });
    Array c = a / b;

    expect_array_eq(c, { 1.0, 1.0, 1.0, 1.0, 1.0 });
}

TEST_F(OperatorTest, ModArrayArray) {
    Array a = to_array<int64_t>({ 5, 7, 9, 11 });
    Array b = to_array<int64_t>({ 2, 3, 4, 5 });
    Array c = a % b;

    const int64_t* data = c.data<int64_t>();
    EXPECT_EQ(data[0], 1);
    EXPECT_EQ(data[1], 1);
    EXPECT_EQ(data[2], 1);
    EXPECT_EQ(data[3], 1);
}

// ========== Arithmetic Operators (Array-Scalar) ==========

TEST_F(OperatorTest, AddArrayScalar) {
    Array a = create_test_array_1d();
    Array b = a + 10.0;

    expect_array_eq(b, { 11.0, 12.0, 13.0, 14.0, 15.0 });
}

TEST_F(OperatorTest, AddScalarArray) {
    Array a = create_test_array_1d();
    Array b = 10.0 + a;

    expect_array_eq(b, { 11.0, 12.0, 13.0, 14.0, 15.0 });
}

TEST_F(OperatorTest, SubArrayScalar) {
    Array a = create_test_array_1d();
    Array b = a - 1.0;

    expect_array_eq(b, { 0.0, 1.0, 2.0, 3.0, 4.0 });
}

TEST_F(OperatorTest, SubScalarArray) {
    Array a = create_test_array_1d();
    Array b = 10.0 - a;

    expect_array_eq(b, { 9.0, 8.0, 7.0, 6.0, 5.0 });
}

TEST_F(OperatorTest, MulArrayScalar) {
    Array a = create_test_array_1d();
    Array b = a * 2.0;

    expect_array_eq(b, { 2.0, 4.0, 6.0, 8.0, 10.0 });
}

TEST_F(OperatorTest, MulScalarArray) {
    Array a = create_test_array_1d();
    Array b = 2.0 * a;

    expect_array_eq(b, { 2.0, 4.0, 6.0, 8.0, 10.0 });
}

TEST_F(OperatorTest, DivArrayScalar) {
    Array a = create_test_array_1d();
    Array b = a / 2.0;

    expect_array_eq(b, { 0.5, 1.0, 1.5, 2.0, 2.5 });
}

TEST_F(OperatorTest, DivScalarArray) {
    Array a = create_test_array_1d();
    Array b = 10.0 / a;

    expect_array_eq(b, { 10.0, 5.0, 10.0 / 3.0, 2.5, 2.0 });
}

// ========== Comparison Operators (Array-Array) ==========

TEST_F(OperatorTest, EqualArrayArray) {
    Array a = create_test_array_1d();
    Array b = create_test_array_1d();
    Array c = a == b;

    const bool* data = c.data<bool>();
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(data[i]);
    }
}

TEST_F(OperatorTest, NotEqualArrayArray) {
    Array a = create_test_array_1d();
    Array b = a + 1.0;
    Array c = a != b;

    const bool* data = c.data<bool>();
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(data[i]);
    }
}

TEST_F(OperatorTest, LessArrayArray) {
    Array a = create_test_array_1d();
    Array b = a + 1.0;
    Array c = a < b;

    const bool* data = c.data<bool>();
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(data[i]);
    }
}

TEST_F(OperatorTest, GreaterArrayArray) {
    Array a = create_test_array_1d();
    Array b = a + 1.0;
    Array c = a > b;

    const bool* data = c.data<bool>();
    for (int i = 0; i < 5; ++i) {
        EXPECT_FALSE(data[i]);
    }
}

TEST_F(OperatorTest, LessEqualArrayArray) {
    Array a = create_test_array_1d();
    Array b = a.copy();
    Array c = a <= b;

    const bool* data = c.data<bool>();
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(data[i]);
    }
}

TEST_F(OperatorTest, GreaterEqualArrayArray) {
    Array a = create_test_array_1d();
    Array b = a.copy();
    Array c = a >= b;

    const bool* data = c.data<bool>();
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(data[i]);
    }
}

// ========== Comparison Operators (Array-Scalar) ==========

TEST_F(OperatorTest, EqualArrayScalar) {
    Array a = to_array({ 3.0, 4.0, 5.0, 4.0, 3.0 });
    Array c = a == 4.0;

    const bool* data = c.data<bool>();
    EXPECT_FALSE(data[0]);
    EXPECT_TRUE(data[1]);
    EXPECT_FALSE(data[2]);
    EXPECT_TRUE(data[3]);
    EXPECT_FALSE(data[4]);
}

TEST_F(OperatorTest, EqualScalarArray) {
    Array a = to_array({ 3.0, 4.0, 5.0, 4.0, 3.0 });
    Array c = 4.0 == a;

    const bool* data = c.data<bool>();
    EXPECT_FALSE(data[0]);
    EXPECT_TRUE(data[1]);
    EXPECT_FALSE(data[2]);
    EXPECT_TRUE(data[3]);
    EXPECT_FALSE(data[4]);
}

TEST_F(OperatorTest, LessArrayScalar) {
    Array a = create_test_array_1d();
    Array c = a < 3.0;

    const bool* data = c.data<bool>();
    EXPECT_TRUE(data[0]);   // 1 < 3
    EXPECT_TRUE(data[1]);   // 2 < 3
    EXPECT_FALSE(data[2]);  // 3 < 3
    EXPECT_FALSE(data[3]);  // 4 < 3
    EXPECT_FALSE(data[4]);  // 5 < 3
}

TEST_F(OperatorTest, LessScalarArray) {
    Array a = create_test_array_1d();
    Array c = 3.0 < a;

    const bool* data = c.data<bool>();
    EXPECT_FALSE(data[0]);  // 3 < 1
    EXPECT_FALSE(data[1]);  // 3 < 2
    EXPECT_FALSE(data[2]);  // 3 < 3
    EXPECT_TRUE(data[3]);   // 3 < 4
    EXPECT_TRUE(data[4]);   // 3 < 5
}

// ========== Compound Assignment Operators ==========

TEST_F(OperatorTest, AddAssignArray) {
    Array a = create_test_array_1d();
    Array b = create_test_array_1d();

    a += b;
    expect_array_eq(a, { 2.0, 4.0, 6.0, 8.0, 10.0 });
}

TEST_F(OperatorTest, SubAssignArray) {
    Array a = create_test_array_1d();
    Array b = create_test_array_1d();

    a -= b;
    expect_array_eq(a, { 0.0, 0.0, 0.0, 0.0, 0.0 });
}

TEST_F(OperatorTest, MulAssignArray) {
    Array a = create_test_array_1d();
    Array b = create_test_array_1d();

    a *= b;
    expect_array_eq(a, { 1.0, 4.0, 9.0, 16.0, 25.0 });
}

TEST_F(OperatorTest, DivAssignArray) {
    Array a = create_test_array_1d();
    Array b = create_test_array_1d();

    a /= b;
    expect_array_eq(a, { 1.0, 1.0, 1.0, 1.0, 1.0 });
}

TEST_F(OperatorTest, AddAssignScalar) {
    Array a = create_test_array_1d();

    a += 10.0;
    expect_array_eq(a, { 11.0, 12.0, 13.0, 14.0, 15.0 });
}

TEST_F(OperatorTest, SubAssignScalar) {
    Array a = create_test_array_1d();

    a -= 1.0;
    expect_array_eq(a, { 0.0, 1.0, 2.0, 3.0, 4.0 });
}

TEST_F(OperatorTest, MulAssignScalar) {
    Array a = create_test_array_1d();

    a *= 2.0;
    expect_array_eq(a, { 2.0, 4.0, 6.0, 8.0, 10.0 });
}

TEST_F(OperatorTest, DivAssignScalar) {
    Array a = create_test_array_1d();

    a /= 2.0;
    expect_array_eq(a, { 0.5, 1.0, 1.5, 2.0, 2.5 });
}

// ========== Increment/Decrement Operators ==========

TEST_F(OperatorTest, PrefixIncrement) {
    Array a = create_test_array_1d();
    Array& ref = ++a;

    expect_array_eq(a, { 2.0, 3.0, 4.0, 5.0, 6.0 });
    EXPECT_EQ(ref.data(), a.data());  // Returns reference to same array
}

TEST_F(OperatorTest, PostfixIncrement) {
    Array a = create_test_array_1d();
    Array b = a++;

    expect_array_eq(b, { 1.0, 2.0, 3.0, 4.0, 5.0 });  // Original values
    expect_array_eq(a, { 2.0, 3.0, 4.0, 5.0, 6.0 });  // Incremented
    EXPECT_NE(b.data(), a.data());  // Different objects
}

TEST_F(OperatorTest, PrefixDecrement) {
    Array a = create_test_array_1d();
    Array& ref = --a;

    expect_array_eq(a, { 0.0, 1.0, 2.0, 3.0, 4.0 });
    EXPECT_EQ(ref.data(), a.data());
}

TEST_F(OperatorTest, PostfixDecrement) {
    Array a = create_test_array_1d();
    Array b = a--;

    expect_array_eq(b, { 1.0, 2.0, 3.0, 4.0, 5.0 });
    expect_array_eq(a, { 0.0, 1.0, 2.0, 3.0, 4.0 });
}

// ========== Chain Operations ==========

TEST_F(OperatorTest, ChainArithmetic) {
    Array a = create_test_array_1d();
    Array b = create_test_array_1d();
    Array c = a + b * 2.0 - 3.0;

    // Expected: (x + x*2 - 3) = 3x - 3 for each x
    expect_array_eq(c, { 0.0, 3.0, 6.0, 9.0, 12.0 });
}

TEST_F(OperatorTest, ChainComparison) {
    Array a = to_array<double>({ 1.0, 2.0, 3.0, 4.0, 5.0 });
    Array mask = (a > 1) & (a < 5);

    const bool* data = mask.data<bool>();
    EXPECT_FALSE(data[0]);  // 1 > 1? false
    EXPECT_TRUE(data[1]);   // 2 > 1 && 2 < 5 -> true
    EXPECT_TRUE(data[2]);   // 3 > 1 && 3 < 5 -> true
    EXPECT_TRUE(data[3]);   // 4 > 1 && 4 < 5 -> true
    EXPECT_FALSE(data[4]);  // 5 < 5? false
}

// ========== Array::bool() Tests ==========

TEST_F(OperatorTest, ScalarBoolTrue) {
    Array a(3.14);
    EXPECT_TRUE(static_cast<bool>(a));
    EXPECT_TRUE(a.any());
    EXPECT_TRUE(a.all());
}

TEST_F(OperatorTest, ScalarBoolFalse) {
    Array a(0.0);
    EXPECT_FALSE(static_cast<bool>(a));
    EXPECT_FALSE(a.any());
    EXPECT_FALSE(a.all());
}

TEST_F(OperatorTest, ScalarIntTrue) {
    Array a(42);
    EXPECT_TRUE(static_cast<bool>(a));
}

TEST_F(OperatorTest, ScalarIntFalse) {
    Array a(0);
    EXPECT_FALSE(static_cast<bool>(a));
}

TEST_F(OperatorTest, MultiElementBoolThrows) {
    Array a = create_test_array_1d();
    EXPECT_ANY_THROW(static_cast<bool>(a));
}

TEST_F(OperatorTest, AnyAllOnMultiElement) {
    Array a = create_test_array_1d();
    EXPECT_TRUE(a.any());   // All values are non-zero
    EXPECT_TRUE(a.all());   // All values are non-zero

    Array b = to_array({ 0.0, 1.0, 0.0, 2.0, 0.0 });
    EXPECT_TRUE(b.any());   // Has non-zero elements
    EXPECT_FALSE(b.all());  // Not all are non-zero

    Array c = zeros({ 3, 3 });
    EXPECT_FALSE(c.any());
    EXPECT_FALSE(c.all());
}

// ========== Edge Cases ==========

TEST_F(OperatorTest, EmptyArray) {
    Array a;
    EXPECT_FALSE(static_cast<bool>(a));
}

TEST_F(OperatorTest, BroadcastingInOperations) {
    Array a = create_test_array_2d();      // 2x3
    Array b = to_array({ 10.0, 20.0, 30.0 }); // 1x3
    Array c = a + b;  // Should broadcast

    // Expected: each row + [10,20,30]
    const double* data = c.data<double>();
    EXPECT_NEAR(data[0], 1 + 10, 1e-6);
    EXPECT_NEAR(data[1], 2 + 20, 1e-6);
    EXPECT_NEAR(data[2], 3 + 30, 1e-6);
    EXPECT_NEAR(data[3], 4 + 10, 1e-6);
    EXPECT_NEAR(data[4], 5 + 20, 1e-6);
    EXPECT_NEAR(data[5], 6 + 30, 1e-6);
}

TEST_F(OperatorTest, DivisionByScalarZero) {
    Array a = create_test_array_1d();
    // Division by zero produces inf (should not crash)
    Array b = a / 0.0;
    const double* data = b.data<double>();
    EXPECT_TRUE(std::isinf(data[0]));
}