// src/ops/operator.cpp
#include "insight/ops/operator.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/reduction.h"
#include "insight/ops/creation.h"

namespace ins {

    // ========== Helper Functions ==========

    static Array scalar_to_array(double s, const Array& ref) {
        return full(Shape({ 1 }), s, ref.dtype(), ref.place());
    }

    static Array scalar_to_array_int(int64_t s, const Array& ref) {
        return full(Shape({ 1 }), static_cast<double>(s), ref.dtype(), ref.place());
    }

    static Array scalar_to_array_uint(uint64_t s, const Array& ref) {
        return full(Shape({ 1 }), static_cast<double>(s), ref.dtype(), ref.place());
    }

    // ========== Unary Operators ==========

    Array operator-(const Array& x) {
        return negative(x);
    }

    Array operator+(const Array& x) {
        return x;
    }

    Array operator~(const Array& x) {
        return bitwise_not(x);
    }

    Array operator!(const Array& x) {
        return logical_not(x);
    }

    // ========== Arithmetic Operators (Array-Array) ==========

    Array operator+(const Array& a, const Array& b) {
        return add(a, b);
    }

    Array operator-(const Array& a, const Array& b) {
        return sub(a, b);
    }

    Array operator*(const Array& a, const Array& b) {
        return mul(a, b);
    }

    Array operator/(const Array& a, const Array& b) {
        return div(a, b);
    }

    Array operator%(const Array& a, const Array& b) {
        return mod(a, b);
    }

    // ========== Arithmetic Operators (Array-Scalar) ==========

    Array operator+(const Array& a, double b) {
        return add(a, scalar_to_array(b, a));
    }

    Array operator+(double a, const Array& b) {
        return add(scalar_to_array(a, b), b);
    }

    Array operator-(const Array& a, double b) {
        return sub(a, scalar_to_array(b, a));
    }

    Array operator-(double a, const Array& b) {
        return sub(scalar_to_array(a, b), b);
    }

    Array operator*(const Array& a, double b) {
        return mul(a, scalar_to_array(b, a));
    }

    Array operator*(double a, const Array& b) {
        return mul(scalar_to_array(a, b), b);
    }

    Array operator/(const Array& a, double b) {
        return div(a, scalar_to_array(b, a));
    }

    Array operator/(double a, const Array& b) {
        return div(scalar_to_array(a, b), b);
    }

    Array operator%(const Array& a, double b) {
        return mod(a, scalar_to_array(b, a));
    }

    Array operator%(double a, const Array& b) {
        return mod(scalar_to_array(a, b), b);
    }

    // ========== Bitwise Operators (Array-Array) ==========

    Array operator&(const Array& a, const Array& b) {
        return bitwise_and(a, b);
    }

    Array operator|(const Array& a, const Array& b) {
        return bitwise_or(a, b);
    }

    Array operator^(const Array& a, const Array& b) {
        return bitwise_xor(a, b);
    }

    Array operator<<(const Array& a, const Array& b) {
        return bitwise_left_shift(a, b);
    }

    Array operator>>(const Array& a, const Array& b) {
        return bitwise_right_shift(a, b);
    }

    // ========== Bitwise Operators (Array-Scalar) ==========

    Array operator&(const Array& a, double b) {
        return bitwise_and(a, scalar_to_array_int(static_cast<int64_t>(b), a));
    }

    Array operator&(double a, const Array& b) {
        return bitwise_and(scalar_to_array_int(static_cast<int64_t>(a), b), b);
    }

    Array operator|(const Array& a, double b) {
        return bitwise_or(a, scalar_to_array_int(static_cast<int64_t>(b), a));
    }

    Array operator|(double a, const Array& b) {
        return bitwise_or(scalar_to_array_int(static_cast<int64_t>(a), b), b);
    }

    Array operator^(const Array& a, double b) {
        return bitwise_xor(a, scalar_to_array_int(static_cast<int64_t>(b), a));
    }

    Array operator^(double a, const Array& b) {
        return bitwise_xor(scalar_to_array_int(static_cast<int64_t>(a), b), b);
    }

    Array operator<<(const Array& a, int64_t b) {
        return bitwise_left_shift(a, scalar_to_array_int(b, a));
    }

    Array operator<<(int64_t a, const Array& b) {
        return bitwise_left_shift(scalar_to_array_int(a, b), b);
    }

    Array operator>>(const Array& a, int64_t b) {
        return bitwise_right_shift(a, scalar_to_array_int(b, a));
    }

    Array operator>>(int64_t a, const Array& b) {
        return bitwise_right_shift(scalar_to_array_int(a, b), b);
    }

    // ========== Comparison Operators (Array-Array) ==========

    Array operator==(const Array& a, const Array& b) {
        return equal(a, b);
    }

    Array operator!=(const Array& a, const Array& b) {
        return not_equal(a, b);
    }

    Array operator<(const Array& a, const Array& b) {
        return less(a, b);
    }

    Array operator<=(const Array& a, const Array& b) {
        return less_equal(a, b);
    }

    Array operator>(const Array& a, const Array& b) {
        return greater(a, b);
    }

    Array operator>=(const Array& a, const Array& b) {
        return greater_equal(a, b);
    }

    // ========== Comparison Operators (Array-Scalar) ==========

    Array operator==(const Array& a, double b) {
        return equal(a, scalar_to_array(b, a));
    }

    Array operator!=(const Array& a, double b) {
        return not_equal(a, scalar_to_array(b, a));
    }

    Array operator<(const Array& a, double b) {
        return less(a, scalar_to_array(b, a));
    }

    Array operator<=(const Array& a, double b) {
        return less_equal(a, scalar_to_array(b, a));
    }

    Array operator>(const Array& a, double b) {
        return greater(a, scalar_to_array(b, a));
    }

    Array operator>=(const Array& a, double b) {
        return greater_equal(a, scalar_to_array(b, a));
    }

    Array operator==(double a, const Array& b) {
        return equal(scalar_to_array(a, b), b);
    }

    Array operator!=(double a, const Array& b) {
        return not_equal(scalar_to_array(a, b), b);
    }

    Array operator<(double a, const Array& b) {
        return less(scalar_to_array(a, b), b);
    }

    Array operator<=(double a, const Array& b) {
        return less_equal(scalar_to_array(a, b), b);
    }

    Array operator>(double a, const Array& b) {
        return greater(scalar_to_array(a, b), b);
    }

    Array operator>=(double a, const Array& b) {
        return greater_equal(scalar_to_array(a, b), b);
    }

    // ========== Compound Assignment Operators (Array-Array) ==========

    Array& operator+=(Array& a, const Array& b) {
        a = a + b;
        return a;
    }

    Array& operator-=(Array& a, const Array& b) {
        a = a - b;
        return a;
    }

    Array& operator*=(Array& a, const Array& b) {
        a = a * b;
        return a;
    }

    Array& operator/=(Array& a, const Array& b) {
        a = a / b;
        return a;
    }

    Array& operator%=(Array& a, const Array& b) {
        a = a % b;
        return a;
    }

    // ========== Compound Assignment Operators (Array-Scalar) ==========

    Array& operator+=(Array& a, double b) {
        a = a + b;
        return a;
    }

    Array& operator-=(Array& a, double b) {
        a = a - b;
        return a;
    }

    Array& operator*=(Array& a, double b) {
        a = a * b;
        return a;
    }

    Array& operator/=(Array& a, double b) {
        a = a / b;
        return a;
    }

    Array& operator%=(Array& a, double b) {
        a = a % b;
        return a;
    }

    // ========== Compound Bitwise Assignment Operators ==========

    Array& operator&=(Array& a, const Array& b) {
        a = a & b;
        return a;
    }

    Array& operator|=(Array& a, const Array& b) {
        a = a | b;
        return a;
    }

    Array& operator^=(Array& a, const Array& b) {
        a = a ^ b;
        return a;
    }

    Array& operator<<=(Array& a, const Array& b) {
        a = a << b;
        return a;
    }

    Array& operator>>=(Array& a, const Array& b) {
        a = a >> b;
        return a;
    }

    Array& operator&=(Array& a, int64_t b) {
        a = a & b;
        return a;
    }

    Array& operator|=(Array& a, int64_t b) {
        a = a | b;
        return a;
    }

    Array& operator^=(Array& a, int64_t b) {
        a = a ^ b;
        return a;
    }

    Array& operator<<=(Array& a, int64_t b) {
        a = a << b;
        return a;
    }

    Array& operator>>=(Array& a, int64_t b) {
        a = a >> b;
        return a;
    }

    // ========== Increment/Decrement Operators ==========

    Array& operator++(Array& a) {
        a = a + 1.0;
        return a;
    }

    Array operator++(Array& a, int) {
        Array copy = a.copy();
        a = a + 1.0;
        return copy;
    }

    Array& operator--(Array& a) {
        a = a - 1.0;
        return a;
    }

    Array operator--(Array& a, int) {
        Array copy = a.copy();
        a = a - 1.0;
        return copy;
    }

} // namespace ins