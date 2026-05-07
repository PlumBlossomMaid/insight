// insight/ops/operator.h
#pragma once
#include "insight/core/array.h"

namespace ins {

    // ========== Unary Operators ==========

    /**
     * @brief Unary minus.
     * @param x Input array
     * @return Negative of input
     */
    Array operator-(const Array& x);

    /**
     * @brief Unary plus (returns a view).
     * @param x Input array
     * @return View of input
     */
    Array operator+(const Array& x);

    /**
     * @brief Bitwise NOT.
     * @param x Input array
     * @return Bitwise complement
     */
    Array operator~(const Array& x);

    /**
     * @brief Logical NOT.
     * @param x Input array
     * @return Boolean array with logical NOT
     */
    Array operator!(const Array& x);

    // ========== Arithmetic Operators (Array-Array) ==========

    Array operator+(const Array& a, const Array& b);
    Array operator-(const Array& a, const Array& b);
    Array operator*(const Array& a, const Array& b);
    Array operator/(const Array& a, const Array& b);
    Array operator%(const Array& a, const Array& b);

    // ========== Arithmetic Operators (Array-Scalar) ==========

    Array operator+(const Array& a, double b);
    Array operator+(double a, const Array& b);
    Array operator-(const Array& a, double b);
    Array operator-(double a, const Array& b);
    Array operator*(const Array& a, double b);
    Array operator*(double a, const Array& b);
    Array operator/(const Array& a, double b);
    Array operator/(double a, const Array& b);
    Array operator%(const Array& a, double b);
    Array operator%(double a, const Array& b);

    // ========== Bitwise Operators (Array-Array) ==========

    /**
     * @brief Bitwise AND.
     * @param a First array
     * @param b Second array
     * @return Bitwise AND of a and b
     */
    Array operator&(const Array& a, const Array& b);

    /**
     * @brief Bitwise OR.
     * @param a First array
     * @param b Second array
     * @return Bitwise OR of a and b
     */
    Array operator|(const Array& a, const Array& b);

    /**
     * @brief Bitwise XOR.
     * @param a First array
     * @param b Second array
     * @return Bitwise XOR of a and b
     */
    Array operator^(const Array& a, const Array& b);

    /**
     * @brief Bitwise left shift.
     * @param a First array
     * @param b Shift amounts (second array)
     * @return Left-shifted array
     */
    Array operator<<(const Array& a, const Array& b);

    /**
     * @brief Bitwise right shift.
     * @param a First array
     * @param b Shift amounts (second array)
     * @return Right-shifted array
     */
    Array operator>>(const Array& a, const Array& b);

    // ========== Bitwise Operators (Array-Scalar) ==========

    Array operator&(const Array& a, double b);
    Array operator&(double a, const Array& b);
    Array operator|(const Array& a, double b);
    Array operator|(double a, const Array& b);
    Array operator^(const Array& a, double b);
    Array operator^(double a, const Array& b);
    Array operator<<(const Array& a, int64_t b);
    Array operator<<(int64_t a, const Array& b);
    Array operator>>(const Array& a, int64_t b);
    Array operator>>(int64_t a, const Array& b);

    // ========== Comparison Operators (Array-Array) ==========

    Array operator==(const Array& a, const Array& b);
    Array operator!=(const Array& a, const Array& b);
    Array operator<(const Array& a, const Array& b);
    Array operator<=(const Array& a, const Array& b);
    Array operator>(const Array& a, const Array& b);
    Array operator>=(const Array& a, const Array& b);

    // ========== Comparison Operators (Array-Scalar) ==========

    Array operator==(const Array& a, double b);
    Array operator!=(const Array& a, double b);
    Array operator<(const Array& a, double b);
    Array operator<=(const Array& a, double b);
    Array operator>(const Array& a, double b);
    Array operator>=(const Array& a, double b);
    Array operator==(double a, const Array& b);
    Array operator!=(double a, const Array& b);
    Array operator<(double a, const Array& b);
    Array operator<=(double a, const Array& b);
    Array operator>(double a, const Array& b);
    Array operator>=(double a, const Array& b);

    // ========== Compound Assignment Operators (Array-Array) ==========

    Array& operator+=(Array& a, const Array& b);
    Array& operator-=(Array& a, const Array& b);
    Array& operator*=(Array& a, const Array& b);
    Array& operator/=(Array& a, const Array& b);
    Array& operator%=(Array& a, const Array& b);

    // ========== Compound Assignment Operators (Array-Scalar) ==========

    Array& operator+=(Array& a, double b);
    Array& operator-=(Array& a, double b);
    Array& operator*=(Array& a, double b);
    Array& operator/=(Array& a, double b);
    Array& operator%=(Array& a, double b);

    // ========== Compound Bitwise Assignment Operators ==========

    Array& operator&=(Array& a, const Array& b);
    Array& operator|=(Array& a, const Array& b);
    Array& operator^=(Array& a, const Array& b);
    Array& operator<<=(Array& a, const Array& b);
    Array& operator>>=(Array& a, const Array& b);

    Array& operator&=(Array& a, int64_t b);
    Array& operator|=(Array& a, int64_t b);
    Array& operator^=(Array& a, int64_t b);
    Array& operator<<=(Array& a, int64_t b);
    Array& operator>>=(Array& a, int64_t b);

    // ========== Increment/Decrement Operators ==========

    /**
     * @brief Prefix increment (++arr).
     * @param a Array to increment
     * @return Reference to incremented array
     */
    Array& operator++(Array& a);

    /**
     * @brief Postfix increment (arr++).
     * @param a Array to increment
     * @return Copy of original array
     */
    Array operator++(Array& a, int);

    /**
     * @brief Prefix decrement (--arr).
     * @param a Array to decrement
     * @return Reference to decremented array
     */
    Array& operator--(Array& a);

    /**
     * @brief Postfix decrement (arr--).
     * @param a Array to decrement
     * @return Copy of original array
     */
    Array operator--(Array& a, int);

} // namespace ins