// include/insight/ops/elementwise.h
#pragma once
#include "insight/core/array.h"

namespace ins {

	// ============================================================================
	// Arithmetic operations
	// ============================================================================
	Array add(const Array& a, const Array& b);
	Array sub(const Array& a, const Array& b);
	Array mul(const Array& a, const Array& b);
	Array div(const Array& a, const Array& b);
	Array pow(const Array& a, const Array& b);
	Array mod(const Array& a, const Array& b);

	// Comparison operations
	Array equal(const Array& a, const Array& b);
	Array not_equal(const Array& a, const Array& b);
	Array greater(const Array& a, const Array& b);
	Array greater_than(const Array& a, const Array& b);
	Array less(const Array& a, const Array& b);
	Array less_than(const Array& a, const Array& b);
	Array greater_equal(const Array& a, const Array& b);
	Array less_equal(const Array& a, const Array& b);

	// Logical operations
	Array logical_and(const Array& a, const Array& b);
	Array logical_or(const Array& a, const Array& b);
	Array logical_xor(const Array& a, const Array& b);
	Array logical_not(const Array& x);

	// Bitwise operations
	Array bitwise_and(const Array& a, const Array& b);
	Array bitwise_or(const Array& a, const Array& b);
	Array bitwise_xor(const Array& a, const Array& b);
	Array bitwise_left_shift(const Array& a, const Array& b);
	Array bitwise_right_shift(const Array& a, const Array& b);

	// Maximum / Minimum
	Array maximum(const Array& a, const Array& b);
	Array minimum(const Array& a, const Array& b);

	// ============================================================================
	// Unary math operations
	// ============================================================================

	/// Absolute value
	Array abs(const Array& x);

	/// Negative (unary minus)
	Array negative(const Array& x);
	inline Array neg(const Array& x) { return negative(x); }

	/// Square (x * x)
	Array square(const Array& x);
	inline Array sqr(const Array& x) { return square(x); }

	/// Exponential
	Array exp(const Array& x);
	Array exp2(const Array& x);
	Array expm1(const Array& x);

	/// Logarithm
	Array log(const Array& x);
	Array log2(const Array& x);
	Array log10(const Array& x);
	Array log1p(const Array& x);

	/// Power / root
	Array sqrt(const Array& x);
	Array cbrt(const Array& x);
	Array reciprocal(const Array& x);

	/// Trigonometric
	Array sin(const Array& x);
	Array cos(const Array& x);
	Array tan(const Array& x);
	Array asin(const Array& x);
	Array acos(const Array& x);
	Array atan(const Array& x);

	/// Hyperbolic
	Array sinh(const Array& x);
	Array cosh(const Array& x);
	Array tanh(const Array& x);
	Array asinh(const Array& x);
	Array acosh(const Array& x);
	Array atanh(const Array& x);

	/// Rounding
	Array floor(const Array& x);
	Array ceil(const Array& x);
	Array trunc(const Array& x);
	Array rint(const Array& x);  // round to nearest integer

	/// Sign function
	Array sign(const Array& x);

	/// Complex conjugate
	Array conj(const Array& x);

	/// Degree/radian conversion
	Array deg2rad(const Array& x);
	Array rad2deg(const Array& x);

	/// Logical not (returns bool)
	Array logical_not(const Array& x);

	/// Bitwise not (integer types only)
	Array bitwise_not(const Array& x);
	inline Array invert(const Array& x) { return bitwise_not(x); }

} // namespace ins