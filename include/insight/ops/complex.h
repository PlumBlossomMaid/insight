// insight/ops/complex.h
#pragma once
#include "insight/core/array.h"

namespace ins {

	/**
	 * @brief Check if tensor has complex data type.
	 *
	 * @param x Input tensor
	 * @return true if dtype is C32 or C64
	 */
	bool is_complex(const Array& x);

	/**
	 * @brief Check if tensor uses legacy complex storage format (last dimension = 2).
	 *
	 * This is for backward compatibility with the old representation.
	 * For modern code, prefer is_complex() which checks the data type.
	 *
	 * @param x Input tensor
	 * @return true if last dimension exists and equals 2
	 */
	bool has_complex_shape(const Array& x);

	/**
	 * @brief Convert real tensor to complex by adding zero imaginary part.
	 *
	 * Input shape: [d1, ..., dn]
	 * Output shape: [d1, ..., dn, 2]
	 *
	 * @param real Real part tensor
	 * @return Complex tensor (real, imag=0)
	 */
	Array to_complex(const Array& real);

	/**
	 * @brief Convert two real tensors to complex.
	 *
	 * Input shapes: both [d1, ..., dn]
	 * Output shape: [d1, ..., dn, 2]
	 *
	 * @param real Real part tensor
	 * @param imag Imaginary part tensor
	 * @return Complex tensor
	 */
	Array to_complex(const Array& real, const Array& imag);

	/**
	 * @brief View real tensor as complex tensor (zero-copy).
	 *
	 * Input must have last dimension = 2 (interleaved real, imag).
	 * Input dtype: F32/F64, Output dtype: C32/C64.
	 *
	 * @param x Real tensor with shape (..., 2)
	 * @return Complex tensor view with shape (...)
	 */
	Array as_complex(const Array& x);

	/**
	 * @brief View complex tensor as real tensor (zero-copy).
	 *
	 * Input dtype: C32/C64, Output dtype: F32/F64.
	 * Output shape: input shape + (2,)
	 *
	 * @param x Complex tensor
	 * @return Real tensor view with last dimension = 2
	 */
	Array as_real(const Array& x);

	/**
	 * @brief Extract real part from complex tensor (view).
	 *
	 * Input shape: (..., 2)
	 * Output shape: (...)
	 *
	 * @param z Complex tensor
	 * @return Real part view
	 */
	Array real(const Array& z);

	/**
	 * @brief Extract imaginary part from complex tensor (view).
	 *
	 * Input shape: (..., 2)
	 * Output shape: (...)
	 *
	 * @param z Complex tensor
	 * @return Imaginary part view
	 */
	Array imag(const Array& z);

} // namespace ins