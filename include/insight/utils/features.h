// insight/utils/features.h
#pragma once

namespace ins {

	/**
	 * @brief Check if Insight was compiled with OpenBLAS support.
	 * @return true if OpenBLAS is available
	 */
	bool is_compiled_with_openblas();

	/**
	 * @brief Check if Insight was compiled with FFTW3 support.
	 * @return true if FFTW3 is available
	 */
	bool is_compiled_with_fftw3();

	/**
	 * @brief Check if Insight was compiled with LAPACK (CLAPACK) support.
	 * @return true if LAPACK is available for linear algebra operations
	 * @note LAPACK provides matrix inverse, determinant, SVD, QR decomposition,
	 *       eigenvalue computation, and linear system solving.
	 */
	bool is_compiled_with_lapack();

	/**
	* @brief Check if Insight was compiled with Thrust support.
	* @return true if Thrust is available for GPU parallel algorithms
	* @note Thrust provides high-level parallel algorithms like sort, reduce, scan, etc. on GPU.
	*/
	bool is_compiled_with_thrust();

} // namespace ins