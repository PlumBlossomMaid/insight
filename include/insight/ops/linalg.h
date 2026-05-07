// insight/ops/linalg.h
#pragma once
#include <tuple>
#include <string>
#include <optional>
#include "insight/core/array.h"

namespace ins {
    namespace linalg {

        // ============================================================================
        // Matrix multiplication
        // ============================================================================

        /**
         * @brief Matrix multiplication.
         * @param a First matrix (2D)
         * @param b Second matrix (2D)
         * @return Matrix product
         */
        Array matmul(const Array& a, const Array& b);

        // ============================================================================
        // Matrix properties
        // ============================================================================

        /**
         * @brief Compute determinant of a square matrix.
         * @param x Input matrix (2D)
         * @return Determinant as scalar array
         */
        Array det(const Array& x);

        /**
         * @brief Compute sign and natural log of determinant.
         * @param x Input matrix (2D)
         * @return Pair of (sign, log(abs(det)))
         */
        std::pair<Array, Array> slogdet(const Array& x);

        /**
         * @brief Compute condition number of a matrix.
         * @param x Input matrix (2D)
         * @param p Norm type: 1, 2, inf, 'fro' (default: 2)
         * @return Condition number
         */
        Array cond(const Array& x, double p = 2.0);

        /**
         * @brief Compute matrix or vector norm.
         * @param x Input array (1D or 2D)
         * @param ord Norm type (default: 2 for matrix, 2 for vector)
         * @return Norm value
         */
        Array norm(const Array& x, double ord = 2);

        /**
         * @brief Compute matrix rank using SVD.
         * @param x Input matrix (2D)
         * @param tol Tolerance (default: max(shape) * eps * max(s))
         * @return Rank as scalar array (int64)
         */
        Array matrix_rank(const Array& x, double tol = -1.0);

        /**
         * @brief Compute trace (sum of diagonal elements).
         * @param x Input matrix (2D)
         * @return Trace value
         */
        Array trace(const Array& x);

        // ============================================================================
        // Matrix computations
        // ============================================================================

        /**
         * @brief Matrix inverse.
         * @param x Input square matrix (2D)
         * @return Inverse matrix
         */
        Array inv(const Array& x);

        /**
         * @brief Pseudo-inverse using SVD.
         * @param x Input matrix (2D)
         * @param rcond Cutoff for small singular values (default: max(shape) * eps)
         * @return Pseudo-inverse matrix
         */
        Array pinv(const Array& x, double rcond = -1.0);

        /**
         * @brief Matrix power (A^n).
         * @param x Input square matrix (2D)
         * @param n Exponent (non-negative integer)
         * @return A^n
         */
        Array matrix_power(const Array& x, int n);

        /**
         * @brief Dot product of two vectors.
         * @param a First vector (1D)
         * @param b Second vector (1D)
         * @return Scalar dot product
         */
        Array dot(const Array& a, const Array& b);

        /**
         * @brief Inner product of two vectors (alias for dot).
         */
        inline Array inner(const Array& a, const Array& b) { return dot(a, b); }

        /**
         * @brief Outer product of two vectors.
         * @param a First vector (1D)
         * @param b Second vector (1D)
         * @return Outer product matrix
         */
        Array outer(const Array& a, const Array& b);

        // ============================================================================
        // Matrix decompositions
        // ============================================================================

        /**
         * @brief LU decomposition.
         * @param x Input matrix (2D)
         * @param pivot Whether to perform pivoting (default: true)
         * @return Tuple of (LU matrix, pivots)
         */
        std::tuple<Array, Array> lu(const Array& x, bool pivot = true);

        /**
         * @brief Unpack LU decomposition into P, L, U.
         * @param LU LU matrix from lu()
         * @param pivots Pivot indices from lu()
         * @return Tuple of (P, L, U)
         */
        std::tuple<Array, Array, Array> lu_unpack(const Array& LU, const Array& pivots);

        /**
         * @brief QR decomposition.
         * @param x Input matrix (2D)
         * @param mode 'reduced' (default) or 'complete'
         * @return Tuple of (Q, R)
         */
        std::tuple<Array, Array> qr(const Array& x, const std::string& mode = "reduced");

        /**
         * @brief LQ decomposition.
         * @param x Input matrix (2D)
         * @param mode 'reduced' (default) or 'complete'
         * @return Tuple of (L, Q)
         */
        std::tuple<Array, Array> lq(const Array& x, const std::string& mode = "reduced");

        /**
         * @brief Cholesky decomposition (symmetric positive definite).
         * @param x Input matrix (2D)
         * @param lower Use lower triangle (true) or upper triangle (false)
         * @return Cholesky factor (L or U)
         */
        Array cholesky(const Array& x, bool lower = true);

        /**
         * @brief Solve AX = B using Cholesky decomposition.
         * @param A Symmetric positive definite matrix
         * @param B Right-hand side
         * @param lower Whether A is stored as lower triangle (default: true)
         * @return Solution X
         */
        Array cholesky_solve(const Array& A, const Array& B, bool lower = true);

        /**
         * @brief Singular value decomposition.
         * @param x Input matrix (2D)
         * @param full_matrices Whether to compute full U and VT (default: true)
         * @return Tuple of (U, S, VT)
         */
        std::tuple<Array, Array, Array> svd(const Array& x, bool full_matrices = true);

        /**
         * @brief Singular values only.
         * @param x Input matrix (2D)
         * @return Singular values array (1D)
         */
        Array svdvals(const Array& x);

        /**
         * @brief Eigenvalues and eigenvectors (general matrix).
         * @param x Input matrix (2D)
         * @return Tuple of (eigenvalues (complex as [:,2]), eigenvectors (complex))
         */
        std::tuple<Array, Array> eig(const Array& x);

        /**
         * @brief Eigenvalues only (general matrix).
         * @param x Input matrix (2D)
         * @return Eigenvalues array (complex as [:,2])
         */
        Array eigvals(const Array& x);

        /**
         * @brief Eigenvalues and eigenvectors (symmetric matrix).
         * @param x Input symmetric matrix (2D)
         * @param uplo 'L' for lower triangle, 'U' for upper triangle (default: 'L')
         * @return Tuple of (eigenvalues, eigenvectors)
         */
        std::tuple<Array, Array> eigh(const Array& x, const std::string& uplo = "L");

        /**
         * @brief Eigenvalues only (symmetric matrix).
         * @param x Input symmetric matrix (2D)
         * @param uplo 'L' for lower triangle, 'U' for upper triangle (default: 'L')
         * @return Eigenvalues array
         */
        Array eigvalsh(const Array& x, const std::string& uplo = "L");

        // ============================================================================
        // Linear system solvers
        // ============================================================================

        /**
         * @brief Solve linear system AX = B.
         * @param A Coefficient matrix (square)
         * @param B Right-hand side (vector or matrix)
         * @return Solution X
         */
        Array solve(const Array& A, const Array& B);

        /**
         * @brief Least squares solution of AX = B.
         * @param A Coefficient matrix (m x n)
         * @param B Right-hand side
         * @param rcond Cutoff for small singular values (default: max(shape) * eps)
         * @return Solution X
         */
        Array lstsq(const Array& A, const Array& B, double rcond = -1.0);

        /**
         * @brief Solve AX = B where A is triangular.
         * @param A Triangular matrix
         * @param B Right-hand side
         * @param lower Whether A is lower triangular (default: true)
         * @param unit_diag Whether diagonal is assumed to be 1 (default: false)
         * @return Solution X
         */
        Array solve_triangular(const Array& A, const Array& B,
            bool lower = true, bool unit_diag = false);

        /**
         * @brief Covariance matrix.
         *
         * @param x Input array (2D). If rowvar is true, each row is a variable; otherwise each column.
         * @param rowvar If true, rows are variables (shape: m, n -> output: m, m).
         *               If false, columns are variables (transposed then same as above).
         * @param ddof Delta degrees of freedom (default: 1 for sample covariance, 0 for population)
         * @return Covariance matrix (square matrix of size m)
         */
        Array cov(const Array& x, bool rowvar = true, int ddof = 1);

    } // namespace linalg
} // namespace ins