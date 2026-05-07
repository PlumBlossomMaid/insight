// src/ops/linalg.cpp
#include "insight/ops/linalg.h"
#include "insight/ops/broadcast.h"
#include "insight/plugin/op_registry.h"
#include "insight/utils/promotion.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/reduction.h"
#include <cmath>

namespace ins {
    namespace linalg {

        static DeviceKind get_device_kind(const Place& place) {
            return place.is_cpu() ? DeviceKind::CPU : DeviceKind::GPU;
        }

        static DType get_working_dtype(DType dtype) {
            if (is_integer(dtype) || dtype == DType::BOOL) {
                return DType::F64;
            }
            if (dtype == DType::F32) {
                return DType::F32;
            }
            return DType::F64;
        }

        static Array ensure_2d_square(const Array& x, const char* func_name) {
            INS_CHECK(x.shape().ndim() == 2, func_name, ": input must be 2D");
            INS_CHECK(x.shape().dim(0) == x.shape().dim(1),
                func_name, ": input must be square matrix");
            return x;
        }

        static Array ensure_2d(const Array& x, const char* func_name) {
            INS_CHECK(x.shape().ndim() == 2, func_name, ": input must be 2D");
            return x;
        }

        static Array ensure_1d(const Array& x, const char* func_name) {
            INS_CHECK(x.shape().ndim() == 1, func_name, ": input must be 1D");
            return x;
        }

        static Array to_working_dtype(const Array& x, DType target_dtype) {
            if (x.dtype() == target_dtype) {
                return x;
            }
            return x.to(target_dtype);
        }

        static Array to_original_dtype(const Array& x, DType original_dtype) {
            if (x.dtype() == original_dtype) {
                return x;
            }
            return x.to(original_dtype);
        }

        // ============================================================================
        // matmul
        // ============================================================================

        Array matmul(const Array& a, const Array& b) {
            INS_CHECK(a.defined() && b.defined(), "matmul: input is undefined");

            int ndim_a = a.shape().ndim();
            int ndim_b = b.shape().ndim();

            // Determine output shape based on input dimensions
            Shape out_shape;
            DType working_dtype = get_working_dtype(a.dtype());
            Array a_work = to_working_dtype(a, working_dtype);
            Array b_work = to_working_dtype(b, working_dtype);

            if (a_work.place() != b_work.place()) {
                b_work = b_work.to(a_work.place());
            }

            // Case 1: vector * vector -> scalar
            if (ndim_a == 1 && ndim_b == 1) {
                INS_CHECK(a.numel() == b.numel(), "matmul: vector length mismatch");
                out_shape = Shape({});
            }
            // Case 2: matrix * vector -> vector
            else if (ndim_a == 2 && ndim_b == 1) {
                INS_CHECK(a.shape().dim(1) == b.numel(), "matmul: shape mismatch");
                out_shape = Shape({ a.shape().dim(0) });
            }
            // Case 3: vector * matrix -> vector
            else if (ndim_a == 1 && ndim_b == 2) {
                INS_CHECK(a.numel() == b.shape().dim(0), "matmul: shape mismatch");
                out_shape = Shape({ b.shape().dim(1) });
            }
            // Case 4: matrix * matrix -> matrix
            else if (ndim_a == 2 && ndim_b == 2) {
                INS_CHECK(a.shape().dim(1) == b.shape().dim(0), "matmul: shape mismatch");
                out_shape = Shape({ a.shape().dim(0), b.shape().dim(1) });
            }
            // Case 5: batched matrix multiplication (N-D)
            else {
                // Both inputs must have at least 1 dimension, and at least one has N > 2
                INS_CHECK(ndim_a >= 1 && ndim_b >= 1, "matmul: invalid dimensions");

                // Broadcast the shapes
                Shape bc_shape = broadcast_shape(a.shape(), b.shape());
                int ndim = bc_shape.ndim();

                // The last two dimensions are the matrix dimensions
                int64_t m = bc_shape.dim(ndim - 2);
                int64_t n = bc_shape.dim(ndim - 1);
                int64_t k_a = (ndim_a >= 2) ? a.shape().dim(ndim_a - 1) : 1;
                int64_t k_b = (ndim_b >= 2) ? b.shape().dim(ndim_b - 2) : 1;

                if (ndim_a == 1) {
                    INS_CHECK(k_a == k_b, "matmul: shape mismatch for vector-matrix batch");
                }
                else if (ndim_b == 1) {
                    INS_CHECK(k_a == k_b, "matmul: shape mismatch for matrix-vector batch");
                }
                else {
                    INS_CHECK(k_a == k_b, "matmul: shape mismatch for batch matrices");
                }

                out_shape = bc_shape;
            }

            Array result(out_shape, working_dtype, a_work.place());

            OpArgs args = { result, a_work, b_work };
            DeviceKind dev = get_device_kind(a_work.place());
            OpArgs output = ops()["matmul"][dev][working_dtype](args);

            Array out = std::any_cast<Array>(output[0]);
            return to_original_dtype(out, a.dtype());
        }

        // ============================================================================
        // det
        // ============================================================================

        Array det(const Array& x) {
            Array x_2d = ensure_2d_square(x, "det");
            DType working_dtype = get_working_dtype(x.dtype());
            Array work = to_working_dtype(x_2d, working_dtype);

            OpArgs args = { work };
            DeviceKind dev = get_device_kind(work.place());
            OpArgs output = ops()["det"][dev][working_dtype](args);

            Array result = std::any_cast<Array>(output[0]);
            return to_original_dtype(result, x.dtype());
        }

        // ============================================================================
        // slogdet
        // ============================================================================

        std::pair<Array, Array> slogdet(const Array& x) {
            Array x_2d = ensure_2d_square(x, "slogdet");
            DType working_dtype = get_working_dtype(x.dtype());
            Array work = to_working_dtype(x_2d, working_dtype);

            OpArgs args = { work };
            DeviceKind dev = get_device_kind(work.place());
            OpArgs output = ops()["slogdet"][dev][working_dtype](args);

            Array sign = std::any_cast<Array>(output[0]);
            Array logdet = std::any_cast<Array>(output[1]);
            return { to_original_dtype(sign, x.dtype()),
                    to_original_dtype(logdet, x.dtype()) };
        }

        // ============================================================================
        // cond
        // ============================================================================

        Array cond(const Array& x, double p) {
            Array x_2d = ensure_2d(x, "cond");
            DType working_dtype = get_working_dtype(x.dtype());
            Array work = to_working_dtype(x_2d, working_dtype);

            OpArgs args = { work, p };
            DeviceKind dev = get_device_kind(work.place());
            OpArgs output = ops()["cond"][dev][working_dtype](args);

            Array result = std::any_cast<Array>(output[0]);
            return to_original_dtype(result, x.dtype());
        }

        // ============================================================================
        // norm
        // ============================================================================

        Array norm(const Array& x, double ord) {
            INS_CHECK(x.defined(), "norm: input is undefined");
            DType working_dtype = get_working_dtype(x.dtype());
            Array work = to_working_dtype(x, working_dtype);

            OpArgs args = { work, ord };
            DeviceKind dev = get_device_kind(work.place());
            OpArgs output = ops()["norm"][dev][working_dtype](args);

            Array result = std::any_cast<Array>(output[0]);
            return to_original_dtype(result, x.dtype());
        }

        // ============================================================================
        // matrix_rank
        // ============================================================================

        Array matrix_rank(const Array& x, double tol) {
            Array x_2d = ensure_2d(x, "matrix_rank");
            DType working_dtype = get_working_dtype(x.dtype());
            Array work = to_working_dtype(x_2d, working_dtype);

            OpArgs args = { work, tol };
            DeviceKind dev = get_device_kind(work.place());
            OpArgs output = ops()["matrix_rank"][dev][working_dtype](args);

            Array result = std::any_cast<Array>(output[0]);
            return result; // rank is int64, no conversion needed
        }

        // ============================================================================
        // trace
        // ============================================================================

        Array trace(const Array& x) {
            Array x_2d = ensure_2d(x, "trace");
            DType working_dtype = get_working_dtype(x.dtype());
            Array work = to_working_dtype(x_2d, working_dtype);

            OpArgs args = { work };
            DeviceKind dev = get_device_kind(work.place());
            OpArgs output = ops()["trace"][dev][working_dtype](args);

            Array result = std::any_cast<Array>(output[0]);
            return to_original_dtype(result, x.dtype());
        }

        // ============================================================================
        // inv
        // ============================================================================

        Array inv(const Array& x) {
            Array x_2d = ensure_2d_square(x, "inv");
            DType working_dtype = get_working_dtype(x.dtype());
            Array work = to_working_dtype(x_2d, working_dtype);

            OpArgs args = { work };
            DeviceKind dev = get_device_kind(work.place());
            OpArgs output = ops()["inv"][dev][working_dtype](args);

            Array result = std::any_cast<Array>(output[0]);
            return to_original_dtype(result, x.dtype());
        }

        // ============================================================================
        // pinv
        // ============================================================================

        Array pinv(const Array& x, double rcond) {
            Array x_2d = ensure_2d(x, "pinv");
            DType working_dtype = get_working_dtype(x.dtype());
            Array work = to_working_dtype(x_2d, working_dtype);

            OpArgs args = { work, rcond };
            DeviceKind dev = get_device_kind(work.place());
            OpArgs output = ops()["pinv"][dev][working_dtype](args);

            Array result = std::any_cast<Array>(output[0]);
            return to_original_dtype(result, x.dtype());
        }

        // ============================================================================
        // matrix_power
        // ============================================================================

        Array matrix_power(const Array& x, int n) {
            Array x_2d = ensure_2d_square(x, "matrix_power");
            INS_CHECK(n >= 0, "matrix_power: exponent must be non-negative");

            if (n == 0) {
                return eye(x_2d.shape().dim(0), x_2d.shape().dim(1), 0,
                    DType::F64, x.place()).to(x.dtype());
            }

            if (n == 1) {
                return x;
            }

            DType working_dtype = get_working_dtype(x.dtype());
            Array work = to_working_dtype(x_2d, working_dtype);

            OpArgs args = { work, n };
            DeviceKind dev = get_device_kind(work.place());
            OpArgs output = ops()["matrix_power"][dev][working_dtype](args);

            Array result = std::any_cast<Array>(output[0]);
            return to_original_dtype(result, x.dtype());
        }

        // ============================================================================
        // dot
        // ============================================================================

        Array dot(const Array& a, const Array& b) {
            Array a_1d = ensure_1d(a, "dot");
            Array b_1d = ensure_1d(b, "dot");
            INS_CHECK(a_1d.numel() == b_1d.numel(), "dot: size mismatch");

            DType working_dtype = get_working_dtype(a.dtype());
            Array a_work = to_working_dtype(a_1d, working_dtype);
            Array b_work = to_working_dtype(b_1d, working_dtype);

            if (a_work.place() != b_work.place()) {
                b_work = b_work.to(a_work.place());
            }

            OpArgs args = { a_work, b_work };
            DeviceKind dev = get_device_kind(a_work.place());
            OpArgs output = ops()["dot"][dev][working_dtype](args);

            Array result = std::any_cast<Array>(output[0]);
            return to_original_dtype(result, a.dtype());
        }

        // ============================================================================
        // outer
        // ============================================================================

        Array outer(const Array& a, const Array& b) {
            Array a_1d = ensure_1d(a, "outer");
            Array b_1d = ensure_1d(b, "outer");

            DType working_dtype = get_working_dtype(a.dtype());
            Array a_work = to_working_dtype(a_1d, working_dtype);
            Array b_work = to_working_dtype(b_1d, working_dtype);

            if (a_work.place() != b_work.place()) {
                b_work = b_work.to(a_work.place());
            }

            OpArgs args = { a_work, b_work };
            DeviceKind dev = get_device_kind(a_work.place());
            OpArgs output = ops()["outer"][dev][working_dtype](args);

            Array result = std::any_cast<Array>(output[0]);
            return to_original_dtype(result, a.dtype());
        }

        // ============================================================================
        // lu
        // ============================================================================

        std::tuple<Array, Array> lu(const Array& x, bool pivot) {
            Array x_2d = ensure_2d_square(x, "lu");
            DType working_dtype = get_working_dtype(x.dtype());
            Array work = to_working_dtype(x_2d, working_dtype);

            OpArgs args = { work, pivot };
            DeviceKind dev = get_device_kind(work.place());
            OpArgs output = ops()["lu"][dev][working_dtype](args);

            Array LU = std::any_cast<Array>(output[0]);
            Array pivots = std::any_cast<Array>(output[1]);
            return { to_original_dtype(LU, x.dtype()), pivots };
        }

        // ============================================================================
        // lu_unpack
        // ============================================================================

        std::tuple<Array, Array, Array> lu_unpack(const Array& LU, const Array& pivots) {
            Array lu_2d = ensure_2d(LU, "lu_unpack");
            DType working_dtype = get_working_dtype(lu_2d.dtype());
            Array work = to_working_dtype(lu_2d, working_dtype);

            OpArgs args = { work, pivots };
            DeviceKind dev = get_device_kind(work.place());
            OpArgs output = ops()["lu_unpack"][dev][working_dtype](args);

            Array P = std::any_cast<Array>(output[0]);
            Array L = std::any_cast<Array>(output[1]);
            Array U = std::any_cast<Array>(output[2]);
            return { to_original_dtype(P, lu_2d.dtype()),
                    to_original_dtype(L, lu_2d.dtype()),
                    to_original_dtype(U, lu_2d.dtype()) };
        }

        // ============================================================================
        // qr
        // ============================================================================

        std::tuple<Array, Array> qr(const Array& x, const std::string& mode) {
            Array x_2d = ensure_2d(x, "qr");
            DType working_dtype = get_working_dtype(x.dtype());
            Array work = to_working_dtype(x_2d, working_dtype);

            OpArgs args = { work, mode };
            DeviceKind dev = get_device_kind(work.place());
            OpArgs output = ops()["qr"][dev][working_dtype](args);

            Array Q = std::any_cast<Array>(output[0]);
            Array R = std::any_cast<Array>(output[1]);
            return { to_original_dtype(Q, x.dtype()),
                    to_original_dtype(R, x.dtype()) };
        }

        // ============================================================================
        // lq
        // ============================================================================

        std::tuple<Array, Array> lq(const Array& x, const std::string& mode) {
            Array x_2d = ensure_2d(x, "lq");
            DType working_dtype = get_working_dtype(x.dtype());
            Array work = to_working_dtype(x_2d, working_dtype);

            OpArgs args = { work, mode };
            DeviceKind dev = get_device_kind(work.place());
            OpArgs output = ops()["lq"][dev][working_dtype](args);

            Array L = std::any_cast<Array>(output[0]);
            Array Q = std::any_cast<Array>(output[1]);
            return { to_original_dtype(L, x.dtype()),
                    to_original_dtype(Q, x.dtype()) };
        }

        // ============================================================================
        // cholesky
        // ============================================================================

        Array cholesky(const Array& x, bool lower) {
            Array x_2d = ensure_2d_square(x, "cholesky");
            DType working_dtype = get_working_dtype(x.dtype());
            Array work = to_working_dtype(x_2d, working_dtype);

            OpArgs args = { work, lower };
            DeviceKind dev = get_device_kind(work.place());
            OpArgs output = ops()["cholesky"][dev][working_dtype](args);

            Array result = std::any_cast<Array>(output[0]);
            return to_original_dtype(result, x.dtype());
        }

        // ============================================================================
        // cholesky_solve
        // ============================================================================

        Array cholesky_solve(const Array& A, const Array& B, bool lower) {
            Array A_2d = ensure_2d_square(A, "cholesky_solve");
            DType working_dtype = get_working_dtype(A.dtype());
            Array A_work = to_working_dtype(A_2d, working_dtype);
            Array B_work = to_working_dtype(B, working_dtype);

            if (A_work.place() != B_work.place()) {
                B_work = B_work.to(A_work.place());
            }

            OpArgs args = { A_work, B_work, lower };
            DeviceKind dev = get_device_kind(A_work.place());
            OpArgs output = ops()["cholesky_solve"][dev][working_dtype](args);

            Array result = std::any_cast<Array>(output[0]);
            return to_original_dtype(result, B.dtype());
        }

        // ============================================================================
        // svd
        // ============================================================================

        std::tuple<Array, Array, Array> svd(const Array& x, bool full_matrices) {
            Array x_2d = ensure_2d(x, "svd");
            DType working_dtype = get_working_dtype(x.dtype());
            Array work = to_working_dtype(x_2d, working_dtype);

            OpArgs args = { work, full_matrices };
            DeviceKind dev = get_device_kind(work.place());
            OpArgs output = ops()["svd"][dev][working_dtype](args);

            Array U = std::any_cast<Array>(output[0]);
            Array S = std::any_cast<Array>(output[1]);
            Array VT = std::any_cast<Array>(output[2]);
            return { to_original_dtype(U, x.dtype()),
                    to_original_dtype(S, x.dtype()),
                    to_original_dtype(VT, x.dtype()) };
        }

        // ============================================================================
        // svdvals
        // ============================================================================

        Array svdvals(const Array& x) {
            Array x_2d = ensure_2d(x, "svdvals");
            DType working_dtype = get_working_dtype(x.dtype());
            Array work = to_working_dtype(x_2d, working_dtype);

            OpArgs args = { work };
            DeviceKind dev = get_device_kind(work.place());
            OpArgs output = ops()["svdvals"][dev][working_dtype](args);

            Array result = std::any_cast<Array>(output[0]);
            return to_original_dtype(result, x.dtype());
        }

        // ============================================================================
        // eig
        // ============================================================================

        std::tuple<Array, Array> eig(const Array& x) {
            Array x_2d = ensure_2d(x, "eig");
            DType working_dtype = get_working_dtype(x.dtype());
            Array work = to_working_dtype(x_2d, working_dtype);

            OpArgs args = { work };
            DeviceKind dev = get_device_kind(work.place());
            OpArgs output = ops()["eig"][dev][working_dtype](args);

            Array eigenvalues = std::any_cast<Array>(output[0]);
            Array eigenvectors = std::any_cast<Array>(output[1]);
            // eigenvalues are complex, keep as is (C32 or C64)
            return { eigenvalues, eigenvectors };
        }

        // ============================================================================
        // eigvals
        // ============================================================================

        Array eigvals(const Array& x) {
            Array x_2d = ensure_2d(x, "eigvals");
            DType working_dtype = get_working_dtype(x.dtype());
            Array work = to_working_dtype(x_2d, working_dtype);

            OpArgs args = { work };
            DeviceKind dev = get_device_kind(work.place());
            OpArgs output = ops()["eigvals"][dev][working_dtype](args);

            Array result = std::any_cast<Array>(output[0]);
            return result; // complex, keep as is
        }

        // ============================================================================
        // eigh
        // ============================================================================

        std::tuple<Array, Array> eigh(const Array& x, const std::string& uplo) {
            Array x_2d = ensure_2d_square(x, "eigh");
            DType working_dtype = get_working_dtype(x.dtype());
            Array work = to_working_dtype(x_2d, working_dtype);

            OpArgs args = { work, uplo };
            DeviceKind dev = get_device_kind(work.place());
            OpArgs output = ops()["eigh"][dev][working_dtype](args);

            Array eigenvalues = std::any_cast<Array>(output[0]);
            Array eigenvectors = std::any_cast<Array>(output[1]);
            return { to_original_dtype(eigenvalues, x.dtype()),
                    to_original_dtype(eigenvectors, x.dtype()) };
        }

        // ============================================================================
        // eigvalsh
        // ============================================================================

        Array eigvalsh(const Array& x, const std::string& uplo) {
            Array x_2d = ensure_2d_square(x, "eigvalsh");
            DType working_dtype = get_working_dtype(x.dtype());
            Array work = to_working_dtype(x_2d, working_dtype);

            OpArgs args = { work, uplo };
            DeviceKind dev = get_device_kind(work.place());
            OpArgs output = ops()["eigvalsh"][dev][working_dtype](args);

            Array result = std::any_cast<Array>(output[0]);
            return to_original_dtype(result, x.dtype());
        }

        // ============================================================================
        // solve
        // ============================================================================

        Array solve(const Array& A, const Array& B) {
            Array A_2d = ensure_2d_square(A, "solve");
            DType working_dtype = get_working_dtype(A.dtype());
            Array A_work = to_working_dtype(A_2d, working_dtype);
            Array B_work = to_working_dtype(B, working_dtype);

            if (A_work.place() != B_work.place()) {
                B_work = B_work.to(A_work.place());
            }

            OpArgs args = { A_work, B_work };
            DeviceKind dev = get_device_kind(A_work.place());
            OpArgs output = ops()["solve"][dev][working_dtype](args);

            Array result = std::any_cast<Array>(output[0]);
            return to_original_dtype(result, B.dtype());
        }

        // ============================================================================
        // lstsq
        // ============================================================================

        Array lstsq(const Array& A, const Array& B, double rcond) {
            Array A_2d = ensure_2d(A, "lstsq");
            DType working_dtype = get_working_dtype(A.dtype());
            Array A_work = to_working_dtype(A_2d, working_dtype);
            Array B_work = to_working_dtype(B, working_dtype);

            if (A_work.place() != B_work.place()) {
                B_work = B_work.to(A_work.place());
            }

            OpArgs args = { A_work, B_work, rcond };
            DeviceKind dev = get_device_kind(A_work.place());
            OpArgs output = ops()["lstsq"][dev][working_dtype](args);

            Array result = std::any_cast<Array>(output[0]);
            return to_original_dtype(result, B.dtype());
        }

        // ============================================================================
        // solve_triangular
        // ============================================================================

        Array solve_triangular(const Array& A, const Array& B,
            bool lower, bool unit_diag) {
            Array A_2d = ensure_2d_square(A, "solve_triangular");
            DType working_dtype = get_working_dtype(A.dtype());
            Array A_work = to_working_dtype(A_2d, working_dtype);
            Array B_work = to_working_dtype(B, working_dtype);

            if (A_work.place() != B_work.place()) {
                B_work = B_work.to(A_work.place());
            }

            OpArgs args = { A_work, B_work, lower, unit_diag };
            DeviceKind dev = get_device_kind(A_work.place());
            OpArgs output = ops()["solve_triangular"][dev][working_dtype](args);

            Array result = std::any_cast<Array>(output[0]);
            return to_original_dtype(result, B.dtype());
        }

        Array cov(const Array& x, bool rowvar, int ddof) {
            INS_CHECK(x.defined(), "cov: input is undefined");
            INS_CHECK(x.shape().ndim() == 2,
                "cov: input must be 2D, got ", x.shape().ndim(), "D");
            INS_CHECK(ddof >= 0 && ddof <= x.shape().dim(1),
                "cov: ddof must be between 0 and number of observations");

            // If rowvar is false, transpose so that rows are variables
            Array data = x;
            if (!rowvar) {
                data = x.transpose();
            }

            // data shape: (m, n) where m = number of variables, n = number of observations
            int64_t m = data.shape().dim(0);
            int64_t n = data.shape().dim(1);

            // Working dtype (float64 for better precision)
            DType working_dtype = (data.dtype() == DType::F32) ? DType::F32 : DType::F64;
            Array data_work = (data.dtype() == working_dtype) ? data : data.to(working_dtype);

            // Subtract mean along observations axis (axis=1)
            Array mean_vals = mean(data_work, 1, true);
            Array centered = sub(data_work, mean_vals);

            // Covariance matrix = (centered @ centered.T) / (n - ddof)
            Array centered_T = centered.transpose();
            Array cov_matrix = matmul(centered, centered_T);
            Array divisor = full(Shape({ 1 }), static_cast<double>(n - ddof), working_dtype, cov_matrix.place());
            Array result = div(cov_matrix, divisor);

            // Convert back to original dtype if needed
            if (result.dtype() != x.dtype()) {
                result = result.to(x.dtype());
            }

            return result;
        }

    } // namespace linalg
} // namespace ins