// tests/cpu/test_linalg.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include "insight/insight.h"
#include "insight/ops/linalg.h"

using namespace ins;

class LinalgTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ins::init();
        set_device(CPUPlace());
    }
};

// ============================================================================
// Helper functions
// ============================================================================

static Array create_matrix_f64(int rows, int cols, const std::vector<double>& data) {
    Array result(Shape({ rows, cols }), DType::F64);
    std::memcpy(result.data<double>(), data.data(), data.size() * sizeof(double));
    return result;
}

static Array create_matrix_f32(int rows, int cols, const std::vector<float>& data) {
    Array result(Shape({ rows, cols }), DType::F32);
    std::memcpy(result.data<float>(), data.data(), data.size() * sizeof(float));
    return result;
}

static Array create_vector_f64(const std::vector<double>& data) {
    Array result(Shape({ static_cast<int64_t>(data.size()) }), DType::F64);
    std::memcpy(result.data<double>(), data.data(), data.size() * sizeof(double));
    return result;
}

static bool approx_equal(double a, double b, double rtol = 1e-6, double atol = 1e-8) {
    return std::abs(a - b) <= atol + rtol * std::abs(b);
}

static bool check_matrix_equal(const Array& A, const Array& B, double rtol = 1e-5) {
    if (A.shape() != B.shape()) return false;
    if (A.dtype() != B.dtype()) return false;
    int64_t n = A.numel();
    if (A.dtype() == DType::F64) {
        const double* a = A.data<double>();
        const double* b = B.data<double>();
        for (int64_t i = 0; i < n; ++i) {
            if (!approx_equal(a[i], b[i], rtol)) return false;
        }
    }
    else if (A.dtype() == DType::F32) {
        const float* a = A.data<float>();
        const float* b = B.data<float>();
        for (int64_t i = 0; i < n; ++i) {
            if (!approx_equal(static_cast<double>(a[i]), static_cast<double>(b[i]), rtol)) {
                return false;
            }
        }
    }
    else {
        return false;
    }
    return true;
}

// ============================================================================
// matmul tests
// ============================================================================

TEST_F(LinalgTest, MatMul2x2F64) {
    Array A = create_matrix_f64(2, 2, { 1, 2, 3, 4 });
    Array B = create_matrix_f64(2, 2, { 5, 6, 7, 8 });
    Array C = linalg::matmul(A, B);

    const double* c = C.data<double>();
    EXPECT_NEAR(c[0], 19, 1e-6);
    EXPECT_NEAR(c[1], 22, 1e-6);
    EXPECT_NEAR(c[2], 43, 1e-6);
    EXPECT_NEAR(c[3], 50, 1e-6);
}

TEST_F(LinalgTest, MatMul2x2F32) {
    Array A = create_matrix_f32(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f });
    Array B = create_matrix_f32(2, 2, { 5.0f, 6.0f, 7.0f, 8.0f });
    Array C = linalg::matmul(A, B);

    const float* c = C.data<float>();
    EXPECT_NEAR(c[0], 19.0f, 1e-5f);
    EXPECT_NEAR(c[1], 22.0f, 1e-5f);
    EXPECT_NEAR(c[2], 43.0f, 1e-5f);
    EXPECT_NEAR(c[3], 50.0f, 1e-5f);
}

TEST_F(LinalgTest, MatMulNonSquare) {
    Array A = create_matrix_f64(2, 3, { 1, 2, 3, 4, 5, 6 });
    Array B = create_matrix_f64(3, 2, { 7, 8, 9, 10, 11, 12 });
    Array C = linalg::matmul(A, B);

    const double* c = C.data<double>();
    // Expected: [[58, 64], [139, 154]]
    EXPECT_NEAR(c[0], 58, 1e-6);
    EXPECT_NEAR(c[1], 64, 1e-6);
    EXPECT_NEAR(c[2], 139, 1e-6);
    EXPECT_NEAR(c[3], 154, 1e-6);
}

// ============================================================================
// det tests
// ============================================================================

TEST_F(LinalgTest, Det2x2) {
    Array A = create_matrix_f64(2, 2, { 1, 2, 3, 4 });
    Array d = linalg::det(A);
    EXPECT_NEAR(d.item<double>(), -2.0, 1e-6);
}

TEST_F(LinalgTest, Det3x3) {
    Array A = create_matrix_f64(3, 3, { 1, 2, 3, 2, 5, 3, 1, 0, 8 });
    Array d = linalg::det(A);
    EXPECT_NEAR(d.item<double>(), -1.0, 1e-6);
}

TEST_F(LinalgTest, DetIdentity) {
    Array A = create_matrix_f64(3, 3, { 1, 0, 0, 0, 1, 0, 0, 0, 1 });
    Array d = linalg::det(A);
    EXPECT_NEAR(d.item<double>(), 1.0, 1e-6);
}

// ============================================================================
// slogdet tests
// ============================================================================

TEST_F(LinalgTest, Slogdet2x2) {
    Array A = create_matrix_f64(2, 2, { 1, 2, 3, 4 });
    auto [sign, logdet] = linalg::slogdet(A);
    EXPECT_NEAR(sign.item<double>(), -1.0, 1e-6);
    EXPECT_NEAR(logdet.item<double>(), std::log(2.0), 1e-6);
}

// ============================================================================
// inv tests
// ============================================================================

TEST_F(LinalgTest, Inv2x2) {
    Array A = create_matrix_f64(2, 2, { 1, 2, 3, 4 });
    Array invA = linalg::inv(A);
    const double* data = invA.data<double>();
    EXPECT_NEAR(data[0], -2.0, 1e-6);
    EXPECT_NEAR(data[1], 1.0, 1e-6);
    EXPECT_NEAR(data[2], 1.5, 1e-6);
    EXPECT_NEAR(data[3], -0.5, 1e-6);

    // Verify A * invA = I
    Array I = linalg::matmul(A, invA);
    const double* i = I.data<double>();
    EXPECT_NEAR(i[0], 1.0, 1e-6);
    EXPECT_NEAR(i[1], 0.0, 1e-6);
    EXPECT_NEAR(i[2], 0.0, 1e-6);
    EXPECT_NEAR(i[3], 1.0, 1e-6);
}

TEST_F(LinalgTest, Inv3x3) {
    Array A = create_matrix_f64(3, 3, { 1, 2, 3, 2, 5, 3, 1, 0, 8 });
    Array invA = linalg::inv(A);
    Array I = linalg::matmul(A, invA);

    const double* i = I.data<double>();
    EXPECT_NEAR(i[0], 1.0, 1e-5);
    EXPECT_NEAR(i[1], 0.0, 1e-5);
    EXPECT_NEAR(i[2], 0.0, 1e-5);
    EXPECT_NEAR(i[3], 0.0, 1e-5);
    EXPECT_NEAR(i[4], 1.0, 1e-5);
    EXPECT_NEAR(i[5], 0.0, 1e-5);
    EXPECT_NEAR(i[6], 0.0, 1e-5);
    EXPECT_NEAR(i[7], 0.0, 1e-5);
    EXPECT_NEAR(i[8], 1.0, 1e-5);
}

// ============================================================================
// solve tests
// ============================================================================

TEST_F(LinalgTest, Solve3x3) {
    Array A = create_matrix_f64(3, 3, { 3, 2, -1, 2, -2, 4, -1, 0.5, -1 });
    Array b = create_vector_f64({ 1.0, -2.0, 0.0 });
    Array x = linalg::solve(A, b);

    const double* data = x.data<double>();
    EXPECT_NEAR(data[0], 1.0, 1e-6);
    EXPECT_NEAR(data[1], -2.0, 1e-6);
    EXPECT_NEAR(data[2], -2.0, 1e-6);
}

TEST_F(LinalgTest, SolveMultipleRHS) {
    Array A = create_matrix_f64(3, 3, { 3, 2, -1, 2, -2, 4, -1, 0.5, -1 });
    Array B = create_matrix_f64(3, 2, { 1, 0, -2, 1, 0, 2 });
    Array X = linalg::solve(A, B);

    EXPECT_EQ(X.shape(), Shape({ 3, 2 }));

    const double* x = X.data<double>();
    // Row-major layout: 
    // row0: index 0, 1
    // row1: index 2, 3
    // row2: index 4, 5

    // First column (solution for B[:,0] = [1, -2, 0]) should be [1, -2, -2]
    EXPECT_NEAR(x[0], 1.0, 1e-6);
    EXPECT_NEAR(x[2], -2.0, 1e-6);
    EXPECT_NEAR(x[4], -2.0, 1e-6);

    // Second column: verify A * X[:,1] == B[:,1]
    Array X_col1 = create_matrix_f64(3, 1, { x[1], x[3], x[5] });
    Array B_col1 = create_matrix_f64(3, 1, { 0.0, 1.0, 2.0 });
    Array check = linalg::matmul(A, X_col1);
    EXPECT_NEAR(check.data<double>()[0], B_col1.data<double>()[0], 1e-5);
    EXPECT_NEAR(check.data<double>()[1], B_col1.data<double>()[1], 1e-5);
    EXPECT_NEAR(check.data<double>()[2], B_col1.data<double>()[2], 1e-5);
}

// ============================================================================
// cholesky tests
// ============================================================================

TEST_F(LinalgTest, Cholesky3x3) {
    Array A = create_matrix_f64(3, 3, { 4, 12, -16, 12, 37, -43, -16, -43, 98 });
    Array L = linalg::cholesky(A, true);
    Array LLT = linalg::matmul(L, transpose(L));

    EXPECT_TRUE(check_matrix_equal(A, LLT, 1e-5));
}

// ============================================================================
// qr tests
// ============================================================================

TEST_F(LinalgTest, QR3x3) {
    Array A = create_matrix_f64(3, 3, { 12, -51, 4, 6, 167, -68, -4, 24, -41 });
    auto [Q, R] = linalg::qr(A, "reduced");
    Array QR = linalg::matmul(Q, R);

    EXPECT_TRUE(check_matrix_equal(A, QR, 1e-5));

    // Check Q is orthogonal: Q^T * Q = I
    Array QTQ = linalg::matmul(transpose(Q), Q);
    EXPECT_NEAR(QTQ.data<double>()[0], 1.0, 1e-5);
    EXPECT_NEAR(QTQ.data<double>()[4], 1.0, 1e-5);
    EXPECT_NEAR(QTQ.data<double>()[8], 1.0, 1e-5);
}

TEST_F(LinalgTest, QR3x2) {
    Array A = create_matrix_f64(3, 2, { 1, 2, 3, 4, 5, 6 });
    auto [Q, R] = linalg::qr(A, "reduced");
    EXPECT_EQ(Q.shape(), Shape({ 3, 2 }));
    EXPECT_EQ(R.shape(), Shape({ 2, 2 }));

    Array QR = linalg::matmul(Q, R);
    EXPECT_TRUE(check_matrix_equal(A, QR, 1e-5));
}

// ============================================================================
// svd tests
// ============================================================================

TEST_F(LinalgTest, SVD3x3) {
    Array A = create_matrix_f64(3, 3, { 1, 0, 0, 0, 2, 0, 0, 0, 3 });
    auto [U, S, VT] = linalg::svd(A, false);

    const double* s = S.data<double>();
    EXPECT_NEAR(s[0], 3.0, 1e-6);
    EXPECT_NEAR(s[1], 2.0, 1e-6);
    EXPECT_NEAR(s[2], 1.0, 1e-6);
}

TEST_F(LinalgTest, SVDvals) {
    Array A = create_matrix_f64(3, 3, { 1, 0, 0, 0, 2, 0, 0, 0, 3 });
    Array S = linalg::svdvals(A);

    const double* s = S.data<double>();
    EXPECT_NEAR(s[0], 3.0, 1e-6);
    EXPECT_NEAR(s[1], 2.0, 1e-6);
    EXPECT_NEAR(s[2], 1.0, 1e-6);
}

// ============================================================================
// eigh / eigvalsh tests
// ============================================================================

TEST_F(LinalgTest, Eigh3x3) {
    Array A = create_matrix_f64(3, 3, { 2, -1, 0, -1, 2, -1, 0, -1, 2 });
    auto [vals, vecs] = linalg::eigh(A, "L");

    const double* v = vals.data<double>();
    EXPECT_NEAR(v[0], 0.5858, 1e-3);
    EXPECT_NEAR(v[1], 2.0, 1e-3);
    EXPECT_NEAR(v[2], 3.4142, 1e-3);
}

TEST_F(LinalgTest, Eigvalsh) {
    Array A = create_matrix_f64(3, 3, { 2, -1, 0, -1, 2, -1, 0, -1, 2 });
    Array vals = linalg::eigvalsh(A, "L");

    const double* v = vals.data<double>();
    EXPECT_NEAR(v[0], 0.5858, 1e-3);
    EXPECT_NEAR(v[1], 2.0, 1e-3);
    EXPECT_NEAR(v[2], 3.4142, 1e-3);
}

// ============================================================================
// pinv tests
// ============================================================================

TEST_F(LinalgTest, Pinv2x2) {
    Array A = create_matrix_f64(2, 2, { 1, 2, 3, 4 });
    Array pinvA = linalg::pinv(A);

    Array I1 = linalg::matmul(A, pinvA);
    Array I2 = linalg::matmul(pinvA, A);

    EXPECT_NEAR(I1.data<double>()[0], 1.0, 1e-5);
    EXPECT_NEAR(I1.data<double>()[3], 1.0, 1e-5);
    EXPECT_NEAR(I2.data<double>()[0], 1.0, 1e-5);
    EXPECT_NEAR(I2.data<double>()[3], 1.0, 1e-5);
}

TEST_F(LinalgTest, PinvRectangular) {
    Array A = create_matrix_f64(3, 2, { 1, 2, 3, 4, 5, 6 });
    Array pinvA = linalg::pinv(A);
    EXPECT_EQ(pinvA.shape(), Shape({ 2, 3 }));

    Array I = linalg::matmul(pinvA, A);
    EXPECT_NEAR(I.data<double>()[0], 1.0, 1e-5);
    EXPECT_NEAR(I.data<double>()[3], 1.0, 1e-5);
}

// ============================================================================
// dot / outer tests
// ============================================================================

TEST_F(LinalgTest, Dot) {
    Array a = create_vector_f64({ 1.0, 2.0, 3.0 });
    Array b = create_vector_f64({ 4.0, 5.0, 6.0 });
    Array d = linalg::dot(a, b);
    EXPECT_NEAR(d.item<double>(), 32.0, 1e-6);
}

TEST_F(LinalgTest, Outer) {
    Array a = create_vector_f64({ 1.0, 2.0, 3.0 });
    Array b = create_vector_f64({ 4.0, 5.0, 6.0 });
    Array o = linalg::outer(a, b);

    const double* data = o.data<double>();
    EXPECT_NEAR(data[0], 4.0, 1e-6);
    EXPECT_NEAR(data[1], 5.0, 1e-6);
    EXPECT_NEAR(data[2], 6.0, 1e-6);
    EXPECT_NEAR(data[3], 8.0, 1e-6);
    EXPECT_NEAR(data[4], 10.0, 1e-6);
    EXPECT_NEAR(data[5], 12.0, 1e-6);
    EXPECT_NEAR(data[6], 12.0, 1e-6);
    EXPECT_NEAR(data[7], 15.0, 1e-6);
    EXPECT_NEAR(data[8], 18.0, 1e-6);
}

// ============================================================================
// matrix_power tests
// ============================================================================

TEST_F(LinalgTest, MatrixPower2x2) {
    Array A = create_matrix_f64(2, 2, { 1, 2, 3, 4 });
    Array A2 = linalg::matrix_power(A, 2);

    const double* data = A2.data<double>();
    EXPECT_NEAR(data[0], 7.0, 1e-6);
    EXPECT_NEAR(data[1], 10.0, 1e-6);
    EXPECT_NEAR(data[2], 15.0, 1e-6);
    EXPECT_NEAR(data[3], 22.0, 1e-6);
}

TEST_F(LinalgTest, MatrixPowerZero) {
    Array A = create_matrix_f64(2, 2, { 1, 2, 3, 4 });
    Array A0 = linalg::matrix_power(A, 0);

    const double* data = A0.data<double>();
    EXPECT_NEAR(data[0], 1.0, 1e-6);
    EXPECT_NEAR(data[1], 0.0, 1e-6);
    EXPECT_NEAR(data[2], 0.0, 1e-6);
    EXPECT_NEAR(data[3], 1.0, 1e-6);
}

// ============================================================================
// matrix_rank tests
// ============================================================================

TEST_F(LinalgTest, MatrixRankFull) {
    Array A = create_matrix_f64(3, 3, { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    Array r = linalg::matrix_rank(A);
    EXPECT_EQ(r.item<int64_t>(), 2);  // Rank 2 (rows 0 and 1 are independent, row2 is dependent)
}

TEST_F(LinalgTest, MatrixRankFullRank) {
    Array A = create_matrix_f64(3, 3, { 1, 2, 3, 2, 5, 3, 1, 0, 8 });
    Array r = linalg::matrix_rank(A);
    EXPECT_EQ(r.item<int64_t>(), 3);
}

// ============================================================================
// trace tests
// ============================================================================

TEST_F(LinalgTest, Trace) {
    Array A = create_matrix_f64(3, 3, { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    Array t = linalg::trace(A);
    EXPECT_NEAR(t.item<double>(), 15.0, 1e-6);
}

// ============================================================================
// norm tests
// ============================================================================

TEST_F(LinalgTest, NormVector2) {
    Array v = create_vector_f64({ 3.0, 4.0 });
    Array n = linalg::norm(v);
    EXPECT_NEAR(n.item<double>(), 5.0, 1e-6);
}

TEST_F(LinalgTest, NormMatrixFrobenius) {
    Array A = create_matrix_f64(2, 2, { 1, 2, 3, 4 });
    Array n = linalg::norm(A, 2);
    double expected = std::sqrt(1.0 + 4.0 + 9.0 + 16.0);
    EXPECT_NEAR(n.item<double>(), expected, 1e-6);
}

TEST_F(LinalgTest, NormMatrix1) {
    Array A = create_matrix_f64(2, 2, { 1, 2, 3, 4 });
    Array n = linalg::norm(A, 1);
    // Column sum norm: max(col0=4, col1=6) = 6
    EXPECT_NEAR(n.item<double>(), 6.0, 1e-6); // 441
}

TEST_F(LinalgTest, NormMatrixInf) {
    Array A = create_matrix_f64(2, 2, { 1, 2, 3, 4 });
    Array n = linalg::norm(A, std::numeric_limits<double>::infinity());
    // Row sum norm: max(row0=3, row1=7) = 7
    EXPECT_NEAR(n.item<double>(), 7.0, 1e-6);
}

// ============================================================================
// cond tests
// ============================================================================

TEST_F(LinalgTest, Cond) {
    Array A = create_matrix_f64(2, 2, { 1, 2, 3, 4 });
    Array c = linalg::cond(A, 1);
    // Condition number in 1-norm ~ 21
    EXPECT_NEAR(c.item<double>(), 21.0, 1.0);
}

// ============================================================================
// solve_triangular tests
// ============================================================================

TEST_F(LinalgTest, SolveTriangularUpper) {
    Array U = create_matrix_f64(3, 3, { 1, 2, 3, 0, 4, 5, 0, 0, 6 });
    Array b = create_vector_f64({ 1.0, 2.0, 3.0 });
    Array x = linalg::solve_triangular(U, b, false, false);

    const double* data = x.data<double>();
    EXPECT_NEAR(data[0], -0.25, 1e-6);
    EXPECT_NEAR(data[1], -0.125, 1e-6);
    EXPECT_NEAR(data[2], 0.5, 1e-6);
}

TEST_F(LinalgTest, SolveTriangularLower) {
    Array L = create_matrix_f64(3, 3, { 2, 0, 0, 1, 2, 0, 1, 1, 2 });
    Array b = create_vector_f64({ 4.0, 5.0, 6.0 });
    Array x = linalg::solve_triangular(L, b, true, false);

    // Solution: x0 = 2, x1 = (5-2)/2 = 1.5, x2 = (6-2-1.5)/2 = 1.25
    const double* data = x.data<double>();
    EXPECT_NEAR(data[0], 2.0, 1e-6);
    EXPECT_NEAR(data[1], 1.5, 1e-6);
    EXPECT_NEAR(data[2], 1.25, 1e-6);
}

// ============================================================================
// lstsq tests
// ============================================================================

TEST_F(LinalgTest, LstsqOverdetermined) {
    Array A = create_matrix_f64(3, 2, { 1, 1, 1, 2, 1, 3 });
    Array b = create_vector_f64({ 2.0, 3.0, 4.0 });
    Array x = linalg::lstsq(A, b);

    const double* data = x.data<double>();
    EXPECT_NEAR(data[0], 1.0, 1e-5);
    EXPECT_NEAR(data[1], 1.0, 1e-5);
}

TEST_F(LinalgTest, LstsqUnderdetermined) {
    Array A = create_matrix_f64(2, 3, { 1, 0, 1, 0, 1, 1 });
    Array b = create_vector_f64({ 2.0, 3.0 });
    Array x = linalg::lstsq(A, b);

    EXPECT_EQ(x.shape(), Shape({ 3 }));

    // Check that A * x ≈ b
    Array Ax = linalg::matmul(A, x);
    EXPECT_EQ(Ax.shape(), Shape({ 2 }));
    const double* ax_data = Ax.data<double>();
    EXPECT_NEAR(ax_data[0], 2.0, 1e-5);
    EXPECT_NEAR(ax_data[1], 3.0, 1e-5);
}

// ============================================================================
// lu tests
// ============================================================================

TEST_F(LinalgTest, LuDecomposition) {
    Array A = create_matrix_f64(3, 3, { 1, 2, 3, 2, 5, 3, 1, 0, 8 });
    auto [LU, pivots] = linalg::lu(A, true);
    auto [P, L, U] = linalg::lu_unpack(LU, pivots);

    Array PA = linalg::matmul(P, A);
    Array LU_mat = linalg::matmul(L, U);

    EXPECT_TRUE(check_matrix_equal(PA, LU_mat, 1e-5));
}

// ============================================================================
// lq tests
// ============================================================================

TEST_F(LinalgTest, LqDecomposition) {
    Array A = create_matrix_f64(3, 3, { 1, 2, 3, 2, 5, 3, 1, 0, 8 });
    auto [L, Q] = linalg::lq(A, "reduced");
    Array LQ = linalg::matmul(L, Q);

    EXPECT_TRUE(check_matrix_equal(A, LQ, 1e-5));
}

// ============================================================================
// Additional data type tests
// ============================================================================

TEST_F(LinalgTest, DetF32) {
    Array A = create_matrix_f32(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f });
    Array d = linalg::det(A);
    EXPECT_NEAR(d.item<float>(), -2.0f, 1e-5f);
}

TEST_F(LinalgTest, InvF32) {
    Array A = create_matrix_f32(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f });
    Array invA = linalg::inv(A);
    const float* data = invA.data<float>();
    EXPECT_NEAR(data[0], -2.0f, 1e-5f);
    EXPECT_NEAR(data[1], 1.0f, 1e-5f);
    EXPECT_NEAR(data[2], 1.5f, 1e-5f);
    EXPECT_NEAR(data[3], -0.5f, 1e-5f);
}

TEST_F(LinalgTest, SvdF32) {
    Array A = create_matrix_f32(3, 3, { 1.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 3.0f });
    auto [U, S, VT] = linalg::svd(A, false);
    const float* s = S.data<float>();
    EXPECT_NEAR(s[0], 3.0f, 1e-5f);
    EXPECT_NEAR(s[1], 2.0f, 1e-5f);
    EXPECT_NEAR(s[2], 1.0f, 1e-5f);
}

TEST_F(LinalgTest, Cov) {
    // Create test data: 3 variables, 4 observations
    std::vector<double> data = {
        1.0, 2.0, 3.0, 4.0,   // variable 1
        5.0, 6.0, 7.0, 8.0,   // variable 2
        9.0, 10.0, 11.0, 12.0 // variable 3
    };
    Array x = to_array(data, Shape({ 3, 4 }));
    Array c = ins::linalg::cov(x);

    EXPECT_EQ(c.shape(), Shape({ 3, 3 }));
    const double* c_data = c.data<double>();
    // Diagonal: variance of each variable = (0^2+0^2+0^2+0^2)/3? Actually (1^2+...)/3
    // With mean 2.5, 4.5, 6.5? Let's check approximation
    // For variable 1: [1,2,3,4] mean=2.5, (1-2.5)^2+(2-2.5)^2+(3-2.5)^2+(4-2.5)^2 = 5, /3 = 1.666...
    EXPECT_NEAR(c_data[0], 5.0 / 3.0, 1e-6);
    // Covariance between var1 and var2: sum((x1-mean1)*(x2-mean2))/3 = 5/3 ≈ 1.666
    EXPECT_NEAR(c_data[1], 5.0 / 3.0, 1e-6);
}