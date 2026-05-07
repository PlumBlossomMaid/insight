// tests/cuda/test_linalg.cu
#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include "insight/insight.h"
#include "insight/ops/linalg.h"

using namespace ins;

class LinalgTestGPU : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ins::init();
        set_device(GPUPlace(0));
    }
};

// ============================================================================
// Helper functions
// ============================================================================

static Array create_matrix_f64(int rows, int cols, const std::vector<double>& data) {
    Array result(Shape({ rows, cols }), DType::F64, CPUPlace());
    std::memcpy(result.data<double>(), data.data(), data.size() * sizeof(double));
    return result.to(GPUPlace(0));
}

static Array create_matrix_f32(int rows, int cols, const std::vector<float>& data) {
    Array result(Shape({ rows, cols }), DType::F32, CPUPlace());
    std::memcpy(result.data<float>(), data.data(), data.size() * sizeof(float));
    return result.to(GPUPlace(0));
}

static Array create_vector_f64(const std::vector<double>& data) {
    Array result(Shape({ static_cast<int64_t>(data.size()) }), DType::F64, CPUPlace());
    std::memcpy(result.data<double>(), data.data(), data.size() * sizeof(double));
    return result.to(GPUPlace(0));
}

static bool approx_equal(double a, double b, double rtol = 1e-6, double atol = 1e-8) {
    return std::abs(a - b) <= atol + rtol * std::abs(b);
}

static bool check_matrix_equal(const Array& A, const Array& B, double rtol = 1e-5) {
    Array A_cpu = A.to(CPUPlace());
    Array B_cpu = B.to(CPUPlace());
    if (A_cpu.shape() != B_cpu.shape()) return false;
    if (A_cpu.dtype() != B_cpu.dtype()) return false;
    int64_t n = A_cpu.numel();
    if (A_cpu.dtype() == DType::F64) {
        const double* a = A_cpu.data<double>();
        const double* b = B_cpu.data<double>();
        for (int64_t i = 0; i < n; ++i) {
            if (!approx_equal(a[i], b[i], rtol)) return false;
        }
    }
    else if (A_cpu.dtype() == DType::F32) {
        const float* a = A_cpu.data<float>();
        const float* b = B_cpu.data<float>();
        for (int64_t i = 0; i < n; ++i) {
            if (!approx_equal(static_cast<double>(a[i]), static_cast<double>(b[i]), rtol)) return false;
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

TEST_F(LinalgTestGPU, MatMul2x2F64) {
    Array A = create_matrix_f64(2, 2, { 1, 2, 3, 4 });
    Array B = create_matrix_f64(2, 2, { 5, 6, 7, 8 });
    Array C = linalg::matmul(A, B);

    Array C_cpu = C.to(CPUPlace());
    const double* c = C_cpu.data<double>();
    EXPECT_NEAR(c[0], 19, 1e-6);
    EXPECT_NEAR(c[1], 22, 1e-6);
    EXPECT_NEAR(c[2], 43, 1e-6);
    EXPECT_NEAR(c[3], 50, 1e-6);
}

TEST_F(LinalgTestGPU, MatMul2x2F32) {
    Array A = create_matrix_f32(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f });
    Array B = create_matrix_f32(2, 2, { 5.0f, 6.0f, 7.0f, 8.0f });
    Array C = linalg::matmul(A, B);

    Array C_cpu = C.to(CPUPlace());
    const float* c = C_cpu.data<float>();
    EXPECT_NEAR(c[0], 19.0f, 1e-5f);
    EXPECT_NEAR(c[1], 22.0f, 1e-5f);
    EXPECT_NEAR(c[2], 43.0f, 1e-5f);
    EXPECT_NEAR(c[3], 50.0f, 1e-5f);
}

TEST_F(LinalgTestGPU, MatMulNonSquare) {
    Array A = create_matrix_f64(2, 3, { 1, 2, 3, 4, 5, 6 });
    Array B = create_matrix_f64(3, 2, { 7, 8, 9, 10, 11, 12 });
    Array C = linalg::matmul(A, B);

    Array C_cpu = C.to(CPUPlace());
    const double* c = C_cpu.data<double>();
    EXPECT_NEAR(c[0], 58, 1e-6);
    EXPECT_NEAR(c[1], 64, 1e-6);
    EXPECT_NEAR(c[2], 139, 1e-6);
    EXPECT_NEAR(c[3], 154, 1e-6);
}

// ============================================================================
// det tests
// ============================================================================

TEST_F(LinalgTestGPU, Det2x2) {
    Array A = create_matrix_f64(2, 2, { 1, 2, 3, 4 });
    Array d = linalg::det(A);
    EXPECT_NEAR(d.to(CPUPlace()).item<double>(), -2.0, 1e-6);
}

TEST_F(LinalgTestGPU, Det3x3) {
    Array A = create_matrix_f64(3, 3, { 1, 2, 3, 2, 5, 3, 1, 0, 8 });
    Array d = linalg::det(A);
    EXPECT_NEAR(d.to(CPUPlace()).item<double>(), -1.0, 1e-6);
}

TEST_F(LinalgTestGPU, DetIdentity) {
    Array A = create_matrix_f64(3, 3, { 1, 0, 0, 0, 1, 0, 0, 0, 1 });
    Array d = linalg::det(A);
    EXPECT_NEAR(d.to(CPUPlace()).item<double>(), 1.0, 1e-6);
}

// ============================================================================
// slogdet tests
// ============================================================================

TEST_F(LinalgTestGPU, Slogdet2x2) {
    Array A = create_matrix_f64(2, 2, { 1, 2, 3, 4 });
    auto [sign, logdet] = linalg::slogdet(A);
    EXPECT_NEAR(sign.to(CPUPlace()).item<double>(), -1.0, 1e-6);
    EXPECT_NEAR(logdet.to(CPUPlace()).item<double>(), std::log(2.0), 1e-6);
}

// ============================================================================
// inv tests
// ============================================================================

TEST_F(LinalgTestGPU, Inv2x2) {
    Array A = create_matrix_f64(2, 2, { 1, 2, 3, 4 });
    Array invA = linalg::inv(A);
    Array invA_cpu = invA.to(CPUPlace());
    const double* data = invA_cpu.data<double>();
    EXPECT_NEAR(data[0], -2.0, 1e-6);
    EXPECT_NEAR(data[1], 1.0, 1e-6);
    EXPECT_NEAR(data[2], 1.5, 1e-6);
    EXPECT_NEAR(data[3], -0.5, 1e-6);

    Array I = linalg::matmul(A, invA);
    Array I_cpu = I.to(CPUPlace());
    const double* i = I_cpu.data<double>();
    EXPECT_NEAR(i[0], 1.0, 1e-6);
    EXPECT_NEAR(i[1], 0.0, 1e-6);
    EXPECT_NEAR(i[2], 0.0, 1e-6);
    EXPECT_NEAR(i[3], 1.0, 1e-6);
}

TEST_F(LinalgTestGPU, Inv3x3) {
    Array A = create_matrix_f64(3, 3, { 1, 2, 3, 2, 5, 3, 1, 0, 8 });
    Array invA = linalg::inv(A);
    Array I = linalg::matmul(A, invA);
    Array I_cpu = I.to(CPUPlace());

    const double* i = I_cpu.data<double>();
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

TEST_F(LinalgTestGPU, Solve3x3) {
    Array A = create_matrix_f64(3, 3, { 3, 2, -1, 2, -2, 4, -1, 0.5, -1 });
    Array b = create_vector_f64({ 1.0, -2.0, 0.0 });
    Array x = linalg::solve(A, b);

    Array x_cpu = x.to(CPUPlace());
    const double* data = x_cpu.data<double>();
    EXPECT_NEAR(data[0], 1.0, 1e-6);
    EXPECT_NEAR(data[1], -2.0, 1e-6);
    EXPECT_NEAR(data[2], -2.0, 1e-6);
}

TEST_F(LinalgTestGPU, SolveMultipleRHS) {
    Array A = create_matrix_f64(3, 3, { 3, 2, -1, 2, -2, 4, -1, 0.5, -1 });
    Array B = create_matrix_f64(3, 2, { 1, 0, -2, 1, 0, 2 });
    Array X = linalg::solve(A, B);

    EXPECT_EQ(X.shape(), Shape({ 3, 2 }));

    Array X_cpu = X.to(CPUPlace());
    const double* x = X_cpu.data<double>();
    EXPECT_NEAR(x[0], 1.0, 1e-6);
    EXPECT_NEAR(x[2], -2.0, 1e-6);
    EXPECT_NEAR(x[4], -2.0, 1e-6);
}

// ============================================================================
// cholesky tests
// ============================================================================

TEST_F(LinalgTestGPU, Cholesky3x3) {
    Array A = create_matrix_f64(3, 3, { 4, 12, -16, 12, 37, -43, -16, -43, 98 });
    Array L = linalg::cholesky(A, true);
    Array LLT = linalg::matmul(L, transpose(L));

    EXPECT_TRUE(check_matrix_equal(A, LLT, 1e-5));
}

// ============================================================================
// qr tests
// ============================================================================

TEST_F(LinalgTestGPU, QR3x3) {
    Array A = create_matrix_f64(3, 3, { 12, -51, 4, 6, 167, -68, -4, 24, -41 });
    auto [Q, R] = linalg::qr(A, "reduced");
    Array QR = linalg::matmul(Q, R);

    EXPECT_TRUE(check_matrix_equal(A, QR, 1e-5));

    Array QTQ = linalg::matmul(transpose(Q), Q);
    Array QTQ_cpu = QTQ.to(CPUPlace());
    EXPECT_NEAR(QTQ_cpu.data<double>()[0], 1.0, 1e-5);
    EXPECT_NEAR(QTQ_cpu.data<double>()[4], 1.0, 1e-5);
    EXPECT_NEAR(QTQ_cpu.data<double>()[8], 1.0, 1e-5);
}

TEST_F(LinalgTestGPU, QR3x2) {
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

TEST_F(LinalgTestGPU, SVD3x3) {
    Array A = create_matrix_f64(3, 3, { 1, 0, 0, 0, 2, 0, 0, 0, 3 });
    auto [U, S, VT] = linalg::svd(A, false);

    Array S_cpu = S.to(CPUPlace());
    const double* s = S_cpu.data<double>();
    EXPECT_NEAR(s[0], 3.0, 1e-6);
    EXPECT_NEAR(s[1], 2.0, 1e-6);
    EXPECT_NEAR(s[2], 1.0, 1e-6);
}

TEST_F(LinalgTestGPU, SVDvals) {
    Array A = create_matrix_f64(3, 3, { 1, 0, 0, 0, 2, 0, 0, 0, 3 });
    Array S = linalg::svdvals(A);

    Array S_cpu = S.to(CPUPlace());
    const double* s = S_cpu.data<double>();
    EXPECT_NEAR(s[0], 3.0, 1e-6);
    EXPECT_NEAR(s[1], 2.0, 1e-6);
    EXPECT_NEAR(s[2], 1.0, 1e-6);
}

// ============================================================================
// eigh / eigvalsh tests
// ============================================================================

TEST_F(LinalgTestGPU, Eigh3x3) {
    Array A = create_matrix_f64(3, 3, { 2, -1, 0, -1, 2, -1, 0, -1, 2 });
    auto [vals, vecs] = linalg::eigh(A, "L");

    Array vals_cpu = vals.to(CPUPlace());
    const double* v = vals_cpu.data<double>();
    EXPECT_NEAR(v[0], 0.5858, 1e-3);
    EXPECT_NEAR(v[1], 2.0, 1e-3);
    EXPECT_NEAR(v[2], 3.4142, 1e-3);
}

TEST_F(LinalgTestGPU, Eigvalsh) {
    Array A = create_matrix_f64(3, 3, { 2, -1, 0, -1, 2, -1, 0, -1, 2 });
    Array vals = linalg::eigvalsh(A, "L");

    Array vals_cpu = vals.to(CPUPlace());
    const double* v = vals_cpu.data<double>();
    EXPECT_NEAR(v[0], 0.5858, 1e-3);
    EXPECT_NEAR(v[1], 2.0, 1e-3);
    EXPECT_NEAR(v[2], 3.4142, 1e-3);
}

// ============================================================================
// pinv tests
// ============================================================================

TEST_F(LinalgTestGPU, Pinv2x2) {
    Array A = create_matrix_f64(2, 2, { 1, 2, 3, 4 });
    Array pinvA = linalg::pinv(A);

    Array I1 = linalg::matmul(A, pinvA);
    Array I2 = linalg::matmul(pinvA, A);

    Array I1_cpu = I1.to(CPUPlace());
    Array I2_cpu = I2.to(CPUPlace());
    EXPECT_NEAR(I1_cpu.data<double>()[0], 1.0, 1e-5);
    EXPECT_NEAR(I1_cpu.data<double>()[3], 1.0, 1e-5);
    EXPECT_NEAR(I2_cpu.data<double>()[0], 1.0, 1e-5);
    EXPECT_NEAR(I2_cpu.data<double>()[3], 1.0, 1e-5);
}

TEST_F(LinalgTestGPU, PinvRectangular) {
    Array A = create_matrix_f64(3, 2, { 1, 2, 3, 4, 5, 6 });
    Array pinvA = linalg::pinv(A);
    EXPECT_EQ(pinvA.shape(), Shape({ 2, 3 }));

    Array I = linalg::matmul(pinvA, A);
    Array I_cpu = I.to(CPUPlace());
    EXPECT_NEAR(I_cpu.data<double>()[0], 1.0, 1e-5);
    EXPECT_NEAR(I_cpu.data<double>()[3], 1.0, 1e-5);
}

// ============================================================================
// dot / outer tests
// ============================================================================

TEST_F(LinalgTestGPU, Dot) {
    Array a = create_vector_f64({ 1.0, 2.0, 3.0 });
    Array b = create_vector_f64({ 4.0, 5.0, 6.0 });
    Array d = linalg::dot(a, b);
    EXPECT_NEAR(d.to(CPUPlace()).item<double>(), 32.0, 1e-6);
}

TEST_F(LinalgTestGPU, Outer) {
    Array a = create_vector_f64({ 1.0, 2.0, 3.0 });
    Array b = create_vector_f64({ 4.0, 5.0, 6.0 });
    Array o = linalg::outer(a, b);

    Array o_cpu = o.to(CPUPlace());
    const double* data = o_cpu.data<double>();
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

TEST_F(LinalgTestGPU, MatrixPower2x2) {
    Array A = create_matrix_f64(2, 2, { 1, 2, 3, 4 });
    Array A2 = linalg::matrix_power(A, 2);

    Array A2_cpu = A2.to(CPUPlace());
    const double* data = A2_cpu.data<double>();
    EXPECT_NEAR(data[0], 7.0, 1e-6);
    EXPECT_NEAR(data[1], 10.0, 1e-6);
    EXPECT_NEAR(data[2], 15.0, 1e-6);
    EXPECT_NEAR(data[3], 22.0, 1e-6);
}

TEST_F(LinalgTestGPU, MatrixPowerZero) {
    Array A = create_matrix_f64(2, 2, { 1, 2, 3, 4 });
    Array A0 = linalg::matrix_power(A, 0);

    Array A0_cpu = A0.to(CPUPlace());
    const double* data = A0_cpu.data<double>();
    EXPECT_NEAR(data[0], 1.0, 1e-6);
    EXPECT_NEAR(data[1], 0.0, 1e-6);
    EXPECT_NEAR(data[2], 0.0, 1e-6);
    EXPECT_NEAR(data[3], 1.0, 1e-6);
}

// ============================================================================
// matrix_rank tests
// ============================================================================

TEST_F(LinalgTestGPU, MatrixRankFull) {
    Array A = create_matrix_f64(3, 3, { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    Array r = linalg::matrix_rank(A);
    EXPECT_EQ(r.to(CPUPlace()).item<int64_t>(), 2);
}

TEST_F(LinalgTestGPU, MatrixRankFullRank) {
    Array A = create_matrix_f64(3, 3, { 1, 2, 3, 2, 5, 3, 1, 0, 8 });
    Array r = linalg::matrix_rank(A);
    EXPECT_EQ(r.to(CPUPlace()).item<int64_t>(), 3);
}

// ============================================================================
// trace tests
// ============================================================================

TEST_F(LinalgTestGPU, Trace) {
    Array A = create_matrix_f64(3, 3, { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    Array t = linalg::trace(A);
    EXPECT_NEAR(t.to(CPUPlace()).item<double>(), 15.0, 1e-6);
}

// ============================================================================
// norm tests
// ============================================================================

TEST_F(LinalgTestGPU, NormVector2) {
    Array v = create_vector_f64({ 3.0, 4.0 });
    Array n = linalg::norm(v);
    EXPECT_NEAR(n.to(CPUPlace()).item<double>(), 5.0, 1e-6);
}

TEST_F(LinalgTestGPU, NormMatrixFrobenius) {
    Array A = create_matrix_f64(2, 2, { 1, 2, 3, 4 });
    Array n = linalg::norm(A, 2);
    double expected = std::sqrt(1.0 + 4.0 + 9.0 + 16.0);
    EXPECT_NEAR(n.to(CPUPlace()).item<double>(), expected, 1e-6);
}

TEST_F(LinalgTestGPU, NormMatrix1) {
    Array A = create_matrix_f64(2, 2, { 1, 2, 3, 4 });
    Array n = linalg::norm(A, 1);
    EXPECT_NEAR(n.to(CPUPlace()).item<double>(), 6.0, 1e-6);
}

TEST_F(LinalgTestGPU, NormMatrixInf) {
    Array A = create_matrix_f64(2, 2, { 1, 2, 3, 4 });
    Array n = linalg::norm(A, std::numeric_limits<double>::infinity());
    EXPECT_NEAR(n.to(CPUPlace()).item<double>(), 7.0, 1e-6);
}

// ============================================================================
// cond tests
// ============================================================================

TEST_F(LinalgTestGPU, Cond) {
    Array A = create_matrix_f64(2, 2, { 1, 2, 3, 4 });
    Array c = linalg::cond(A, 1);
    EXPECT_NEAR(c.to(CPUPlace()).item<double>(), 21.0, 1.0);
}

// ============================================================================
// solve_triangular tests
// ============================================================================

TEST_F(LinalgTestGPU, SolveTriangularUpper) {
    Array U = create_matrix_f64(3, 3, { 1, 2, 3, 0, 4, 5, 0, 0, 6 });
    Array b = create_vector_f64({ 1.0, 2.0, 3.0 });
    Array x = linalg::solve_triangular(U, b, false, false);

    Array x_cpu = x.to(CPUPlace());
    const double* data = x_cpu.data<double>();
    EXPECT_NEAR(data[0], -0.25, 1e-6);
    EXPECT_NEAR(data[1], -0.125, 1e-6);
    EXPECT_NEAR(data[2], 0.5, 1e-6);
}

TEST_F(LinalgTestGPU, SolveTriangularLower) {
    Array L = create_matrix_f64(3, 3, { 2, 0, 0, 1, 2, 0, 1, 1, 2 });
    Array b = create_vector_f64({ 4.0, 5.0, 6.0 });
    Array x = linalg::solve_triangular(L, b, true, false);

    Array x_cpu = x.to(CPUPlace());
    const double* data = x_cpu.data<double>();
    EXPECT_NEAR(data[0], 2.0, 1e-6);
    EXPECT_NEAR(data[1], 1.5, 1e-6);
    EXPECT_NEAR(data[2], 1.25, 1e-6);
}

// ============================================================================
// lstsq tests
// ============================================================================

TEST_F(LinalgTestGPU, LstsqOverdetermined) {
    Array A = create_matrix_f64(3, 2, { 1, 1, 1, 2, 1, 3 });
    Array b = create_vector_f64({ 2.0, 3.0, 4.0 });
    Array x = linalg::lstsq(A, b);

    Array x_cpu = x.to(CPUPlace());
    const double* data = x_cpu.data<double>();
    EXPECT_NEAR(data[0], 1.0, 1e-5);
    EXPECT_NEAR(data[1], 1.0, 1e-5);
}

TEST_F(LinalgTestGPU, LstsqUnderdetermined) {
    Array A = create_matrix_f64(2, 3, { 1, 0, 1, 0, 1, 1 });
    Array b = create_vector_f64({ 2.0, 3.0 });
    Array x = linalg::lstsq(A, b);

    EXPECT_EQ(x.shape(), Shape({ 3 }));

    Array Ax = linalg::matmul(A, x);
    EXPECT_EQ(Ax.shape(), Shape({ 2 }));
    Array Ax_cpu = Ax.to(CPUPlace());
    const double* ax_data = Ax_cpu.data<double>();
    EXPECT_NEAR(ax_data[0], 2.0, 1e-5);
    EXPECT_NEAR(ax_data[1], 3.0, 1e-5);
}

// ============================================================================
// lu tests
// ============================================================================

TEST_F(LinalgTestGPU, LuDecomposition) {
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

TEST_F(LinalgTestGPU, LqDecomposition) {
    Array A = create_matrix_f64(3, 3, { 1, 2, 3, 2, 5, 3, 1, 0, 8 });
    auto [L, Q] = linalg::lq(A, "reduced");
    Array LQ = linalg::matmul(L, Q);

    EXPECT_TRUE(check_matrix_equal(A, LQ, 1e-5));
}

// ============================================================================
// Additional data type tests
// ============================================================================

TEST_F(LinalgTestGPU, DetF32) {
    Array A = create_matrix_f32(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f });
    Array d = linalg::det(A);
    EXPECT_NEAR(d.to(CPUPlace()).item<float>(), -2.0f, 1e-5f);
}

TEST_F(LinalgTestGPU, InvF32) {
    Array A = create_matrix_f32(2, 2, { 1.0f, 2.0f, 3.0f, 4.0f });
    Array invA = linalg::inv(A);
    Array invA_cpu = invA.to(CPUPlace());
    const float* data = invA_cpu.data<float>();
    EXPECT_NEAR(data[0], -2.0f, 1e-5f);
    EXPECT_NEAR(data[1], 1.0f, 1e-5f);
    EXPECT_NEAR(data[2], 1.5f, 1e-5f);
    EXPECT_NEAR(data[3], -0.5f, 1e-5f);
}

TEST_F(LinalgTestGPU, SvdF32) {
    Array A = create_matrix_f32(3, 3, { 1.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 3.0f });
    auto [U, S, VT] = linalg::svd(A, false);
    Array S_cpu = S.to(CPUPlace());
    const float* s = S_cpu.data<float>();
    EXPECT_NEAR(s[0], 3.0f, 1e-5f);
    EXPECT_NEAR(s[1], 2.0f, 1e-5f);
    EXPECT_NEAR(s[2], 1.0f, 1e-5f);
}

TEST_F(LinalgTestGPU, Cov) {
    std::vector<double> data = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0
    };
    Array x_cpu = to_array(data, Shape({ 3, 4 }));
    Array x = x_cpu.to(GPUPlace(0));
    Array c = ins::linalg::cov(x);

    EXPECT_EQ(c.shape(), Shape({ 3, 3 }));
    Array c_cpu = c.to(CPUPlace());
    const double* c_data = c_cpu.data<double>();
    EXPECT_NEAR(c_data[0], 5.0 / 3.0, 1e-6);
    EXPECT_NEAR(c_data[1], 5.0 / 3.0, 1e-6);
}