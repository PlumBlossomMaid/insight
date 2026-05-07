// backends/cpu/linalg.cpp
#include "insight/plugin/op_registry.h"
#include "insight/core/array.h"
#include "insight/ops/broadcast.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <openblas/cblas.h>

// LAPACK Fortran interfaces
extern "C" {
    // Level 3 BLAS
    void dgemm_(char* transa, char* transb, int* m, int* n, int* k,
        double* alpha, double* a, int* lda, double* b, int* ldb,
        double* beta, double* c, int* ldc);
    void sgemm_(char* transa, char* transb, int* m, int* n, int* k,
        float* alpha, float* a, int* lda, float* b, int* ldb,
        float* beta, float* c, int* ldc);
    void dger_(int* m, int* n, double* alpha, double* x, int* incx,
        double* y, int* incy, double* a, int* lda);
    void sger_(int* m, int* n, float* alpha, float* x, int* incx,
        float* y, int* incy, float* a, int* lda);
    double ddot_(int* n, double* x, int* incx, double* y, int* incy);
    float sdot_(int* n, float* x, int* incx, float* y, int* incy);

    // LAPACK
    void dgetrf_(int* m, int* n, double* a, int* lda, int* ipiv, int* info);
    void sgetrf_(int* m, int* n, float* a, int* lda, int* ipiv, int* info);
    void dgetri_(int* n, double* a, int* lda, int* ipiv, double* work, int* lwork, int* info);
    void sgetri_(int* n, float* a, int* lda, int* ipiv, float* work, int* lwork, int* info);
    void dgesv_(int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb, int* info);
    void sgesv_(int* n, int* nrhs, float* a, int* lda, int* ipiv, float* b, int* ldb, int* info);
    void dgecon_(char* norm, int* n, double* a, int* lda, double* anorm, double* rcond,
        double* work, int* iwork, int* info);
    void sgecon_(char* norm, int* n, float* a, int* lda, float* anorm, float* rcond,
        float* work, int* iwork, int* info);
    double dlange_(char* norm, int* m, int* n, double* a, int* lda, double* work);
    float slange_(char* norm, int* m, int* n, float* a, int* lda, float* work);
    void dgeqrf_(int* m, int* n, double* a, int* lda, double* tau, double* work, int* lwork, int* info);
    void sgeqrf_(int* m, int* n, float* a, int* lda, float* tau, float* work, int* lwork, int* info);
    void dorgqr_(int* m, int* n, int* k, double* a, int* lda, double* tau,
        double* work, int* lwork, int* info);
    void sorgqr_(int* m, int* n, int* k, float* a, int* lda, float* tau,
        float* work, int* lwork, int* info);
    void dgelqf_(int* m, int* n, double* a, int* lda, double* tau, double* work, int* lwork, int* info);
    void sgelqf_(int* m, int* n, float* a, int* lda, float* tau, float* work, int* lwork, int* info);
    void dorglq_(int* m, int* n, int* k, double* a, int* lda, double* tau,
        double* work, int* lwork, int* info);
    void sorglq_(int* m, int* n, int* k, float* a, int* lda, float* tau,
        float* work, int* lwork, int* info);
    void dpotrf_(char* uplo, int* n, double* a, int* lda, int* info);
    void spotrf_(char* uplo, int* n, float* a, int* lda, int* info);
    void dpotrs_(char* uplo, int* n, int* nrhs, double* a, int* lda,
        double* b, int* ldb, int* info);
    void spotrs_(char* uplo, int* n, int* nrhs, float* a, int* lda,
        float* b, int* ldb, int* info);
    void dgesvd_(char* jobu, char* jobvt, int* m, int* n, double* a, int* lda,
        double* s, double* u, int* ldu, double* vt, int* ldvt,
        double* work, int* lwork, int* info);
    void sgesvd_(char* jobu, char* jobvt, int* m, int* n, float* a, int* lda,
        float* s, float* u, int* ldu, float* vt, int* ldvt,
        float* work, int* lwork, int* info);
    void dgeev_(char* jobvl, char* jobvr, int* n, double* a, int* lda,
        double* wr, double* wi, double* vl, int* ldvl, double* vr, int* ldvr,
        double* work, int* lwork, int* info);
    void sgeev_(char* jobvl, char* jobvr, int* n, float* a, int* lda,
        float* wr, float* wi, float* vl, int* ldvl, float* vr, int* ldvr,
        float* work, int* lwork, int* info);
    void dsyev_(char* jobz, char* uplo, int* n, double* a, int* lda,
        double* w, double* work, int* lwork, int* info);
    void ssyev_(char* jobz, char* uplo, int* n, float* a, int* lda,
        float* w, float* work, int* lwork, int* info);
    void dgels_(char* trans, int* m, int* n, int* nrhs, double* a, int* lda,
        double* b, int* ldb, double* work, int* lwork, int* info);
    void sgels_(char* trans, int* m, int* n, int* nrhs, float* a, int* lda,
        float* b, int* ldb, float* work, int* lwork, int* info);
    void dtrtrs_(char* uplo, char* trans, char* diag, int* n, int* nrhs,
        double* a, int* lda, double* b, int* ldb, int* info);
    void strtrs_(char* uplo, char* trans, char* diag, int* n, int* nrhs,
        float* a, int* lda, float* b, int* ldb, int* info);
}

namespace ins::cpu {
#ifdef INSIGHT_USE_OPENBLAS
    // ============================================================================
    // Helper: Manual LU decomposition with OpenMP (column-major)
    // ============================================================================

    template<typename T>
    static void lu_decompose(std::vector<T>& A, int n, std::vector<int>& ipiv) {
        ipiv.resize(n);
        for (int i = 0; i < n; ++i) {
            ipiv[i] = i;
        }

        for (int i = 0; i < n; ++i) {
            // Find pivot (sequential, cannot parallelize)
            int pivot_row = i;
            T max_val = std::abs(A[i + i * n]);
            for (int k = i + 1; k < n; ++k) {
                T val = std::abs(A[k + i * n]);
                if (val > max_val) {
                    max_val = val;
                    pivot_row = k;
                }
            }

            if (max_val < std::numeric_limits<T>::epsilon()) {
                continue;  // singular matrix
            }

            // Swap rows if needed
            if (pivot_row != i) {
                for (int j = i; j < n; ++j) {
                    std::swap(A[i + j * n], A[pivot_row + j * n]);
                }
                std::swap(ipiv[i], ipiv[pivot_row]);
            }

            // Compute multipliers
            T pivot = A[i + i * n];
            for (int k = i + 1; k < n; ++k) {
                A[k + i * n] /= pivot;
            }

            // Update trailing matrix (can be parallelized)
#pragma omp parallel for
            for (int k = i + 1; k < n; ++k) {
                T factor = A[k + i * n];
                for (int j = i + 1; j < n; ++j) {
                    A[k + j * n] -= factor * A[i + j * n];
                }
            }
        }
    }

    // Helper: row-major to column-major conversion
    template<typename T>
    static std::vector<T> rowmajor_to_colmajor(const T* src, int rows, int cols) {
        std::vector<T> dst(rows * cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                dst[i + j * rows] = src[i * cols + j];
            }
        }
        return dst;
    }

    // Helper: column-major to row-major conversion
    template<typename T>
    static void colmajor_to_rowmajor(T* dst, const T* src, int rows, int cols) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                dst[i * cols + j] = src[i + j * rows];
            }
        }
    }

    static void check_lapack_info(int info, const char* func_name) {
        if (info < 0) {
            INS_THROW(func_name, ": illegal value at argument ", -info);
        }
        else if (info > 0) {
            if (std::string(func_name).find("getrf") != std::string::npos ||
                std::string(func_name).find("getri") != std::string::npos) {
                INS_THROW(func_name, ": singular matrix (U(", info, ",", info, ") is zero)");
            }
            else if (std::string(func_name).find("potrf") != std::string::npos) {
                INS_THROW(func_name, ": matrix not positive definite at leading minor ", info);
            }
        }
    }

    // ============================================================================
    // matmul
    // ============================================================================


    static OpArgs matmul_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& a = std::any_cast<const Array&>(args[1]);
        const Array& b = std::any_cast<const Array&>(args[2]);

        Array& mutable_out = const_cast<Array&>(out);
        DType dtype = out.dtype();

        int ndim_a = a.shape().ndim();
        int ndim_b = b.shape().ndim();

        // Case 1: vector * vector -> scalar
        if (ndim_a == 1 && ndim_b == 1) {
            Array a_contig = a.is_contiguous() ? a : a.contiguous();
            Array b_contig = b.is_contiguous() ? b : b.contiguous();
            int n = static_cast<int>(a.numel());
            if (dtype == DType::F64) {
                const double* a_data = a_contig.data<double>();
                const double* b_data = b_contig.data<double>();
                double result = cblas_ddot(n, a_data, 1, b_data, 1);
                *mutable_out.data<double>() = result;
            }
            else {
                const float* a_data = a_contig.data<float>();
                const float* b_data = b_contig.data<float>();
                float result = cblas_sdot(n, a_data, 1, b_data, 1);
                *mutable_out.data<float>() = result;
            }
            return { mutable_out };
        }

        // Case 2: matrix * vector -> vector
        if (ndim_a == 2 && ndim_b == 1) {
            Array a_contig = a.is_contiguous() ? a : a.contiguous();
            Array b_contig = b.is_contiguous() ? b : b.contiguous();
            int m = static_cast<int>(a.shape().dim(0));
            int n = static_cast<int>(a.shape().dim(1));
            if (dtype == DType::F64) {
                const double* a_data = a_contig.data<double>();
                const double* b_data = b_contig.data<double>();
                double* c_data = mutable_out.data<double>();
                cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n,
                    1.0, a_data, n, b_data, 1, 0.0, c_data, 1);
            }
            else {
                const float* a_data = a_contig.data<float>();
                const float* b_data = b_contig.data<float>();
                float* c_data = mutable_out.data<float>();
                cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n,
                    1.0f, a_data, n, b_data, 1, 0.0f, c_data, 1);
            }
            return { mutable_out };
        }

        // Case 3: vector * matrix -> vector
        if (ndim_a == 1 && ndim_b == 2) {
            Array a_contig = a.is_contiguous() ? a : a.contiguous();
            Array b_contig = b.is_contiguous() ? b : b.contiguous();
            int m = static_cast<int>(b.shape().dim(0));
            int n = static_cast<int>(b.shape().dim(1));
            if (dtype == DType::F64) {
                const double* a_data = a_contig.data<double>();
                const double* b_data = b_contig.data<double>();
                double* c_data = mutable_out.data<double>();
                cblas_dgemv(CblasRowMajor, CblasTrans, m, n,
                    1.0, b_data, n, a_data, 1, 0.0, c_data, 1);
            }
            else {
                const float* a_data = a_contig.data<float>();
                const float* b_data = b_contig.data<float>();
                float* c_data = mutable_out.data<float>();
                cblas_sgemv(CblasRowMajor, CblasTrans, m, n,
                    1.0f, b_data, n, a_data, 1, 0.0f, c_data, 1);
            }
            return { mutable_out };
        }

        // Case 4: matrix * matrix -> matrix
        if (ndim_a == 2 && ndim_b == 2) {
            Array a_contig = a.is_contiguous() ? a : a.contiguous();
            Array b_contig = b.is_contiguous() ? b : b.contiguous();
            int m = static_cast<int>(a_contig.shape().dim(0));
            int k = static_cast<int>(a_contig.shape().dim(1));
            int n = static_cast<int>(b_contig.shape().dim(1));
            if (dtype == DType::F64) {
                double* c = mutable_out.data<double>();
                const double* a_data = a_contig.data<double>();
                const double* b_data = b_contig.data<double>();
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, 1.0, a_data, k, b_data, n, 0.0, c, n);
            }
            else {
                float* c = mutable_out.data<float>();
                const float* a_data = a_contig.data<float>();
                const float* b_data = b_contig.data<float>();
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, 1.0f, a_data, k, b_data, n, 0.0f, c, n);
            }
            return { mutable_out };
        }

        // Case 5: batched matrix multiplication (N-D)
        auto broadcasted = broadcast_arrays({ a, b });
        Array a_bc = broadcasted[0];
        Array b_bc = broadcasted[1];
        int ndim = a_bc.shape().ndim();
        int batch_dims = ndim - 2;
        int64_t batch_size = 1;
        for (int i = 0; i < batch_dims; ++i) {
            batch_size *= a_bc.shape().dim(i);
        }
        int m = static_cast<int>(a_bc.shape().dim(batch_dims));
        int k = static_cast<int>(a_bc.shape().dim(batch_dims + 1));
        int n = static_cast<int>(b_bc.shape().dim(batch_dims + 1));

        Array a_flat = a_bc.reshape(Shape({ batch_size, m, k }));
        Array b_flat = b_bc.reshape(Shape({ batch_size, k, n }));

        if (dtype == DType::F64) {
            const double* a_data = a_flat.data<double>();
            const double* b_data = b_flat.data<double>();
            double* c_data = mutable_out.data<double>();
            for (int64_t batch = 0; batch < batch_size; ++batch) {
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, 1.0,
                    a_data + batch * m * k, k,
                    b_data + batch * k * n, n,
                    0.0, c_data + batch * m * n, n);
            }
        }
        else {
            const float* a_data = a_flat.data<float>();
            const float* b_data = b_flat.data<float>();
            float* c_data = mutable_out.data<float>();
            for (int64_t batch = 0; batch < batch_size; ++batch) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, 1.0f,
                    a_data + batch * m * k, k,
                    b_data + batch * k * n, n,
                    0.0f, c_data + batch * m * n, n);
            }
        }
        return { mutable_out };
    }

    REGISTER_KERNEL(matmul, CPU, F64, matmul_kernel);
    REGISTER_KERNEL(matmul, CPU, F32, matmul_kernel);

    // ============================================================================
    // det
    // ============================================================================

    static Array det_f64_impl(const Array& x) {
        int n = static_cast<int>(x.shape().dim(0));
        std::vector<double> work(x.numel());
        std::memcpy(work.data(), x.data<double>(), x.numel() * sizeof(double));

        std::vector<int> ipiv(n);
        int info;

        dgetrf_(&n, &n, work.data(), &n, ipiv.data(), &info);
        check_lapack_info(info, "det(dgetrf)");

        double det = 1.0;
        int sign_changes = 0;
        for (int i = 0; i < n; ++i) {
            det *= work[i * n + i];
            if (ipiv[i] != i + 1) {
                sign_changes++;
            }
        }
        if (sign_changes % 2 == 1) {
            det = -det;
        }

        Array result(Shape({}), DType::F64, x.place());
        *result.data<double>() = det;
        return result;
    }

    static Array det_f32_impl(const Array& x) {
        int n = static_cast<int>(x.shape().dim(0));
        std::vector<float> work(x.numel());
        std::memcpy(work.data(), x.data<float>(), x.numel() * sizeof(float));

        std::vector<int> ipiv(n);
        int info;

        sgetrf_(&n, &n, work.data(), &n, ipiv.data(), &info);
        check_lapack_info(info, "det(sgetrf)");

        float det = 1.0f;
        int sign_changes = 0;
        for (int i = 0; i < n; ++i) {
            det *= work[i * n + i];
            if (ipiv[i] != i + 1) {
                sign_changes++;
            }
        }
        if (sign_changes % 2 == 1) {
            det = -det;
        }

        Array result(Shape({}), DType::F32, x.place());
        *result.data<float>() = det;
        return result;
    }

    static OpArgs det_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        if (x.dtype() == DType::F64) {
            return { det_f64_impl(x) };
        }
        else {
            return { det_f32_impl(x) };
        }
    }

    REGISTER_KERNEL(det, CPU, F64, det_kernel);
    REGISTER_KERNEL(det, CPU, F32, det_kernel);

    // ============================================================================
    // slogdet
    // ============================================================================

    static std::pair<Array, Array> slogdet_f64_impl(const Array& x) {
        int n = static_cast<int>(x.shape().dim(0));
        std::vector<double> work(x.numel());
        std::memcpy(work.data(), x.data<double>(), x.numel() * sizeof(double));

        std::vector<int> ipiv(n);
        int info;

        dgetrf_(&n, &n, work.data(), &n, ipiv.data(), &info);
        check_lapack_info(info, "slogdet(dgetrf)");

        double det = 1.0;
        int sign_changes = 0;
        for (int i = 0; i < n; ++i) {
            det *= work[i * n + i];
            if (ipiv[i] != i + 1) {
                sign_changes++;
            }
        }
        double sign = (sign_changes % 2 == 1) ? -1.0 : 1.0;
        double logdet = std::log(std::abs(det));

        Array sign_arr(Shape({}), DType::F64, x.place());
        Array logdet_arr(Shape({}), DType::F64, x.place());
        *sign_arr.data<double>() = sign;
        *logdet_arr.data<double>() = logdet;
        return { sign_arr, logdet_arr };
    }

    static std::pair<Array, Array> slogdet_f32_impl(const Array& x) {
        int n = static_cast<int>(x.shape().dim(0));
        std::vector<float> work(x.numel());
        std::memcpy(work.data(), x.data<float>(), x.numel() * sizeof(float));

        std::vector<int> ipiv(n);
        int info;

        sgetrf_(&n, &n, work.data(), &n, ipiv.data(), &info);
        check_lapack_info(info, "slogdet(sgetrf)");

        float det = 1.0f;
        int sign_changes = 0;
        for (int i = 0; i < n; ++i) {
            det *= work[i * n + i];
            if (ipiv[i] != i + 1) {
                sign_changes++;
            }
        }
        float sign = (sign_changes % 2 == 1) ? -1.0f : 1.0f;
        double logdet = std::log(std::abs(det));

        Array sign_arr(Shape({}), DType::F32, x.place());
        Array logdet_arr(Shape({}), DType::F64, x.place());
        *sign_arr.data<float>() = sign;
        *logdet_arr.data<double>() = logdet;
        return { sign_arr, logdet_arr };
    }

    static OpArgs slogdet_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        if (x.dtype() == DType::F64) {
            auto [sign, logdet] = slogdet_f64_impl(x);
            return { sign, logdet };
        }
        else {
            auto [sign, logdet] = slogdet_f32_impl(x);
            return { sign, logdet };
        }
    }

    REGISTER_KERNEL(slogdet, CPU, F64, slogdet_kernel);
    REGISTER_KERNEL(slogdet, CPU, F32, slogdet_kernel);

    // ============================================================================
    // cond
    // ============================================================================

    static Array cond_f64_impl(const Array& x, double p) {
        int m = static_cast<int>(x.shape().dim(0));
        int n = static_cast<int>(x.shape().dim(1));
        std::vector<double> work(x.numel());
        std::memcpy(work.data(), x.data<double>(), x.numel() * sizeof(double));

        char norm_char = '1';
        double anorm = dlange_(&norm_char, &m, &n, work.data(), &m, nullptr);

        std::vector<int> ipiv(n);
        int info;
        dgetrf_(&n, &n, work.data(), &n, ipiv.data(), &info);
        check_lapack_info(info, "cond(dgetrf)");

        double rcond;
        std::vector<double> work_gecon(4 * n);
        std::vector<int> iwork(n);
        dgecon_(&norm_char, &n, work.data(), &n, &anorm, &rcond,
            work_gecon.data(), iwork.data(), &info);

        double cond_val = 1.0 / rcond;
        Array result(Shape({}), DType::F64, x.place());
        *result.data<double>() = cond_val;
        return result;
    }

    static Array cond_f32_impl(const Array& x, double p) {
        int m = static_cast<int>(x.shape().dim(0));
        int n = static_cast<int>(x.shape().dim(1));
        std::vector<float> work(x.numel());
        std::memcpy(work.data(), x.data<float>(), x.numel() * sizeof(float));

        char norm_char = '1';
        float anorm = slange_(&norm_char, &m, &n, work.data(), &m, nullptr);

        std::vector<int> ipiv(n);
        int info;
        sgetrf_(&n, &n, work.data(), &n, ipiv.data(), &info);
        check_lapack_info(info, "cond(sgetrf)");

        float rcond;
        std::vector<float> work_gecon(4 * n);
        std::vector<int> iwork(n);
        sgecon_(&norm_char, &n, work.data(), &n, &anorm, &rcond,
            work_gecon.data(), iwork.data(), &info);

        float cond_val = 1.0f / rcond;
        Array result(Shape({}), DType::F32, x.place());
        *result.data<float>() = cond_val;
        return result;
    }

    static OpArgs cond_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        double p = std::any_cast<double>(args[1]);
        if (x.dtype() == DType::F64) {
            return { cond_f64_impl(x, p) };
        }
        else {
            return { cond_f32_impl(x, p) };
        }
    }

    REGISTER_KERNEL(cond, CPU, F64, cond_kernel);
    REGISTER_KERNEL(cond, CPU, F32, cond_kernel);

    // ============================================================================
    // norm
    // ============================================================================

    static Array norm_f64_impl(const Array& x, double ord) {
        double norm_val;
        int ndim = x.shape().ndim();

        if (ndim == 1) {
            int n = static_cast<int>(x.numel());
            const double* data = x.data<double>();
            if (ord == 2.0) {
                int incx = 1;
                norm_val = ddot_(&n, const_cast<double*>(data), &incx,
                    const_cast<double*>(data), &incx);
                norm_val = std::sqrt(norm_val);
            }
            else if (ord == 1.0) {
                double sum = 0.0;
                for (int i = 0; i < n; ++i) {
                    sum += std::abs(data[i]);
                }
                norm_val = sum;
            }
            else if (ord == std::numeric_limits<double>::infinity()) {
                double max_val = 0.0;
                for (int i = 0; i < n; ++i) {
                    max_val = std::max(max_val, std::abs(data[i]));
                }
                norm_val = max_val;
            }
            else {
                double sum = 0.0;
                for (int i = 0; i < n; ++i) {
                    sum += std::pow(std::abs(data[i]), ord);
                }
                norm_val = std::pow(sum, 1.0 / ord);
            }
        }
        else {
            int m = static_cast<int>(x.shape().dim(0));
            int n = static_cast<int>(x.shape().dim(1));
            const double* data = x.data<double>();

            char norm_char[2];
            if (ord == 1.0) {
                norm_char[0] = 'I';
                norm_char[1] = '\0';
                std::vector<double> work(m);
                norm_val = dlange_(norm_char, &m, &n, const_cast<double*>(data), &m, work.data());
            }
            else if (ord == std::numeric_limits<double>::infinity()) {
                norm_char[0] = '1';
                norm_char[1] = '\0';
                norm_val = dlange_(norm_char, &m, &n, const_cast<double*>(data), &m, nullptr);
            }
            else {
                norm_char[0] = 'F';
                norm_char[1] = '\0';
                norm_val = dlange_(norm_char, &m, &n, const_cast<double*>(data), &m, nullptr);
            }
        }

        Array result(Shape({}), DType::F64, x.place());
        *result.data<double>() = norm_val;
        return result;
    }

    static Array norm_f32_impl(const Array& x, double ord) {
        float norm_val;
        int ndim = x.shape().ndim();

        if (ndim == 1) {
            int n = static_cast<int>(x.numel());
            const float* data = x.data<float>();
            if (ord == 2.0) {
                int incx = 1;
                norm_val = sdot_(&n, const_cast<float*>(data), &incx,
                    const_cast<float*>(data), &incx);
                norm_val = std::sqrt(norm_val);
            }
            else if (ord == 1.0) {
                float sum = 0.0f;
                for (int i = 0; i < n; ++i) {
                    sum += std::abs(data[i]);
                }
                norm_val = sum;
            }
            else if (ord == std::numeric_limits<float>::infinity()) {
                float max_val = 0.0f;
                for (int i = 0; i < n; ++i) {
                    max_val = std::max(max_val, std::abs(data[i]));
                }
                norm_val = max_val;
            }
            else {
                double sum = 0.0;
                for (int i = 0; i < n; ++i) {
                    sum += std::pow(std::abs(data[i]), ord);
                }
                norm_val = static_cast<float>(std::pow(sum, 1.0 / ord));
            }
        }
        else {
            int m = static_cast<int>(x.shape().dim(0));
            int n = static_cast<int>(x.shape().dim(1));
            const float* data = x.data<float>();

            char norm_char[2];
            if (ord == 1.0) {
                norm_char[0] = 'I';
                norm_char[1] = '\0';
                std::vector<float> work(m);
                norm_val = slange_(norm_char, &m, &n, const_cast<float*>(data), &m, work.data());
            }
            else if (ord == std::numeric_limits<float>::infinity()) {
                norm_char[0] = '1';
                norm_char[1] = '\0';
                norm_val = slange_(norm_char, &m, &n, const_cast<float*>(data), &m, nullptr);
            }
            else {
                norm_char[0] = 'F';
                norm_char[1] = '\0';
                norm_val = slange_(norm_char, &m, &n, const_cast<float*>(data), &m, nullptr);
            }
        }

        Array result(Shape({}), DType::F32, x.place());
        *result.data<float>() = norm_val;
        return result;
    }

    static OpArgs norm_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        double ord = std::any_cast<double>(args[1]);
        if (x.dtype() == DType::F64) {
            return { norm_f64_impl(x, ord) };
        }
        else {
            return { norm_f32_impl(x, ord) };
        }
    }

    REGISTER_KERNEL(norm, CPU, F64, norm_kernel);
    REGISTER_KERNEL(norm, CPU, F32, norm_kernel);

    // ============================================================================
    // matrix_rank
    // ============================================================================

    static Array matrix_rank_f64_impl(const Array& x, double tol) {
        int m = static_cast<int>(x.shape().dim(0));
        int n = static_cast<int>(x.shape().dim(1));
        int min_mn = std::min(m, n);

        std::vector<double> work(x.numel());
        std::memcpy(work.data(), x.data<double>(), x.numel() * sizeof(double));

        std::vector<double> S(min_mn);
        std::vector<double> U(m * m);
        std::vector<double> VT(n * n);
        std::vector<double> work_svd(10 * min_mn);
        int lwork = static_cast<int>(work_svd.size());
        int info;

        char jobu = 'N';
        char jobvt = 'N';
        dgesvd_(&jobu, &jobvt, &m, &n, work.data(), &m,
            S.data(), U.data(), &m, VT.data(), &n,
            work_svd.data(), &lwork, &info);
        check_lapack_info(info, "matrix_rank(dgesvd)");

        double max_s = S[0];
        double actual_tol;
        if (tol < 0) {
            actual_tol = max_s * std::max(m, n) * std::numeric_limits<double>::epsilon();
        }
        else {
            actual_tol = tol;
        }

        int rank = 0;
        for (int i = 0; i < min_mn; ++i) {
            if (S[i] > actual_tol) {
                ++rank;
            }
        }

        Array result(Shape({}), DType::I64, x.place());
        *result.data<int64_t>() = rank;
        return result;
    }

    static Array matrix_rank_f32_impl(const Array& x, double tol) {
        int m = static_cast<int>(x.shape().dim(0));
        int n = static_cast<int>(x.shape().dim(1));
        int min_mn = std::min(m, n);

        std::vector<float> work(x.numel());
        std::memcpy(work.data(), x.data<float>(), x.numel() * sizeof(float));

        std::vector<float> S(min_mn);
        std::vector<float> U(m * m);
        std::vector<float> VT(n * n);
        std::vector<float> work_svd(10 * min_mn);
        int lwork = static_cast<int>(work_svd.size());
        int info;

        char jobu = 'N';
        char jobvt = 'N';
        sgesvd_(&jobu, &jobvt, &m, &n, work.data(), &m,
            S.data(), U.data(), &m, VT.data(), &n,
            work_svd.data(), &lwork, &info);
        check_lapack_info(info, "matrix_rank(sgesvd)");

        float max_s = S[0];
        float actual_tol;
        if (tol < 0) {
            actual_tol = max_s * static_cast<float>(std::max(m, n)) * std::numeric_limits<float>::epsilon();
        }
        else {
            actual_tol = static_cast<float>(tol);
        }

        int rank = 0;
        for (int i = 0; i < min_mn; ++i) {
            if (S[i] > actual_tol) {
                ++rank;
            }
        }

        Array result(Shape({}), DType::I64, x.place());
        *result.data<int64_t>() = rank;
        return result;
    }

    static OpArgs matrix_rank_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        double tol = std::any_cast<double>(args[1]);
        if (x.dtype() == DType::F64) {
            return { matrix_rank_f64_impl(x, tol) };
        }
        else {
            return { matrix_rank_f32_impl(x, tol) };
        }
    }

    REGISTER_KERNEL(matrix_rank, CPU, F64, matrix_rank_kernel);
    REGISTER_KERNEL(matrix_rank, CPU, F32, matrix_rank_kernel);

    // ============================================================================
    // trace
    // ============================================================================

    static Array trace_f64_impl(const Array& x) {
        int n = static_cast<int>(x.shape().dim(0));
        const double* data = x.data<double>();
        double trace_val = 0;
        for (int i = 0; i < n; ++i) {
            trace_val += data[i * n + i];
        }
        Array result(Shape({}), DType::F64, x.place());
        *result.data<double>() = trace_val;
        return result;
    }

    static Array trace_f32_impl(const Array& x) {
        int n = static_cast<int>(x.shape().dim(0));
        const float* data = x.data<float>();
        float trace_val = 0;
        for (int i = 0; i < n; ++i) {
            trace_val += data[i * n + i];
        }
        Array result(Shape({}), DType::F32, x.place());
        *result.data<float>() = trace_val;
        return result;
    }

    static OpArgs trace_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        if (x.dtype() == DType::F64) {
            return { trace_f64_impl(x) };
        }
        else {
            return { trace_f32_impl(x) };
        }
    }

    REGISTER_KERNEL(trace, CPU, F64, trace_kernel);
    REGISTER_KERNEL(trace, CPU, F32, trace_kernel);

    // ============================================================================
    // inv
    // ============================================================================

    static Array inv_f64_impl(const Array& x) {
        int n = static_cast<int>(x.shape().dim(0));
        std::vector<double> work(x.numel());
        std::memcpy(work.data(), x.data<double>(), x.numel() * sizeof(double));

        std::vector<int> ipiv(n);
        int info;

        dgetrf_(&n, &n, work.data(), &n, ipiv.data(), &info);
        check_lapack_info(info, "inv(dgetrf)");

        std::vector<double> work_inv(4 * n);
        int lwork = static_cast<int>(work_inv.size());
        dgetri_(&n, work.data(), &n, ipiv.data(), work_inv.data(), &lwork, &info);
        check_lapack_info(info, "inv(dgetri)");

        Array result(x.shape(), DType::F64, x.place());
        std::memcpy(result.data<double>(), work.data(), x.numel() * sizeof(double));
        return result;
    }

    static Array inv_f32_impl(const Array& x) {
        int n = static_cast<int>(x.shape().dim(0));
        std::vector<float> work(x.numel());
        std::memcpy(work.data(), x.data<float>(), x.numel() * sizeof(float));

        std::vector<int> ipiv(n);
        int info;

        sgetrf_(&n, &n, work.data(), &n, ipiv.data(), &info);
        check_lapack_info(info, "inv(sgetrf)");

        std::vector<float> work_inv(4 * n);
        int lwork = static_cast<int>(work_inv.size());
        sgetri_(&n, work.data(), &n, ipiv.data(), work_inv.data(), &lwork, &info);
        check_lapack_info(info, "inv(sgetri)");

        Array result(x.shape(), DType::F32, x.place());
        std::memcpy(result.data<float>(), work.data(), x.numel() * sizeof(float));
        return result;
    }

    static OpArgs inv_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        if (x.dtype() == DType::F64) {
            return { inv_f64_impl(x) };
        }
        else {
            return { inv_f32_impl(x) };
        }
    }

    REGISTER_KERNEL(inv, CPU, F64, inv_kernel);
    REGISTER_KERNEL(inv, CPU, F32, inv_kernel);

    // ============================================================================
    // pinv
    // ============================================================================

    static Array pinv_f64_impl(const Array& x, double rcond) {
        int m = static_cast<int>(x.shape().dim(0));
        int n = static_cast<int>(x.shape().dim(1));
        int min_mn = std::min(m, n);

        // Convert input to column-major
        std::vector<double> A_work = rowmajor_to_colmajor(x.data<double>(), m, n);

        // Query workspace size
        int lwork = -1;
        int info;
        char jobu = 'A';     // Compute all columns of U (m x m)
        char jobvt = 'A';    // Compute all rows of VT (n x n)

        std::vector<double> u(m * m);
        std::vector<double> s(min_mn);
        std::vector<double> vt(n * n);
        std::vector<double> work(1);

        dgesvd_(&jobu, &jobvt, &m, &n, A_work.data(), &m,
            s.data(), u.data(), &m, vt.data(), &n,
            work.data(), &lwork, &info);

        lwork = static_cast<int>(work[0]);
        work.resize(lwork);

        dgesvd_(&jobu, &jobvt, &m, &n, A_work.data(), &m,
            s.data(), u.data(), &m, vt.data(), &n,
            work.data(), &lwork, &info);
        check_lapack_info(info, "pinv(dgesvd)");

        // Compute tolerance
        double max_s = s[0];
        double actual_rcond;
        if (rcond < 0) {
            actual_rcond = max_s * std::max(m, n) * std::numeric_limits<double>::epsilon();
        }
        else {
            actual_rcond = rcond;
        }

        // Compute S_pinv
        std::vector<double> S_pinv(min_mn, 0.0);
        for (int i = 0; i < min_mn; ++i) {
            if (s[i] > actual_rcond) {
                S_pinv[i] = 1.0 / s[i];
            }
        }

        // Compute pinv = V * diag(S_pinv) * U^T
        // u is column-major: u[row + col * m] = U(row, col)
        // vt is column-major: vt[row + col * n] = VT(row, col) = V(col, row)
        Array result(Shape({ n, m }), DType::F64, x.place());
        double* result_data = result.data<double>();

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                double sum = 0.0;
                for (int k = 0; k < min_mn; ++k) {
                    // V(i, k) = vt[k + i * n]  (since vt = V^T, so vt[k][i] = V[i][k])
                    double v_ik = vt[k + i * n];
                    // S_pinv[k]
                    double s_inv = S_pinv[k];
                    // U^T(k, j) = U(j, k) = u[j + k * m]
                    double uT_kj = u[j + k * m];
                    sum += v_ik * s_inv * uT_kj;
                }
                result_data[i * m + j] = sum;
            }
        }

        return result;
    }

    static Array pinv_f32_impl(const Array& x, double rcond) {
        int m = static_cast<int>(x.shape().dim(0));
        int n = static_cast<int>(x.shape().dim(1));
        int min_mn = std::min(m, n);

        std::vector<float> A_work = rowmajor_to_colmajor(x.data<float>(), m, n);

        int lwork = -1;
        int info;
        char jobu = 'A';
        char jobvt = 'A';

        std::vector<float> u(m * m);
        std::vector<float> s(min_mn);
        std::vector<float> vt(n * n);
        std::vector<float> work(1);

        sgesvd_(&jobu, &jobvt, &m, &n, A_work.data(), &m,
            s.data(), u.data(), &m, vt.data(), &n,
            work.data(), &lwork, &info);

        lwork = static_cast<int>(work[0]);
        work.resize(lwork);

        sgesvd_(&jobu, &jobvt, &m, &n, A_work.data(), &m,
            s.data(), u.data(), &m, vt.data(), &n,
            work.data(), &lwork, &info);
        check_lapack_info(info, "pinv(sgesvd)");

        float max_s = s[0];
        float actual_rcond;
        if (rcond < 0) {
            actual_rcond = max_s * static_cast<float>(std::max(m, n)) * std::numeric_limits<float>::epsilon();
        }
        else {
            actual_rcond = static_cast<float>(rcond);
        }

        std::vector<float> S_pinv(min_mn, 0.0f);
        for (int i = 0; i < min_mn; ++i) {
            if (s[i] > actual_rcond) {
                S_pinv[i] = 1.0f / s[i];
            }
        }

        Array result(Shape({ n, m }), DType::F32, x.place());
        float* result_data = result.data<float>();

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < min_mn; ++k) {
                    float v_ik = vt[k + i * n];
                    float s_inv = S_pinv[k];
                    float uT_kj = u[j + k * m];
                    sum += v_ik * s_inv * uT_kj;
                }
                result_data[i * m + j] = sum;
            }
        }

        return result;
    }

    static OpArgs pinv_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        double rcond = std::any_cast<double>(args[1]);
        if (x.dtype() == DType::F64) {
            return { pinv_f64_impl(x, rcond) };
        }
        else {
            return { pinv_f32_impl(x, rcond) };
        }
    }

    REGISTER_KERNEL(pinv, CPU, F64, pinv_kernel);
    REGISTER_KERNEL(pinv, CPU, F32, pinv_kernel);

    // ============================================================================
    // matrix_power
    // ============================================================================

    static Array matrix_power_f64_impl(const Array& x, int n) {
        int size = static_cast<int>(x.shape().dim(0));
        std::vector<double> result_data(size * size, 0.0);
        for (int i = 0; i < size; ++i) {
            result_data[i * size + i] = 1.0;
        }

        std::vector<double> base_data(x.numel());
        std::memcpy(base_data.data(), x.data<double>(), x.numel() * sizeof(double));

        std::vector<double> temp_data(size * size);

        int exp = n;
        while (exp > 0) {
            if (exp & 1) {
                for (int i = 0; i < size; ++i) {
                    for (int j = 0; j < size; ++j) {
                        double sum = 0;
                        for (int k = 0; k < size; ++k) {
                            sum += result_data[i * size + k] * base_data[k * size + j];
                        }
                        temp_data[i * size + j] = sum;
                    }
                }
                result_data.swap(temp_data);
            }
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    double sum = 0;
                    for (int k = 0; k < size; ++k) {
                        sum += base_data[i * size + k] * base_data[k * size + j];
                    }
                    temp_data[i * size + j] = sum;
                }
            }
            base_data.swap(temp_data);
            exp >>= 1;
        }

        Array result(x.shape(), DType::F64, x.place());
        std::memcpy(result.data<double>(), result_data.data(), x.numel() * sizeof(double));
        return result;
    }

    static Array matrix_power_f32_impl(const Array& x, int n) {
        int size = static_cast<int>(x.shape().dim(0));
        std::vector<float> result_data(size * size, 0.0f);
        for (int i = 0; i < size; ++i) {
            result_data[i * size + i] = 1.0f;
        }

        std::vector<float> base_data(x.numel());
        std::memcpy(base_data.data(), x.data<float>(), x.numel() * sizeof(float));

        std::vector<float> temp_data(size * size);

        int exp = n;
        while (exp > 0) {
            if (exp & 1) {
                for (int i = 0; i < size; ++i) {
                    for (int j = 0; j < size; ++j) {
                        float sum = 0;
                        for (int k = 0; k < size; ++k) {
                            sum += result_data[i * size + k] * base_data[k * size + j];
                        }
                        temp_data[i * size + j] = sum;
                    }
                }
                result_data.swap(temp_data);
            }
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    float sum = 0;
                    for (int k = 0; k < size; ++k) {
                        sum += base_data[i * size + k] * base_data[k * size + j];
                    }
                    temp_data[i * size + j] = sum;
                }
            }
            base_data.swap(temp_data);
            exp >>= 1;
        }

        Array result(x.shape(), DType::F32, x.place());
        std::memcpy(result.data<float>(), result_data.data(), x.numel() * sizeof(float));
        return result;
    }

    static OpArgs matrix_power_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        int n = std::any_cast<int>(args[1]);
        if (x.dtype() == DType::F64) {
            return { matrix_power_f64_impl(x, n) };
        }
        else {
            return { matrix_power_f32_impl(x, n) };
        }
    }

    REGISTER_KERNEL(matrix_power, CPU, F64, matrix_power_kernel);
    REGISTER_KERNEL(matrix_power, CPU, F32, matrix_power_kernel);

    // ============================================================================
    // dot
    // ============================================================================

    static Array dot_f64_impl(const Array& a, const Array& b) {
        int n = static_cast<int>(a.numel());
        const double* a_data = a.data<double>();
        const double* b_data = b.data<double>();

        int incx = 1;
        int incy = 1;
        double result_val = ddot_(&n, const_cast<double*>(a_data), &incx,
            const_cast<double*>(b_data), &incy);

        Array result(Shape({}), DType::F64, a.place());
        *result.data<double>() = result_val;
        return result;
    }

    static Array dot_f32_impl(const Array& a, const Array& b) {
        int n = static_cast<int>(a.numel());
        const float* a_data = a.data<float>();
        const float* b_data = b.data<float>();

        int incx = 1;
        int incy = 1;
        float result_val = sdot_(&n, const_cast<float*>(a_data), &incx,
            const_cast<float*>(b_data), &incy);

        Array result(Shape({}), DType::F32, a.place());
        *result.data<float>() = result_val;
        return result;
    }

    static OpArgs dot_kernel(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        if (a.dtype() == DType::F64) {
            return { dot_f64_impl(a, b) };
        }
        else {
            return { dot_f32_impl(a, b) };
        }
    }

    REGISTER_KERNEL(dot, CPU, F64, dot_kernel);
    REGISTER_KERNEL(dot, CPU, F32, dot_kernel);

    // ============================================================================
    // outer
    // ============================================================================

    static Array outer_f64_impl(const Array& a, const Array& b) {
        int m = static_cast<int>(a.numel());
        int n = static_cast<int>(b.numel());
        const double* a_data = a.data<double>();
        const double* b_data = b.data<double>();

        Array result(Shape({ m, n }), DType::F64, a.place());
        double* result_data = result.data<double>();

#pragma omp parallel for collapse(2)
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                result_data[i * n + j] = a_data[i] * b_data[j];
            }
        }

        return result;
    }

    static Array outer_f32_impl(const Array& a, const Array& b) {
        int m = static_cast<int>(a.numel());
        int n = static_cast<int>(b.numel());
        const float* a_data = a.data<float>();
        const float* b_data = b.data<float>();

        Array result(Shape({ m, n }), DType::F32, a.place());
        float* result_data = result.data<float>();

#pragma omp parallel for collapse(2)
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                result_data[i * n + j] = a_data[i] * b_data[j];
            }
        }

        return result;
    }

    static OpArgs outer_kernel(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        if (a.dtype() == DType::F64) {
            return { outer_f64_impl(a, b) };
        }
        else {
            return { outer_f32_impl(a, b) };
        }
    }

    REGISTER_KERNEL(outer, CPU, F64, outer_kernel);
    REGISTER_KERNEL(outer, CPU, F32, outer_kernel);

    // ============================================================================
    // lu
    // ============================================================================

    static std::pair<Array, Array> lu_f64_impl(const Array& x, bool pivot) {
        int n = static_cast<int>(x.shape().dim(0));

        // Convert input to column-major
        std::vector<double> A_colmajor = rowmajor_to_colmajor(x.data<double>(), n, n);

        std::vector<int> ipiv;
        lu_decompose(A_colmajor, n, ipiv);

        // Return LU in column-major (user will unpack via lu_unpack)
        Array LU(Shape({ n, n }), DType::F64, x.place());
        std::memcpy(LU.data<double>(), A_colmajor.data(), n * n * sizeof(double));

        // Convert ipiv to 1-based for LAPACK compatibility
        Array pivots(Shape({ n }), DType::I32, x.place());
        int32_t* pivots_data = pivots.data<int32_t>();
        for (int i = 0; i < n; ++i) {
            pivots_data[i] = ipiv[i] + 1;
        }

        return { LU, pivots };
    }

    static std::pair<Array, Array> lu_f32_impl(const Array& x, bool pivot) {
        int n = static_cast<int>(x.shape().dim(0));

        // Convert input to column-major
        std::vector<float> A_colmajor = rowmajor_to_colmajor(x.data<float>(), n, n);

        std::vector<int> ipiv;
        lu_decompose(A_colmajor, n, ipiv);

        Array LU(Shape({ n, n }), DType::F32, x.place());
        std::memcpy(LU.data<float>(), A_colmajor.data(), n * n * sizeof(float));

        Array pivots(Shape({ n }), DType::I32, x.place());
        int32_t* pivots_data = pivots.data<int32_t>();
        for (int i = 0; i < n; ++i) {
            pivots_data[i] = ipiv[i] + 1;
        }

        return { LU, pivots };
    }


    static OpArgs lu_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        bool pivot = std::any_cast<bool>(args[1]);
        if (x.dtype() == DType::F64) {
            auto [LU, pivots] = lu_f64_impl(x, pivot);
            return { LU, pivots };
        }
        else {
            auto [LU, pivots] = lu_f32_impl(x, pivot);
            return { LU, pivots };
        }
    }

    REGISTER_KERNEL(lu, CPU, F64, lu_kernel);
    REGISTER_KERNEL(lu, CPU, F32, lu_kernel);

    // ============================================================================
    // lu_unpack
    // ============================================================================

    static std::tuple<Array, Array, Array> lu_unpack_f64_impl(const Array& LU, const Array& pivots) {
        int n = static_cast<int>(LU.shape().dim(0));
        const double* lu = LU.data<double>();      // column-major
        const int32_t* piv = pivots.data<int32_t>();

        // Permutation matrix P (row-major for output)
        Array P(Shape({ n, n }), DType::F64, LU.place());
        double* p = P.data<double>();
        std::fill(p, p + n * n, 0.0);
        for (int i = 0; i < n; ++i) {
            p[i * n + (piv[i] - 1)] = 1.0;
        }

        // L and U (row-major for output)
        Array L(Shape({ n, n }), DType::F64, LU.place());
        Array U(Shape({ n, n }), DType::F64, LU.place());
        double* l = L.data<double>();
        double* u = U.data<double>();

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double val = lu[i + j * n];  // column-major access
                if (i == j) {
                    l[i * n + j] = 1.0;
                    u[i * n + j] = val;
                }
                else if (i > j) {
                    l[i * n + j] = val;
                    u[i * n + j] = 0.0;
                }
                else {
                    l[i * n + j] = 0.0;
                    u[i * n + j] = val;
                }
            }
        }

        return { P, L, U };
    }

    static std::tuple<Array, Array, Array> lu_unpack_f32_impl(const Array& LU, const Array& pivots) {
        int n = static_cast<int>(LU.shape().dim(0));
        const float* lu = LU.data<float>();      // column-major
        const int32_t* piv = pivots.data<int32_t>();

        // Permutation matrix P (row-major for output)
        Array P(Shape({ n, n }), DType::F32, LU.place());
        float* p = P.data<float>();
        std::fill(p, p + n * n, 0.0f);
        for (int i = 0; i < n; ++i) {
            p[i * n + (piv[i] - 1)] = 1.0f;
        }

        // L and U (row-major for output)
        Array L(Shape({ n, n }), DType::F32, LU.place());
        Array U(Shape({ n, n }), DType::F32, LU.place());
        float* l = L.data<float>();
        float* u = U.data<float>();

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                float val = lu[i + j * n];  // column-major access
                if (i == j) {
                    l[i * n + j] = 1.0f;
                    u[i * n + j] = val;
                }
                else if (i > j) {
                    l[i * n + j] = val;
                    u[i * n + j] = 0.0f;
                }
                else {
                    l[i * n + j] = 0.0f;
                    u[i * n + j] = val;
                }
            }
        }

        return { P, L, U };
    }

    static OpArgs lu_unpack_kernel(const OpArgs& args) {
        const Array& LU = std::any_cast<const Array&>(args[0]);
        const Array& pivots = std::any_cast<const Array&>(args[1]);
        if (LU.dtype() == DType::F64) {
            auto [P, L, U] = lu_unpack_f64_impl(LU, pivots);
            return { P, L, U };
        }
        else {
            auto [P, L, U] = lu_unpack_f32_impl(LU, pivots);
            return { P, L, U };
        }
    }

    REGISTER_KERNEL(lu_unpack, CPU, F64, lu_unpack_kernel);
    REGISTER_KERNEL(lu_unpack, CPU, F32, lu_unpack_kernel);

    // ============================================================================
    // qr
    // ============================================================================

    static std::pair<Array, Array> qr_f64_impl(const Array& x, const std::string& mode) {
        int m = static_cast<int>(x.shape().dim(0));
        int n = static_cast<int>(x.shape().dim(1));
        int k = std::min(m, n);

        // Convert input to column-major
        std::vector<double> A_colmajor = rowmajor_to_colmajor(x.data<double>(), m, n);

        std::vector<double> tau(k);
        std::vector<double> work_qr(1);
        int lwork = -1;
        int info;

        // Query optimal workspace
        dgeqrf_(&m, &n, A_colmajor.data(), &m, tau.data(), work_qr.data(), &lwork, &info);
        lwork = static_cast<int>(work_qr[0]);
        work_qr.resize(lwork);

        // Compute QR decomposition
        dgeqrf_(&m, &n, A_colmajor.data(), &m, tau.data(), work_qr.data(), &lwork, &info);
        check_lapack_info(info, "qr(dgeqrf)");

        // Compute Q (column-major)
        std::vector<double> Q_colmajor(m * m, 0.0);
        for (int i = 0; i < m; ++i) {
            Q_colmajor[i + i * m] = 1.0;
        }
        // Copy the Householder vectors from A_colmajor
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < k && j < i; ++j) {
                Q_colmajor[i + j * m] = A_colmajor[i + j * m];
            }
        }

        lwork = -1;
        dorgqr_(&m, &m, &k, Q_colmajor.data(), &m, tau.data(), work_qr.data(), &lwork, &info);
        lwork = static_cast<int>(work_qr[0]);
        work_qr.resize(lwork);
        dorgqr_(&m, &m, &k, Q_colmajor.data(), &m, tau.data(), work_qr.data(), &lwork, &info);
        check_lapack_info(info, "qr(dorgqr)");

        // Extract R (upper triangular part of A_colmajor)
        std::vector<double> R_colmajor(k * n, 0.0);
        for (int i = 0; i < k; ++i) {
            for (int j = i; j < n; ++j) {
                R_colmajor[i + j * k] = A_colmajor[i + j * m];
            }
        }

        // Convert results to row-major
        Array Q, R;
        if (mode == "complete") {
            Q = Array(Shape({ m, m }), DType::F64, x.place());
            colmajor_to_rowmajor(Q.data<double>(), Q_colmajor.data(), m, m);
        }
        else {
            // Reduced: first k columns of Q
            std::vector<double> Q_reduced_colmajor(m * k);
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < k; ++j) {
                    Q_reduced_colmajor[i + j * m] = Q_colmajor[i + j * m];
                }
            }
            Q = Array(Shape({ m, k }), DType::F64, x.place());
            colmajor_to_rowmajor(Q.data<double>(), Q_reduced_colmajor.data(), m, k);
        }

        R = Array(Shape({ k, n }), DType::F64, x.place());
        colmajor_to_rowmajor(R.data<double>(), R_colmajor.data(), k, n);

        return { Q, R };
    }

    static std::pair<Array, Array> qr_f32_impl(const Array& x, const std::string& mode) {
        int m = static_cast<int>(x.shape().dim(0));
        int n = static_cast<int>(x.shape().dim(1));
        int k = std::min(m, n);

        // Convert input to column-major
        std::vector<float> A_colmajor = rowmajor_to_colmajor(x.data<float>(), m, n);

        std::vector<float> tau(k);
        std::vector<float> work_qr(1);
        int lwork = -1;
        int info;

        // Query optimal workspace
        sgeqrf_(&m, &n, A_colmajor.data(), &m, tau.data(), work_qr.data(), &lwork, &info);
        lwork = static_cast<int>(work_qr[0]);
        work_qr.resize(lwork);

        // Compute QR decomposition
        sgeqrf_(&m, &n, A_colmajor.data(), &m, tau.data(), work_qr.data(), &lwork, &info);
        check_lapack_info(info, "qr(sgeqrf)");

        // Compute Q (column-major)
        std::vector<float> Q_colmajor(m * m, 0.0f);
        for (int i = 0; i < m; ++i) {
            Q_colmajor[i + i * m] = 1.0f;
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < k && j < i; ++j) {
                Q_colmajor[i + j * m] = A_colmajor[i + j * m];
            }
        }

        lwork = -1;
        sorgqr_(&m, &m, &k, Q_colmajor.data(), &m, tau.data(), work_qr.data(), &lwork, &info);
        lwork = static_cast<int>(work_qr[0]);
        work_qr.resize(lwork);
        sorgqr_(&m, &m, &k, Q_colmajor.data(), &m, tau.data(), work_qr.data(), &lwork, &info);
        check_lapack_info(info, "qr(sorgqr)");

        // Extract R
        std::vector<float> R_colmajor(k * n, 0.0f);
        for (int i = 0; i < k; ++i) {
            for (int j = i; j < n; ++j) {
                R_colmajor[i + j * k] = A_colmajor[i + j * m];
            }
        }

        // Convert results to row-major
        Array Q, R;
        if (mode == "complete") {
            Q = Array(Shape({ m, m }), DType::F32, x.place());
            colmajor_to_rowmajor(Q.data<float>(), Q_colmajor.data(), m, m);
        }
        else {
            std::vector<float> Q_reduced_colmajor(m * k);
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < k; ++j) {
                    Q_reduced_colmajor[i + j * m] = Q_colmajor[i + j * m];
                }
            }
            Q = Array(Shape({ m, k }), DType::F32, x.place());
            colmajor_to_rowmajor(Q.data<float>(), Q_reduced_colmajor.data(), m, k);
        }

        R = Array(Shape({ k, n }), DType::F32, x.place());
        colmajor_to_rowmajor(R.data<float>(), R_colmajor.data(), k, n);

        return { Q, R };
    }

    static OpArgs qr_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        std::string mode = std::any_cast<std::string>(args[1]);
        if (x.dtype() == DType::F64) {
            auto [Q, R] = qr_f64_impl(x, mode);
            return { Q, R };
        }
        else {
            auto [Q, R] = qr_f32_impl(x, mode);
            return { Q, R };
        }
    }

    REGISTER_KERNEL(qr, CPU, F64, qr_kernel);
    REGISTER_KERNEL(qr, CPU, F32, qr_kernel);

    // ============================================================================
    // lq
    // ============================================================================

    static std::pair<Array, Array> lq_f64_impl(const Array& x, const std::string& mode) {
        int m = static_cast<int>(x.shape().dim(0));
        int n = static_cast<int>(x.shape().dim(1));
        int k = std::min(m, n);

        // Convert input to column-major (LAPACK expects column-major)
        std::vector<double> A_colmajor = rowmajor_to_colmajor(x.data<double>(), m, n);

        std::vector<double> tau(k);
        std::vector<double> work(1);
        int lwork = -1;
        int info;

        // Query optimal workspace
        dgelqf_(&m, &n, A_colmajor.data(), &m, tau.data(), work.data(), &lwork, &info);
        lwork = static_cast<int>(work[0]);
        work.resize(lwork);

        // Compute LQ decomposition
        dgelqf_(&m, &n, A_colmajor.data(), &m, tau.data(), work.data(), &lwork, &info);
        check_lapack_info(info, "lq(dgelqf)");

        // Compute Q (column-major)
        std::vector<double> Q_colmajor(n * n, 0.0);
        // Copy the Householder vectors
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (j >= i) {
                    Q_colmajor[i + j * n] = A_colmajor[i + j * m];
                }
                else {
                    Q_colmajor[i + j * n] = A_colmajor[i + j * m];
                }
            }
        }
        // Set diagonal to 1 for rows beyond m
        for (int i = m; i < n; ++i) {
            Q_colmajor[i + i * n] = 1.0;
        }

        lwork = -1;
        dorglq_(&n, &n, &k, Q_colmajor.data(), &n, tau.data(), work.data(), &lwork, &info);
        lwork = static_cast<int>(work[0]);
        work.resize(lwork);
        dorglq_(&n, &n, &k, Q_colmajor.data(), &n, tau.data(), work.data(), &lwork, &info);
        check_lapack_info(info, "lq(dorglq)");

        // Extract L (lower triangular part of A_colmajor)
        std::vector<double> L_colmajor(m * k, 0.0);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j <= i && j < k; ++j) {
                L_colmajor[i + j * m] = A_colmajor[i + j * m];
            }
        }

        // Convert results to row-major
        Array L, Q;
        if (mode == "complete") {
            Q = Array(Shape({ n, n }), DType::F64, x.place());
            colmajor_to_rowmajor(Q.data<double>(), Q_colmajor.data(), n, n);
        }
        else {
            // Reduced: first m columns of Q
            std::vector<double> Q_reduced_colmajor(n * m);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < m; ++j) {
                    Q_reduced_colmajor[i + j * n] = Q_colmajor[i + j * n];
                }
            }
            Q = Array(Shape({ n, m }), DType::F64, x.place());
            colmajor_to_rowmajor(Q.data<double>(), Q_reduced_colmajor.data(), n, m);
        }

        L = Array(Shape({ m, k }), DType::F64, x.place());
        colmajor_to_rowmajor(L.data<double>(), L_colmajor.data(), m, k);

        return { L, Q };
    }

    static std::pair<Array, Array> lq_f32_impl(const Array& x, const std::string& mode) {
        int m = static_cast<int>(x.shape().dim(0));
        int n = static_cast<int>(x.shape().dim(1));
        int k = std::min(m, n);

        std::vector<float> A_colmajor = rowmajor_to_colmajor(x.data<float>(), m, n);

        std::vector<float> tau(k);
        std::vector<float> work(1);
        int lwork = -1;
        int info;

        sgelqf_(&m, &n, A_colmajor.data(), &m, tau.data(), work.data(), &lwork, &info);
        lwork = static_cast<int>(work[0]);
        work.resize(lwork);

        sgelqf_(&m, &n, A_colmajor.data(), &m, tau.data(), work.data(), &lwork, &info);
        check_lapack_info(info, "lq(sgelqf)");

        std::vector<float> Q_colmajor(n * n, 0.0f);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                Q_colmajor[i + j * n] = A_colmajor[i + j * m];
            }
        }
        for (int i = m; i < n; ++i) {
            Q_colmajor[i + i * n] = 1.0f;
        }

        lwork = -1;
        sorglq_(&n, &n, &k, Q_colmajor.data(), &n, tau.data(), work.data(), &lwork, &info);
        lwork = static_cast<int>(work[0]);
        work.resize(lwork);
        sorglq_(&n, &n, &k, Q_colmajor.data(), &n, tau.data(), work.data(), &lwork, &info);
        check_lapack_info(info, "lq(sorglq)");

        std::vector<float> L_colmajor(m * k, 0.0f);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j <= i && j < k; ++j) {
                L_colmajor[i + j * m] = A_colmajor[i + j * m];
            }
        }

        Array L, Q;
        if (mode == "complete") {
            Q = Array(Shape({ n, n }), DType::F32, x.place());
            colmajor_to_rowmajor(Q.data<float>(), Q_colmajor.data(), n, n);
        }
        else {
            std::vector<float> Q_reduced_colmajor(n * m);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < m; ++j) {
                    Q_reduced_colmajor[i + j * n] = Q_colmajor[i + j * n];
                }
            }
            Q = Array(Shape({ n, m }), DType::F32, x.place());
            colmajor_to_rowmajor(Q.data<float>(), Q_reduced_colmajor.data(), n, m);
        }

        L = Array(Shape({ m, k }), DType::F32, x.place());
        colmajor_to_rowmajor(L.data<float>(), L_colmajor.data(), m, k);

        return { L, Q };
    }

    static OpArgs lq_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        std::string mode = std::any_cast<std::string>(args[1]);
        if (x.dtype() == DType::F64) {
            auto [L, Q] = lq_f64_impl(x, mode);
            return { L, Q };
        }
        else {
            auto [L, Q] = lq_f32_impl(x, mode);
            return { L, Q };
        }
    }

    REGISTER_KERNEL(lq, CPU, F64, lq_kernel);
    REGISTER_KERNEL(lq, CPU, F32, lq_kernel);

    // ============================================================================
    // cholesky
    // ============================================================================
    static Array cholesky_f64_impl(const Array& x, bool lower) {
        int n = static_cast<int>(x.shape().dim(0));

        // 1. 将输入转换为列主序（如果还不是）
        std::vector<double> a_colmajor(n * n);
        const double* a_rowmajor = x.data<double>();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                a_colmajor[i + j * n] = a_rowmajor[i * n + j];
            }
        }

        // 2. 调用 LAPACK（列主序）
        int info;
        char uplo[] = { lower ? 'L' : 'U', '\0' };
        dpotrf_(uplo, &n, a_colmajor.data(), &n, &info);
        check_lapack_info(info, "cholesky(dpotrf)");

        // 3. 将结果转换回行主序
        Array result(x.shape(), DType::F64, x.place());
        double* result_data = result.data<double>();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (lower && i >= j) {
                    result_data[i * n + j] = a_colmajor[i + j * n];
                }
                else if (!lower && i <= j) {
                    result_data[i * n + j] = a_colmajor[i + j * n];
                }
                else {
                    result_data[i * n + j] = 0.0;
                }
            }
        }

        return result;
    }

    static Array cholesky_f32_impl(const Array& x, bool lower) {
        int n = static_cast<int>(x.shape().dim(0));
        Array result(x.shape(), DType::F32, x.place());
        std::memcpy(result.data<float>(), x.data<float>(), x.numel() * sizeof(float));

        char uplo[2] = { lower ? 'L' : 'U', '\0' };
        int info;

        spotrf_(uplo, &n, result.data<float>(), &n, &info);
        check_lapack_info(info, "cholesky(spotrf)");

        float* data = result.data<float>();
        if (lower) {
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    data[i * n + j] = 0.0f;
                }
            }
        }
        else {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < i; ++j) {
                    data[i * n + j] = 0.0f;
                }
            }
        }

        return result;
    }

    static OpArgs cholesky_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        bool lower = std::any_cast<bool>(args[1]);
        if (x.dtype() == DType::F64) {
            return { cholesky_f64_impl(x, lower) };
        }
        else {
            return { cholesky_f32_impl(x, lower) };
        }
    }

    REGISTER_KERNEL(cholesky, CPU, F64, cholesky_kernel);
    REGISTER_KERNEL(cholesky, CPU, F32, cholesky_kernel);

    // ============================================================================
    // cholesky_solve
    // ============================================================================

    static Array cholesky_solve_f64_impl(const Array& A, const Array& B, bool lower) {
        int n = static_cast<int>(A.shape().dim(0));
        int nrhs = (B.shape().ndim() == 1) ? 1 : static_cast<int>(B.shape().dim(1));
        bool B_is_1d = (B.shape().ndim() == 1);

        std::vector<double> A_work(A.numel());
        std::memcpy(A_work.data(), A.data<double>(), A.numel() * sizeof(double));

        Array B_work_arr = B.copy();
        if (B_is_1d) {
            B_work_arr = B_work_arr.reshape(Shape({ n, 1 }));
        }
        double* B_work = B_work_arr.data<double>();

        char uplo = lower ? 'L' : 'U';
        int info;
        int ldb = n;

        dpotrs_(&uplo, &n, &nrhs, A_work.data(), &n, B_work, &ldb, &info);
        check_lapack_info(info, "cholesky_solve(dpotrs)");

        if (B_is_1d) {
            B_work_arr = B_work_arr.reshape(Shape({ n }));
        }
        return B_work_arr;
    }

    static Array cholesky_solve_f32_impl(const Array& A, const Array& B, bool lower) {
        int n = static_cast<int>(A.shape().dim(0));
        int nrhs = (B.shape().ndim() == 1) ? 1 : static_cast<int>(B.shape().dim(1));
        bool B_is_1d = (B.shape().ndim() == 1);

        std::vector<float> A_work(A.numel());
        std::memcpy(A_work.data(), A.data<float>(), A.numel() * sizeof(float));

        Array B_work_arr = B.copy();
        if (B_is_1d) {
            B_work_arr = B_work_arr.reshape(Shape({ n, 1 }));
        }
        float* B_work = B_work_arr.data<float>();

        char uplo = lower ? 'L' : 'U';
        int info;
        int ldb = n;

        spotrs_(&uplo, &n, &nrhs, A_work.data(), &n, B_work, &ldb, &info);
        check_lapack_info(info, "cholesky_solve(spotrs)");

        if (B_is_1d) {
            B_work_arr = B_work_arr.reshape(Shape({ n }));
        }
        return B_work_arr;
    }

    static OpArgs cholesky_solve_kernel(const OpArgs& args) {
        const Array& A = std::any_cast<const Array&>(args[0]);
        const Array& B = std::any_cast<const Array&>(args[1]);
        bool lower = std::any_cast<bool>(args[2]);
        if (A.dtype() == DType::F64) {
            return { cholesky_solve_f64_impl(A, B, lower) };
        }
        else {
            return { cholesky_solve_f32_impl(A, B, lower) };
        }
    }

    REGISTER_KERNEL(cholesky_solve, CPU, F64, cholesky_solve_kernel);
    REGISTER_KERNEL(cholesky_solve, CPU, F32, cholesky_solve_kernel);

    // ============================================================================
    // svd
    // ============================================================================

    static std::tuple<Array, Array, Array> svd_f64_impl(const Array& x, bool full_matrices) {
        int m = static_cast<int>(x.shape().dim(0));
        int n = static_cast<int>(x.shape().dim(1));
        int min_mn = std::min(m, n);

        // Convert input to column-major
        std::vector<double> A_work = rowmajor_to_colmajor(x.data<double>(), m, n);

        int lwork = -1;
        int info;

        char jobu = full_matrices ? 'A' : 'S';
        char jobvt = full_matrices ? 'A' : 'S';

        int u_rows = m;
        int u_cols = full_matrices ? m : min_mn;
        int vt_rows = full_matrices ? n : min_mn;
        int vt_cols = n;

        std::vector<double> u(u_rows * u_cols);
        std::vector<double> s(min_mn);
        std::vector<double> vt(vt_rows * vt_cols);
        std::vector<double> work(1);

        // Query optimal workspace
        dgesvd_(&jobu, &jobvt, &m, &n, A_work.data(), &m,
            s.data(), u.data(), &u_rows, vt.data(), &vt_rows,
            work.data(), &lwork, &info);

        lwork = static_cast<int>(work[0]);
        work.resize(lwork);

        // Compute SVD
        dgesvd_(&jobu, &jobvt, &m, &n, A_work.data(), &m,
            s.data(), u.data(), &u_rows, vt.data(), &vt_rows,
            work.data(), &lwork, &info);
        check_lapack_info(info, "svd(dgesvd)");

        // Convert U to row-major (note: u is already in column-major format)
        Array U_arr(Shape({ u_rows, u_cols }), DType::F64, x.place());
        colmajor_to_rowmajor(U_arr.data<double>(), u.data(), u_rows, u_cols);

        // Convert S (singular values) - 1D array, no conversion needed
        Array S_arr(Shape({ min_mn }), DType::F64, x.place());
        std::memcpy(S_arr.data<double>(), s.data(), min_mn * sizeof(double));

        // Convert VT to row-major
        Array VT_arr(Shape({ vt_rows, vt_cols }), DType::F64, x.place());
        colmajor_to_rowmajor(VT_arr.data<double>(), vt.data(), vt_rows, vt_cols);

        return { U_arr, S_arr, VT_arr };
    }

    static std::tuple<Array, Array, Array> svd_f32_impl(const Array& x, bool full_matrices) {
        int m = static_cast<int>(x.shape().dim(0));
        int n = static_cast<int>(x.shape().dim(1));
        int min_mn = std::min(m, n);

        // Convert input to column-major
        std::vector<float> A_work = rowmajor_to_colmajor(x.data<float>(), m, n);

        int lwork = -1;
        int info;

        char jobu = full_matrices ? 'A' : 'S';
        char jobvt = full_matrices ? 'A' : 'S';

        int u_rows = m;
        int u_cols = full_matrices ? m : min_mn;
        int vt_rows = full_matrices ? n : min_mn;
        int vt_cols = n;

        std::vector<float> u(u_rows * u_cols);
        std::vector<float> s(min_mn);
        std::vector<float> vt(vt_rows * vt_cols);
        std::vector<float> work(1);

        // Query optimal workspace
        sgesvd_(&jobu, &jobvt, &m, &n, A_work.data(), &m,
            s.data(), u.data(), &u_rows, vt.data(), &vt_rows,
            work.data(), &lwork, &info);

        lwork = static_cast<int>(work[0]);
        work.resize(lwork);

        // Compute SVD
        sgesvd_(&jobu, &jobvt, &m, &n, A_work.data(), &m,
            s.data(), u.data(), &u_rows, vt.data(), &vt_rows,
            work.data(), &lwork, &info);
        check_lapack_info(info, "svd(sgesvd)");

        // Convert U to row-major
        Array U_arr(Shape({ u_rows, u_cols }), DType::F32, x.place());
        colmajor_to_rowmajor(U_arr.data<float>(), u.data(), u_rows, u_cols);

        // S - no conversion needed
        Array S_arr(Shape({ min_mn }), DType::F32, x.place());
        std::memcpy(S_arr.data<float>(), s.data(), min_mn * sizeof(float));

        // Convert VT to row-major
        Array VT_arr(Shape({ vt_rows, vt_cols }), DType::F32, x.place());
        colmajor_to_rowmajor(VT_arr.data<float>(), vt.data(), vt_rows, vt_cols);

        return { U_arr, S_arr, VT_arr };
    }

    static OpArgs svd_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        bool full_matrices = std::any_cast<bool>(args[1]);
        if (x.dtype() == DType::F64) {
            auto [U, S, VT] = svd_f64_impl(x, full_matrices);
            return { U, S, VT };
        }
        else {
            auto [U, S, VT] = svd_f32_impl(x, full_matrices);
            return { U, S, VT };
        }
    }

    REGISTER_KERNEL(svd, CPU, F64, svd_kernel);
    REGISTER_KERNEL(svd, CPU, F32, svd_kernel);

    // ============================================================================
    // svdvals
    // ============================================================================

    static Array svdvals_f64_impl(const Array& x) {
        int m = static_cast<int>(x.shape().dim(0));
        int n = static_cast<int>(x.shape().dim(1));
        int min_mn = std::min(m, n);

        // Convert input to column-major
        std::vector<double> A_work = rowmajor_to_colmajor(x.data<double>(), m, n);

        int lwork = -1;
        int info;
        char jobu = 'N';
        char jobvt = 'N';

        std::vector<double> s(min_mn);
        std::vector<double> work(1);

        dgesvd_(&jobu, &jobvt, &m, &n, A_work.data(), &m,
            s.data(), nullptr, &m, nullptr, &n,
            work.data(), &lwork, &info);

        lwork = static_cast<int>(work[0]);
        work.resize(lwork);

        dgesvd_(&jobu, &jobvt, &m, &n, A_work.data(), &m,
            s.data(), nullptr, &m, nullptr, &n,
            work.data(), &lwork, &info);
        check_lapack_info(info, "svdvals(dgesvd)");

        Array result(Shape({ min_mn }), DType::F64, x.place());
        std::memcpy(result.data<double>(), s.data(), min_mn * sizeof(double));
        return result;
    }

    static Array svdvals_f32_impl(const Array& x) {
        int m = static_cast<int>(x.shape().dim(0));
        int n = static_cast<int>(x.shape().dim(1));
        int min_mn = std::min(m, n);

        // Convert input to column-major
        std::vector<float> A_work = rowmajor_to_colmajor(x.data<float>(), m, n);

        int lwork = -1;
        int info;
        char jobu = 'N';
        char jobvt = 'N';

        std::vector<float> s(min_mn);
        std::vector<float> work(1);

        sgesvd_(&jobu, &jobvt, &m, &n, A_work.data(), &m,
            s.data(), nullptr, &m, nullptr, &n,
            work.data(), &lwork, &info);

        lwork = static_cast<int>(work[0]);
        work.resize(lwork);

        sgesvd_(&jobu, &jobvt, &m, &n, A_work.data(), &m,
            s.data(), nullptr, &m, nullptr, &n,
            work.data(), &lwork, &info);
        check_lapack_info(info, "svdvals(sgesvd)");

        Array result(Shape({ min_mn }), DType::F32, x.place());
        std::memcpy(result.data<float>(), s.data(), min_mn * sizeof(float));
        return result;
    }


    static OpArgs svdvals_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        if (x.dtype() == DType::F64) {
            return { svdvals_f64_impl(x) };
        }
        else {
            return { svdvals_f32_impl(x) };
        }
    }

    REGISTER_KERNEL(svdvals, CPU, F64, svdvals_kernel);
    REGISTER_KERNEL(svdvals, CPU, F32, svdvals_kernel);

    // ============================================================================
    // eig
    // ============================================================================

    static std::pair<Array, Array> eig_f64_impl(const Array& x) {
        int n = static_cast<int>(x.shape().dim(0));
        std::vector<double> A_work(x.numel());
        std::memcpy(A_work.data(), x.data<double>(), x.numel() * sizeof(double));

        std::vector<double> wr(n), wi(n);
        std::vector<double> vr(n * n);
        std::vector<double> work_eig(1);
        int lwork = -1;
        int info;
        char jobvl[2] = { 'N', ' ' };
        char jobvr[2] = { 'V', ' ' };

        dgeev_(jobvl, jobvr, &n, A_work.data(), &n,
            wr.data(), wi.data(), nullptr, &n, vr.data(), &n,
            work_eig.data(), &lwork, &info);

        lwork = static_cast<int>(work_eig[0]);
        work_eig.resize(lwork);

        dgeev_(jobvl, jobvr, &n, A_work.data(), &n,
            wr.data(), wi.data(), nullptr, &n, vr.data(), &n,
            work_eig.data(), &lwork, &info);
        check_lapack_info(info, "eig(dgeev)");

        Array eigvals_arr(Shape({ n, 2 }), DType::F64, x.place());
        double* eigvals_data = eigvals_arr.data<double>();
        for (int i = 0; i < n; ++i) {
            eigvals_data[i * 2] = wr[i];
            eigvals_data[i * 2 + 1] = wi[i];
        }

        Array eigvecs_arr(Shape({ n, n, 2 }), DType::F64, x.place());
        double* eigvecs_data = eigvecs_arr.data<double>();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                eigvecs_data[((i * n + j) * 2)] = vr[j * n + i];
                eigvecs_data[((i * n + j) * 2) + 1] = (wi[j] != 0) ? vr[j * n + i + n] : 0.0;
            }
        }

        return { eigvals_arr, eigvecs_arr };
    }

    static std::pair<Array, Array> eig_f32_impl(const Array& x) {
        int n = static_cast<int>(x.shape().dim(0));
        std::vector<float> A_work(x.numel());
        std::memcpy(A_work.data(), x.data<float>(), x.numel() * sizeof(float));

        std::vector<float> wr(n), wi(n);
        std::vector<float> vr(n * n);
        std::vector<float> work_eig(1);
        int lwork = -1;
        int info;
        char jobvl = 'N';
        char jobvr = 'V';

        sgeev_(&jobvl, &jobvr, &n, A_work.data(), &n,
            wr.data(), wi.data(), nullptr, &n, vr.data(), &n,
            work_eig.data(), &lwork, &info);

        lwork = static_cast<int>(work_eig[0]);
        work_eig.resize(lwork);

        sgeev_(&jobvl, &jobvr, &n, A_work.data(), &n,
            wr.data(), wi.data(), nullptr, &n, vr.data(), &n,
            work_eig.data(), &lwork, &info);
        check_lapack_info(info, "eig(sgeev)");

        Array eigvals_arr(Shape({ n, 2 }), DType::F32, x.place());
        float* eigvals_data = eigvals_arr.data<float>();
        for (int i = 0; i < n; ++i) {
            eigvals_data[i * 2] = wr[i];
            eigvals_data[i * 2 + 1] = wi[i];
        }

        Array eigvecs_arr(Shape({ n, n, 2 }), DType::F32, x.place());
        float* eigvecs_data = eigvecs_arr.data<float>();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                eigvecs_data[((i * n + j) * 2)] = vr[j * n + i];
                eigvecs_data[((i * n + j) * 2) + 1] = (wi[j] != 0) ? vr[j * n + i + n] : 0.0f;
            }
        }

        return { eigvals_arr, eigvecs_arr };
    }

    static OpArgs eig_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        if (x.dtype() == DType::F64) {
            auto [vals, vecs] = eig_f64_impl(x);
            return { vals, vecs };
        }
        else {
            auto [vals, vecs] = eig_f32_impl(x);
            return { vals, vecs };
        }
    }

    REGISTER_KERNEL(eig, CPU, F64, eig_kernel);
    REGISTER_KERNEL(eig, CPU, F32, eig_kernel);

    // ============================================================================
    // eigvals
    // ============================================================================

    static Array eigvals_f64_impl(const Array& x) {
        int n = static_cast<int>(x.shape().dim(0));
        std::vector<double> A_work(x.numel());
        std::memcpy(A_work.data(), x.data<double>(), x.numel() * sizeof(double));

        std::vector<double> wr(n), wi(n);
        std::vector<double> work_eig(1);
        int lwork = -1;
        int info;
        char jobvl[2] = { 'N', ' ' };
        char jobvr[2] = { 'V', ' ' };

        dgeev_(jobvl, jobvr, &n, A_work.data(), &n,
            wr.data(), wi.data(), nullptr, &n, nullptr, &n,
            work_eig.data(), &lwork, &info);

        lwork = static_cast<int>(work_eig[0]);
        work_eig.resize(lwork);

        dgeev_(jobvl, jobvr, &n, A_work.data(), &n,
            wr.data(), wi.data(), nullptr, &n, nullptr, &n,
            work_eig.data(), &lwork, &info);
        check_lapack_info(info, "eigvals(dgeev)");

        Array result(Shape({ n, 2 }), DType::F64, x.place());
        double* result_data = result.data<double>();
        for (int i = 0; i < n; ++i) {
            result_data[i * 2] = wr[i];
            result_data[i * 2 + 1] = wi[i];
        }
        return result;
    }

    static Array eigvals_f32_impl(const Array& x) {
        int n = static_cast<int>(x.shape().dim(0));
        std::vector<float> A_work(x.numel());
        std::memcpy(A_work.data(), x.data<float>(), x.numel() * sizeof(float));

        std::vector<float> wr(n), wi(n);
        std::vector<float> work_eig(1);
        int lwork = -1;
        int info;
        char jobvl = 'N';
        char jobvr = 'N';

        sgeev_(&jobvl, &jobvr, &n, A_work.data(), &n,
            wr.data(), wi.data(), nullptr, &n, nullptr, &n,
            work_eig.data(), &lwork, &info);

        lwork = static_cast<int>(work_eig[0]);
        work_eig.resize(lwork);

        sgeev_(&jobvl, &jobvr, &n, A_work.data(), &n,
            wr.data(), wi.data(), nullptr, &n, nullptr, &n,
            work_eig.data(), &lwork, &info);
        check_lapack_info(info, "eigvals(sgeev)");

        Array result(Shape({ n, 2 }), DType::F32, x.place());
        float* result_data = result.data<float>();
        for (int i = 0; i < n; ++i) {
            result_data[i * 2] = wr[i];
            result_data[i * 2 + 1] = wi[i];
        }
        return result;
    }

    static OpArgs eigvals_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        if (x.dtype() == DType::F64) {
            return { eigvals_f64_impl(x) };
        }
        else {
            return { eigvals_f32_impl(x) };
        }
    }

    REGISTER_KERNEL(eigvals, CPU, F64, eigvals_kernel);
    REGISTER_KERNEL(eigvals, CPU, F32, eigvals_kernel);

    // ============================================================================
    // eigh
    // ============================================================================

    static std::pair<Array, Array> eigh_f64_impl(const Array& x, const std::string& uplo) {
        int n = static_cast<int>(x.shape().dim(0));
        Array A(x.shape(), DType::F64, x.place());
        std::memcpy(A.data<double>(), x.data<double>(), x.numel() * sizeof(double));

        std::vector<double> w(n);

        // 正确：以 null 结尾
        char jobz[] = { 'V', '\0' };
        char uplo_char[] = { (uplo == "L") ? 'L' : 'U', '\0' };

        int lwork = -1;
        int info;

        std::vector<double> work_query(1);
        dsyev_(jobz, uplo_char, &n, A.data<double>(), &n,
            w.data(), work_query.data(), &lwork, &info);

        if (info != 0) {
            lwork = 3 * n - 1;
        }
        else {
            lwork = static_cast<int>(work_query[0]);
        }

        std::vector<double> work(lwork);

        // 重置 A
        std::memcpy(A.data<double>(), x.data<double>(), x.numel() * sizeof(double));

        dsyev_(jobz, uplo_char, &n, A.data<double>(), &n,
            w.data(), work.data(), &lwork, &info);
        check_lapack_info(info, "eigh(dsyev)");

        Array eigenvalues(Shape({ n }), DType::F64, x.place());
        std::memcpy(eigenvalues.data<double>(), w.data(), n * sizeof(double));

        Array eigenvectors = A;

        return { eigenvalues, eigenvectors };
    }

    static std::pair<Array, Array> eigh_f32_impl(const Array& x, const std::string& uplo) {
        int n = static_cast<int>(x.shape().dim(0));
        Array A(x.shape(), DType::F32, x.place());
        std::memcpy(A.data<float>(), x.data<float>(), x.numel() * sizeof(float));

        std::vector<float> w(n);

        char jobz[] = { 'V', '\0' };
        char uplo_char[] = { (uplo == "L") ? 'L' : 'U', '\0' };

        int lwork = -1;
        int info;

        std::vector<float> work_query(1);
        ssyev_(jobz, uplo_char, &n, A.data<float>(), &n,
            w.data(), work_query.data(), &lwork, &info);

        if (info != 0) {
            lwork = 3 * n - 1;
        }
        else {
            lwork = static_cast<int>(work_query[0]);
        }

        std::vector<float> work(lwork);

        std::memcpy(A.data<float>(), x.data<float>(), x.numel() * sizeof(float));

        ssyev_(jobz, uplo_char, &n, A.data<float>(), &n,
            w.data(), work.data(), &lwork, &info);
        check_lapack_info(info, "eigh(ssyev)");

        Array eigenvalues(Shape({ n }), DType::F32, x.place());
        std::memcpy(eigenvalues.data<float>(), w.data(), n * sizeof(float));

        Array eigenvectors = A;

        return { eigenvalues, eigenvectors };
    }

    static OpArgs eigh_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        std::string uplo = std::any_cast<std::string>(args[1]);
        if (x.dtype() == DType::F64) {
            auto [vals, vecs] = eigh_f64_impl(x, uplo);
            return { vals, vecs };
        }
        else {
            auto [vals, vecs] = eigh_f32_impl(x, uplo);
            return { vals, vecs };
        }
    }

    REGISTER_KERNEL(eigh, CPU, F64, eigh_kernel);
    REGISTER_KERNEL(eigh, CPU, F32, eigh_kernel);

    // ============================================================================
    // eigvalsh
    // ============================================================================

    static Array eigvalsh_f64_impl(const Array& x, const std::string& uplo) {
        int n = static_cast<int>(x.shape().dim(0));
        Array A(x.shape(), DType::F64, x.place());
        std::memcpy(A.data<double>(), x.data<double>(), x.numel() * sizeof(double));

        std::vector<double> w(n);

        char jobz[] = { 'N', '\0' };
        char uplo_char[] = { (uplo == "L") ? 'L' : 'U', '\0' };

        int lwork = -1;
        int info;

        std::vector<double> work_query(1);
        dsyev_(jobz, uplo_char, &n, A.data<double>(), &n,
            w.data(), work_query.data(), &lwork, &info);

        if (info != 0) {
            lwork = 3 * n - 1;
        }
        else {
            lwork = static_cast<int>(work_query[0]);
        }

        std::vector<double> work(lwork);

        dsyev_(jobz, uplo_char, &n, A.data<double>(), &n,
            w.data(), work.data(), &lwork, &info);
        check_lapack_info(info, "eigvalsh(dsyev)");

        Array eigenvalues(Shape({ n }), DType::F64, x.place());
        std::memcpy(eigenvalues.data<double>(), w.data(), n * sizeof(double));
        return eigenvalues;
    }

    static Array eigvalsh_f32_impl(const Array& x, const std::string& uplo) {
        int n = static_cast<int>(x.shape().dim(0));
        Array A(x.shape(), DType::F32, x.place());
        std::memcpy(A.data<float>(), x.data<float>(), x.numel() * sizeof(float));

        std::vector<float> w(n);

        char jobz[] = { 'N', '\0' };
        char uplo_char[] = { (uplo == "L") ? 'L' : 'U', '\0' };

        int lwork = -1;
        int info;

        std::vector<float> work_query(1);
        ssyev_(jobz, uplo_char, &n, A.data<float>(), &n,
            w.data(), work_query.data(), &lwork, &info);

        if (info != 0) {
            lwork = 3 * n - 1;
        }
        else {
            lwork = static_cast<int>(work_query[0]);
        }

        std::vector<float> work(lwork);

        ssyev_(jobz, uplo_char, &n, A.data<float>(), &n,
            w.data(), work.data(), &lwork, &info);
        check_lapack_info(info, "eigvalsh(ssyev)");

        Array eigenvalues(Shape({ n }), DType::F32, x.place());
        std::memcpy(eigenvalues.data<float>(), w.data(), n * sizeof(float));
        return eigenvalues;
    }

    static OpArgs eigvalsh_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        std::string uplo = std::any_cast<std::string>(args[1]);
        if (x.dtype() == DType::F64) {
            return { eigvalsh_f64_impl(x, uplo) };
        }
        else {
            return { eigvalsh_f32_impl(x, uplo) };
        }
    }

    REGISTER_KERNEL(eigvalsh, CPU, F64, eigvalsh_kernel);
    REGISTER_KERNEL(eigvalsh, CPU, F32, eigvalsh_kernel);

    // ============================================================================
    // solve
    // ============================================================================

    static Array solve_f64_impl(const Array& A, const Array& B) {
        int n = static_cast<int>(A.shape().dim(0));
        int nrhs = (B.shape().ndim() == 1) ? 1 : static_cast<int>(B.shape().dim(1));
        bool B_is_1d = (B.shape().ndim() == 1);

        // Convert A to column-major
        std::vector<double> A_work = rowmajor_to_colmajor(A.data<double>(), n, n);

        // Convert B to column-major and store in a single buffer dgesv will overwrite
        std::vector<double> B_work;
        if (B_is_1d) {
            B_work = rowmajor_to_colmajor(B.data<double>(), n, 1);
        }
        else {
            int b_rows = B.shape().dim(0);
            int b_cols = B.shape().dim(1);
            B_work = rowmajor_to_colmajor(B.data<double>(), b_rows, b_cols);
        }

        std::vector<int> ipiv(n);
        int info;

        dgesv_(&n, &nrhs, A_work.data(), &n, ipiv.data(), B_work.data(), &n, &info);
        check_lapack_info(info, "solve(dgesv)");

        // Convert result back to row-major
        Array result;
        if (B_is_1d) {
            result = Array(Shape({ n }), DType::F64, A.place());
            colmajor_to_rowmajor(result.data<double>(), B_work.data(), n, 1);
        }
        else {
            result = Array(Shape({ n, nrhs }), DType::F64, A.place());
            double* result_data = result.data<double>();
            // B_work is column-major: index = row + col * n
            // Convert to row-major: result[row][col] = B_work[row + col * n]
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < nrhs; ++j) {
                    result_data[i * nrhs + j] = B_work[i + j * n];
                }
            }
        }

        return result;
    }

    static Array solve_f32_impl(const Array& A, const Array& B) {
        int n = static_cast<int>(A.shape().dim(0));
        int nrhs = (B.shape().ndim() == 1) ? 1 : static_cast<int>(B.shape().dim(1));
        bool B_is_1d = (B.shape().ndim() == 1);

        // Convert A to column-major
        std::vector<float> A_work = rowmajor_to_colmajor(A.data<float>(), n, n);

        // Convert B to column-major
        std::vector<float> B_work;
        if (B_is_1d) {
            B_work = rowmajor_to_colmajor(B.data<float>(), n, 1);
        }
        else {
            int b_rows = B.shape().dim(0);
            int b_cols = B.shape().dim(1);
            B_work = rowmajor_to_colmajor(B.data<float>(), b_rows, b_cols);
        }

        std::vector<int> ipiv(n);
        int info;

        sgesv_(&n, &nrhs, A_work.data(), &n, ipiv.data(), B_work.data(), &n, &info);
        check_lapack_info(info, "solve(sgesv)");

        // Convert result back to row-major
        Array result;
        if (B_is_1d) {
            result = Array(Shape({ n }), DType::F32, A.place());
            colmajor_to_rowmajor(result.data<float>(), B_work.data(), n, 1);
        }
        else {
            result = Array(Shape({ n, nrhs }), DType::F32, A.place());
            float* result_data = result.data<float>();
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < nrhs; ++j) {
                    result_data[i * nrhs + j] = B_work[i + j * n];
                }
            }
        }

        return result;
    }

    static OpArgs solve_kernel(const OpArgs& args) {
        const Array& A = std::any_cast<const Array&>(args[0]);
        const Array& B = std::any_cast<const Array&>(args[1]);
        if (A.dtype() == DType::F64) {
            return { solve_f64_impl(A, B) };
        }
        else {
            return { solve_f32_impl(A, B) };
        }
    }

    REGISTER_KERNEL(solve, CPU, F64, solve_kernel);
    REGISTER_KERNEL(solve, CPU, F32, solve_kernel);

    // ============================================================================
    // lstsq
    // ============================================================================

    static Array lstsq_f64_impl(const Array& A, const Array& B, double rcond) {
        int m = static_cast<int>(A.shape().dim(0));
        int n = static_cast<int>(A.shape().dim(1));
        int nrhs = (B.shape().ndim() == 1) ? 1 : static_cast<int>(B.shape().dim(1));
        bool B_is_1d = (B.shape().ndim() == 1);

        // Convert A to column-major
        std::vector<double> A_work = rowmajor_to_colmajor(A.data<double>(), m, n);

        // Convert B to column-major
        std::vector<double> B_work;
        int b_rows = std::max(m, n);
        int b_cols = nrhs;

        if (B_is_1d) {
            B_work.resize(b_rows);
            const double* b_data = B.data<double>();
            for (int i = 0; i < m; ++i) {
                B_work[i] = b_data[i];
            }
            for (int i = m; i < b_rows; ++i) {
                B_work[i] = 0.0;
            }
        }
        else {
            B_work.resize(b_rows * b_cols, 0.0);
            const double* b_data = B.data<double>();
            // B is row-major, convert to column-major with padding
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < nrhs; ++j) {
                    B_work[i + j * b_rows] = b_data[i * nrhs + j];
                }
            }
        }

        // Query workspace size
        int lwork = -1;
        int info;
        char trans = 'N';
        std::vector<double> work(1);

        dgels_(&trans, &m, &n, &nrhs, A_work.data(), &m,
            B_work.data(), &b_rows, work.data(), &lwork, &info);

        lwork = static_cast<int>(work[0]);
        work.resize(lwork);

        // Solve least squares
        dgels_(&trans, &m, &n, &nrhs, A_work.data(), &m,
            B_work.data(), &b_rows, work.data(), &lwork, &info);
        check_lapack_info(info, "lstsq(dgels)");

        // Extract solution X (first n rows of B_work)
        Array result;
        if (B_is_1d) {
            result = Array(Shape({ n }), DType::F64, A.place());
            double* result_data = result.data<double>();
            for (int i = 0; i < n; ++i) {
                result_data[i] = B_work[i];  // Column-major, first column, first n rows
            }
        }
        else {
            result = Array(Shape({ n, nrhs }), DType::F64, A.place());
            double* result_data = result.data<double>();
            // Convert from column-major to row-major
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < nrhs; ++j) {
                    result_data[i * nrhs + j] = B_work[i + j * b_rows];
                }
            }
        }

        return result;
    }

    static Array lstsq_f32_impl(const Array& A, const Array& B, double rcond) {
        int m = static_cast<int>(A.shape().dim(0));
        int n = static_cast<int>(A.shape().dim(1));
        int nrhs = (B.shape().ndim() == 1) ? 1 : static_cast<int>(B.shape().dim(1));
        bool B_is_1d = (B.shape().ndim() == 1);

        // Convert A to column-major
        std::vector<float> A_work = rowmajor_to_colmajor(A.data<float>(), m, n);

        // Convert B to column-major
        std::vector<float> B_work;
        int b_rows = std::max(m, n);
        int b_cols = nrhs;

        if (B_is_1d) {
            B_work.resize(b_rows);
            const float* b_data = B.data<float>();
            for (int i = 0; i < m; ++i) {
                B_work[i] = b_data[i];
            }
            for (int i = m; i < b_rows; ++i) {
                B_work[i] = 0.0f;
            }
        }
        else {
            B_work.resize(b_rows * b_cols, 0.0f);
            const float* b_data = B.data<float>();
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < nrhs; ++j) {
                    B_work[i + j * b_rows] = b_data[i * nrhs + j];
                }
            }
        }

        // Query workspace size
        int lwork = -1;
        int info;
        char trans = 'N';
        std::vector<float> work(1);

        sgels_(&trans, &m, &n, &nrhs, A_work.data(), &m,
            B_work.data(), &b_rows, work.data(), &lwork, &info);

        lwork = static_cast<int>(work[0]);
        work.resize(lwork);

        // Solve least squares
        sgels_(&trans, &m, &n, &nrhs, A_work.data(), &m,
            B_work.data(), &b_rows, work.data(), &lwork, &info);
        check_lapack_info(info, "lstsq(sgels)");

        // Extract solution X (first n rows of B_work)
        Array result;
        if (B_is_1d) {
            result = Array(Shape({ n }), DType::F32, A.place());
            float* result_data = result.data<float>();
            for (int i = 0; i < n; ++i) {
                result_data[i] = B_work[i];
            }
        }
        else {
            result = Array(Shape({ n, nrhs }), DType::F32, A.place());
            float* result_data = result.data<float>();
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < nrhs; ++j) {
                    result_data[i * nrhs + j] = B_work[i + j * b_rows];
                }
            }
        }

        return result;
    }

    static OpArgs lstsq_kernel(const OpArgs& args) {
        const Array& A = std::any_cast<const Array&>(args[0]);
        const Array& B = std::any_cast<const Array&>(args[1]);
        double rcond = std::any_cast<double>(args[2]);
        if (A.dtype() == DType::F64) {
            return { lstsq_f64_impl(A, B, rcond) };
        }
        else {
            return { lstsq_f32_impl(A, B, rcond) };
        }
    }

    REGISTER_KERNEL(lstsq, CPU, F64, lstsq_kernel);
    REGISTER_KERNEL(lstsq, CPU, F32, lstsq_kernel);

    // ============================================================================
    // solve_triangular
    // ============================================================================

    static Array solve_triangular_f64_impl(const Array& A, const Array& B,
        bool lower, bool unit_diag) {
        int n = static_cast<int>(A.shape().dim(0));
        int nrhs = (B.shape().ndim() == 1) ? 1 : static_cast<int>(B.shape().dim(1));
        bool B_is_1d = (B.shape().ndim() == 1);

        // Convert A to column-major
        std::vector<double> A_colmajor = rowmajor_to_colmajor(A.data<double>(), n, n);

        // Convert B to column-major
        std::vector<double> B_work;
        int b_rows = n;
        int b_cols = B_is_1d ? 1 : nrhs;

        if (B_is_1d) {
            B_work = rowmajor_to_colmajor(B.data<double>(), n, 1);
        }
        else {
            B_work = rowmajor_to_colmajor(B.data<double>(), n, nrhs);
        }

        char uplo = lower ? 'L' : 'U';
        char trans = 'N';
        char diag = unit_diag ? 'U' : 'N';
        int info;
        int ldb = n;

        dtrtrs_(&uplo, &trans, &diag, &n, &nrhs,
            A_colmajor.data(), &n, B_work.data(), &ldb, &info);
        check_lapack_info(info, "solve_triangular(dtrtrs)");

        // Convert result back to row-major
        Array result;
        if (B_is_1d) {
            result = Array(Shape({ n }), DType::F64, A.place());
            colmajor_to_rowmajor(result.data<double>(), B_work.data(), n, 1);
        }
        else {
            result = Array(Shape({ n, nrhs }), DType::F64, A.place());
            colmajor_to_rowmajor(result.data<double>(), B_work.data(), n, nrhs);
        }

        return result;
    }

    static Array solve_triangular_f32_impl(const Array& A, const Array& B,
        bool lower, bool unit_diag) {
        int n = static_cast<int>(A.shape().dim(0));
        int nrhs = (B.shape().ndim() == 1) ? 1 : static_cast<int>(B.shape().dim(1));
        bool B_is_1d = (B.shape().ndim() == 1);

        // Convert A to column-major
        std::vector<float> A_colmajor = rowmajor_to_colmajor(A.data<float>(), n, n);

        // Convert B to column-major
        std::vector<float> B_work;
        int b_rows = n;
        int b_cols = B_is_1d ? 1 : nrhs;

        if (B_is_1d) {
            B_work = rowmajor_to_colmajor(B.data<float>(), n, 1);
        }
        else {
            B_work = rowmajor_to_colmajor(B.data<float>(), n, nrhs);
        }

        char uplo = lower ? 'L' : 'U';
        char trans = 'N';
        char diag = unit_diag ? 'U' : 'N';
        int info;
        int ldb = n;

        strtrs_(&uplo, &trans, &diag, &n, &nrhs,
            A_colmajor.data(), &n, B_work.data(), &ldb, &info);
        check_lapack_info(info, "solve_triangular(strtrs)");

        // Convert result back to row-major
        Array result;
        if (B_is_1d) {
            result = Array(Shape({ n }), DType::F32, A.place());
            colmajor_to_rowmajor(result.data<float>(), B_work.data(), n, 1);
        }
        else {
            result = Array(Shape({ n, nrhs }), DType::F32, A.place());
            colmajor_to_rowmajor(result.data<float>(), B_work.data(), n, nrhs);
        }

        return result;
    }

    static OpArgs solve_triangular_kernel(const OpArgs& args) {
        const Array& A = std::any_cast<const Array&>(args[0]);
        const Array& B = std::any_cast<const Array&>(args[1]);
        bool lower = std::any_cast<bool>(args[2]);
        bool unit_diag = std::any_cast<bool>(args[3]);
        if (A.dtype() == DType::F64) {
            return { solve_triangular_f64_impl(A, B, lower, unit_diag) };
        }
        else {
            return { solve_triangular_f32_impl(A, B, lower, unit_diag) };
        }
    }

    REGISTER_KERNEL(solve_triangular, CPU, F64, solve_triangular_kernel);
    REGISTER_KERNEL(solve_triangular, CPU, F32, solve_triangular_kernel);
#endif INSIGHT_USE_OPENBLAS
    // ============================================================================
    // Module registration
    // ============================================================================

    REGISTER_MODULE(linalg, CPU);

} // namespace ins::cpu