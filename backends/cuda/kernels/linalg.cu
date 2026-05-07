// backends/cuda/kernels/linalg.cu
#include "insight/plugin/op_registry.h"
#include "insight/core/array.h"
#include "insight/core/exception.h"
#include "insight/core/launch_config.h"
#include "insight/ops/broadcast.h"
#include <cmath>
#include <cstring>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
#include "insight/io/print.h"
namespace ins::gpu {

    // ============================================================================
    // cuBLAS / cuSOLVER handle management
    // ============================================================================

    struct CublasHandle {
        cublasHandle_t handle = nullptr;
        cusolverDnHandle_t solver = nullptr;
        int device_id = -1;

        ~CublasHandle() {
            if (handle) cublasDestroy(handle);
            if (solver) cusolverDnDestroy(solver);
        }

        void ensure(int dev) {
            if (device_id != dev) {
                if (handle) cublasDestroy(handle);
                if (solver) cusolverDnDestroy(solver);
                cublasCreate(&handle);
                cusolverDnCreate(&solver);
                device_id = dev;
            }
        }
    };

    static thread_local CublasHandle tls_handle;

    // ============================================================================
    // CPU fallback helpers
    // ============================================================================

    template<typename Func>
    static Array cpu_fallback(const Array& x, Func&& func) {
        Array x_cpu = x.to(CPUPlace());
        Array result_cpu = func(x_cpu);
        return result_cpu.to(x.place());
    }

    template<typename Func>
    static Array cpu_fallback2(const Array& a, const Array& b, Func&& func) {
        Array a_cpu = a.to(CPUPlace());
        Array b_cpu = b.to(CPUPlace());
        Array result_cpu = func(a_cpu, b_cpu);
        return result_cpu.to(a.place());
    }

    template<typename Func>
    static std::vector<Array> cpu_fallback_multi(const Array& x, Func&& func) {
        Array x_cpu = x.to(CPUPlace());
        auto results_cpu = func(x_cpu);
        std::vector<Array> results_gpu;
        for (auto& r : results_cpu) {
            results_gpu.push_back(r.to(x.place()));
        }
        return results_gpu;
    }

    // ============================================================================
    // matmul (cuBLAS)
    // ============================================================================

    static OpArgs matmul_wrapper(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& a_ = std::any_cast<const Array&>(args[1]);
        const Array& b_ = std::any_cast<const Array&>(args[2]);

        // Ensure contiguous inputs for cuBLAS
        Array a = (a_.is_contiguous() ? a_ : a_.contiguous());
        Array b = (b_.is_contiguous() ? b_ : b_.contiguous());

        tls_handle.ensure(a.place().device_id());

        int ndim_a = a.shape().ndim();
        int ndim_b = b.shape().ndim();
        DType dtype = out.dtype();

        // Case 1: vector * vector -> scalar
        if (ndim_a == 1 && ndim_b == 1) {
            int n = static_cast<int>(a.numel());
            if (dtype == DType::F64) {
                double result_val;
                cublasDdot(tls_handle.handle, n, a.data<double>(), 1, b.data<double>(), 1, &result_val);
                cudaMemcpy(const_cast<double*>(out.data<double>()), &result_val, sizeof(double), cudaMemcpyHostToDevice);
            }
            else {
                float result_val;
                cublasSdot(tls_handle.handle, n, a.data<float>(), 1, b.data<float>(), 1, &result_val);
                cudaMemcpy(const_cast<float*>(out.data<float>()), &result_val, sizeof(float), cudaMemcpyHostToDevice);
            }
            return { out };
        }

        // Case 2: matrix * vector
        if (ndim_a == 2 && ndim_b == 1) {
            int m = static_cast<int>(a.shape().dim(0));
            int n = static_cast<int>(a.shape().dim(1));
            if (dtype == DType::F64) {
                const double alpha = 1.0, beta = 0.0;
                // 行主序的 A * x 等价于 cuBLAS 的 gemv(TRANSPOSE, n, m, A, n, x, 1, y, 1)
                // 因为 cuBLAS 默认列主序，所以传 TRANSPOSE + 交换 m,n 来对应行主序
                cublasDgemv(tls_handle.handle, CUBLAS_OP_T, n, m,
                    &alpha, a.data<double>(), n, b.data<double>(), 1, &beta,
                    const_cast<double*>(out.data<double>()), 1);
            }
            else {
                const float alpha = 1.0f, beta = 0.0f;
                cublasSgemv(tls_handle.handle, CUBLAS_OP_T, n, m,
                    &alpha, a.data<float>(), n, b.data<float>(), 1, &beta,
                    const_cast<float*>(out.data<float>()), 1);
            }
            return { out };
        }

        // Case 3: vector * matrix
        if (ndim_a == 1 && ndim_b == 2) {
            int m = static_cast<int>(b.shape().dim(0));
            int n = static_cast<int>(b.shape().dim(1));
            if (dtype == DType::F64) {
                const double alpha = 1.0, beta = 0.0;
                cublasDgemv(tls_handle.handle, CUBLAS_OP_T, m, n,
                    &alpha, b.data<double>(), m, a.data<double>(), 1, &beta,
                    const_cast<double*>(out.data<double>()), 1);
            }
            else {
                const float alpha = 1.0f, beta = 0.0f;
                cublasSgemv(tls_handle.handle, CUBLAS_OP_T, m, n,
                    &alpha, b.data<float>(), m, a.data<float>(), 1, &beta,
                    const_cast<float*>(out.data<float>()), 1);
            }
            return { out };
        }

        // Case 4: matrix * matrix (2D)
        if (ndim_a == 2 && ndim_b == 2) {
            int m = static_cast<int>(a.shape().dim(0));
            int k = static_cast<int>(a.shape().dim(1));
            int n = static_cast<int>(b.shape().dim(1));
            if (dtype == DType::F64) {
                const double alpha = 1.0, beta = 0.0;
                cublasDgemm(tls_handle.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k, &alpha, b.data<double>(), n, a.data<double>(), k, &beta,
                    const_cast<double*>(out.data<double>()), n);
            }
            else {
                const float alpha = 1.0f, beta = 0.0f;
                cublasSgemm(tls_handle.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k, &alpha, b.data<float>(), n, a.data<float>(), k, &beta,
                    const_cast<float*>(out.data<float>()), n);
            }
            return { out };
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

        int64_t a_stride = m * k;
        int64_t b_stride = k * n;
        int64_t c_stride = m * n;

        if (dtype == DType::F64) {
            const double* a_data = a_flat.data<double>();
            const double* b_data = b_flat.data<double>();
            double* c_data = const_cast<double*>(out.data<double>());
            const double alpha = 1.0, beta = 0.0;
            for (int64_t batch = 0; batch < batch_size; ++batch) {
                cublasDgemm(tls_handle.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k, &alpha,
                    b_data + batch * b_stride, n,
                    a_data + batch * a_stride, k, &beta,
                    c_data + batch * c_stride, n);
            }
        }
        else {
            const float* a_data = a_flat.data<float>();
            const float* b_data = b_flat.data<float>();
            float* c_data = const_cast<float*>(out.data<float>());
            const float alpha = 1.0f, beta = 0.0f;
            for (int64_t batch = 0; batch < batch_size; ++batch) {
                cublasSgemm(tls_handle.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k, &alpha,
                    b_data + batch * b_stride, n,
                    a_data + batch * a_stride, k, &beta,
                    c_data + batch * c_stride, n);
            }
        }
        return { out };
    }

    REGISTER_KERNEL(matmul, GPU, F32, matmul_wrapper);
    REGISTER_KERNEL(matmul, GPU, F64, matmul_wrapper);

    // ============================================================================
    // dot (cuBLAS)
    // ============================================================================

    static OpArgs dot_wrapper(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);

        tls_handle.ensure(a.place().device_id());
        int n = static_cast<int>(a.numel());
        Array result(Shape({}), a.dtype(), a.place());

        if (a.dtype() == DType::F64) {
            double result_val;
            cublasDdot(tls_handle.handle, n, a.data<double>(), 1, b.data<double>(), 1, &result_val);
            cudaMemcpy(result.data<double>(), &result_val, sizeof(double), cudaMemcpyHostToDevice);
        }
        else {
            float result_val;
            cublasSdot(tls_handle.handle, n, a.data<float>(), 1, b.data<float>(), 1, &result_val);
            cudaMemcpy(result.data<float>(), &result_val, sizeof(float), cudaMemcpyHostToDevice);
        }
        return { result };
    }

    REGISTER_KERNEL(dot, GPU, F32, dot_wrapper);
    REGISTER_KERNEL(dot, GPU, F64, dot_wrapper);

    // ============================================================================
    // trace (GPU kernel)
    // ============================================================================

    template<typename T>
    __global__ void trace_kernel(T* dst, const T* src, int n) {
        T sum = T(0);
        for (int i = 0; i < n; ++i) {
            sum += src[i * n + i];
        }
        *dst = sum;
    }

    static OpArgs trace_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        int n = static_cast<int>(x.shape().dim(0));
        Array result(Shape({}), x.dtype(), x.place());

        if (x.dtype() == DType::F64) {
            trace_kernel<double> << <1, 1 >> > (
                result.data<double>(), x.data<double>(), n);
        }
        else {
            trace_kernel<float> << <1, 1 >> > (
                result.data<float>(), x.data<float>(), n);
        }
        return { result };
    }

    REGISTER_KERNEL(trace, GPU, F32, trace_wrapper);
    REGISTER_KERNEL(trace, GPU, F64, trace_wrapper);

    // ============================================================================
    // outer (GPU kernel)
    // ============================================================================

    template<typename T>
    __global__ void outer_kernel(T* dst, const T* a, const T* b, int m, int n) {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < m && j < n) {
            dst[i * n + j] = a[i] * b[j];
        }
    }

    static OpArgs outer_wrapper(const OpArgs& args) {
        const Array& a = std::any_cast<const Array&>(args[0]);
        const Array& b = std::any_cast<const Array&>(args[1]);
        int m = static_cast<int>(a.numel());
        int n = static_cast<int>(b.numel());
        Array result(Shape({ m, n }), a.dtype(), a.place());

        dim3 threads(16, 16);
        dim3 blocks((n + 15) / 16, (m + 15) / 16);

        if (a.dtype() == DType::F64) {
            outer_kernel<double> << <blocks, threads >> > (
                result.data<double>(), a.data<double>(), b.data<double>(), m, n);
        }
        else {
            outer_kernel<float> << <blocks, threads >> > (
                result.data<float>(), a.data<float>(), b.data<float>(), m, n);
        }
        return { result };
    }

    REGISTER_KERNEL(outer, GPU, F32, outer_wrapper);
    REGISTER_KERNEL(outer, GPU, F64, outer_wrapper);

    // ============================================================================
    // norm
    // ============================================================================

    template<typename T>
    __global__ void norm_frobenius_kernel(T* dst, const T* src, int total) {
        T sum = T(0);
        for (int i = 0; i < total; ++i) {
            sum += src[i] * src[i];
        }
        *dst = sqrt(sum);
    }

    template<typename T>
    __global__ void norm_absmax_kernel(T* dst, const T* src, int total) {
        T max_val = T(0);
        for (int i = 0; i < total; ++i) {
            T val = src[i];
            if (val < T(0)) val = -val;
            if (val > max_val) max_val = val;
        }
        *dst = max_val;
    }

    static OpArgs norm_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        double ord = std::any_cast<double>(args[1]);

        tls_handle.ensure(x.place().device_id());
        int ndim = x.shape().ndim();

        // Vector 2-norm → cuBLAS nrm2
        if (ndim == 1 && ord == 2.0) {
            int n = static_cast<int>(x.numel());
            Array result(Shape({}), x.dtype(), x.place());
            if (x.dtype() == DType::F64) {
                double norm_val;
                cublasDnrm2(tls_handle.handle, n, x.data<double>(), 1, &norm_val);
                cudaMemcpy(result.data<double>(), &norm_val, sizeof(double), cudaMemcpyHostToDevice);
            }
            else {
                float norm_val;
                cublasSnrm2(tls_handle.handle, n, x.data<float>(), 1, &norm_val);
                cudaMemcpy(result.data<float>(), &norm_val, sizeof(float), cudaMemcpyHostToDevice);
            }
            return { result };
        }

        // Matrix Frobenius norm → GPU kernel
        if (ndim == 2 && ord == 2.0) {
            Array result(Shape({}), x.dtype(), x.place());
            int total = static_cast<int>(x.numel());
            if (x.dtype() == DType::F64) {
                norm_frobenius_kernel<double> << <1, 1 >> > (
                    result.data<double>(), x.data<double>(), total);
            }
            else {
                norm_frobenius_kernel<float> << <1, 1 >> > (
                    result.data<float>(), x.data<float>(), total);
            }
            return { result };
        }

        // Vector 1-norm → GPU kernel
        if (ndim == 1 && ord == 1.0) {
            Array result(Shape({}), x.dtype(), x.place());
            int n = static_cast<int>(x.numel());
            if (x.dtype() == DType::F64) {
                norm_frobenius_kernel<double> << <1, 1 >> > (
                    result.data<double>(), x.data<double>(), n);
            }
            else {
                norm_frobenius_kernel<float> << <1, 1 >> > (
                    result.data<float>(), x.data<float>(), n);
            }
            return { result };
        }

        // Vector inf-norm → GPU kernel
        if (ndim == 1 && ord == std::numeric_limits<double>::infinity()) {
            Array result(Shape({}), x.dtype(), x.place());
            int n = static_cast<int>(x.numel());
            if (x.dtype() == DType::F64) {
                norm_absmax_kernel<double> << <1, 1 >> > (
                    result.data<double>(), x.data<double>(), n);
            }
            else {
                norm_absmax_kernel<float> << <1, 1 >> > (
                    result.data<float>(), x.data<float>(), n);
            }
            return { result };
        }

        // Matrix 1-norm / inf-norm / other → CPU fallback
        return { cpu_fallback(x, [ord](const Array& cx) {
            OpArgs cpu_args = { cx, ord };
            return std::any_cast<Array>(ops()["norm"][DeviceKind::CPU][cx.dtype()](cpu_args)[0]);
        }) };
    }

    REGISTER_KERNEL(norm, GPU, F32, norm_wrapper);
    REGISTER_KERNEL(norm, GPU, F64, norm_wrapper);

    // ============================================================================
    // Helper kernels for matrix_power
    // ============================================================================

    template<typename T>
    __global__ void matrix_init_identity_kernel(T* dst, int size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < size && j < size) {
            dst[i * size + j] = (i == j) ? T(1) : T(0);
        }
    }

    template<typename T>
    __global__ void matrix_copy_kernel(T* dst, const T* src, int size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < size && j < size) {
            dst[i * size + j] = src[i * size + j];
        }
    }

    // ============================================================================
    // matrix_power (GPU via cuBLAS gemm + exponentiation by squaring)
    // ============================================================================

    static OpArgs matrix_power_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        int n = std::any_cast<int>(args[1]);

        tls_handle.ensure(x.place().device_id());
        int size = static_cast<int>(x.shape().dim(0));
        DType dtype = x.dtype();

        dim3 threads(16, 16);
        dim3 blocks((size + 15) / 16, (size + 15) / 16);

        // Allocate result (identity), base (copy of x), work (temp)
        Array result(x.shape(), dtype, x.place());
        Array base(x.shape(), dtype, x.place());
        Array work(x.shape(), dtype, x.place());

        // Initialize result = I, base = x
        if (dtype == DType::F64) {
            matrix_init_identity_kernel<double> << <blocks, threads >> > (result.data<double>(), size);
            matrix_copy_kernel<double> << <blocks, threads >> > (base.data<double>(), x.data<double>(), size);
        }
        else {
            matrix_init_identity_kernel<float> << <blocks, threads >> > (result.data<float>(), size);
            matrix_copy_kernel<float> << <blocks, threads >> > (base.data<float>(), x.data<float>(), size);
        }

        int exp = n;
        while (exp > 0) {
            if (exp & 1) {
                // result = result * base
                if (dtype == DType::F64) {
                    matrix_copy_kernel<double> << <blocks, threads >> > (work.data<double>(), result.data<double>(), size);
                    const double alpha = 1.0, beta = 0.0;
                    cublasDgemm(tls_handle.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        size, size, size, &alpha, base.data<double>(), size, work.data<double>(), size, &beta,
                        result.data<double>(), size);
                }
                else {
                    matrix_copy_kernel<float> << <blocks, threads >> > (work.data<float>(), result.data<float>(), size);
                    const float alpha = 1.0f, beta = 0.0f;
                    cublasSgemm(tls_handle.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        size, size, size, &alpha, base.data<float>(), size, work.data<float>(), size, &beta,
                        result.data<float>(), size);
                }
            }
            // base = base * base
            if (dtype == DType::F64) {
                matrix_copy_kernel<double> << <blocks, threads >> > (work.data<double>(), base.data<double>(), size);
                const double alpha = 1.0, beta = 0.0;
                cublasDgemm(tls_handle.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    size, size, size, &alpha, work.data<double>(), size, work.data<double>(), size, &beta,
                    base.data<double>(), size);
            }
            else {
                matrix_copy_kernel<float> << <blocks, threads >> > (work.data<float>(), base.data<float>(), size);
                const float alpha = 1.0f, beta = 0.0f;
                cublasSgemm(tls_handle.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    size, size, size, &alpha, work.data<float>(), size, work.data<float>(), size, &beta,
                    base.data<float>(), size);
            }
            exp >>= 1;
        }

        return { result };
    }

    REGISTER_KERNEL(matrix_power, GPU, F32, matrix_power_wrapper);
    REGISTER_KERNEL(matrix_power, GPU, F64, matrix_power_wrapper);

    // ============================================================================
    // det, slogdet, inv (cuSOLVER LU decomposition)
    // ============================================================================

    static OpArgs det_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        return { cpu_fallback(x, [](const Array& cx) {
            return std::any_cast<Array>(ops()["det"][DeviceKind::CPU][cx.dtype()]({cx})[0]);
        }) };
    }

    REGISTER_KERNEL(det, GPU, F32, det_wrapper);
    REGISTER_KERNEL(det, GPU, F64, det_wrapper);

    static OpArgs slogdet_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        auto results = cpu_fallback_multi(x, [](const Array& cx) {
            auto output = ops()["slogdet"][DeviceKind::CPU][cx.dtype()]({ cx });
            return std::vector<Array>{
                std::any_cast<Array>(output[0]),
                    std::any_cast<Array>(output[1])
            };
            });
        return { results[0], results[1] };
    }

    REGISTER_KERNEL(slogdet, GPU, F32, slogdet_wrapper);
    REGISTER_KERNEL(slogdet, GPU, F64, slogdet_wrapper);

    static OpArgs inv_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        return { cpu_fallback(x, [](const Array& cx) {
            return std::any_cast<Array>(ops()["inv"][DeviceKind::CPU][cx.dtype()]({cx})[0]);
        }) };
    }

    REGISTER_KERNEL(inv, GPU, F32, inv_wrapper);
    REGISTER_KERNEL(inv, GPU, F64, inv_wrapper);

    // ============================================================================
    // solve (cuSOLVER)
    // ============================================================================

    static OpArgs solve_wrapper(const OpArgs& args) {
        const Array& A = std::any_cast<const Array&>(args[0]);
        const Array& B = std::any_cast<const Array&>(args[1]);
        return { cpu_fallback2(A, B, [](const Array& cA, const Array& cB) {
            OpArgs cpu_args = { cA, cB };
            return std::any_cast<Array>(ops()["solve"][DeviceKind::CPU][cA.dtype()](cpu_args)[0]);
        }) };
    }

    REGISTER_KERNEL(solve, GPU, F32, solve_wrapper);
    REGISTER_KERNEL(solve, GPU, F64, solve_wrapper);

    // ============================================================================
    // cholesky (cuSOLVER)
    // ============================================================================

    static OpArgs cholesky_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        bool lower = std::any_cast<bool>(args[1]);
        return { cpu_fallback(x, [lower](const Array& cx) {
            OpArgs cpu_args = { cx, lower };
            return std::any_cast<Array>(ops()["cholesky"][DeviceKind::CPU][cx.dtype()](cpu_args)[0]);
        }) };
    }

    REGISTER_KERNEL(cholesky, GPU, F32, cholesky_wrapper);
    REGISTER_KERNEL(cholesky, GPU, F64, cholesky_wrapper);

    static OpArgs cholesky_solve_wrapper(const OpArgs& args) {
        const Array& A = std::any_cast<const Array&>(args[0]);
        const Array& B = std::any_cast<const Array&>(args[1]);
        bool lower = std::any_cast<bool>(args[2]);
        return { cpu_fallback2(A, B, [lower](const Array& cA, const Array& cB) {
            OpArgs cpu_args = { cA, cB, lower };
            return std::any_cast<Array>(ops()["cholesky_solve"][DeviceKind::CPU][cA.dtype()](cpu_args)[0]);
        }) };
    }

    REGISTER_KERNEL(cholesky_solve, GPU, F32, cholesky_solve_wrapper);
    REGISTER_KERNEL(cholesky_solve, GPU, F64, cholesky_solve_wrapper);

    // ============================================================================
    // solve_triangular (cuBLAS trsm)
    // ============================================================================

    static OpArgs solve_triangular_wrapper(const OpArgs& args) {
        const Array& A = std::any_cast<const Array&>(args[0]);
        const Array& B = std::any_cast<const Array&>(args[1]);
        bool lower = std::any_cast<bool>(args[2]);
        bool unit_diag = std::any_cast<bool>(args[3]);
        return { cpu_fallback2(A, B, [lower, unit_diag](const Array& cA, const Array& cB) {
            OpArgs cpu_args = { cA, cB, lower, unit_diag };
            return std::any_cast<Array>(ops()["solve_triangular"][DeviceKind::CPU][cA.dtype()](cpu_args)[0]);
        }) };
    }

    REGISTER_KERNEL(solve_triangular, GPU, F32, solve_triangular_wrapper);
    REGISTER_KERNEL(solve_triangular, GPU, F64, solve_triangular_wrapper);

    // ============================================================================
    // cond (CPU fallback)
    // ============================================================================

    static OpArgs cond_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        double p = std::any_cast<double>(args[1]);
        return { cpu_fallback(x, [p](const Array& cx) {
            OpArgs cpu_args = { cx, p };
            return std::any_cast<Array>(ops()["cond"][DeviceKind::CPU][cx.dtype()](cpu_args)[0]);
        }) };
    }

    REGISTER_KERNEL(cond, GPU, F32, cond_wrapper);
    REGISTER_KERNEL(cond, GPU, F64, cond_wrapper);

    // ============================================================================
    // Complex matrix decompositions → CPU fallback
    // ============================================================================

    static OpArgs svd_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        bool full_matrices = std::any_cast<bool>(args[1]);
        auto results = cpu_fallback_multi(x, [full_matrices](const Array& cx) {
            OpArgs cpu_args = { cx, full_matrices };
            auto output = ops()["svd"][DeviceKind::CPU][cx.dtype()](cpu_args);
            return std::vector<Array>{
                std::any_cast<Array>(output[0]),
                    std::any_cast<Array>(output[1]),
                    std::any_cast<Array>(output[2])
            };
            });
        return { results[0], results[1], results[2] };
    }

    REGISTER_KERNEL(svd, GPU, F32, svd_wrapper);
    REGISTER_KERNEL(svd, GPU, F64, svd_wrapper);

    static OpArgs svdvals_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        return { cpu_fallback(x, [](const Array& cx) {
            return std::any_cast<Array>(ops()["svdvals"][DeviceKind::CPU][cx.dtype()]({cx})[0]);
        }) };
    }

    REGISTER_KERNEL(svdvals, GPU, F32, svdvals_wrapper);
    REGISTER_KERNEL(svdvals, GPU, F64, svdvals_wrapper);

    static OpArgs qr_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        std::string mode = std::any_cast<std::string>(args[1]);
        auto results = cpu_fallback_multi(x, [mode](const Array& cx) {
            OpArgs cpu_args = { cx, mode };
            auto output = ops()["qr"][DeviceKind::CPU][cx.dtype()](cpu_args);
            return std::vector<Array>{
                std::any_cast<Array>(output[0]),
                    std::any_cast<Array>(output[1])
            };
            });
        return { results[0], results[1] };
    }

    REGISTER_KERNEL(qr, GPU, F32, qr_wrapper);
    REGISTER_KERNEL(qr, GPU, F64, qr_wrapper);

    static OpArgs lq_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        std::string mode = std::any_cast<std::string>(args[1]);
        auto results = cpu_fallback_multi(x, [mode](const Array& cx) {
            OpArgs cpu_args = { cx, mode };
            auto output = ops()["lq"][DeviceKind::CPU][cx.dtype()](cpu_args);
            return std::vector<Array>{
                std::any_cast<Array>(output[0]),
                    std::any_cast<Array>(output[1])
            };
            });
        return { results[0], results[1] };
    }

    REGISTER_KERNEL(lq, GPU, F32, lq_wrapper);
    REGISTER_KERNEL(lq, GPU, F64, lq_wrapper);

    static OpArgs lu_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        bool pivot = std::any_cast<bool>(args[1]);
        auto results = cpu_fallback_multi(x, [pivot](const Array& cx) {
            OpArgs cpu_args = { cx, pivot };
            auto output = ops()["lu"][DeviceKind::CPU][cx.dtype()](cpu_args);
            return std::vector<Array>{
                std::any_cast<Array>(output[0]),
                    std::any_cast<Array>(output[1])
            };
            });
        return { results[0], results[1] };
    }

    REGISTER_KERNEL(lu, GPU, F32, lu_wrapper);
    REGISTER_KERNEL(lu, GPU, F64, lu_wrapper);

    static OpArgs lu_unpack_wrapper(const OpArgs& args) {
        const Array& LU = std::any_cast<const Array&>(args[0]);
        const Array& pivots = std::any_cast<const Array&>(args[1]);
        Array LU_cpu = LU.to(CPUPlace());
        Array pivots_cpu = pivots.to(CPUPlace());
        auto output = ops()["lu_unpack"][DeviceKind::CPU][LU_cpu.dtype()]({ LU_cpu, pivots_cpu });
        return {
            std::any_cast<Array>(output[0]).to(LU.place()),
            std::any_cast<Array>(output[1]).to(LU.place()),
            std::any_cast<Array>(output[2]).to(LU.place())
        };
    }

    REGISTER_KERNEL(lu_unpack, GPU, F32, lu_unpack_wrapper);
    REGISTER_KERNEL(lu_unpack, GPU, F64, lu_unpack_wrapper);

    static OpArgs eig_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        auto results = cpu_fallback_multi(x, [](const Array& cx) {
            auto output = ops()["eig"][DeviceKind::CPU][cx.dtype()]({ cx });
            return std::vector<Array>{
                std::any_cast<Array>(output[0]),
                    std::any_cast<Array>(output[1])
            };
            });
        return { results[0], results[1] };
    }

    REGISTER_KERNEL(eig, GPU, F32, eig_wrapper);
    REGISTER_KERNEL(eig, GPU, F64, eig_wrapper);

    static OpArgs eigh_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        std::string uplo = std::any_cast<std::string>(args[1]);
        auto results = cpu_fallback_multi(x, [uplo](const Array& cx) {
            OpArgs cpu_args = { cx, uplo };
            auto output = ops()["eigh"][DeviceKind::CPU][cx.dtype()](cpu_args);
            return std::vector<Array>{
                std::any_cast<Array>(output[0]),
                    std::any_cast<Array>(output[1])
            };
            });
        return { results[0], results[1] };
    }

    REGISTER_KERNEL(eigh, GPU, F32, eigh_wrapper);
    REGISTER_KERNEL(eigh, GPU, F64, eigh_wrapper);

    static OpArgs eigvals_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        return { cpu_fallback(x, [](const Array& cx) {
            return std::any_cast<Array>(ops()["eigvals"][DeviceKind::CPU][cx.dtype()]({cx})[0]);
        }) };
    }

    REGISTER_KERNEL(eigvals, GPU, F32, eigvals_wrapper);
    REGISTER_KERNEL(eigvals, GPU, F64, eigvals_wrapper);

    static OpArgs eigvalsh_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        std::string uplo = std::any_cast<std::string>(args[1]);
        return { cpu_fallback(x, [uplo](const Array& cx) {
            OpArgs cpu_args = { cx, uplo };
            return std::any_cast<Array>(ops()["eigvalsh"][DeviceKind::CPU][cx.dtype()](cpu_args)[0]);
        }) };
    }

    REGISTER_KERNEL(eigvalsh, GPU, F32, eigvalsh_wrapper);
    REGISTER_KERNEL(eigvalsh, GPU, F64, eigvalsh_wrapper);

    static OpArgs pinv_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        double rcond = std::any_cast<double>(args[1]);
        return { cpu_fallback(x, [rcond](const Array& cx) {
            OpArgs cpu_args = { cx, rcond };
            return std::any_cast<Array>(ops()["pinv"][DeviceKind::CPU][cx.dtype()](cpu_args)[0]);
        }) };
    }

    REGISTER_KERNEL(pinv, GPU, F32, pinv_wrapper);
    REGISTER_KERNEL(pinv, GPU, F64, pinv_wrapper);

    static OpArgs matrix_rank_wrapper(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        double tol = std::any_cast<double>(args[1]);
        return { cpu_fallback(x, [tol](const Array& cx) {
            OpArgs cpu_args = { cx, tol };
            return std::any_cast<Array>(ops()["matrix_rank"][DeviceKind::CPU][cx.dtype()](cpu_args)[0]);
        }) };
    }

    REGISTER_KERNEL(matrix_rank, GPU, F32, matrix_rank_wrapper);
    REGISTER_KERNEL(matrix_rank, GPU, F64, matrix_rank_wrapper);

    static OpArgs lstsq_wrapper(const OpArgs& args) {
        const Array& A = std::any_cast<const Array&>(args[0]);
        const Array& B = std::any_cast<const Array&>(args[1]);
        double rcond = std::any_cast<double>(args[2]);
        return { cpu_fallback2(A, B, [rcond](const Array& cA, const Array& cB) {
            OpArgs cpu_args = { cA, cB, rcond };
            return std::any_cast<Array>(ops()["lstsq"][DeviceKind::CPU][cA.dtype()](cpu_args)[0]);
        }) };
    }

    REGISTER_KERNEL(lstsq, GPU, F32, lstsq_wrapper);
    REGISTER_KERNEL(lstsq, GPU, F64, lstsq_wrapper);

} // namespace ins::gpu

// ============================================================================
// Module Registration
// ============================================================================

REGISTER_MODULE(linalg, GPU);