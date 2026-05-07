// backends/cuda/kernels/random.cu
#include "insight/plugin/op_registry.h"
#include "insight/core/array.h"
#include "insight/core/exception.h"
#include "insight/core/launch_config.h"
#include "insight/ops/random.h"
#include <curand_kernel.h>
#include <random>
#include <algorithm>
#include <cmath>

namespace ins::gpu {

    // ============================================================================
    // CUDA curand state initialization
    // ============================================================================

    __global__ void init_curand_states(
        curandState* states, unsigned long long seed, int64_t n
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            curand_init(seed, idx, 0, &states[idx]);
        }
    }

    // ============================================================================
    // Unified launcher - handles seed and curand state management
    // ============================================================================

    template<typename KernelFunc, typename... Args>
    static void launch_random_with_seed(
        const Array& out, KernelFunc kernel, Args... args
    ) {
        int64_t n = out.numel();
        if (n == 0) return;

        LaunchConfig config(n);

        // Allocate and initialize curand states with global seed
        curandState* states;
        cudaMalloc(&states, n * sizeof(curandState));

        uint64_t seed = get_seed();
        init_curand_states << <config.blocks, config.threads >> > (
            states, static_cast<unsigned long long>(seed), n
            );

        // Launch the actual random generation kernel
        kernel << <config.blocks, config.threads >> > (
            states, n, args...
            );

        cudaFree(states);
    }

    // ============================================================================
    // Distribution generation functions for curand
    // ============================================================================

    __device__ float uniform_f32(curandState* state) {
        return curand_uniform(state);
    }

    __device__ double uniform_f64(curandState* state) {
        return curand_uniform_double(state);
    }

    __device__ float normal_f32(curandState* state) {
        return curand_normal(state);
    }

    __device__ double normal_f64(curandState* state) {
        return curand_normal_double(state);
    }

    // ============================================================================
    // gamma distribution using Marsaglia-Tsang method
    // ============================================================================

    __device__ float gamma_f32(curandState* state, float shape) {
        float d, c, x, v, u;

        if (shape < 1.0f) {
            // Gamma(shape) = Gamma(1+shape) * Uniform^(1/shape)
            d = 1.0f + shape - 1.0f / 3.0f;
            c = 1.0f / sqrtf(9.0f * d);
            do {
                do {
                    x = normal_f32(state);
                    v = 1.0f + c * x;
                } while (v <= 0.0f);
                v = v * v * v;
                u = uniform_f32(state);
            } while (u > 1.0f - 0.0331f * (x * x) * (x * x) &&
                logf(u) > 0.5f * x * x + d * (1.0f - v + logf(v)));
            return d * v * powf(uniform_f32(state), 1.0f / shape);
        }

        // shape >= 1
        d = shape - 1.0f / 3.0f;
        c = 1.0f / sqrtf(9.0f * d);
        do {
            do {
                x = normal_f32(state);
                v = 1.0f + c * x;
            } while (v <= 0.0f);
            v = v * v * v;
            u = uniform_f32(state);
        } while (u > 1.0f - 0.0331f * (x * x) * (x * x) &&
            logf(u) > 0.5f * x * x + d * (1.0f - v + logf(v)));
        return d * v;
    }

    __device__ double gamma_f64(curandState* state, double shape) {
        double d, c, x, v, u;

        if (shape < 1.0) {
            d = 1.0 + shape - 1.0 / 3.0;
            c = 1.0 / sqrt(9.0 * d);
            do {
                do {
                    x = normal_f64(state);
                    v = 1.0 + c * x;
                } while (v <= 0.0);
                v = v * v * v;
                u = uniform_f64(state);
            } while (u > 1.0 - 0.0331 * (x * x) * (x * x) &&
                log(u) > 0.5 * x * x + d * (1.0 - v + log(v)));
            return d * v * pow(uniform_f64(state), 1.0 / shape);
        }

        d = shape - 1.0 / 3.0;
        c = 1.0 / sqrt(9.0 * d);
        do {
            do {
                x = normal_f64(state);
                v = 1.0 + c * x;
            } while (v <= 0.0);
            v = v * v * v;
            u = uniform_f64(state);
        } while (u > 1.0 - 0.0331 * (x * x) * (x * x) &&
            log(u) > 0.5 * x * x + d * (1.0 - v + log(v)));
        return d * v;
    }

    // ============================================================================
    // Random generation kernels
    // ============================================================================

    // rand: uniform [0, 1)
    template<typename T>
    __global__ void rand_kernel(curandState* states, int64_t n, T* out) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = uniform_f32(states + idx);
        }
    }

    template<>
    __global__ void rand_kernel<double>(curandState* states, int64_t n, double* out) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = uniform_f64(states + idx);
        }
    }

    // randn: normal(0, 1)
    template<typename T>
    __global__ void randn_kernel(curandState* states, int64_t n, T* out) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = normal_f32(states + idx);
        }
    }

    template<>
    __global__ void randn_kernel<double>(curandState* states, int64_t n, double* out) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = normal_f64(states + idx);
        }
    }

    // randint: uniform integer [low, high)
    template<typename T>
    __global__ void randint_kernel(curandState* states, int64_t n, T* out, int64_t low, int64_t high) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            T range = static_cast<T>(high - low);
            out[idx] = static_cast<T>(low) +
                static_cast<T>(curand_uniform(states + idx) * range);
        }
    }

    // normal(mean, std)
    template<typename T>
    __global__ void normal_kernel(curandState* states, int64_t n, T* out, T mean, T std) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = mean + std * normal_f32(states + idx);
        }
    }

    template<>
    __global__ void normal_kernel<double>(curandState* states, int64_t n, double* out, double mean, double std) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = mean + std * normal_f64(states + idx);
        }
    }

    // uniform(low, high)
    template<typename T>
    __global__ void uniform_kernel(curandState* states, int64_t n, T* out, T low, T high) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = low + (high - low) * uniform_f32(states + idx);
        }
    }

    template<>
    __global__ void uniform_kernel<double>(curandState* states, int64_t n, double* out, double low, double high) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = low + (high - low) * uniform_f64(states + idx);
        }
    }

    // exponential(scale)
    template<typename T>
    __global__ void exponential_kernel(curandState* states, int64_t n, T* out, T inv_scale) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            T u = uniform_f32(states + idx);
            out[idx] = -logf(1.0f - u) / inv_scale;
        }
    }

    template<>
    __global__ void exponential_kernel<double>(curandState* states, int64_t n, double* out, double inv_scale) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            double u = uniform_f64(states + idx);
            out[idx] = -log(1.0 - u) / inv_scale;
        }
    }

    // gamma(shape, rate)
    template<typename T>
    __global__ void gamma_kernel(curandState* states, int64_t n, T* out, T shape, T rate) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = gamma_f32(states + idx, shape) * rate;
        }
    }

    template<>
    __global__ void gamma_kernel<double>(curandState* states, int64_t n, double* out, double shape, double rate) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = gamma_f64(states + idx, shape) * rate;
        }
    }

    // chisquare(df) = Gamma(df/2, 2)
    template<typename T>
    __global__ void chisquare_kernel(curandState* states, int64_t n, T* out, T df) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = 2.0f * gamma_f32(states + idx, df / 2.0f);
        }
    }

    template<>
    __global__ void chisquare_kernel<double>(curandState* states, int64_t n, double* out, double df) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = 2.0 * gamma_f64(states + idx, df / 2.0);
        }
    }

    // poisson(lambda)
    template<typename T>
    __global__ void poisson_kernel(curandState* states, int64_t n, T* out, double lambda) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            double L = exp(-lambda);
            T k = 0;
            double p = 1.0;
            do {
                k++;
                p *= uniform_f64(states + idx);
            } while (p > L);
            out[idx] = k - 1;
        }
    }

    // binomial(n, p)
    template<typename T>
    __global__ void binomial_kernel(curandState* states, int64_t n, T* out, int64_t n_trials, double p) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            T count = 0;
            for (int64_t i = 0; i < n_trials; ++i) {
                if (uniform_f64(states + idx) < p) {
                    count++;
                }
            }
            out[idx] = count;
        }
    }

    // beta(a, b) = Gamma(a) / (Gamma(a) + Gamma(b))
    template<typename T>
    __global__ void beta_kernel(curandState* states, int64_t n, T* out, T a, T b) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            T x = gamma_f32(states + idx, a);
            T y = gamma_f32(states + idx, b);
            out[idx] = x / (x + y);
        }
    }

    template<>
    __global__ void beta_kernel<double>(curandState* states, int64_t n, double* out, double a, double b) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            double x = gamma_f64(states + idx, a);
            double y = gamma_f64(states + idx, b);
            out[idx] = x / (x + y);
        }
    }

    // ============================================================================
    // randperm: CPU-implemented (Fisher-Yates not GPU-friendly)
    // ============================================================================

    template<typename T>
    static Array randperm_impl(Array& out) {
        int64_t n = out.numel();
        std::vector<T> perm(n);
        for (int64_t i = 0; i < n; ++i) {
            perm[i] = static_cast<T>(i);
        }

        // Use a deterministic RNG seeded from global seed
        uint64_t seed = get_seed();
        std::mt19937 rng(static_cast<unsigned int>(seed));

        for (int64_t i = n - 1; i > 0; --i) {
            std::uniform_int_distribution<int64_t> dist(0, i);
            int64_t j = dist(rng);
            std::swap(perm[i], perm[j]);
        }

        cudaMemcpy(out.data<T>(), perm.data(), n * sizeof(T), cudaMemcpyHostToDevice);
        return out;
    }

    // ============================================================================
    // Wrapper functions (for operator registration)
    // ============================================================================

    static OpArgs rand_wrapper(const OpArgs& args) {
        Array& out = const_cast<Array&>(std::any_cast<const Array&>(args[0]));

        switch (out.dtype()) {
        case DType::F32:
            launch_random_with_seed(out, rand_kernel<float>, out.data<float>());
            break;
        case DType::F64:
            launch_random_with_seed(out, rand_kernel<double>, out.data<double>());
            break;
        default:
            INS_THROW("rand: unsupported dtype");
        }
        return { out };
    }

    static OpArgs randn_wrapper(const OpArgs& args) {
        Array& out = const_cast<Array&>(std::any_cast<const Array&>(args[0]));

        switch (out.dtype()) {
        case DType::F32:
            launch_random_with_seed(out, randn_kernel<float>, out.data<float>());
            break;
        case DType::F64:
            launch_random_with_seed(out, randn_kernel<double>, out.data<double>());
            break;
        default:
            INS_THROW("randn: unsupported dtype");
        }
        return { out };
    }

    static OpArgs randint_wrapper(const OpArgs& args) {
        Array& out = const_cast<Array&>(std::any_cast<const Array&>(args[0]));
        int64_t low = std::any_cast<int64_t>(args[1]);
        int64_t high = std::any_cast<int64_t>(args[2]);

        switch (out.dtype()) {
        case DType::I32:
            launch_random_with_seed(out, randint_kernel<int32_t>,
                out.data<int32_t>(), low, high);
            break;
        case DType::I64:
            launch_random_with_seed(out, randint_kernel<int64_t>,
                out.data<int64_t>(), low, high);
            break;
        default:
            INS_THROW("randint: unsupported dtype");
        }
        return { out };
    }

    static OpArgs normal_wrapper(const OpArgs& args) {
        Array& out = const_cast<Array&>(std::any_cast<const Array&>(args[0]));
        double mean = std::any_cast<double>(args[1]);
        double std_val = std::any_cast<double>(args[2]);

        switch (out.dtype()) {
        case DType::F32:
            launch_random_with_seed(out, normal_kernel<float>,
                out.data<float>(), static_cast<float>(mean), static_cast<float>(std_val));
            break;
        case DType::F64:
            launch_random_with_seed(out, normal_kernel<double>,
                out.data<double>(), mean, std_val);
            break;
        default:
            INS_THROW("normal: unsupported dtype");
        }
        return { out };
    }

    static OpArgs uniform_wrapper(const OpArgs& args) {
        Array& out = const_cast<Array&>(std::any_cast<const Array&>(args[0]));
        double low = std::any_cast<double>(args[1]);
        double high = std::any_cast<double>(args[2]);

        switch (out.dtype()) {
        case DType::F32:
            launch_random_with_seed(out, uniform_kernel<float>,
                out.data<float>(), static_cast<float>(low), static_cast<float>(high));
            break;
        case DType::F64:
            launch_random_with_seed(out, uniform_kernel<double>,
                out.data<double>(), low, high);
            break;
        default:
            INS_THROW("uniform: unsupported dtype");
        }
        return { out };
    }

    static OpArgs randperm_wrapper(const OpArgs& args) {
        Array& out = const_cast<Array&>(std::any_cast<const Array&>(args[0]));

        switch (out.dtype()) {
        case DType::I32: return { randperm_impl<int32_t>(out) };
        case DType::I64: return { randperm_impl<int64_t>(out) };
        default: INS_THROW("randperm: unsupported dtype");
        }
    }

    static OpArgs exponential_wrapper(const OpArgs& args) {
        Array& out = const_cast<Array&>(std::any_cast<const Array&>(args[0]));
        double scale = std::any_cast<double>(args[1]);

        switch (out.dtype()) {
        case DType::F32:
            launch_random_with_seed(out, exponential_kernel<float>,
                out.data<float>(), static_cast<float>(1.0 / scale));
            break;
        case DType::F64:
            launch_random_with_seed(out, exponential_kernel<double>,
                out.data<double>(), 1.0 / scale);
            break;
        default:
            INS_THROW("exponential: unsupported dtype");
        }
        return { out };
    }

    static OpArgs gamma_wrapper(const OpArgs& args) {
        Array& out = const_cast<Array&>(std::any_cast<const Array&>(args[0]));
        double shape_param = std::any_cast<double>(args[1]);
        double rate = std::any_cast<double>(args[2]);

        switch (out.dtype()) {
        case DType::F32:
            launch_random_with_seed(out, gamma_kernel<float>,
                out.data<float>(), static_cast<float>(shape_param), static_cast<float>(1.0 / rate));
            break;
        case DType::F64:
            launch_random_with_seed(out, gamma_kernel<double>,
                out.data<double>(), shape_param, 1.0 / rate);
            break;
        default:
            INS_THROW("gamma: unsupported dtype");
        }
        return { out };
    }

    static OpArgs chisquare_wrapper(const OpArgs& args) {
        Array& out = const_cast<Array&>(std::any_cast<const Array&>(args[0]));
        double df = std::any_cast<double>(args[1]);

        switch (out.dtype()) {
        case DType::F32:
            launch_random_with_seed(out, chisquare_kernel<float>,
                out.data<float>(), static_cast<float>(df));
            break;
        case DType::F64:
            launch_random_with_seed(out, chisquare_kernel<double>,
                out.data<double>(), df);
            break;
        default:
            INS_THROW("chisquare: unsupported dtype");
        }
        return { out };
    }

    static OpArgs poisson_wrapper(const OpArgs& args) {
        Array& out = const_cast<Array&>(std::any_cast<const Array&>(args[0]));
        double lam = std::any_cast<double>(args[1]);

        switch (out.dtype()) {
        case DType::I32:
            launch_random_with_seed(out, poisson_kernel<int32_t>,
                out.data<int32_t>(), lam);
            break;
        case DType::I64:
            launch_random_with_seed(out, poisson_kernel<int64_t>,
                out.data<int64_t>(), lam);
            break;
        default:
            INS_THROW("poisson: unsupported dtype");
        }
        return { out };
    }

    static OpArgs binomial_wrapper(const OpArgs& args) {
        Array& out = const_cast<Array&>(std::any_cast<const Array&>(args[0]));
        int64_t n_trials = std::any_cast<int64_t>(args[1]);
        double p = std::any_cast<double>(args[2]);

        switch (out.dtype()) {
        case DType::I32:
            launch_random_with_seed(out, binomial_kernel<int32_t>,
                out.data<int32_t>(), n_trials, p);
            break;
        case DType::I64:
            launch_random_with_seed(out, binomial_kernel<int64_t>,
                out.data<int64_t>(), n_trials, p);
            break;
        default:
            INS_THROW("binomial: unsupported dtype");
        }
        return { out };
    }

    static OpArgs beta_wrapper(const OpArgs& args) {
        Array& out = const_cast<Array&>(std::any_cast<const Array&>(args[0]));
        double a = std::any_cast<double>(args[1]);
        double b = std::any_cast<double>(args[2]);

        switch (out.dtype()) {
        case DType::F32:
            launch_random_with_seed(out, beta_kernel<float>,
                out.data<float>(), static_cast<float>(a), static_cast<float>(b));
            break;
        case DType::F64:
            launch_random_with_seed(out, beta_kernel<double>,
                out.data<double>(), a, b);
            break;
        default:
            INS_THROW("beta: unsupported dtype");
        }
        return { out };
    }

    // ============================================================================
    // Kernel Registration
    // ============================================================================

    REGISTER_KERNEL(rand, GPU, F32, rand_wrapper);
    REGISTER_KERNEL(rand, GPU, F64, rand_wrapper);

    REGISTER_KERNEL(randn, GPU, F32, randn_wrapper);
    REGISTER_KERNEL(randn, GPU, F64, randn_wrapper);

    REGISTER_KERNEL(randint, GPU, I32, randint_wrapper);
    REGISTER_KERNEL(randint, GPU, I64, randint_wrapper);

    REGISTER_KERNEL(normal, GPU, F32, normal_wrapper);
    REGISTER_KERNEL(normal, GPU, F64, normal_wrapper);

    REGISTER_KERNEL(uniform, GPU, F32, uniform_wrapper);
    REGISTER_KERNEL(uniform, GPU, F64, uniform_wrapper);

    REGISTER_KERNEL(randperm, GPU, I32, randperm_wrapper);
    REGISTER_KERNEL(randperm, GPU, I64, randperm_wrapper);

    REGISTER_KERNEL(exponential, GPU, F32, exponential_wrapper);
    REGISTER_KERNEL(exponential, GPU, F64, exponential_wrapper);

    REGISTER_KERNEL(gamma, GPU, F32, gamma_wrapper);
    REGISTER_KERNEL(gamma, GPU, F64, gamma_wrapper);

    REGISTER_KERNEL(chisquare, GPU, F32, chisquare_wrapper);
    REGISTER_KERNEL(chisquare, GPU, F64, chisquare_wrapper);

    REGISTER_KERNEL(poisson, GPU, I32, poisson_wrapper);
    REGISTER_KERNEL(poisson, GPU, I64, poisson_wrapper);

    REGISTER_KERNEL(binomial, GPU, I32, binomial_wrapper);
    REGISTER_KERNEL(binomial, GPU, I64, binomial_wrapper);

    REGISTER_KERNEL(beta, GPU, F32, beta_wrapper);
    REGISTER_KERNEL(beta, GPU, F64, beta_wrapper);

} // namespace ins::gpu

REGISTER_MODULE(random, GPU);