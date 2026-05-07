// backends/cpu/random.cpp
#include "insight/plugin/op_registry.h"
#include "insight/core/array.h"
#include "insight/ops/random.h"
#include <random>
#include <algorithm>
#include <cmath>

namespace ins::cpu {

    // ========== Thread-local RNG (simplified strategy) ==========
    static std::mt19937& get_rng() {
        static thread_local std::mt19937 rng(std::random_device{}() + ins::get_seed());
        return rng;
    }

    // ========== Helper: fill contiguous with distribution ==========
    template<typename T, typename Dist>
    static void fill_contiguous(T* data, int64_t n, Dist&& dist) {
#pragma omp parallel for
        for (int64_t i = 0; i < n; ++i) {
            data[i] = dist(get_rng());
        }
    }

    // ========== Helper: fill strided with distribution ==========
    template<typename T, typename Dist>
    static void fill_strided(Array& out, Dist&& dist) {
        const Shape& shape = out.shape();
        const Strides& strides = out.strides();
        int64_t offset = out.offset();
        T* base = out.data<T>();
        int64_t total = out.numel();
        int ndim = shape.ndim();

        std::vector<int64_t> indices(ndim, 0);
        for (int64_t i = 0; i < total; ++i) {
            int64_t idx = offset;
            for (int d = 0; d < ndim; ++d) {
                idx += indices[d] * strides[d];
            }
            base[idx] = dist(get_rng());

            for (int d = ndim - 1; d >= 0; --d) {
                if (++indices[d] < shape.dim(d)) break;
                indices[d] = 0;
            }
        }
    }

    template<typename T, typename Dist>
    static void fill_array(Array& out, Dist&& dist) {
        if (out.is_contiguous()) {
            fill_contiguous(out.data<T>(), out.numel(), std::forward<Dist>(dist));
        }
        else {
            fill_strided<T>(out, std::forward<Dist>(dist));
        }
    }

    // ========== rand ==========
    template<typename T>
    static Array rand_impl(Array& out) {
        std::uniform_real_distribution<T> dist(0, 1);
        fill_array<T>(out, dist);
        return out;
    }

    static OpArgs rand_kernel(const OpArgs& args) {
        Array& out = const_cast<Array&>(std::any_cast<const Array&>(args[0]));
        switch (out.dtype()) {
        case DType::F32: return { rand_impl<float>(out) };
        case DType::F64: return { rand_impl<double>(out) };
        default: INS_THROW("rand: unsupported dtype");
        }
    }

    REGISTER_KERNEL(rand, CPU, F32, rand_kernel);
    REGISTER_KERNEL(rand, CPU, F64, rand_kernel);

    // ========== randn ==========
    template<typename T>
    static Array randn_impl(Array& out) {
        std::normal_distribution<T> dist(0, 1);
        fill_array<T>(out, dist);
        return out;
    }

    static OpArgs randn_kernel(const OpArgs& args) {
        Array& out = const_cast<Array&>(std::any_cast<const Array&>(args[0]));
        switch (out.dtype()) {
        case DType::F32: return { randn_impl<float>(out) };
        case DType::F64: return { randn_impl<double>(out) };
        default: INS_THROW("randn: unsupported dtype");
        }
    }

    REGISTER_KERNEL(randn, CPU, F32, randn_kernel);
    REGISTER_KERNEL(randn, CPU, F64, randn_kernel);

    // ========== randint ==========
    template<typename T>
    static Array randint_impl(Array& out, int64_t low, int64_t high) {
        std::uniform_int_distribution<T> dist(static_cast<T>(low), static_cast<T>(high - 1));
        fill_array<T>(out, dist);
        return out;
    }

    static OpArgs randint_kernel(const OpArgs& args) {
        Array& out = const_cast<Array&>(std::any_cast<const Array&>(args[0]));
        int64_t low = std::any_cast<int64_t>(args[1]);
        int64_t high = std::any_cast<int64_t>(args[2]);
        switch (out.dtype()) {
        case DType::I32: return { randint_impl<int32_t>(out, low, high) };
        case DType::I64: return { randint_impl<int64_t>(out, low, high) };
        default: INS_THROW("randint: unsupported dtype");
        }
    }

    REGISTER_KERNEL(randint, CPU, I32, randint_kernel);
    REGISTER_KERNEL(randint, CPU, I64, randint_kernel);

    // ========== normal ==========
    template<typename T>
    static Array normal_impl(Array& out, double mean, double std_val) {
        std::normal_distribution<T> dist(static_cast<T>(mean), static_cast<T>(std_val));
        fill_array<T>(out, dist);
        return out;
    }

    static OpArgs normal_kernel(const OpArgs& args) {
        Array& out = const_cast<Array&>(std::any_cast<const Array&>(args[0]));
        double mean = std::any_cast<double>(args[1]);
        double std_val = std::any_cast<double>(args[2]);
        switch (out.dtype()) {
        case DType::F32: return { normal_impl<float>(out, mean, std_val) };
        case DType::F64: return { normal_impl<double>(out, mean, std_val) };
        default: INS_THROW("normal: unsupported dtype");
        }
    }

    REGISTER_KERNEL(normal, CPU, F32, normal_kernel);
    REGISTER_KERNEL(normal, CPU, F64, normal_kernel);

    // ========== uniform ==========
    template<typename T>
    static Array uniform_impl(Array& out, double low, double high) {
        std::uniform_real_distribution<T> dist(static_cast<T>(low), static_cast<T>(high));
        fill_array<T>(out, dist);
        return out;
    }

    static OpArgs uniform_kernel(const OpArgs& args) {
        Array& out = const_cast<Array&>(std::any_cast<const Array&>(args[0]));
        double low = std::any_cast<double>(args[1]);
        double high = std::any_cast<double>(args[2]);
        switch (out.dtype()) {
        case DType::F32: return { uniform_impl<float>(out, low, high) };
        case DType::F64: return { uniform_impl<double>(out, low, high) };
        default: INS_THROW("uniform: unsupported dtype");
        }
    }

    REGISTER_KERNEL(uniform, CPU, F32, uniform_kernel);
    REGISTER_KERNEL(uniform, CPU, F64, uniform_kernel);

    // ========== randperm ==========
    template<typename T>
    static Array randperm_impl(Array& out) {
        T* data = out.data<T>();
        int64_t n = out.numel();
        // Fill with 0..n-1
        for (int64_t i = 0; i < n; ++i) {
            data[i] = static_cast<T>(i);
        }
        // Fisher-Yates shuffle (serial - cannot parallelize)
        for (int64_t i = n - 1; i > 0; --i) {
            std::uniform_int_distribution<int64_t> dist(0, i);
            int64_t j = dist(get_rng());
            std::swap(data[i], data[j]);
        }
        return out;
    }

    static OpArgs randperm_kernel(const OpArgs& args) {
        Array& out = const_cast<Array&>(std::any_cast<const Array&>(args[0]));
        switch (out.dtype()) {
        case DType::I32: return { randperm_impl<int32_t>(out) };
        case DType::I64: return { randperm_impl<int64_t>(out) };
        default: INS_THROW("randperm: unsupported dtype");
        }
    }

    REGISTER_KERNEL(randperm, CPU, I32, randperm_kernel);
    REGISTER_KERNEL(randperm, CPU, I64, randperm_kernel);

    // ========== exponential ==========
    template<typename T>
    static Array exponential_impl(Array& out, double scale) {
        std::exponential_distribution<T> dist(static_cast<T>(1.0 / scale));
        fill_array<T>(out, dist);
        return out;
    }

    static OpArgs exponential_kernel(const OpArgs& args) {
        Array& out = const_cast<Array&>(std::any_cast<const Array&>(args[0]));
        double scale = std::any_cast<double>(args[1]);
        switch (out.dtype()) {
        case DType::F32: return { exponential_impl<float>(out, scale) };
        case DType::F64: return { exponential_impl<double>(out, scale) };
        default: INS_THROW("exponential: unsupported dtype");
        }
    }

    REGISTER_KERNEL(exponential, CPU, F32, exponential_kernel);
    REGISTER_KERNEL(exponential, CPU, F64, exponential_kernel);

    // ========== gamma ==========
    template<typename T>
    static Array gamma_impl(Array& out, double shape_param, double rate) {
        std::gamma_distribution<T> dist(static_cast<T>(shape_param), static_cast<T>(1.0 / rate));
        fill_array<T>(out, dist);
        return out;
    }

    static OpArgs gamma_kernel(const OpArgs& args) {
        Array& out = const_cast<Array&>(std::any_cast<const Array&>(args[0]));
        double shape_param = std::any_cast<double>(args[1]);
        double rate = std::any_cast<double>(args[2]);
        switch (out.dtype()) {
        case DType::F32: return { gamma_impl<float>(out, shape_param, rate) };
        case DType::F64: return { gamma_impl<double>(out, shape_param, rate) };
        default: INS_THROW("gamma: unsupported dtype");
        }
    }

    REGISTER_KERNEL(gamma, CPU, F32, gamma_kernel);
    REGISTER_KERNEL(gamma, CPU, F64, gamma_kernel);

    // ========== beta (using Gamma distribution relation) ==========
    template<typename T>
    static Array beta_impl(Array& out, double a, double b) {
        T* data = out.data<T>();
        int64_t n = out.numel();
        std::gamma_distribution<T> dist_a(static_cast<T>(a), static_cast<T>(1));
        std::gamma_distribution<T> dist_b(static_cast<T>(b), static_cast<T>(1));

        if (out.is_contiguous()) {
#pragma omp parallel for
            for (int64_t i = 0; i < n; ++i) {
                T x = dist_a(get_rng());
                T y = dist_b(get_rng());
                data[i] = x / (x + y);
            }
        }
        else {
            const Shape& shape = out.shape();
            const Strides& strides = out.strides();
            int64_t offset = out.offset();
            T* base = out.data<T>();
            int ndim = shape.ndim();
            std::vector<int64_t> indices(ndim, 0);
            for (int64_t i = 0; i < n; ++i) {
                int64_t idx = offset;
                for (int d = 0; d < ndim; ++d) {
                    idx += indices[d] * strides[d];
                }
                T x = dist_a(get_rng());
                T y = dist_b(get_rng());
                base[idx] = x / (x + y);
                // Update indices
                for (int d = ndim - 1; d >= 0; --d) {
                    if (++indices[d] < shape.dim(d)) break;
                    indices[d] = 0;
                }
            }
        }
        return out;
    }

    static OpArgs beta_kernel(const OpArgs& args) {
        Array& out = const_cast<Array&>(std::any_cast<const Array&>(args[0]));
        double a = std::any_cast<double>(args[1]);
        double b = std::any_cast<double>(args[2]);
        switch (out.dtype()) {
        case DType::F32: return { beta_impl<float>(out, a, b) };
        case DType::F64: return { beta_impl<double>(out, a, b) };
        default: INS_THROW("beta: unsupported dtype");
        }
    }

    REGISTER_KERNEL(beta, CPU, F32, beta_kernel);
    REGISTER_KERNEL(beta, CPU, F64, beta_kernel);

    // ========== binomial ==========
    template<typename T>
    static Array binomial_impl(Array& out, int64_t n, double p) {
        std::binomial_distribution<T> dist(static_cast<T>(n), p);
        fill_array<T>(out, dist);
        return out;
    }

    static OpArgs binomial_kernel(const OpArgs& args) {
        Array& out = const_cast<Array&>(std::any_cast<const Array&>(args[0]));
        int64_t n_trials = std::any_cast<int64_t>(args[1]);
        double p = std::any_cast<double>(args[2]);
        switch (out.dtype()) {
        case DType::I32: return { binomial_impl<int32_t>(out, n_trials, p) };
        case DType::I64: return { binomial_impl<int64_t>(out, n_trials, p) };
        default: INS_THROW("binomial: unsupported dtype");
        }
    }

    REGISTER_KERNEL(binomial, CPU, I32, binomial_kernel);
    REGISTER_KERNEL(binomial, CPU, I64, binomial_kernel);

    // ========== poisson ==========
    template<typename T>
    static Array poisson_impl(Array& out, double lam) {
        std::poisson_distribution<T> dist(lam);
        fill_array<T>(out, dist);
        return out;
    }

    static OpArgs poisson_kernel(const OpArgs& args) {
        Array& out = const_cast<Array&>(std::any_cast<const Array&>(args[0]));
        double lam = std::any_cast<double>(args[1]);
        switch (out.dtype()) {
        case DType::I32: return { poisson_impl<int32_t>(out, lam) };
        case DType::I64: return { poisson_impl<int64_t>(out, lam) };
        default: INS_THROW("poisson: unsupported dtype");
        }
    }

    REGISTER_KERNEL(poisson, CPU, I32, poisson_kernel);
    REGISTER_KERNEL(poisson, CPU, I64, poisson_kernel);

    // ========== chisquare ==========
    template<typename T>
    static Array chisquare_impl(Array& out, double df) {
        std::chi_squared_distribution<T> dist(df);
        fill_array<T>(out, dist);
        return out;
    }

    static OpArgs chisquare_kernel(const OpArgs& args) {
        Array& out = const_cast<Array&>(std::any_cast<const Array&>(args[0]));
        double df = std::any_cast<double>(args[1]);
        switch (out.dtype()) {
        case DType::F32: return { chisquare_impl<float>(out, df) };
        case DType::F64: return { chisquare_impl<double>(out, df) };
        default: INS_THROW("chisquare: unsupported dtype");
        }
    }

    REGISTER_KERNEL(chisquare, CPU, F32, chisquare_kernel);
    REGISTER_KERNEL(chisquare, CPU, F64, chisquare_kernel);

    // ========== Module Registration ==========
    REGISTER_MODULE(random, CPU);

} // namespace ins::cpu