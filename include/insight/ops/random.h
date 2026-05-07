// insight/ops/random.h
#pragma once
#include "insight/core/array.h"

namespace ins {

    // ========== Seed Management ==========

    /**
     * @brief Set global random seed.
     *
     * Note: Due to thread_local optimization, calling seed() after some threads
     * have already used RNG may not affect those threads. For deterministic
     * behavior, call seed() before any random operations.
     */
    void seed(uint64_t s);

    /**
     * @brief Get current global random seed.
     */
    uint64_t get_seed();

    // ========== Basic Distributions ==========

    /**
     * @brief Create an array with random values uniformly distributed in [0, 1).
     */
    Array rand(const Shape& shape, DType dtype = DType::F32, const Place& place = get_device());

    /**
     * @brief Create an array with random values from standard normal distribution N(0, 1).
     */
    Array randn(const Shape& shape, DType dtype = DType::F32, const Place& place = get_device());

    /**
     * @brief Create an array with random integers in [low, high).
     */
    Array randint(int64_t low, int64_t high, const Shape& shape,
        DType dtype = DType::I64, const Place& place = get_device());

    /**
     * @brief Create an array with random values from normal distribution N(mean, std).
     */
    Array normal(double mean, double std, const Shape& shape,
        DType dtype = DType::F32, const Place& place = get_device());

    /**
     * @brief Create an array with random values from uniform distribution [low, high).
     */
    Array uniform(double low, double high, const Shape& shape,
        DType dtype = DType::F32, const Place& place = get_device());

    /**
     * @brief Create a random permutation of integers from 0 to n-1.
     */
    Array randperm(int64_t n, DType dtype = DType::I64, const Place& place = get_device());

    // ========== Like Functions ==========

    /**
     * @brief Create a random uniform array with same shape/dtype/place as another array.
     */
    inline Array rand_like(const Array& x) {
        return rand(x.shape(), x.dtype(), x.place());
    }

    /**
     * @brief Create a random normal array with same shape/dtype/place as another array.
     */
    inline Array randn_like(const Array& x) {
        return randn(x.shape(), x.dtype(), x.place());
    }

    /**
     * @brief Create a random integer array with same shape as another array.
     */
    Array randint_like(const Array& x, int64_t low, int64_t high);

    // ========== Additional Distributions ==========

    /**
     * @brief Create an array with random values from exponential distribution.
     * @param scale The scale parameter (beta = 1/lambda)
     */
    Array exponential(double scale, const Shape& shape,
        DType dtype = DType::F32, const Place& place = get_device());

    /**
     * @brief Create an array with random values from gamma distribution.
     * @param shape_param Shape parameter (alpha)
     * @param rate Rate parameter (beta = 1/theta)
     */
    Array gamma(double shape_param, double rate, const Shape& shape,
        DType dtype = DType::F32, const Place& place = get_device());

    /**
     * @brief Create an array with random values from beta distribution.
     * @param a Alpha parameter
     * @param b Beta parameter
     */
    Array beta(double a, double b, const Shape& shape,
        DType dtype = DType::F32, const Place& place = get_device());

    /**
     * @brief Create an array with random values from binomial distribution.
     * @param n Number of trials
     * @param p Probability of success (0 <= p <= 1)
     */
    Array binomial(int64_t n, double p, const Shape& shape,
        DType dtype = DType::I64, const Place& place = get_device());

    /**
     * @brief Create an array with random values from Poisson distribution.
     * @param lam Lambda (mean)
     */
    Array poisson(double lam, const Shape& shape,
        DType dtype = DType::I64, const Place& place = get_device());

    /**
     * @brief Create an array with random values from chi-square distribution.
     * @param df Degrees of freedom
     */
    Array chisquare(double df, const Shape& shape,
        DType dtype = DType::F32, const Place& place = get_device());

} // namespace ins