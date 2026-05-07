// src/ops/random.cpp
#include "insight/ops/random.h"
#include "insight/plugin/op_registry.h"
#include <atomic>

namespace ins {

    static std::atomic<uint64_t> g_global_seed{ 0 };

    static DeviceKind get_device_kind(const Place& place) {
        return place.is_cpu() ? DeviceKind::CPU : DeviceKind::GPU;
    }

    // ========== Seed Management ==========

    void seed(uint64_t s) {
        g_global_seed = s;
    }

    uint64_t get_seed() {
        return g_global_seed.load();
    }

    // ========== Basic Distributions ==========

    Array rand(const Shape& shape, DType dtype, const Place& place) {
        Array result(shape, dtype, place);
        OpArgs args = { result };
        DeviceKind dev = get_device_kind(place);
        OpArgs output = ops()["rand"][dev][dtype](args);
        return std::any_cast<Array>(output[0]);
    }

    Array randn(const Shape& shape, DType dtype, const Place& place) {
        Array result(shape, dtype, place);
        OpArgs args = { result };
        DeviceKind dev = get_device_kind(place);
        OpArgs output = ops()["randn"][dev][dtype](args);
        return std::any_cast<Array>(output[0]);
    }

    Array randint(int64_t low, int64_t high, const Shape& shape, DType dtype, const Place& place) {
        Array result(shape, dtype, place);
        OpArgs args = { result, low, high };
        DeviceKind dev = get_device_kind(place);
        OpArgs output = ops()["randint"][dev][dtype](args);
        return std::any_cast<Array>(output[0]);
    }

    Array normal(double mean, double std, const Shape& shape, DType dtype, const Place& place) {
        Array result(shape, dtype, place);
        OpArgs args = { result, mean, std };
        DeviceKind dev = get_device_kind(place);
        OpArgs output = ops()["normal"][dev][dtype](args);
        return std::any_cast<Array>(output[0]);
    }

    Array uniform(double low, double high, const Shape& shape, DType dtype, const Place& place) {
        Array result(shape, dtype, place);
        OpArgs args = { result, low, high };
        DeviceKind dev = get_device_kind(place);
        OpArgs output = ops()["uniform"][dev][dtype](args);
        return std::any_cast<Array>(output[0]);
    }

    Array randperm(int64_t n, DType dtype, const Place& place) {
        Array result(Shape({ n }), dtype, place);
        OpArgs args = { result };
        DeviceKind dev = get_device_kind(place);
        OpArgs output = ops()["randperm"][dev][dtype](args);
        return std::any_cast<Array>(output[0]);
    }

    // ========== Like Functions ==========

    Array randint_like(const Array& x, int64_t low, int64_t high) {
        DType out_dtype = (is_integer(x.dtype())) ? x.dtype() : DType::I64;
        return randint(low, high, x.shape(), out_dtype, x.place());
    }

    // ========== Additional Distributions ==========

    Array exponential(double scale, const Shape& shape, DType dtype, const Place& place) {
        Array result(shape, dtype, place);
        OpArgs args = { result, scale };
        DeviceKind dev = get_device_kind(place);
        OpArgs output = ops()["exponential"][dev][dtype](args);
        return std::any_cast<Array>(output[0]);
    }

    Array gamma(double shape_param, double rate, const Shape& shape, DType dtype, const Place& place) {
        Array result(shape, dtype, place);
        OpArgs args = { result, shape_param, rate };
        DeviceKind dev = get_device_kind(place);
        OpArgs output = ops()["gamma"][dev][dtype](args);
        return std::any_cast<Array>(output[0]);
    }

    Array beta(double a, double b, const Shape& shape, DType dtype, const Place& place) {
        Array result(shape, dtype, place);
        OpArgs args = { result, a, b };
        DeviceKind dev = get_device_kind(place);
        OpArgs output = ops()["beta"][dev][dtype](args);
        return std::any_cast<Array>(output[0]);
    }

    Array binomial(int64_t n, double p, const Shape& shape, DType dtype, const Place& place) {
        Array result(shape, dtype, place);
        OpArgs args = { result, n, p };
        DeviceKind dev = get_device_kind(place);
        OpArgs output = ops()["binomial"][dev][dtype](args);
        return std::any_cast<Array>(output[0]);
    }

    Array poisson(double lam, const Shape& shape, DType dtype, const Place& place) {
        Array result(shape, dtype, place);
        OpArgs args = { result, lam };
        DeviceKind dev = get_device_kind(place);
        OpArgs output = ops()["poisson"][dev][dtype](args);
        return std::any_cast<Array>(output[0]);
    }

    Array chisquare(double df, const Shape& shape, DType dtype, const Place& place) {
        Array result(shape, dtype, place);
        OpArgs args = { result, df };
        DeviceKind dev = get_device_kind(place);
        OpArgs output = ops()["chisquare"][dev][dtype](args);
        return std::any_cast<Array>(output[0]);
    }

} // namespace ins