// src/ops/creation.cpp
#include "insight/ops/creation.h"
#include "insight/plugin/op_registry.h"
#include "insight/utils/promotion.h"

namespace ins {

    static DeviceKind get_device_kind(const Place& place) {
        return place.is_cpu() ? DeviceKind::CPU : DeviceKind::GPU;
    }

    // ========== Basic Creation ==========

    Array zeros(const Shape& shape, DType dtype, const Place& place) {
        return full(shape, 0.0, dtype, place);
    }

    Array ones(const Shape& shape, DType dtype, const Place& place) {
        return full(shape, 1.0, dtype, place);
    }

    Array full(const Shape& shape, double fill_value, DType dtype, const Place& place) {
        // Create array
        Array result(shape, dtype, place);

        // Call fill kernel
        OpArgs args = { result, fill_value };
        DeviceKind dev = get_device_kind(place);
        OpArgs output = ops()["full"][dev][dtype](args);

        return std::any_cast<Array>(output[0]);
    }

    Array eye(int64_t n, int64_t m, int64_t k, DType dtype, const Place& place) {
        if (m < 0) m = n;
        Shape shape({ n, m });
        Array result(shape, dtype, place);

        OpArgs args = { result, k };
        DeviceKind dev = get_device_kind(place);
        OpArgs output = ops()["eye"][dev][dtype](args);

        return std::any_cast<Array>(output[0]);
    }

    // ========== Range Creation ==========

    Array arange(double end, DType dtype, const Place& place) {
        return arange(0.0, end, 1.0, dtype, place);
    }

    Array arange(double start, double end, double step, DType dtype, const Place& place) {
        // Compute number of elements
        int64_t num = static_cast<int64_t>(std::ceil((end - start) / step));
        Array result(Shape({ num }), dtype, place);

        OpArgs args = { result, start, step };
        DeviceKind dev = get_device_kind(place);
        OpArgs output = ops()["arange"][dev][dtype](args);

        return std::any_cast<Array>(output[0]);
    }

    Array linspace(double start, double stop, int64_t num, DType dtype, const Place& place) {
        Array result(Shape({ num }), dtype, place);

        OpArgs args = { result, start, stop };
        DeviceKind dev = get_device_kind(place);
        OpArgs output = ops()["linspace"][dev][dtype](args);

        return std::any_cast<Array>(output[0]);
    }

    Array logspace(double start, double stop, int64_t num, double base, DType dtype, const Place& place) {
        Array result(Shape({ num }), dtype, place);

        OpArgs args = { result, start, stop, base };
        DeviceKind dev = get_device_kind(place);
        OpArgs output = ops()["logspace"][dev][dtype](args);

        return std::any_cast<Array>(output[0]);
    }

    // ========== Like Creation ==========

    Array zeros_like(const Array& arr) {
        return zeros(arr.shape(), arr.dtype(), arr.place());
    }

    Array ones_like(const Array& arr) {
        return ones(arr.shape(), arr.dtype(), arr.place());
    }

} // namespace ins