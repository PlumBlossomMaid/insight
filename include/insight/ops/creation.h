// insight/ops/creation.h
#pragma once
#include <vector>
#include <initializer_list>
#include <complex>
#include "insight/core/array.h"
#include "insight/core/place.h"
#include "insight/core/dtype.h"

namespace ins {

    // ========== Basic Creation ==========

    /**
     * @brief Create an array filled with zeros.
     */
    Array zeros(const Shape& shape, DType dtype = DType::F32, const Place& place = get_device());

    /**
     * @brief Create an array filled with ones.
     */
    Array ones(const Shape& shape, DType dtype = DType::F32, const Place& place = get_device());

    /**
     * @brief Create an array filled with a constant value.
     */
    Array full(const Shape& shape, double fill_value, DType dtype = DType::F32, const Place& place = get_device());

    /**
     * @brief Create an identity matrix (2D square with ones on diagonal).
     */
    Array eye(int64_t n, int64_t m = -1, int64_t k = 0, DType dtype = DType::F32, const Place& place = get_device());

    // ========== Range Creation ==========

    /**
     * @brief Create a 1D array with evenly spaced values.
     *
     * Single argument: arange(stop) -> [0, stop)
     * Three arguments: arange(start, stop, step)
     */
    Array arange(double end, DType dtype = DType::I64, const Place& place = get_device());
    Array arange(double start, double end, double step = 1.0, DType dtype = DType::I64, const Place& place = get_device());

    /**
     * @brief Create a 1D array with linearly spaced values.
     */
    Array linspace(double start, double stop, int64_t num, DType dtype = DType::F32, const Place& place = get_device());

    /**
     * @brief Create a 1D array with logarithmically spaced values.
     */
    Array logspace(double start, double stop, int64_t num, double base = 10.0, DType dtype = DType::F32, const Place& place = get_device());

    // ========== Create from Existing Data ==========

    /**
     * @brief Create an array from a vector (auto-deduced dtype).
     */
    template<typename T>
    Array to_array(const std::vector<T>& data, const Place& place = get_device()) {
        Shape shape({ static_cast<int64_t>(data.size()) });
        Array result(shape, dtype_of<T>(), ins::CPUPlace());
        T* dst = result.data<T>();
        std::copy(data.begin(), data.end(), dst);
        return result.to(place);
    }

    /**
     * @brief Create an array from a vector with explicit dtype.
     */
    template<typename T>
    Array to_array(const std::vector<T>& data, DType dtype, const Place& place = get_device()) {
        Shape shape({ static_cast<int64_t>(data.size()) });

        if (dtype_of<T>() == dtype) {
            Array result(shape, dtype, ins::CPUPlace());
            T* dst = result.data<T>();
            std::copy(data.begin(), data.end(), dst);
            return result.to(place);
        }
        else {
            // Create source array and convert
            Array src = to_array(data, place);  // auto-deduced dtype
            return src.to(dtype).to(place);
        }
    }

    /**
     * @brief Create an array from an initializer list (auto-deduced dtype).
     */
    template<typename T>
    Array to_array(std::initializer_list<T> data, const Place& place = get_device()) {
        std::vector<T> vec(data);
        return to_array(vec, place);
    }

    /**
     * @brief Create an array from a raw pointer and shape (auto-deduced dtype).
     */
    template<typename T>
    Array to_array(const T* data, const Shape& shape, const Place& place = get_device()) {
        Array result(shape, dtype_of<T>(), ins::CPUPlace());
        T* dst = result.data<T>();
        std::memcpy(dst, data, shape.numel() * sizeof(T));
        return result.to(place);
    }

    /**
     * @brief Create an array from a raw pointer and shape with explicit dtype.
     */
    template<typename T>
    Array to_array(const T* data, const Shape& shape, DType dtype, const Place& place = get_device()) {
        if (dtype_of<T>() == dtype) {
            Array result(shape, dtype, ins::CPUPlace());
            T* dst = result.data<T>();
            std::memcpy(dst, data, shape.numel() * sizeof(T));
            return result.to(place);
        }
        else {
            Array src = to_array(data, shape, ins::CPUPlace());  // auto-deduced dtype
            return src.to(dtype).to(place);
        }
    }

    /**
     * @brief Create an array from a vector with explicit shape.
     */
    template<typename T>
    Array to_array(const std::vector<T>& data, const Shape& shape, const Place& place = get_device()) {
        INS_CHECK(shape.numel() == static_cast<int64_t>(data.size()),
            "to_array: shape.numel() must equal data.size()");
        Array result(shape, dtype_of<T>(), ins::CPUPlace());
        T* dst = result.data<T>();
        std::copy(data.begin(), data.end(), dst);
        return result.to(place);
    }

    /**
     * @brief Create an array from a vector with explicit shape and dtype.
     */
    template<typename T>
    Array to_array(const std::vector<T>& data, const Shape& shape, DType dtype, const Place& place = get_device()) {
        INS_CHECK(shape.numel() == static_cast<int64_t>(data.size()),
            "to_array: shape.numel() must equal data.size()");

        if (dtype_of<T>() == dtype) {
            Array result(shape, dtype, ins::CPUPlace());
            T* dst = result.data<T>();
            std::copy(data.begin(), data.end(), dst);
            return result.to(place);
        }
        else {
            Array src = to_array(data, shape, place);
            return src.to(dtype).to(place);
        }
    }

    /**
     * @brief Create an array from an initializer list with explicit shape.
     */
    template<typename T>
    Array to_array(std::initializer_list<T> data, const Shape& shape, const Place& place = get_device()) {
        std::vector<T> vec(data);
        return to_array(vec, shape, place);
    }

    /**
     * @brief Create an array from an initializer list with explicit shape and dtype.
     */
    template<typename T>
    Array to_array(std::initializer_list<T> data, const Shape& shape, DType dtype, const Place& place = get_device()) {
        std::vector<T> vec(data);
        return to_array(vec, shape, dtype, place);
    }

    // ========== Like Creation ==========

    /**
     * @brief Create a zero-filled array with same shape/dtype/place as another array.
     */
    Array zeros_like(const Array& arr);

    /**
     * @brief Create a one-filled array with same shape/dtype/place as another array.
     */
    Array ones_like(const Array& arr);

} // namespace ins