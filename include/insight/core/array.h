// insight/core/array.h
#pragma once
#include "insight/core/shape.h"
#include "insight/core/dtype.h"
#include "insight/core/place.h"
#include "insight/core/slice.h"
#include "insight/core/exception.h"
#include "insight/core/strides.h"
#include <memory>
#include <vector>
#include <cstdint>

namespace ins {

    // Forward declaration
    class ArrayImpl;

    /**
     * @brief Multi-dimensional array container.
     *
     * Array is the core data structure of Insight. It represents a dense
     * multi-dimensional array with a given shape, data type, and device placement.
     *
     * Examples:
     *   Array a({2, 3});                    // 2x3 float32 array on CPU
     *   Array b({2, 3}, DType::F64);        // 2x3 float64 array on CPU
     *   Array c({2, 3}, DType::F32, Place::GPU());  // 2x3 on GPU
     */
    class Array {
    public:
        // ========== Constructors ==========

        /// Create an empty (undefined) array
        Array();

        /// Create a new array with given shape, dtype, and place
        Array(const Shape& shape, DType dtype = DType::F32, const Place& place = Place::CPU());

        /// Create a scalar array from a value
        // Boolean
        explicit Array(bool value);

        // Unsigned integers
        explicit Array(uint8_t value);
        explicit Array(uint16_t value);
        explicit Array(uint32_t value);
        explicit Array(uint64_t value);

        // Signed integers
        explicit Array(int8_t value);
        explicit Array(int16_t value);
        explicit Array(int32_t value);
        explicit Array(int64_t value);

        // Floating point
        explicit Array(float value);
        explicit Array(double value);

        // Complex
        explicit Array(std::complex<float> value);
        explicit Array(std::complex<double> value);

        Array(const Array& other) = default;

        /// Move constructor
        Array(Array&& other) noexcept = default;

        /// Destructor
        ~Array();

        // ========== Assignment ==========

        /// Deep copy assignment
        Array& operator=(const Array& other);

        /// Move assignment
        Array& operator=(Array&& other) noexcept;

        // ========== State ==========

        /// Check if array is initialized
        bool defined() const;

        // ========== Metadata ==========

        /// Return the shape of the array
        Shape shape() const;

        /// Return the data type
        DType dtype() const;

        /// Return the device placement
        Place place() const;

        /// Return the total number of elements
        int64_t numel() const;

        // ========== Memory Layout ==========

        /// Check if the array is contiguous in memory
        bool is_contiguous() const;

        /// Ensure the array is contiguous (creates a copy if needed)
        Array contiguous() const;

        // ========== Data Access ==========

        /// Raw pointer access (use with caution)
        void* data();
        const void* data() const;

        /// Typed raw pointer access
        template<typename T>
        T* data() {
            return static_cast<T*>(data());
        }

        template<typename T>
        const T* data() const {
            return static_cast<const T*>(data());
        }

        // ========== Element Access ==========

        /// Return a zero-dimensional array (scalar) at given indices
        /// Use .item<T>() to extract the value
        template<typename... Indices>
        Array at(Indices... indices) const {
            static_assert(sizeof...(Indices) > 0, "at() requires at least one index");
            std::vector<int64_t> idx = { static_cast<int64_t>(indices)... };
            return at(idx);
        }

        Array at(const std::vector<int64_t>& indices) const;

        template<typename T>
        T item() const {
            static_assert(std::is_arithmetic_v<T> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>,
                "item<T>() requires arithmetic type or complex type");
            INS_CHECK(defined(), "Array is not initialized");
            INS_CHECK(numel() == 1, "item() only works for scalar arrays, got ", numel());

            // Check type compatibility
            DType expected = dtype_of<T>();
            INS_CHECK(dtype() == expected,
                "item<", dtype_name(expected), ">() called on array of dtype ", dtype_name(dtype()));

            return this->to(ins::CPUPlace()).data<T>()[0];
        }

        // ========== Slicing (Views) ==========

        /// Slice a single dimension
        Array slice(int dim, int64_t start, int64_t stop, int64_t step = 1) const;

        /// Multi-dimensional slice with Slice objects
        Array slice(const std::vector<Slice>& slices) const;

        // Python-style syntactic sugar
        Array operator[](const std::string& spec) const;
        Array operator[](const Slice& slice) const;

        // ========== View Operations (Zero-Copy) ==========

        /// Reshape the array (view if possible, otherwise copy)
        Array reshape(const Shape& new_shape) const;

        /// Transpose the last two dimensions (view)
        Array transpose() const;

        /// Transpose with arbitrary permutation (advanced)
        Array transpose(const std::vector<int>& perm) const;

        /**
         * @brief Remove dimensions of size 1.
         * @param axis If provided, remove only the specified dimension (must be size 1).
         *             If not provided, remove all dimensions of size 1.
         * @return Squeezed view (zero-copy if possible)
         */
        Array squeeze(std::optional<int> axis = std::nullopt) const;

        /// Add a dimension of size 1 (view)
        Array unsqueeze(int dim) const;

        /**
         * @brief Create a view with a different shape (zero-copy).
         *
         * @param new_shape New shape, must have same total number of elements.
         * @return View array sharing the same storage.
         */
        Array view(const Shape& new_shape) const;

        /**
         * @brief Create a view with a different dtype (zero-copy reinterpretation).
         *
         * The new dtype must have the same size as the original dtype.
         *
         * @param new_dtype Target dtype (must have same size)
         * @return View array with reinterpreted dtype.
         */
        Array view(DType new_dtype) const;

        /**
         * @brief Create a view with different shape and dtype (zero-copy).
         *
         * Total bytes: shape.numel() * dtype_size(new_dtype) must equal
         * total bytes of original array.
         *
         * @param new_shape New shape
         * @param new_dtype Target dtype
         * @return View array with new shape and reinterpreted dtype.
         */
        Array view(const Shape& new_shape, DType new_dtype) const;

        // ========== Device/Type Conversion ==========

        /// Convert to a different device
        Array to(const Place& target) const;

        /// Convert to a different data type
        Array to(DType target) const;

        /// Convert to a different device and data type
        Array to(const Place& target, DType target_dtype) const;

        /// Convert to same device and dtype as another array
        Array to(const Array& other) const;

        // ========== Copy ==========

        /// Create a deep copy of the array
        Array copy() const;

        // ========== Advanced Accessors ==========

        /**
         * @brief Get the stride information of the array.
         *
         * Strides represent the number of elements to skip to move to the next
         * element along each dimension. This is primarily used for implementing
         * zero-copy views and broadcasting.
         *
         * @return const reference to the strides object
         */
        const Strides strides() const;

        /**
         * @brief Get the offset of the first element in the underlying storage.
         *
         * The offset is measured in number of elements from the start of the
         * shared storage to the first element of this array view.
         *
         * @return Offset in elements (0 for non-view arrays)
         */
        int64_t offset() const;

        // ========== Boolean conversion ==========

        /**
         * @brief Convert scalar array to bool.
         *
         * For arrays with more than one element, throws an exception to avoid ambiguity.
         * This matches NumPy/PyTorch/Paddle behavior.
         *
         * @throws Exception if numel() != 1
         * @return true if the scalar value is non-zero
         */
        explicit operator bool() const;

        /**
         * @brief Returns true if any element is non-zero.
         */
        bool any() const;

        /**
         * @brief Returns true if all elements are non-zero.
         */
        bool all() const;

    private:
        std::shared_ptr<ArrayImpl> impl_;

        friend Array broadcast_to(const Array& x, const Shape& target_shape);
        friend std::vector<Array> broadcast_arrays(const std::vector<Array>& tensors);

        /// View constructor (internal, zero-copy)
        Array(std::shared_ptr<ArrayImpl> parent_impl,
            const Shape& shape, const Strides& strides, int64_t offset);
    };

} // namespace ins