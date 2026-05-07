// src/internal/strides.h
#pragma once
#include <cstdint>
#include <vector>

namespace ins {

    class Shape;

    /**
     * @brief Stride information for multi-dimensional array (internal use only).
     *
     * Strides represent the number of elements to skip to move to the next
     * element along each dimension. For a contiguous row-major array:
     *   strides[ndim-1] = 1
     *   strides[i] = shape[i+1] * strides[i+1]
     *
     * This class is used internally for view implementation and is not
     * exposed to users.
     */
    class Strides {
    public:
        // Constructors
        Strides() = default;

        /// Compute row-major contiguous strides from shape
        explicit Strides(const Shape& shape);

        /// Custom strides
        explicit Strides(const std::vector<int64_t>& strides);

        // Accessors
        int ndim() const { return static_cast<int>(strides_.size()); }
        int64_t operator[](int i) const;
        const std::vector<int64_t>& data() const { return strides_; }

        // Query
        bool is_contiguous(const Shape& shape) const;
        int64_t offset(const std::vector<int64_t>& indices) const;

        // Comparison
        bool operator==(const Strides& other) const;
        bool operator!=(const Strides& other) const;

        // Utility
        bool empty() const { return strides_.empty(); }

    private:
        std::vector<int64_t> strides_;
    };

} // namespace ins