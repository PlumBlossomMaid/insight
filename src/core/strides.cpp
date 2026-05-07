// src/internal/strides.cpp
#include "insight/core/strides.h"
#include "insight/core/shape.h"
#include <cstddef>

namespace ins {

    // ========== Constructors ==========

    Strides::Strides(const Shape& shape) {
        int nd = shape.ndim();
        if (nd == 0) return;

        strides_.resize(nd);

        // Row-major: last dimension stride = 1
        int64_t stride = 1;
        for (int i = nd - 1; i >= 0; --i) {
            strides_[i] = stride;
            stride *= shape.dim(i);
        }
    }

    Strides::Strides(const std::vector<int64_t>& strides)
        : strides_(strides) {
    }

    // ========== Accessors ==========

    int64_t Strides::operator[](int i) const {
        int nd = ndim();
        if (i < 0) i += nd;
        // No bounds check for internal use (caller ensures validity)
        return strides_[i];
    }

    // ========== Query ==========

    bool Strides::is_contiguous(const Shape& shape) const {
        if (shape.numel() == 0) return true;
        if (strides_.size() != static_cast<size_t>(shape.ndim())) return false;

        int64_t expected = 1;
        for (int i = shape.ndim() - 1; i >= 0; --i) {
            if (strides_[i] != expected) return false;
            expected *= shape.dim(i);
        }
        return true;
    }

    int64_t Strides::offset(const std::vector<int64_t>& indices) const {
        int64_t off = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            off += indices[i] * strides_[i];
        }
        return off;
    }

    // ========== Comparison ==========

    bool Strides::operator==(const Strides& other) const {
        return strides_ == other.strides_;
    }

    bool Strides::operator!=(const Strides& other) const {
        return !(*this == other);
    }

} // namespace ins