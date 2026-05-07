// src/core/shape.cpp
#include "insight/core/shape.h"
#include "insight/core/exception.h"
#include "insight/core/array.h"
#include <algorithm>
#include <numeric>
#include <sstream>

namespace ins {

    // ========== Constructors ==========

    Shape::Shape(std::initializer_list<int64_t> dims) {
        ndim_ = static_cast<int>(dims.size());
        INS_CHECK(ndim_ <= INSIGHT_MAX_NDIM,
            "Shape exceeds maximum dimensions: ", ndim_, " > ", INSIGHT_MAX_NDIM);
        numel_ = 1;
        int i = 0;
        for (int64_t d : dims) {
            dims_[i++] = d;
            numel_ *= d;
        }
    }

    Shape::Shape(const std::vector<int64_t>& dims) {
        ndim_ = static_cast<int>(dims.size());
        INS_CHECK(ndim_ <= INSIGHT_MAX_NDIM,
            "Shape exceeds maximum dimensions: ", ndim_, " > ", INSIGHT_MAX_NDIM);
        numel_ = 1;
        for (int i = 0; i < ndim_; ++i) {
            dims_[i] = dims[i];
            numel_ *= dims_[i];
        }
    }

    // ========== Accessors ==========

    int64_t Shape::dim(int i) const {
        int nd = ndim();
        if (i < 0) i += nd;
        INS_CHECK(i >= 0 && i < nd, "Shape dimension index out of range: ", i, " (ndim=", nd, ")");
        return dims_[i];
    }

    // ========== Shape Manipulation ==========

    Shape Shape::squeeze() const {
        std::vector<int64_t> new_dims;
        for (int64_t d : dims_) {
            if (d != 1) {
                new_dims.push_back(d);
            }
        }
        return Shape(new_dims);
    }

    Shape Shape::squeeze(int dim) const {
        if (dim < 0) dim += ndim_;
        INS_CHECK(dim >= 0 && dim < ndim_, "squeeze: dimension out of range: ", dim);
        INS_CHECK(dims_[dim] == 1, "squeeze: dimension ", dim, " is not 1 (value=", dims_[dim], ")");

        int64_t new_dims_arr[INSIGHT_MAX_NDIM];
        int new_ndim = 0;
        for (int i = 0; i < ndim_; ++i) {
            if (i != dim) {
                new_dims_arr[new_ndim++] = dims_[i];
            }
        }
        return Shape(std::vector<int64_t>(new_dims_arr, new_dims_arr + new_ndim));
    }

    Shape Shape::unsqueeze(int dim) const {
        if (dim < 0) dim += ndim_ + 1;
        INS_CHECK(dim >= 0 && dim <= ndim_, "unsqueeze: dimension out of range: ", dim, " (ndim=", ndim_, ")");

        int64_t new_dims_arr[INSIGHT_MAX_NDIM];
        for (int i = 0, j = 0; i <= ndim_; ++i) {
            if (i == dim) {
                new_dims_arr[i] = 1;
            }
            else {
                new_dims_arr[i] = dims_[j++];
            }
        }
        return Shape(std::vector<int64_t>(new_dims_arr, new_dims_arr + ndim_ + 1));
    }

    // ========== Comparison ==========

    bool Shape::operator==(const Shape& other) const {
        if (ndim_ != other.ndim_) return false;
        for (int i = 0; i < ndim_; ++i) {
            if (dims_[i] != other.dims_[i]) return false;
        }
        return true;
    }

    bool Shape::operator!=(const Shape& other) const {
        return !(*this == other);
    }

    // ========== String Conversion ==========

    std::string Shape::to_string() const {
        std::ostringstream os;
        os << *this;
        return os.str();
    }

    std::ostream& operator<<(std::ostream& os, const Shape& shape) {
        os << "[";
        for (int i = 0; i < shape.ndim(); ++i) {
            if (i > 0) os << ", ";
            os << shape.dim(i);
        }
        os << "]";
        return os;
    }

} // namespace ins