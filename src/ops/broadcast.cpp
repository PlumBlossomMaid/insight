// src/ops/broadcast.cpp
#include "insight/core/strides.h"
#include "insight/ops/broadcast.h"
#include "insight/core/exception.h"
#include "insight/core/array.h"
#include "../internal/array_impl.h"
#include <algorithm>

namespace ins {

    // ========== Helper Functions ==========

    static Shape broadcast_shape_impl(const Shape& a, const Shape& b) {
        int a_ndim = a.ndim();
        int b_ndim = b.ndim();
        int max_ndim = std::max(a_ndim, b_ndim);

        std::vector<int64_t> out_dims(max_ndim);

        for (int i = 0; i < max_ndim; ++i) {
            int64_t a_dim = (i < a_ndim) ? a.dim(a_ndim - 1 - i) : 1;
            int64_t b_dim = (i < b_ndim) ? b.dim(b_ndim - 1 - i) : 1;

            if (a_dim != 1 && b_dim != 1 && a_dim != b_dim) {
                INS_THROW("Shapes not broadcastable: ", a, " vs ", b);
            }

            out_dims[max_ndim - 1 - i] = std::max(a_dim, b_dim);
        }

        return Shape(out_dims);
    }

    static Strides broadcast_strides(const Shape& src_shape,
        const Strides& src_strides,
        const Shape& target_shape) {
        int src_ndim = src_shape.ndim();
        int tgt_ndim = target_shape.ndim();

        std::vector<int64_t> new_strides(tgt_ndim);

        for (int i = 0; i < tgt_ndim; ++i) {
            int src_idx = i - (tgt_ndim - src_ndim);

            if (src_idx >= 0 && src_shape.dim(src_idx) == target_shape.dim(i)) {
                new_strides[i] = src_strides[src_idx];
            }
            else {
                new_strides[i] = 0;
            }
        }

        return Strides(new_strides);
    }

    static bool is_broadcastable_to(const Shape& src, const Shape& target) {
        int src_ndim = src.ndim();
        int tgt_ndim = target.ndim();

        if (src_ndim > tgt_ndim) {
            return false;
        }

        for (int i = 0; i < src_ndim; ++i) {
            int src_idx = src_ndim - 1 - i;
            int tgt_idx = tgt_ndim - 1 - i;

            int64_t src_dim = src.dim(src_idx);
            int64_t tgt_dim = target.dim(tgt_idx);

            if (src_dim != 1 && src_dim != tgt_dim) {
                return false;
            }
        }

        return true;
    }

    // ========== Public API ==========

    Shape broadcast_shape(const Shape& a, const Shape& b) {
        return broadcast_shape_impl(a, b);
    }

    Array broadcast_to(const Array& x, const Shape& target_shape) {
        INS_CHECK(x.defined(), "broadcast_to: input array is undefined");
        INS_CHECK(is_broadcastable_to(x.shape(), target_shape),
            "broadcast_to: cannot broadcast shape ", x.shape(), " to ", target_shape);

        Strides new_strides = broadcast_strides(x.impl_->shape, x.impl_->strides, target_shape);

        return Array(x.impl_, target_shape, new_strides, x.impl_->offset);
    }

    std::vector<Array> broadcast_arrays(const std::vector<Array>& tensors) {
        INS_CHECK(!tensors.empty(), "broadcast_arrays: input list cannot be empty");

        // Check all tensors are defined
        for (size_t i = 0; i < tensors.size(); ++i) {
            INS_CHECK(tensors[i].defined(), "broadcast_arrays: tensor ", i, " is undefined");
        }

        // Compute common shape by broadcasting all input shapes
        Shape common = tensors[0].shape();
        for (size_t i = 1; i < tensors.size(); ++i) {
            common = broadcast_shape_impl(common, tensors[i].shape());
        }

        // Broadcast each tensor to the common shape
        std::vector<Array> result;
        result.reserve(tensors.size());
        for (const auto& t : tensors) {
            result.push_back(broadcast_to(t, common));
        }

        return result;
    }

} // namespace ins