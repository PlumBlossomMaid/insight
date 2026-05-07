// backends/cpu/indexing.cpp
#include "insight/plugin/op_registry.h"
#include "insight/core/array.h"
#include <algorithm>
#include <vector>
#include <cstring>
#include <limits>
#include <numeric>

namespace ins::cpu {

    // ========== Helper functions ==========

    static inline std::vector<int64_t> compute_stride(const Shape& shape) {
        int ndim = shape.ndim();
        std::vector<int64_t> stride(ndim);
        if (ndim == 0) return stride;
        stride[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            stride[i] = stride[i + 1] * shape.dim(i + 1);
        }
        return stride;
    }

    static int64_t linear_index(const std::vector<int64_t>& coords,
        const std::vector<int64_t>& strides) {
        int64_t idx = 0;
        for (size_t i = 0; i < coords.size(); ++i) {
            idx += coords[i] * strides[i];
        }
        return idx;
    }

    template<typename T>
    static void fill_with_value(T* data, int64_t n, T val) {
        for (int64_t i = 0; i < n; ++i) {
            data[i] = val;
        }
    }

    // ========== take ==========

    template<typename T>
    static Array take_impl(const Array& out, const Array& x, const Array& indices,
        int orig_axis, bool has_axis) {
        T* dst = const_cast<T*>(out.data<T>());
        const T* src = x.data<T>();
        const int64_t* idx = indices.data<int64_t>();
        int64_t n = out.numel();

        if (!has_axis) {
            int64_t x_size = x.numel();
            for (int64_t i = 0; i < n; ++i) {
                int64_t pos = idx[i];
                if (pos < 0) pos += x_size;
                if (pos < 0 || pos >= x_size) {
                    INS_THROW("take: index ", pos, " out of bounds for size ", x_size);
                }
                dst[i] = src[pos];
            }
        }
        else {
            const Shape& x_shape = x.shape();
            const Shape& out_shape = out.shape();
            int ndim = x_shape.ndim();
            int axis = orig_axis;
            if (axis < 0) axis += ndim;

            std::vector<int64_t> x_stride = compute_stride(x_shape);
            std::vector<int64_t> out_stride = compute_stride(out_shape);

            std::vector<int64_t> coord(ndim, 0);
            int64_t total = out.numel();

            for (int64_t linear = 0; linear < total; ++linear) {
                int64_t tmp = linear;
                for (int d = ndim - 1; d >= 0; --d) {
                    coord[d] = tmp % out_shape.dim(d);
                    tmp /= out_shape.dim(d);
                }

                int64_t src_idx = 0;
                for (int d = 0; d < ndim; ++d) {
                    if (d == axis) {
                        int64_t pos = idx[coord[d]];
                        int64_t dim_size = x_shape.dim(d);
                        if (pos < 0) pos += dim_size;
                        if (pos < 0 || pos >= dim_size) {
                            INS_THROW("take: index ", pos, " out of bounds for dim ", d, " size ", dim_size);
                        }
                        src_idx += pos * x_stride[d];
                    }
                    else {
                        src_idx += coord[d] * x_stride[d];
                    }
                }
                dst[linear] = src[src_idx];
            }
        }
        return out;
    }

    static OpArgs take_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& x = std::any_cast<const Array&>(args[1]);
        const Array& indices = std::any_cast<const Array&>(args[2]);
        int orig_axis = std::any_cast<int>(args[3]);
        bool has_axis = std::any_cast<bool>(args[4]);

        Array& mutable_out = const_cast<Array&>(out);
        DType dtype = x.dtype();
        switch (dtype) {
        case DType::F32: return { take_impl<float>(mutable_out, x, indices, orig_axis, has_axis) };
        case DType::F64: return { take_impl<double>(mutable_out, x, indices, orig_axis, has_axis) };
        case DType::I32: return { take_impl<int32_t>(mutable_out, x, indices, orig_axis, has_axis) };
        case DType::I64: return { take_impl<int64_t>(mutable_out, x, indices, orig_axis, has_axis) };
        case DType::U8:  return { take_impl<uint8_t>(mutable_out, x, indices, orig_axis, has_axis) };
        case DType::BOOL:return { take_impl<bool>(mutable_out, x, indices, orig_axis, has_axis) };
        default: INS_THROW("take: unsupported dtype");
        }
    }

    REGISTER_KERNEL(take, CPU, F32, take_kernel);
    REGISTER_KERNEL(take, CPU, F64, take_kernel);
    REGISTER_KERNEL(take, CPU, I32, take_kernel);
    REGISTER_KERNEL(take, CPU, I64, take_kernel);
    REGISTER_KERNEL(take, CPU, U8, take_kernel);
    REGISTER_KERNEL(take, CPU, BOOL, take_kernel);

    // ========== take_along_axis ==========

    template<typename T>
    static Array take_along_axis_impl(const Array& out, const Array& x, const Array& indices, int axis) {
        T* dst = const_cast<T*>(out.data<T>());
        const T* src = x.data<T>();
        const int64_t* idx = indices.data<int64_t>();

        const Shape& x_shape = x.shape();
        const Shape& out_shape = out.shape();
        const Shape& idx_shape = indices.shape();

        int ndim = x_shape.ndim();
        if (axis < 0) axis += ndim;

        std::vector<int64_t> x_stride = compute_stride(x_shape);
        std::vector<int64_t> out_stride = compute_stride(out_shape);
        std::vector<int64_t> idx_stride = compute_stride(idx_shape);

        std::vector<int64_t> coord(ndim, 0);
        int64_t total = out.numel();

        for (int64_t linear = 0; linear < total; ++linear) {
            int64_t tmp = linear;
            for (int d = ndim - 1; d >= 0; --d) {
                coord[d] = tmp % out_shape.dim(d);
                tmp /= out_shape.dim(d);
            }

            // Compute index position in indices array (may be broadcasted)
            int64_t idx_pos = 0;
            int64_t idx_mult = 1;
            for (int d = ndim - 1; d >= 0; --d) {
                int64_t idx_coord = (coord[d] < idx_shape.dim(d)) ? coord[d] : 0;
                idx_pos += idx_coord * idx_mult;
                idx_mult *= idx_shape.dim(d);
            }

            int64_t pos = idx[idx_pos];
            int64_t dim_size = x_shape.dim(axis);
            if (pos < 0) pos += dim_size;
            if (pos < 0 || pos >= dim_size) {
                INS_THROW("take_along_axis: index ", pos, " out of bounds for dim ", axis);
            }

            int64_t src_idx = 0;
            for (int d = 0; d < ndim; ++d) {
                if (d == axis) {
                    src_idx += pos * x_stride[d];
                }
                else {
                    int64_t x_coord = coord[d];
                    src_idx += x_coord * x_stride[d];
                }
            }
            dst[linear] = src[src_idx];
        }
        return out;
    }

    static OpArgs take_along_axis_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& x = std::any_cast<const Array&>(args[1]);
        const Array& indices = std::any_cast<const Array&>(args[2]);
        int axis = std::any_cast<int>(args[3]);

        Array& mutable_out = const_cast<Array&>(out);
        DType dtype = x.dtype();
        switch (dtype) {
        case DType::F32: return { take_along_axis_impl<float>(mutable_out, x, indices, axis) };
        case DType::F64: return { take_along_axis_impl<double>(mutable_out, x, indices, axis) };
        case DType::I32: return { take_along_axis_impl<int32_t>(mutable_out, x, indices, axis) };
        case DType::I64: return { take_along_axis_impl<int64_t>(mutable_out, x, indices, axis) };
        case DType::U8:  return { take_along_axis_impl<uint8_t>(mutable_out, x, indices, axis) };
        case DType::BOOL:return { take_along_axis_impl<bool>(mutable_out, x, indices, axis) };
        default: INS_THROW("take_along_axis: unsupported dtype");
        }
    }

    REGISTER_KERNEL(take_along_axis, CPU, F32, take_along_axis_kernel);
    REGISTER_KERNEL(take_along_axis, CPU, F64, take_along_axis_kernel);
    REGISTER_KERNEL(take_along_axis, CPU, I32, take_along_axis_kernel);
    REGISTER_KERNEL(take_along_axis, CPU, I64, take_along_axis_kernel);
    REGISTER_KERNEL(take_along_axis, CPU, U8, take_along_axis_kernel);
    REGISTER_KERNEL(take_along_axis, CPU, BOOL, take_along_axis_kernel);

    // ========== put ==========

    template<typename T>
    static Array put_impl(const Array& out, const Array& indices, const Array& values) {
        T* dst = const_cast<T*>(out.data<T>());
        const int64_t* idx = indices.data<int64_t>();
        const T* val = values.data<T>();
        int64_t n = indices.numel();
        int64_t out_size = out.numel();
        int64_t val_size = values.numel();

        for (int64_t i = 0; i < n; ++i) {
            int64_t pos = idx[i];
            if (pos < 0) pos += out_size;
            if (pos < 0 || pos >= out_size) {
                INS_THROW("put: index ", pos, " out of bounds for size ", out_size);
            }
            dst[pos] = val[i % val_size];
        }
        return out;
    }

    static OpArgs put_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& indices = std::any_cast<const Array&>(args[1]);
        const Array& values = std::any_cast<const Array&>(args[2]);

        Array& mutable_out = const_cast<Array&>(out);
        DType dtype = out.dtype();
        switch (dtype) {
        case DType::F32: return { put_impl<float>(mutable_out, indices, values) };
        case DType::F64: return { put_impl<double>(mutable_out, indices, values) };
        case DType::I32: return { put_impl<int32_t>(mutable_out, indices, values) };
        case DType::I64: return { put_impl<int64_t>(mutable_out, indices, values) };
        case DType::U8:  return { put_impl<uint8_t>(mutable_out, indices, values) };
        case DType::BOOL:return { put_impl<bool>(mutable_out, indices, values) };
        default: INS_THROW("put: unsupported dtype");
        }
    }

    REGISTER_KERNEL(put, CPU, F32, put_kernel);
    REGISTER_KERNEL(put, CPU, F64, put_kernel);
    REGISTER_KERNEL(put, CPU, I32, put_kernel);
    REGISTER_KERNEL(put, CPU, I64, put_kernel);
    REGISTER_KERNEL(put, CPU, U8, put_kernel);
    REGISTER_KERNEL(put, CPU, BOOL, put_kernel);

    // ========== put_along_axis ==========

    template<typename T>
    static Array put_along_axis_impl(const Array& out, const Array& indices,
        const Array& values, int axis) {
        T* dst = const_cast<T*>(out.data<T>());
        const int64_t* idx = indices.data<int64_t>();
        const T* val = values.data<T>();

        const Shape& out_shape = out.shape();
        const Shape& idx_shape = indices.shape();
        int ndim = out_shape.ndim();
        if (axis < 0) axis += ndim;

        std::vector<int64_t> out_stride = compute_stride(out_shape);
        std::vector<int64_t> idx_stride = compute_stride(idx_shape);

        std::vector<int64_t> coord(ndim, 0);
        int64_t total = indices.numel();

        for (int64_t linear = 0; linear < total; ++linear) {
            // Decode coordinate from indices shape (this gives the output coordinate
            // except on the axis dimension)
            int64_t tmp = linear;
            for (int d = ndim - 1; d >= 0; --d) {
                coord[d] = tmp % idx_shape.dim(d);
                tmp /= idx_shape.dim(d);
            }

            int64_t pos = idx[linear];
            if (pos < 0) pos += out_shape.dim(axis);
            if (pos < 0 || pos >= out_shape.dim(axis)) {
                INS_THROW("put_along_axis: index ", pos, " out of bounds for dim ", axis);
            }

            // Build output index: use pos for axis, original coord for others
            int64_t out_idx = 0;
            for (int d = 0; d < ndim; ++d) {
                if (d == axis) {
                    out_idx += pos * out_stride[d];
                }
                else {
                    out_idx += coord[d] * out_stride[d];
                }
            }
            dst[out_idx] = val[linear % values.numel()];
        }
        return out;
    }

    static OpArgs put_along_axis_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& indices = std::any_cast<const Array&>(args[1]);
        const Array& values = std::any_cast<const Array&>(args[2]);
        int axis = std::any_cast<int>(args[3]);

        Array& mutable_out = const_cast<Array&>(out);
        DType dtype = out.dtype();
        switch (dtype) {
        case DType::F32: return { put_along_axis_impl<float>(mutable_out, indices, values, axis) };
        case DType::F64: return { put_along_axis_impl<double>(mutable_out, indices, values, axis) };
        case DType::I32: return { put_along_axis_impl<int32_t>(mutable_out, indices, values, axis) };
        case DType::I64: return { put_along_axis_impl<int64_t>(mutable_out, indices, values, axis) };
        case DType::U8:  return { put_along_axis_impl<uint8_t>(mutable_out, indices, values, axis) };
        case DType::BOOL:return { put_along_axis_impl<bool>(mutable_out, indices, values, axis) };
        default: INS_THROW("put_along_axis: unsupported dtype");
        }
    }

    REGISTER_KERNEL(put_along_axis, CPU, F32, put_along_axis_kernel);
    REGISTER_KERNEL(put_along_axis, CPU, F64, put_along_axis_kernel);
    REGISTER_KERNEL(put_along_axis, CPU, I32, put_along_axis_kernel);
    REGISTER_KERNEL(put_along_axis, CPU, I64, put_along_axis_kernel);
    REGISTER_KERNEL(put_along_axis, CPU, U8, put_along_axis_kernel);
    REGISTER_KERNEL(put_along_axis, CPU, BOOL, put_along_axis_kernel);

    // ========== scatter_reduce ==========

    template<typename T>
    static Array scatter_reduce_impl(const Array& out, const Array& indices,
        const Array& src, int dim, const std::string& reduce) {
        T* dst = const_cast<T*>(out.data<T>());
        const int64_t* idx = indices.data<int64_t>();
        const T* val = src.data<T>();

        const Shape& out_shape = out.shape();
        const Shape& idx_shape = indices.shape();
        int ndim = out_shape.ndim();
        if (dim < 0) dim += ndim;

        std::vector<int64_t> out_stride = compute_stride(out_shape);
        std::vector<int64_t> idx_stride = compute_stride(idx_shape);

        std::vector<int64_t> coord(ndim, 0);
        int64_t total = indices.numel();

        for (int64_t linear = 0; linear < total; ++linear) {
            int64_t tmp = linear;
            for (int d = ndim - 1; d >= 0; --d) {
                coord[d] = tmp % idx_shape.dim(d);
                tmp /= idx_shape.dim(d);
            }

            int64_t pos = idx[linear];
            if (pos < 0) pos += out_shape.dim(dim);
            if (pos < 0 || pos >= out_shape.dim(dim)) {
                INS_THROW("scatter_reduce: index ", pos, " out of bounds for dim ", dim);
            }

            int64_t out_idx = 0;
            for (int d = 0; d < ndim; ++d) {
                if (d == dim) {
                    out_idx += pos * out_stride[d];
                }
                else {
                    out_idx += coord[d] * out_stride[d];
                }
            }

            T val_cur = val[linear % src.numel()];
            if (reduce == "replace") {
                dst[out_idx] = val_cur;
            }
            else if (reduce == "add") {
                dst[out_idx] += val_cur;
            }
            else if (reduce == "mul") {
                dst[out_idx] *= val_cur;
            }
            else if (reduce == "max") {
                if (val_cur > dst[out_idx]) dst[out_idx] = val_cur;
            }
            else if (reduce == "min") {
                if (val_cur < dst[out_idx]) dst[out_idx] = val_cur;
            }
            else {
                INS_THROW("scatter_reduce: unknown reduce mode: ", reduce);
            }
        }
        return out;
    }

    static OpArgs scatter_reduce_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& indices = std::any_cast<const Array&>(args[1]);
        const Array& src = std::any_cast<const Array&>(args[2]);
        int dim = std::any_cast<int>(args[3]);
        std::string reduce = std::any_cast<std::string>(args[4]);

        Array& mutable_out = const_cast<Array&>(out);
        DType dtype = out.dtype();
        switch (dtype) {
        case DType::F32: return { scatter_reduce_impl<float>(mutable_out, indices, src, dim, reduce) };
        case DType::F64: return { scatter_reduce_impl<double>(mutable_out, indices, src, dim, reduce) };
        case DType::I32: return { scatter_reduce_impl<int32_t>(mutable_out, indices, src, dim, reduce) };
        case DType::I64: return { scatter_reduce_impl<int64_t>(mutable_out, indices, src, dim, reduce) };
        case DType::U8:  return { scatter_reduce_impl<uint8_t>(mutable_out, indices, src, dim, reduce) };
        case DType::BOOL:return { scatter_reduce_impl<bool>(mutable_out, indices, src, dim, reduce) };
        default: INS_THROW("scatter_reduce: unsupported dtype");
        }
    }

    REGISTER_KERNEL(scatter_reduce, CPU, F32, scatter_reduce_kernel);
    REGISTER_KERNEL(scatter_reduce, CPU, F64, scatter_reduce_kernel);
    REGISTER_KERNEL(scatter_reduce, CPU, I32, scatter_reduce_kernel);
    REGISTER_KERNEL(scatter_reduce, CPU, I64, scatter_reduce_kernel);
    REGISTER_KERNEL(scatter_reduce, CPU, U8, scatter_reduce_kernel);
    REGISTER_KERNEL(scatter_reduce, CPU, BOOL, scatter_reduce_kernel);

    // ========== masked_select ==========

    template<typename T>
    static Array masked_select_impl(const Array& out, const Array& x, const Array& mask) {
        T* dst = const_cast<T*>(out.data<T>());
        const T* src = x.data<T>();
        const bool* msk = mask.data<bool>();
        int64_t total = x.numel();
        int64_t out_idx = 0;

        // For simplicity, flatten and iterate
        // In production, handle strides
        if (x.is_contiguous() && mask.is_contiguous()) {
            const T* src_flat = src;
            const bool* msk_flat = msk;
            for (int64_t i = 0; i < total; ++i) {
                if (msk_flat[i]) {
                    dst[out_idx++] = src_flat[i];
                }
            }
        }
        else {
            // Generic strided access
            const Shape& shape = x.shape();
            int ndim = shape.ndim();
            std::vector<int64_t> strides_x = compute_stride(shape);
            std::vector<int64_t> strides_m = compute_stride(mask.shape());
            std::vector<int64_t> coord(ndim, 0);
            for (int64_t linear = 0; linear < total; ++linear) {
                int64_t tmp = linear;
                for (int d = ndim - 1; d >= 0; --d) {
                    coord[d] = tmp % shape.dim(d);
                    tmp /= shape.dim(d);
                }
                int64_t x_idx = linear_index(coord, strides_x);
                int64_t m_idx = linear_index(coord, strides_m);
                if (msk[m_idx]) {
                    dst[out_idx++] = src[x_idx];
                }
            }
        }
        return out;
    }

    static OpArgs masked_select_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& x = std::any_cast<const Array&>(args[1]);
        const Array& mask = std::any_cast<const Array&>(args[2]);

        Array& mutable_out = const_cast<Array&>(out);
        DType dtype = x.dtype();
        switch (dtype) {
        case DType::F32: return { masked_select_impl<float>(mutable_out, x, mask) };
        case DType::F64: return { masked_select_impl<double>(mutable_out, x, mask) };
        case DType::I32: return { masked_select_impl<int32_t>(mutable_out, x, mask) };
        case DType::I64: return { masked_select_impl<int64_t>(mutable_out, x, mask) };
        case DType::U8:  return { masked_select_impl<uint8_t>(mutable_out, x, mask) };
        case DType::BOOL:return { masked_select_impl<bool>(mutable_out, x, mask) };
        default: INS_THROW("masked_select: unsupported dtype");
        }
    }

    REGISTER_KERNEL(masked_select, CPU, F32, masked_select_kernel);
    REGISTER_KERNEL(masked_select, CPU, F64, masked_select_kernel);
    REGISTER_KERNEL(masked_select, CPU, I32, masked_select_kernel);
    REGISTER_KERNEL(masked_select, CPU, I64, masked_select_kernel);
    REGISTER_KERNEL(masked_select, CPU, U8, masked_select_kernel);
    REGISTER_KERNEL(masked_select, CPU, BOOL, masked_select_kernel);

    // ========== compress ==========

    template<typename T>
    static Array compress_impl(const Array& out, const Array& x,
        const Array& condition, int axis) {
        T* dst = const_cast<T*>(out.data<T>());
        const T* src = x.data<T>();
        const bool* cond = condition.data<bool>();

        const Shape& x_shape = x.shape();
        const Shape& out_shape = out.shape();
        int ndim = x_shape.ndim();
        if (axis < 0) axis += ndim;

        std::vector<int64_t> x_stride = compute_stride(x_shape);
        std::vector<int64_t> out_stride = compute_stride(out_shape);

        int64_t axis_dim = x_shape.dim(axis);
        int64_t keep_count = out_shape.dim(axis);

        std::vector<int64_t> coord(ndim, 0);
        int64_t total = out.numel();

        for (int64_t linear = 0; linear < total; ++linear) {
            int64_t tmp = linear;
            for (int d = ndim - 1; d >= 0; --d) {
                coord[d] = tmp % out_shape.dim(d);
                tmp /= out_shape.dim(d);
            }

            // Find the original axis index based on compression
            int64_t orig_axis_idx = 0;
            int64_t cond_idx = 0;
            for (int64_t i = 0; i < axis_dim; ++i) {
                if (cond[i]) {
                    if (cond_idx == coord[axis]) {
                        orig_axis_idx = i;
                        break;
                    }
                    cond_idx++;
                }
            }

            int64_t src_idx = 0;
            for (int d = 0; d < ndim; ++d) {
                if (d == axis) {
                    src_idx += orig_axis_idx * x_stride[d];
                }
                else {
                    src_idx += coord[d] * x_stride[d];
                }
            }
            dst[linear] = src[src_idx];
        }
        return out;
    }

    static OpArgs compress_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& x = std::any_cast<const Array&>(args[1]);
        const Array& condition = std::any_cast<const Array&>(args[2]);
        int axis = std::any_cast<int>(args[3]);

        Array& mutable_out = const_cast<Array&>(out);
        DType dtype = x.dtype();
        switch (dtype) {
        case DType::F32: return { compress_impl<float>(mutable_out, x, condition, axis) };
        case DType::F64: return { compress_impl<double>(mutable_out, x, condition, axis) };
        case DType::I32: return { compress_impl<int32_t>(mutable_out, x, condition, axis) };
        case DType::I64: return { compress_impl<int64_t>(mutable_out, x, condition, axis) };
        case DType::U8:  return { compress_impl<uint8_t>(mutable_out, x, condition, axis) };
        case DType::BOOL:return { compress_impl<bool>(mutable_out, x, condition, axis) };
        default: INS_THROW("compress: unsupported dtype");
        }
    }

    REGISTER_KERNEL(compress, CPU, F32, compress_kernel);
    REGISTER_KERNEL(compress, CPU, F64, compress_kernel);
    REGISTER_KERNEL(compress, CPU, I32, compress_kernel);
    REGISTER_KERNEL(compress, CPU, I64, compress_kernel);
    REGISTER_KERNEL(compress, CPU, U8, compress_kernel);
    REGISTER_KERNEL(compress, CPU, BOOL, compress_kernel);

    // ========== where ==========

    template<typename T>
    static Array where_impl(const Array& out, const Array& condition,
        const Array& x, const Array& y) {
        T* dst = const_cast<T*>(out.data<T>());
        const bool* cond = condition.data<bool>();
        const T* x_data = x.data<T>();
        const T* y_data = y.data<T>();
        int64_t n = out.numel();

        if (condition.is_contiguous() && x.is_contiguous() && y.is_contiguous()) {
            for (int64_t i = 0; i < n; ++i) {
                dst[i] = cond[i] ? x_data[i] : y_data[i];
            }
        }
        else {
            // Strided version - assume broadcasting done, all same shape
            const Shape& shape = out.shape();
            int ndim = shape.ndim();
            std::vector<int64_t> stride_c = compute_stride(condition.shape());
            std::vector<int64_t> stride_x = compute_stride(x.shape());
            std::vector<int64_t> stride_y = compute_stride(y.shape());
            std::vector<int64_t> stride_out = compute_stride(shape);
            std::vector<int64_t> coord(ndim, 0);
            for (int64_t linear = 0; linear < n; ++linear) {
                int64_t tmp = linear;
                for (int d = ndim - 1; d >= 0; --d) {
                    coord[d] = tmp % shape.dim(d);
                    tmp /= shape.dim(d);
                }
                int64_t c_idx = linear_index(coord, stride_c);
                int64_t x_idx = linear_index(coord, stride_x);
                int64_t y_idx = linear_index(coord, stride_y);
                dst[linear] = cond[c_idx] ? x_data[x_idx] : y_data[y_idx];
            }
        }
        return out;
    }

    static OpArgs where_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& condition = std::any_cast<const Array&>(args[1]);
        const Array& x = std::any_cast<const Array&>(args[2]);
        const Array& y = std::any_cast<const Array&>(args[3]);

        Array& mutable_out = const_cast<Array&>(out);
        DType dtype = x.dtype();
        switch (dtype) {
        case DType::F32: return { where_impl<float>(mutable_out, condition, x, y) };
        case DType::F64: return { where_impl<double>(mutable_out, condition, x, y) };
        case DType::I32: return { where_impl<int32_t>(mutable_out, condition, x, y) };
        case DType::I64: return { where_impl<int64_t>(mutable_out, condition, x, y) };
        case DType::U8:  return { where_impl<uint8_t>(mutable_out, condition, x, y) };
        case DType::BOOL:return { where_impl<bool>(mutable_out, condition, x, y) };
        default: INS_THROW("where: unsupported dtype");
        }
    }

    REGISTER_KERNEL(where, CPU, F32, where_kernel);
    REGISTER_KERNEL(where, CPU, F64, where_kernel);
    REGISTER_KERNEL(where, CPU, I32, where_kernel);
    REGISTER_KERNEL(where, CPU, I64, where_kernel);
    REGISTER_KERNEL(where, CPU, U8, where_kernel);
    REGISTER_KERNEL(where, CPU, BOOL, where_kernel);

    // ========== nonzero ==========

    template<typename T>
    static Array nonzero_impl(const Array& x) {
        const T* data = x.data<T>();
        const Shape& shape = x.shape();
        int ndim = shape.ndim();
        int64_t total = x.numel();

        // First pass: count non-zero elements
        std::vector<int64_t> count_per_dim(ndim, 0);
        int64_t nz_count = 0;

        if (x.is_contiguous()) {
            for (int64_t i = 0; i < total; ++i) {
                if (data[i] != T(0)) {
                    nz_count++;
                }
            }
        }
        else {
            std::vector<int64_t> strides = compute_stride(shape);
            std::vector<int64_t> coord(ndim, 0);
            for (int64_t linear = 0; linear < total; ++linear) {
                int64_t tmp = linear;
                for (int d = ndim - 1; d >= 0; --d) {
                    coord[d] = tmp % shape.dim(d);
                    tmp /= shape.dim(d);
                }
                int64_t idx = linear_index(coord, strides);
                if (data[idx] != T(0)) {
                    nz_count++;
                }
            }
        }

        if (nz_count == 0) {
            // Return empty array of shape (ndim, 0)
            return Array(Shape({ ndim, 0 }), DType::I64, x.place());
        }

        // Second pass: fill indices
        Array result(Shape({ ndim, nz_count }), DType::I64, x.place());
        int64_t* result_data = result.data<int64_t>();
        int64_t out_idx = 0;

        if (x.is_contiguous()) {
            for (int64_t linear = 0; linear < total; ++linear) {
                if (data[linear] != T(0)) {
                    int64_t tmp = linear;
                    for (int d = ndim - 1; d >= 0; --d) {
                        result_data[d * nz_count + out_idx] = tmp % shape.dim(d);
                        tmp /= shape.dim(d);
                    }
                    out_idx++;
                }
            }
        }
        else {
            std::vector<int64_t> strides = compute_stride(shape);
            std::vector<int64_t> coord(ndim, 0);
            for (int64_t linear = 0; linear < total; ++linear) {
                int64_t tmp = linear;
                for (int d = ndim - 1; d >= 0; --d) {
                    coord[d] = tmp % shape.dim(d);
                    tmp /= shape.dim(d);
                }
                int64_t idx = linear_index(coord, strides);
                if (data[idx] != T(0)) {
                    for (int d = 0; d < ndim; ++d) {
                        result_data[d * nz_count + out_idx] = coord[d];
                    }
                    out_idx++;
                }
            }
        }
        return result;
    }

    static OpArgs nonzero_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        DType dtype = x.dtype();
        switch (dtype) {
        case DType::F32: return { nonzero_impl<float>(x) };
        case DType::F64: return { nonzero_impl<double>(x) };
        case DType::I32: return { nonzero_impl<int32_t>(x) };
        case DType::I64: return { nonzero_impl<int64_t>(x) };
        case DType::U8:  return { nonzero_impl<uint8_t>(x) };
        case DType::BOOL:return { nonzero_impl<bool>(x) };
        default: INS_THROW("nonzero: unsupported dtype");
        }
    }

    REGISTER_KERNEL(nonzero, CPU, F32, nonzero_kernel);
    REGISTER_KERNEL(nonzero, CPU, F64, nonzero_kernel);
    REGISTER_KERNEL(nonzero, CPU, I32, nonzero_kernel);
    REGISTER_KERNEL(nonzero, CPU, I64, nonzero_kernel);
    REGISTER_KERNEL(nonzero, CPU, U8, nonzero_kernel);
    REGISTER_KERNEL(nonzero, CPU, BOOL, nonzero_kernel);

    // ========== argsort ==========

    template<typename T>
    static Array argsort_impl(const Array& out, const Array& x, bool descending) {
        int64_t* dst = const_cast<int64_t*>(out.data<int64_t>());
        const T* src = x.data<T>();
        const Shape& shape = x.shape();
        int ndim = shape.ndim();
        int64_t last_dim = shape.dim(ndim - 1);
        int64_t batch_size = out.numel() / last_dim;

        for (int64_t batch = 0; batch < batch_size; ++batch) {
            // Build index-value pairs
            std::vector<std::pair<T, int64_t>> pairs(last_dim);
            for (int64_t i = 0; i < last_dim; ++i) {
                pairs[i] = { src[batch * last_dim + i], i };
            }
            if (descending) {
                std::sort(pairs.begin(), pairs.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
            }
            else {
                std::sort(pairs.begin(), pairs.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });
            }
            for (int64_t i = 0; i < last_dim; ++i) {
                dst[batch * last_dim + i] = pairs[i].second;
            }
        }
        return out;
    }

    static OpArgs argsort_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& x = std::any_cast<const Array&>(args[1]);
        bool descending = std::any_cast<bool>(args[2]);

        Array& mutable_out = const_cast<Array&>(out);
        DType dtype = x.dtype();
        switch (dtype) {
        case DType::F32: return { argsort_impl<float>(mutable_out, x, descending) };
        case DType::F64: return { argsort_impl<double>(mutable_out, x, descending) };
        case DType::I32: return { argsort_impl<int32_t>(mutable_out, x, descending) };
        case DType::I64: return { argsort_impl<int64_t>(mutable_out, x, descending) };
        case DType::U8:  return { argsort_impl<uint8_t>(mutable_out, x, descending) };
        case DType::BOOL:return { argsort_impl<bool>(mutable_out, x, descending) };
        default: INS_THROW("argsort: unsupported dtype");
        }
    }

    REGISTER_KERNEL(argsort, CPU, F32, argsort_kernel);
    REGISTER_KERNEL(argsort, CPU, F64, argsort_kernel);
    REGISTER_KERNEL(argsort, CPU, I32, argsort_kernel);
    REGISTER_KERNEL(argsort, CPU, I64, argsort_kernel);
    REGISTER_KERNEL(argsort, CPU, U8, argsort_kernel);
    REGISTER_KERNEL(argsort, CPU, BOOL, argsort_kernel);

    // ========== topk ==========

    template<typename T>
    static Array topk_values_impl(const Array& out, const Array& x, int64_t k, bool largest, bool sorted) {
        T* dst = const_cast<T*>(out.data<T>());
        const T* src = x.data<T>();
        const Shape& shape = x.shape();
        int ndim = shape.ndim();
        int64_t last_dim = shape.dim(ndim - 1);
        int64_t batch_size = out.numel() / k;

        for (int64_t batch = 0; batch < batch_size; ++batch) {
            std::vector<std::pair<T, int64_t>> pairs(last_dim);
            for (int64_t i = 0; i < last_dim; ++i) {
                pairs[i] = { src[batch * last_dim + i], i };
            }
            if (largest) {
                std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
            }
            else {
                std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });
            }
            for (int64_t i = 0; i < k; ++i) {
                dst[batch * k + i] = pairs[i].first;
            }
            if (sorted && !largest) {
                // Already sorted
            }
        }
        return out;
    }

    template<typename T>
    static Array topk_indices_impl(const Array& out, const Array& x, int64_t k, bool largest, bool sorted) {
        int64_t* dst = const_cast<int64_t*>(out.data<int64_t>());
        const T* src = x.data<T>();
        const Shape& shape = x.shape();
        int ndim = shape.ndim();
        int64_t last_dim = shape.dim(ndim - 1);
        int64_t batch_size = out.numel() / k;

        for (int64_t batch = 0; batch < batch_size; ++batch) {
            std::vector<std::pair<T, int64_t>> pairs(last_dim);
            for (int64_t i = 0; i < last_dim; ++i) {
                pairs[i] = { src[batch * last_dim + i], i };
            }
            if (largest) {
                std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
            }
            else {
                std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });
            }
            for (int64_t i = 0; i < k; ++i) {
                dst[batch * k + i] = pairs[i].second;
            }
        }
        return out;
    }

    static OpArgs topk_kernel(const OpArgs& args) {
        const Array& values = std::any_cast<const Array&>(args[0]);
        const Array& indices = std::any_cast<const Array&>(args[1]);
        const Array& x = std::any_cast<const Array&>(args[2]);
        int64_t k = std::any_cast<int64_t>(args[3]);
        bool largest = std::any_cast<bool>(args[4]);
        bool sorted = std::any_cast<bool>(args[5]);

        Array& mutable_vals = const_cast<Array&>(values);
        Array& mutable_idxs = const_cast<Array&>(indices);
        DType dtype = x.dtype();
        switch (dtype) {
        case DType::F32:
            topk_values_impl<float>(mutable_vals, x, k, largest, sorted);
            topk_indices_impl<float>(mutable_idxs, x, k, largest, sorted);
            return { mutable_vals, mutable_idxs };
        case DType::F64:
            topk_values_impl<double>(mutable_vals, x, k, largest, sorted);
            topk_indices_impl<double>(mutable_idxs, x, k, largest, sorted);
            return { mutable_vals, mutable_idxs };
        case DType::I32:
            topk_values_impl<int32_t>(mutable_vals, x, k, largest, sorted);
            topk_indices_impl<int32_t>(mutable_idxs, x, k, largest, sorted);
            return { mutable_vals, mutable_idxs };
        case DType::I64:
            topk_values_impl<int64_t>(mutable_vals, x, k, largest, sorted);
            topk_indices_impl<int64_t>(mutable_idxs, x, k, largest, sorted);
            return { mutable_vals, mutable_idxs };
        case DType::U8:
            topk_values_impl<uint8_t>(mutable_vals, x, k, largest, sorted);
            topk_indices_impl<uint8_t>(mutable_idxs, x, k, largest, sorted);
            return { mutable_vals, mutable_idxs };
        case DType::BOOL:
            topk_values_impl<bool>(mutable_vals, x, k, largest, sorted);
            topk_indices_impl<bool>(mutable_idxs, x, k, largest, sorted);
            return { mutable_vals, mutable_idxs };
        default: INS_THROW("topk: unsupported dtype");
        }
    }

    REGISTER_KERNEL(topk, CPU, F32, topk_kernel);
    REGISTER_KERNEL(topk, CPU, F64, topk_kernel);
    REGISTER_KERNEL(topk, CPU, I32, topk_kernel);
    REGISTER_KERNEL(topk, CPU, I64, topk_kernel);
    REGISTER_KERNEL(topk, CPU, U8, topk_kernel);
    REGISTER_KERNEL(topk, CPU, BOOL, topk_kernel);

    // ========== searchsorted ==========

    template<typename T>
    static Array searchsorted_impl(const Array& out, const Array& x,
        const Array& v, const std::string& side) {
        int64_t* dst = const_cast<int64_t*>(out.data<int64_t>());
        const T* src = x.data<T>();
        const T* vals = v.data<T>();
        int64_t n = v.numel();
        int64_t m = x.numel();

        for (int64_t i = 0; i < n; ++i) {
            T val = vals[i];
            int64_t lo = 0, hi = m;
            while (lo < hi) {
                int64_t mid = (lo + hi) / 2;
                if (side == "left") {
                    if (src[mid] < val) {
                        lo = mid + 1;
                    }
                    else {
                        hi = mid;
                    }
                }
                else {
                    if (src[mid] <= val) {
                        lo = mid + 1;
                    }
                    else {
                        hi = mid;
                    }
                }
            }
            dst[i] = lo;
        }
        return out;
    }

    static OpArgs searchsorted_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& x = std::any_cast<const Array&>(args[1]);
        const Array& v = std::any_cast<const Array&>(args[2]);
        std::string side = std::any_cast<std::string>(args[3]);

        Array& mutable_out = const_cast<Array&>(out);
        DType dtype = x.dtype();
        switch (dtype) {
        case DType::F32: return { searchsorted_impl<float>(mutable_out, x, v, side) };
        case DType::F64: return { searchsorted_impl<double>(mutable_out, x, v, side) };
        case DType::I32: return { searchsorted_impl<int32_t>(mutable_out, x, v, side) };
        case DType::I64: return { searchsorted_impl<int64_t>(mutable_out, x, v, side) };
        case DType::U8:  return { searchsorted_impl<uint8_t>(mutable_out, x, v, side) };
        default: INS_THROW("searchsorted: unsupported dtype");
        }
    }

    REGISTER_KERNEL(searchsorted, CPU, F32, searchsorted_kernel);
    REGISTER_KERNEL(searchsorted, CPU, F64, searchsorted_kernel);
    REGISTER_KERNEL(searchsorted, CPU, I32, searchsorted_kernel);
    REGISTER_KERNEL(searchsorted, CPU, I64, searchsorted_kernel);
    REGISTER_KERNEL(searchsorted, CPU, U8, searchsorted_kernel);

    // ========== unique ==========

    template<typename T>
    static std::vector<Array> unique_impl(const Array& flattened, bool return_indices,
        bool return_inverse, bool return_counts) {
        const T* data = flattened.data<T>();
        int64_t n = flattened.numel();

        // Build index-value pairs
        std::vector<std::pair<T, int64_t>> pairs(n);
        for (int64_t i = 0; i < n; ++i) {
            pairs[i] = { data[i], i };
        }
        std::sort(pairs.begin(), pairs.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

        // Extract unique elements and first occurrence indices
        std::vector<T> unique_vals;
        std::vector<int64_t> first_occurrence;
        std::vector<int64_t> counts;

        for (int64_t i = 0; i < n; ++i) {
            if (i == 0 || pairs[i].first != pairs[i - 1].first) {
                unique_vals.push_back(pairs[i].first);
                first_occurrence.push_back(pairs[i].second);
                counts.push_back(1);
            }
            else {
                counts.back()++;
            }
        }

        int64_t nu = unique_vals.size();

        // Build inverse indices
        std::vector<int64_t> inverse(n);
        int64_t unique_idx = 0;
        for (int64_t i = 0; i < n; ++i) {
            if (i > 0 && pairs[i].first != pairs[i - 1].first) {
                unique_idx++;
            }
            inverse[pairs[i].second] = unique_idx;
        }

        std::vector<Array> results;

        // Unique array
        Array unique(Shape({ nu }), flattened.dtype(), flattened.place());
        T* unique_data = unique.data<T>();
        for (int64_t i = 0; i < nu; ++i) {
            unique_data[i] = unique_vals[i];
        }
        results.push_back(unique);

        // Indices (first occurrence)
        if (return_indices) {
            Array indices(Shape({ nu }), DType::I64, flattened.place());
            int64_t* idx_data = indices.data<int64_t>();
            for (int64_t i = 0; i < nu; ++i) {
                idx_data[i] = first_occurrence[i];
            }
            results.push_back(indices);
        }

        // Inverse
        if (return_inverse) {
            Array inv(Shape({ n }), DType::I64, flattened.place());
            int64_t* inv_data = inv.data<int64_t>();
            for (int64_t i = 0; i < n; ++i) {
                inv_data[i] = inverse[i];
            }
            results.push_back(inv);
        }

        // Counts
        if (return_counts) {
            Array cnt(Shape({ nu }), DType::I64, flattened.place());
            int64_t* cnt_data = cnt.data<int64_t>();
            for (int64_t i = 0; i < nu; ++i) {
                cnt_data[i] = counts[i];
            }
            results.push_back(cnt);
        }

        return results;
    }

    static OpArgs unique_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        bool return_indices = std::any_cast<bool>(args[1]);
        bool return_inverse = std::any_cast<bool>(args[2]);
        bool return_counts = std::any_cast<bool>(args[3]);

        DType dtype = x.dtype();
        std::vector<Array> results;

        switch (dtype) {
        case DType::F32: results = unique_impl<float>(x, return_indices, return_inverse, return_counts); break;
        case DType::F64: results = unique_impl<double>(x, return_indices, return_inverse, return_counts); break;
        case DType::I32: results = unique_impl<int32_t>(x, return_indices, return_inverse, return_counts); break;
        case DType::I64: results = unique_impl<int64_t>(x, return_indices, return_inverse, return_counts); break;
        case DType::U8:  results = unique_impl<uint8_t>(x, return_indices, return_inverse, return_counts); break;
        case DType::BOOL:results = unique_impl<bool>(x, return_indices, return_inverse, return_counts); break;
        default: INS_THROW("unique: unsupported dtype");
        }

        OpArgs output;
        for (const auto& arr : results) {
            output.push_back(arr);
        }
        return output;
    }

    REGISTER_KERNEL(unique, CPU, F32, unique_kernel);
    REGISTER_KERNEL(unique, CPU, F64, unique_kernel);
    REGISTER_KERNEL(unique, CPU, I32, unique_kernel);
    REGISTER_KERNEL(unique, CPU, I64, unique_kernel);
    REGISTER_KERNEL(unique, CPU, U8, unique_kernel);
    REGISTER_KERNEL(unique, CPU, BOOL, unique_kernel);

    // ========== lexsort ==========
    template<typename T>
    static Array lexsort_impl(const Array& out, const Array& transposed,
        int64_t batch_size, int64_t last_dim, int64_t nkeys) {
        int64_t* dst = const_cast<int64_t*>(out.data<int64_t>());
        const T* src = transposed.data<T>();

        // Memory layout after transpose: 
        // [batch][key0][element0...elementN][key1][element0...elementN]...
        // Actually, the data is stored in row-major order: 
        // For a 3D array (nkeys, batch, last_dim) after flattening...

        for (int64_t batch = 0; batch < batch_size; ++batch) {
            // 收集该 batch 的所有键值
            std::vector<std::vector<T>> batch_keys(nkeys);
            for (int64_t k = 0; k < nkeys; ++k) {
                batch_keys[k].resize(last_dim);
                const T* key_start = src + (batch * nkeys + k) * last_dim;
                for (int64_t i = 0; i < last_dim; ++i) {
                    batch_keys[k][i] = key_start[i];
                }
            }

            // 构建索引数组
            std::vector<int64_t> indices(last_dim);
            for (int64_t i = 0; i < last_dim; ++i) indices[i] = i;

            // 多键排序
            std::sort(indices.begin(), indices.end(),
                [&](int64_t a, int64_t b) {
                    for (int64_t k = 0; k < nkeys; ++k) {
                        if (batch_keys[k][a] < batch_keys[k][b]) return true;
                        if (batch_keys[k][a] > batch_keys[k][b]) return false;
                    }
                    return false;
                });

            // 写入输出
            int64_t* batch_dst = dst + batch * last_dim;
            for (int64_t i = 0; i < last_dim; ++i) {
                batch_dst[i] = indices[i];
            }
        }

        return out;
    }

    static OpArgs lexsort_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& transposed = std::any_cast<const Array&>(args[1]);
        int64_t batch_size = std::any_cast<int64_t>(args[2]);
        int64_t last_dim = std::any_cast<int64_t>(args[3]);
        int64_t nkeys = std::any_cast<int64_t>(args[4]);

        Array& mutable_out = const_cast<Array&>(out);
        DType dtype = transposed.dtype();
        switch (dtype) {
        case DType::F32: return { lexsort_impl<float>(mutable_out, transposed, batch_size, last_dim, nkeys) };
        case DType::F64: return { lexsort_impl<double>(mutable_out, transposed, batch_size, last_dim, nkeys) };
        case DType::I32: return { lexsort_impl<int32_t>(mutable_out, transposed, batch_size, last_dim, nkeys) };
        case DType::I64: return { lexsort_impl<int64_t>(mutable_out, transposed, batch_size, last_dim, nkeys) };
        case DType::U8:  return { lexsort_impl<uint8_t>(mutable_out, transposed, batch_size, last_dim, nkeys) };
        default: INS_THROW("lexsort: unsupported dtype");
        }
    }

    REGISTER_KERNEL(lexsort, CPU, F32, lexsort_kernel);
    REGISTER_KERNEL(lexsort, CPU, F64, lexsort_kernel);
    REGISTER_KERNEL(lexsort, CPU, I32, lexsort_kernel);
    REGISTER_KERNEL(lexsort, CPU, I64, lexsort_kernel);
    REGISTER_KERNEL(lexsort, CPU, U8, lexsort_kernel);

    // ========== indices ==========

    static Array indices_impl(const Array& out, const Shape& shape) {
        int64_t* dst = const_cast<int64_t*>(out.data<int64_t>());
        int ndim = shape.ndim();
        int64_t total = shape.numel();

        std::vector<int64_t> dims = shape.dims();
        std::vector<int64_t> strides(ndim);
        if (ndim > 0) {
            strides[ndim - 1] = 1;
            for (int i = ndim - 2; i >= 0; --i) {
                strides[i] = strides[i + 1] * dims[i + 1];
            }
        }

        for (int d = 0; d < ndim; ++d) {
            int64_t* plane = dst + d * total;
            std::vector<int64_t> coord(ndim, 0);
            for (int64_t linear = 0; linear < total; ++linear) {
                int64_t tmp = linear;
                for (int i = ndim - 1; i >= 0; --i) {
                    coord[i] = tmp % dims[i];
                    tmp /= dims[i];
                }
                plane[linear] = coord[d];
            }
        }
        return out;
    }

    static OpArgs indices_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Shape& shape = std::any_cast<const Shape&>(args[1]);

        Array& mutable_out = const_cast<Array&>(out);
        return { indices_impl(mutable_out, shape) };
    }

    REGISTER_KERNEL(indices, CPU, I64, indices_kernel);

    // ========== partition ==========

    template<typename T>
    static Array partition_impl(const Array& out, const Array& x, int64_t kth, int axis) {
        T* dst = const_cast<T*>(out.data<T>());
        const T* src = x.data<T>();
        const Shape& shape = x.shape();
        int ndim = shape.ndim();
        if (axis < 0) axis += ndim;

        // Move axis to last dimension
        std::vector<int> perm(ndim);
        for (int i = 0; i < ndim; ++i) perm[i] = i;
        if (axis != ndim - 1) {
            std::swap(perm[axis], perm[ndim - 1]);
        }

        // For simplicity, handle 1D case first
        if (ndim == 1) {
            std::vector<T> data(src, src + x.numel());
            std::nth_element(data.begin(), data.begin() + kth, data.end());
            for (int64_t i = 0; i < x.numel(); ++i) {
                dst[i] = data[i];
            }
            return out;
        }

        // For multi-dim, need to process each slice
        // Calculate batch size and last dim size
        std::vector<int64_t> dims(ndim);
        for (int i = 0; i < ndim; ++i) {
            dims[i] = shape.dim(perm[i]);
        }
        int64_t last_dim = dims.back();
        int64_t batch_size = 1;
        for (int i = 0; i < ndim - 1; ++i) {
            batch_size *= dims[i];
        }

        // Compute input and output strides
        std::vector<int64_t> in_strides = compute_stride(shape);
        std::vector<int64_t> out_strides = compute_stride(out.shape());

        // Process each batch
        for (int64_t batch = 0; batch < batch_size; ++batch) {
            // Collect data for this batch
            std::vector<T> data(last_dim);
            std::vector<int64_t> batch_coords(ndim - 1);
            int64_t tmp = batch;
            for (int i = ndim - 2; i >= 0; --i) {
                batch_coords[i] = tmp % dims[i];
                tmp /= dims[i];
            }

            for (int64_t j = 0; j < last_dim; ++j) {
                int64_t src_idx = 0;
                // Build coordinates
                for (int d = 0; d < ndim - 1; ++d) {
                    src_idx += batch_coords[d] * in_strides[perm[d]];
                }
                src_idx += j * in_strides[perm[ndim - 1]];
                data[j] = src[src_idx];
            }

            // Partition
            std::nth_element(data.begin(), data.begin() + kth, data.end());

            // Write back
            for (int64_t j = 0; j < last_dim; ++j) {
                int64_t dst_idx = 0;
                for (int d = 0; d < ndim - 1; ++d) {
                    dst_idx += batch_coords[d] * out_strides[perm[d]];
                }
                dst_idx += j * out_strides[perm[ndim - 1]];
                dst[dst_idx] = data[j];
            }
        }

        return out;
    }

    static OpArgs partition_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& x = std::any_cast<const Array&>(args[1]);
        int64_t kth = std::any_cast<int64_t>(args[2]);
        int axis = std::any_cast<int>(args[3]);

        Array& mutable_out = const_cast<Array&>(out);
        DType dtype = x.dtype();
        switch (dtype) {
        case DType::F32: return { partition_impl<float>(mutable_out, x, kth, axis) };
        case DType::F64: return { partition_impl<double>(mutable_out, x, kth, axis) };
        case DType::I32: return { partition_impl<int32_t>(mutable_out, x, kth, axis) };
        case DType::I64: return { partition_impl<int64_t>(mutable_out, x, kth, axis) };
        case DType::U8:  return { partition_impl<uint8_t>(mutable_out, x, kth, axis) };
        default: INS_THROW("partition: unsupported dtype");
        }
    }

    REGISTER_KERNEL(partition, CPU, F32, partition_kernel);
    REGISTER_KERNEL(partition, CPU, F64, partition_kernel);
    REGISTER_KERNEL(partition, CPU, I32, partition_kernel);
    REGISTER_KERNEL(partition, CPU, I64, partition_kernel);
    REGISTER_KERNEL(partition, CPU, U8, partition_kernel);

    // ========== argpartition ==========

    template<typename T>
    static Array argpartition_impl(const Array& out, const Array& x, int64_t kth, int axis) {
        int64_t* dst = const_cast<int64_t*>(out.data<int64_t>());
        const T* src = x.data<T>();
        const Shape& shape = x.shape();
        int ndim = shape.ndim();
        if (axis < 0) axis += ndim;

        // For simplicity, handle 1D case first
        if (ndim == 1) {
            int64_t n = x.numel();
            std::vector<std::pair<T, int64_t>> pairs(n);
            for (int64_t i = 0; i < n; ++i) {
                pairs[i] = { src[i], i };
            }
            // Use nth_element to partially order around kth
            std::nth_element(pairs.begin(), pairs.begin() + kth, pairs.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
            for (int64_t i = 0; i < n; ++i) {
                dst[i] = pairs[i].second;
            }
            return out;
        }

        // Move axis to last dimension
        std::vector<int> perm(ndim);
        for (int i = 0; i < ndim; ++i) perm[i] = i;
        if (axis != ndim - 1) {
            std::swap(perm[axis], perm[ndim - 1]);
        }

        // Build dimension array in permuted order
        std::vector<int64_t> dims(ndim);
        for (int i = 0; i < ndim; ++i) {
            dims[i] = shape.dim(perm[i]);
        }
        int64_t last_dim = dims.back();
        int64_t batch_size = 1;
        for (int i = 0; i < ndim - 1; ++i) {
            batch_size *= dims[i];
        }

        // Compute input strides
        std::vector<int64_t> in_strides = compute_stride(shape);
        std::vector<int64_t> out_strides = compute_stride(out.shape());

        // Process each batch
        for (int64_t batch = 0; batch < batch_size; ++batch) {
            // Decompose batch into coordinates
            std::vector<int64_t> batch_coords(ndim - 1);
            int64_t tmp = batch;
            for (int i = ndim - 2; i >= 0; --i) {
                batch_coords[i] = tmp % dims[i];
                tmp /= dims[i];
            }

            // Build pairs for this batch
            std::vector<std::pair<T, int64_t>> pairs(last_dim);
            for (int64_t j = 0; j < last_dim; ++j) {
                int64_t src_idx = 0;
                for (int d = 0; d < ndim - 1; ++d) {
                    src_idx += batch_coords[d] * in_strides[perm[d]];
                }
                src_idx += j * in_strides[perm[ndim - 1]];
                pairs[j] = { src[src_idx], j };
            }

            // Partial sort around kth
            std::nth_element(pairs.begin(), pairs.begin() + kth, pairs.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });

            // Write indices to output
            for (int64_t j = 0; j < last_dim; ++j) {
                int64_t dst_idx = 0;
                for (int d = 0; d < ndim - 1; ++d) {
                    dst_idx += batch_coords[d] * out_strides[perm[d]];
                }
                dst_idx += j * out_strides[perm[ndim - 1]];
                dst[dst_idx] = pairs[j].second;
            }
        }

        return out;
    }

    static OpArgs argpartition_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& x = std::any_cast<const Array&>(args[1]);
        int64_t kth = std::any_cast<int64_t>(args[2]);
        int axis = std::any_cast<int>(args[3]);

        Array& mutable_out = const_cast<Array&>(out);
        DType dtype = x.dtype();
        switch (dtype) {
        case DType::F32: return { argpartition_impl<float>(mutable_out, x, kth, axis) };
        case DType::F64: return { argpartition_impl<double>(mutable_out, x, kth, axis) };
        case DType::I32: return { argpartition_impl<int32_t>(mutable_out, x, kth, axis) };
        case DType::I64: return { argpartition_impl<int64_t>(mutable_out, x, kth, axis) };
        case DType::U8:  return { argpartition_impl<uint8_t>(mutable_out, x, kth, axis) };
        default: INS_THROW("argpartition: unsupported dtype");
        }
    }

    REGISTER_KERNEL(argpartition, CPU, F32, argpartition_kernel);
    REGISTER_KERNEL(argpartition, CPU, F64, argpartition_kernel);
    REGISTER_KERNEL(argpartition, CPU, I32, argpartition_kernel);
    REGISTER_KERNEL(argpartition, CPU, I64, argpartition_kernel);
    REGISTER_KERNEL(argpartition, CPU, U8, argpartition_kernel);

    // ========== Module registration ==========

    REGISTER_MODULE(indexing, CPU);

} // namespace ins::cpu