// backends/cpu/manipulation.cpp
#include "insight/plugin/op_registry.h"
#include "insight/core/array.h"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <numeric>
#include <iostream>
namespace ins::cpu {

    // ============================================================================
    // Helper: fill with zeros
    // ============================================================================

    template<typename T>
    static void fill_zero(T* data, int64_t n) {
#pragma omp parallel for
        for (int64_t i = 0; i < n; ++i) {
            data[i] = T(0);
        }
    }

    // ============================================================================
    // flip: reverse elements along axis
    // ============================================================================

    template<typename T>
    static void flip_impl(const Array& x, Array& out, int axis) {
        const T* src = x.data<T>();
        T* dst = out.data<T>();

        const Shape& shape = x.shape();
        const Strides& strides = x.strides();
        int ndim = shape.ndim();

        int64_t axis_size = shape.dim(axis);
        int64_t total = x.numel();

#pragma omp parallel for
        for (int64_t linear = 0; linear < total; ++linear) {
            // Convert linear index to multi-dim indices
            int64_t tmp = linear;
            std::vector<int64_t> idx(ndim);
            for (int d = ndim - 1; d >= 0; --d) {
                idx[d] = tmp % shape.dim(d);
                tmp /= shape.dim(d);
            }

            // Flip the target axis
            idx[axis] = axis_size - 1 - idx[axis];

            // Compute source offset using strides
            int64_t src_offset = x.offset();
            for (int d = 0; d < ndim; ++d) {
                src_offset += idx[d] * strides[d];
            }

            dst[linear] = src[src_offset];
        }
    }

    static OpArgs flip_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        int axis = std::any_cast<int>(args[1]);

        Array out(x.shape(), x.dtype(), x.place());

        switch (x.dtype()) {
        case DType::BOOL: flip_impl<bool>(x, out, axis); break;
        case DType::U8:   flip_impl<uint8_t>(x, out, axis); break;
        case DType::I8:   flip_impl<int8_t>(x, out, axis); break;
        case DType::I16:  flip_impl<int16_t>(x, out, axis); break;
        case DType::I32:  flip_impl<int32_t>(x, out, axis); break;
        case DType::I64:  flip_impl<int64_t>(x, out, axis); break;
        case DType::U16:  flip_impl<uint16_t>(x, out, axis); break;
        case DType::U32:  flip_impl<uint32_t>(x, out, axis); break;
        case DType::U64:  flip_impl<uint64_t>(x, out, axis); break;
        case DType::F16:  flip_impl<uint16_t>(x, out, axis); break;
        case DType::BF16: flip_impl<uint16_t>(x, out, axis); break;
        case DType::F32:  flip_impl<float>(x, out, axis); break;
        case DType::F64:  flip_impl<double>(x, out, axis); break;
        case DType::C32:  flip_impl<std::complex<float>>(x, out, axis); break;
        case DType::C64:  flip_impl<std::complex<double>>(x, out, axis); break;
        default: INS_THROW("flip: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(flip, CPU, BOOL, flip_kernel);
    REGISTER_KERNEL(flip, CPU, U8, flip_kernel);
    REGISTER_KERNEL(flip, CPU, I8, flip_kernel);
    REGISTER_KERNEL(flip, CPU, I16, flip_kernel);
    REGISTER_KERNEL(flip, CPU, I32, flip_kernel);
    REGISTER_KERNEL(flip, CPU, I64, flip_kernel);
    REGISTER_KERNEL(flip, CPU, U16, flip_kernel);
    REGISTER_KERNEL(flip, CPU, U32, flip_kernel);
    REGISTER_KERNEL(flip, CPU, U64, flip_kernel);
    REGISTER_KERNEL(flip, CPU, F16, flip_kernel);
    REGISTER_KERNEL(flip, CPU, BF16, flip_kernel);
    REGISTER_KERNEL(flip, CPU, F32, flip_kernel);
    REGISTER_KERNEL(flip, CPU, F64, flip_kernel);
    REGISTER_KERNEL(flip, CPU, C32, flip_kernel);
    REGISTER_KERNEL(flip, CPU, C64, flip_kernel);

    // ============================================================================
    // concat: concatenate multiple arrays along axis
    // ============================================================================

    template<typename T>
    static void concat_impl(
        const std::vector<Array>& tensors,
        Array& out,
        int axis
    ) {
        T* dst = out.data<T>();
        const Shape& out_shape = out.shape();
        int ndim = out_shape.ndim();

        // Precompute output strides once
        std::vector<int64_t> out_strides(ndim);
        out_strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            out_strides[i] = out_strides[i + 1] * out_shape.dim(i + 1);
        }

        // Precompute cumulative sizes along axis
        std::vector<int64_t> cum_sizes;
        cum_sizes.push_back(0);
        for (const auto& t : tensors) {
            cum_sizes.push_back(cum_sizes.back() + t.shape().dim(axis));
        }

        // Precompute input strides for each tensor
        std::vector<std::vector<int64_t>> all_in_strides(tensors.size());
        for (size_t idx = 0; idx < tensors.size(); ++idx) {
            const Shape& in_shape = tensors[idx].shape();
            int in_ndim = in_shape.ndim();
            all_in_strides[idx].resize(in_ndim);
            all_in_strides[idx][in_ndim - 1] = 1;
            for (int i = in_ndim - 2; i >= 0; --i) {
                all_in_strides[idx][i] = all_in_strides[idx][i + 1] * in_shape.dim(i + 1);
            }
        }

        int64_t total = out.numel();
#pragma omp parallel for
        for (int64_t linear = 0; linear < total; ++linear) {
            // Convert linear to multi-dim indices
            int64_t tmp = linear;
            std::vector<int64_t> idx(ndim);
            for (int d = ndim - 1; d >= 0; --d) {
                idx[d] = tmp % out_shape.dim(d);
                tmp /= out_shape.dim(d);
            }

            // Find source tensor
            int64_t axis_idx = idx[axis];
            size_t tensor_idx = 0;
            for (size_t i = 1; i < cum_sizes.size(); ++i) {
                if (axis_idx < cum_sizes[i]) {
                    tensor_idx = i - 1;
                    break;
                }
            }

            // Compute source index
            const Array& t = tensors[tensor_idx];
            const T* src = t.data<T>();
            const std::vector<int64_t>& in_strides = all_in_strides[tensor_idx];

            int64_t src_linear = 0;
            for (int d = 0; d < ndim; ++d) {
                int64_t src_idx;
                if (d == axis) {
                    src_idx = axis_idx - cum_sizes[tensor_idx];
                }
                else {
                    src_idx = idx[d];
                }
                src_linear += src_idx * in_strides[d];
            }

            dst[linear] = src[src_linear];
        }
    }

    static OpArgs concat_kernel(const OpArgs& args) {
        const std::vector<Array>& tensors = std::any_cast<const std::vector<Array>&>(args[0]);
        int axis = std::any_cast<int>(args[1]);
        const Shape& out_shape = std::any_cast<const Shape&>(args[2]);

        DType dtype = tensors[0].dtype();
        Place place = tensors[0].place();
        Array out(out_shape, dtype, place);

        switch (dtype) {
        case DType::BOOL: concat_impl<bool>(tensors, out, axis); break;
        case DType::U8:   concat_impl<uint8_t>(tensors, out, axis); break;
        case DType::I8:   concat_impl<int8_t>(tensors, out, axis); break;
        case DType::I16:  concat_impl<int16_t>(tensors, out, axis); break;
        case DType::I32:  concat_impl<int32_t>(tensors, out, axis); break;
        case DType::I64:  concat_impl<int64_t>(tensors, out, axis); break;
        case DType::U16:  concat_impl<uint16_t>(tensors, out, axis); break;
        case DType::U32:  concat_impl<uint32_t>(tensors, out, axis); break;
        case DType::U64:  concat_impl<uint64_t>(tensors, out, axis); break;
        case DType::F16:  concat_impl<uint16_t>(tensors, out, axis); break;  // placeholder
        case DType::BF16: concat_impl<uint16_t>(tensors, out, axis); break;
        case DType::F32:  concat_impl<float>(tensors, out, axis); break;
        case DType::F64:  concat_impl<double>(tensors, out, axis); break;
        case DType::C32:  concat_impl<std::complex<float>>(tensors, out, axis); break;
        case DType::C64:  concat_impl<std::complex<double>>(tensors, out, axis); break;
        default: INS_THROW("concat: unsupported dtype");
        }
        return { out };
    }

    // Register concat for all types
    REGISTER_KERNEL(concat, CPU, BOOL, concat_kernel);
    REGISTER_KERNEL(concat, CPU, U8, concat_kernel);
    REGISTER_KERNEL(concat, CPU, I8, concat_kernel);
    REGISTER_KERNEL(concat, CPU, I16, concat_kernel);
    REGISTER_KERNEL(concat, CPU, I32, concat_kernel);
    REGISTER_KERNEL(concat, CPU, I64, concat_kernel);
    REGISTER_KERNEL(concat, CPU, U16, concat_kernel);
    REGISTER_KERNEL(concat, CPU, U32, concat_kernel);
    REGISTER_KERNEL(concat, CPU, U64, concat_kernel);
    REGISTER_KERNEL(concat, CPU, F16, concat_kernel);
    REGISTER_KERNEL(concat, CPU, BF16, concat_kernel);
    REGISTER_KERNEL(concat, CPU, F32, concat_kernel);
    REGISTER_KERNEL(concat, CPU, F64, concat_kernel);
    REGISTER_KERNEL(concat, CPU, C32, concat_kernel);
    REGISTER_KERNEL(concat, CPU, C64, concat_kernel);

    // ============================================================================
    // repeat: repeat elements along axis
    // ============================================================================

    template<typename T>
    static void repeat_impl(const Array& x, Array& out, int repeats, int axis) {
        const T* src = x.data<T>();
        T* dst = out.data<T>();

        const Shape& in_shape = x.shape();
        const Shape& out_shape = out.shape();
        int ndim = in_shape.ndim();

        // Precompute strides
        std::vector<int64_t> in_strides(ndim);
        in_strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            in_strides[i] = in_strides[i + 1] * in_shape.dim(i + 1);
        }

        // Precompute dimension sizes
        std::vector<int64_t> dims(ndim);
        for (int i = 0; i < ndim; ++i) {
            dims[i] = in_shape.dim(i);
        }

        int64_t total = out.numel();
#pragma omp parallel for
        for (int64_t linear = 0; linear < total; ++linear) {
            // Convert linear to multi-index
            int64_t tmp = linear;
            std::vector<int64_t> idx(ndim);
            for (int d = ndim - 1; d >= 0; --d) {
                int64_t out_dim = out_shape.dim(d);
                idx[d] = tmp % out_dim;
                tmp /= out_dim;
            }

            // Compute source index
            int64_t src_idx = 0;
            for (int d = 0; d < ndim; ++d) {
                if (d == axis) {
                    src_idx += (idx[d] / repeats) * in_strides[d];
                }
                else {
                    src_idx += idx[d] * in_strides[d];
                }
            }

            dst[linear] = src[src_idx];
        }
    }

    static OpArgs repeat_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        int repeats = std::any_cast<int>(args[1]);
        int axis = std::any_cast<int>(args[2]);

        // Compute output shape
        Shape in_shape = x.shape();
        std::vector<int64_t> out_dims = in_shape.dims();
        out_dims[axis] *= repeats;
        Shape out_shape(out_dims);

        Array out(out_shape, x.dtype(), x.place());

        switch (x.dtype()) {
        case DType::BOOL: repeat_impl<bool>(x, out, repeats, axis); break;
        case DType::U8:   repeat_impl<uint8_t>(x, out, repeats, axis); break;
        case DType::I8:   repeat_impl<int8_t>(x, out, repeats, axis); break;
        case DType::I16:  repeat_impl<int16_t>(x, out, repeats, axis); break;
        case DType::I32:  repeat_impl<int32_t>(x, out, repeats, axis); break;
        case DType::I64:  repeat_impl<int64_t>(x, out, repeats, axis); break;
        case DType::U16:  repeat_impl<uint16_t>(x, out, repeats, axis); break;
        case DType::U32:  repeat_impl<uint32_t>(x, out, repeats, axis); break;
        case DType::U64:  repeat_impl<uint64_t>(x, out, repeats, axis); break;
        case DType::F16:  repeat_impl<uint16_t>(x, out, repeats, axis); break;
        case DType::BF16: repeat_impl<uint16_t>(x, out, repeats, axis); break;
        case DType::F32:  repeat_impl<float>(x, out, repeats, axis); break;
        case DType::F64:  repeat_impl<double>(x, out, repeats, axis); break;
        case DType::C32:  repeat_impl<std::complex<float>>(x, out, repeats, axis); break;
        case DType::C64:  repeat_impl<std::complex<double>>(x, out, repeats, axis); break;
        default: INS_THROW("repeat: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(repeat, CPU, BOOL, repeat_kernel);
    REGISTER_KERNEL(repeat, CPU, U8, repeat_kernel);
    REGISTER_KERNEL(repeat, CPU, I8, repeat_kernel);
    REGISTER_KERNEL(repeat, CPU, I16, repeat_kernel);
    REGISTER_KERNEL(repeat, CPU, I32, repeat_kernel);
    REGISTER_KERNEL(repeat, CPU, I64, repeat_kernel);
    REGISTER_KERNEL(repeat, CPU, U16, repeat_kernel);
    REGISTER_KERNEL(repeat, CPU, U32, repeat_kernel);
    REGISTER_KERNEL(repeat, CPU, U64, repeat_kernel);
    REGISTER_KERNEL(repeat, CPU, F16, repeat_kernel);
    REGISTER_KERNEL(repeat, CPU, BF16, repeat_kernel);
    REGISTER_KERNEL(repeat, CPU, F32, repeat_kernel);
    REGISTER_KERNEL(repeat, CPU, F64, repeat_kernel);
    REGISTER_KERNEL(repeat, CPU, C32, repeat_kernel);
    REGISTER_KERNEL(repeat, CPU, C64, repeat_kernel);

    // ============================================================================
    // tile: tile array along axes
    // ============================================================================

    template<typename T>
    static void tile_impl(const Array& x, Array& out) {
        const T* src = x.data<T>();
        T* dst = out.data<T>();

        const Shape& in_shape = x.shape();
        const Shape& out_shape = out.shape();
        int ndim = out_shape.ndim();
        int src_ndim = in_shape.ndim();

        // Precompute strides
        std::vector<int64_t> out_strides(ndim);
        out_strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            out_strides[i] = out_strides[i + 1] * out_shape.dim(i + 1);
        }

        // Precompute source strides and dimensions (padded with 1 on left)
        std::vector<int64_t> in_strides(ndim, 0);
        std::vector<int64_t> in_dims(ndim, 1);
        for (int i = 0; i < src_ndim; ++i) {
            in_dims[ndim - src_ndim + i] = in_shape.dim(i);
        }
        in_strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            if (i >= ndim - src_ndim) {
                in_strides[i] = in_strides[i + 1] * in_dims[i + 1];
            }
            else {
                in_strides[i] = 0;
            }
        }

        int64_t total = out.numel();
#pragma omp parallel for
        for (int64_t linear = 0; linear < total; ++linear) {
            int64_t tmp = linear;
            std::vector<int64_t> idx(ndim);
            for (int d = ndim - 1; d >= 0; --d) {
                idx[d] = tmp % out_shape.dim(d);
                tmp /= out_shape.dim(d);
            }

            // Compute source index
            int64_t src_idx = 0;
            for (int d = 0; d < ndim; ++d) {
                src_idx += (idx[d] % in_dims[d]) * in_strides[d];
            }

            dst[linear] = src[src_idx];
        }
    }

    static OpArgs tile_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        const Shape& reps = std::any_cast<const Shape&>(args[1]);

        // Compute output shape
        Shape in_shape = x.shape();
        int in_ndim = in_shape.ndim();
        int out_ndim = std::max(in_ndim, reps.ndim());
        std::vector<int64_t> out_dims(out_ndim, 1);
        for (int i = 0; i < out_ndim; ++i) {
            int in_idx = i - (out_ndim - in_ndim);
            int64_t in_dim = (in_idx >= 0) ? in_shape.dim(in_idx) : 1;
            int64_t rep = (i < reps.ndim()) ? reps.dim(i) : 1;
            out_dims[i] = in_dim * rep;
        }
        Shape out_shape(out_dims);

        Array out(out_shape, x.dtype(), x.place());

        switch (x.dtype()) {
        case DType::BOOL: tile_impl<bool>(x, out); break;
        case DType::U8:   tile_impl<uint8_t>(x, out); break;
        case DType::I8:   tile_impl<int8_t>(x, out); break;
        case DType::I16:  tile_impl<int16_t>(x, out); break;
        case DType::I32:  tile_impl<int32_t>(x, out); break;
        case DType::I64:  tile_impl<int64_t>(x, out); break;
        case DType::U16:  tile_impl<uint16_t>(x, out); break;
        case DType::U32:  tile_impl<uint32_t>(x, out); break;
        case DType::U64:  tile_impl<uint64_t>(x, out); break;
        case DType::F16:  tile_impl<uint16_t>(x, out); break;
        case DType::BF16: tile_impl<uint16_t>(x, out); break;
        case DType::F32:  tile_impl<float>(x, out); break;
        case DType::F64:  tile_impl<double>(x, out); break;
        case DType::C32:  tile_impl<std::complex<float>>(x, out); break;
        case DType::C64:  tile_impl<std::complex<double>>(x, out); break;
        default: INS_THROW("tile: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(tile, CPU, BOOL, tile_kernel);
    REGISTER_KERNEL(tile, CPU, U8, tile_kernel);
    REGISTER_KERNEL(tile, CPU, I8, tile_kernel);
    REGISTER_KERNEL(tile, CPU, I16, tile_kernel);
    REGISTER_KERNEL(tile, CPU, I32, tile_kernel);
    REGISTER_KERNEL(tile, CPU, I64, tile_kernel);
    REGISTER_KERNEL(tile, CPU, U16, tile_kernel);
    REGISTER_KERNEL(tile, CPU, U32, tile_kernel);
    REGISTER_KERNEL(tile, CPU, U64, tile_kernel);
    REGISTER_KERNEL(tile, CPU, F16, tile_kernel);
    REGISTER_KERNEL(tile, CPU, BF16, tile_kernel);
    REGISTER_KERNEL(tile, CPU, F32, tile_kernel);
    REGISTER_KERNEL(tile, CPU, F64, tile_kernel);
    REGISTER_KERNEL(tile, CPU, C32, tile_kernel);
    REGISTER_KERNEL(tile, CPU, C64, tile_kernel);

    // ============================================================================
    // pad: pad array with constant value
    // ============================================================================

    template<typename T>
    static void pad_impl(const Array& x, Array& out, const std::vector<int64_t>& pad_width, T constant_value) {
        const T* src = x.data<T>();
        T* dst = out.data<T>();

        const Shape& in_shape = x.shape();
        const Shape& out_shape = out.shape();
        int ndim = in_shape.ndim();

        // 1. Fill entire output with constant value
        int64_t total_out = out.numel();
#pragma omp parallel for
        for (int64_t i = 0; i < total_out; ++i) {
            dst[i] = constant_value;
        }

        // 2. Calculate start offsets for each dimension
        std::vector<int64_t> start(ndim);
        for (int i = 0; i < ndim; ++i) {
            start[i] = pad_width[2 * i];
        }

        // 3. Precompute strides for input and output
        std::vector<int64_t> in_strides(ndim);
        in_strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            in_strides[i] = in_strides[i + 1] * in_shape.dim(i + 1);
        }

        std::vector<int64_t> out_strides(ndim);
        out_strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            out_strides[i] = out_strides[i + 1] * out_shape.dim(i + 1);
        }

        // 4. Copy original data to the padded region
        std::vector<int64_t> in_dims = in_shape.dims();
        int64_t total_in = in_shape.numel();

#pragma omp parallel for
        for (int64_t linear = 0; linear < total_in; ++linear) {
            // Convert linear index to multi-dimensional indices for input
            int64_t tmp = linear;
            std::vector<int64_t> idx(ndim);
            for (int d = ndim - 1; d >= 0; --d) {
                idx[d] = tmp % in_dims[d];
                tmp /= in_dims[d];
            }

            // Compute output offset: output index = start + idx
            int64_t out_offset = 0;
            for (int d = 0; d < ndim; ++d) {
                out_offset += (start[d] + idx[d]) * out_strides[d];
            }

            // Compute input offset
            int64_t in_offset = 0;
            for (int d = 0; d < ndim; ++d) {
                in_offset += idx[d] * in_strides[d];
            }

            dst[out_offset] = src[in_offset];
        }
    }

    static OpArgs pad_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        const std::vector<int64_t>& pad_width = std::any_cast<const std::vector<int64_t>&>(args[1]);
        double constant_value = std::any_cast<double>(args[2]);

        Shape in_shape = x.shape();
        int ndim = in_shape.ndim();
        INS_CHECK(pad_width.size() == static_cast<size_t>(2 * ndim),
            "pad: pad_width size mismatch");

        // Compute output shape
        std::vector<int64_t> out_dims(ndim);
        for (int i = 0; i < ndim; ++i) {
            out_dims[i] = in_shape.dim(i) + pad_width[2 * i] + pad_width[2 * i + 1];
        }
        Shape out_shape(out_dims);

        Array out(out_shape, x.dtype(), x.place());

        switch (x.dtype()) {
        case DType::BOOL: pad_impl<bool>(x, out, pad_width, constant_value != 0); break;
        case DType::U8:   pad_impl<uint8_t>(x, out, pad_width, static_cast<uint8_t>(constant_value)); break;
        case DType::I8:   pad_impl<int8_t>(x, out, pad_width, static_cast<int8_t>(constant_value)); break;
        case DType::I16:  pad_impl<int16_t>(x, out, pad_width, static_cast<int16_t>(constant_value)); break;
        case DType::I32:  pad_impl<int32_t>(x, out, pad_width, static_cast<int32_t>(constant_value)); break;
        case DType::I64:  pad_impl<int64_t>(x, out, pad_width, static_cast<int64_t>(constant_value)); break;
        case DType::U16:  pad_impl<uint16_t>(x, out, pad_width, static_cast<uint16_t>(constant_value)); break;
        case DType::U32:  pad_impl<uint32_t>(x, out, pad_width, static_cast<uint32_t>(constant_value)); break;
        case DType::U64:  pad_impl<uint64_t>(x, out, pad_width, static_cast<uint64_t>(constant_value)); break;
        case DType::F16:  pad_impl<uint16_t>(x, out, pad_width, static_cast<uint16_t>(constant_value)); break;
        case DType::BF16: pad_impl<uint16_t>(x, out, pad_width, static_cast<uint16_t>(constant_value)); break;
        case DType::F32:  pad_impl<float>(x, out, pad_width, static_cast<float>(constant_value)); break;
        case DType::F64:  pad_impl<double>(x, out, pad_width, constant_value); break;
        case DType::C32:  pad_impl<std::complex<float>>(x, out, pad_width, std::complex<float>(constant_value, 0)); break;
        case DType::C64:  pad_impl<std::complex<double>>(x, out, pad_width, std::complex<double>(constant_value, 0)); break;
        default: INS_THROW("pad: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(pad, CPU, BOOL, pad_kernel);
    REGISTER_KERNEL(pad, CPU, U8, pad_kernel);
    REGISTER_KERNEL(pad, CPU, I8, pad_kernel);
    REGISTER_KERNEL(pad, CPU, I16, pad_kernel);
    REGISTER_KERNEL(pad, CPU, I32, pad_kernel);
    REGISTER_KERNEL(pad, CPU, I64, pad_kernel);
    REGISTER_KERNEL(pad, CPU, U16, pad_kernel);
    REGISTER_KERNEL(pad, CPU, U32, pad_kernel);
    REGISTER_KERNEL(pad, CPU, U64, pad_kernel);
    REGISTER_KERNEL(pad, CPU, F16, pad_kernel);
    REGISTER_KERNEL(pad, CPU, BF16, pad_kernel);
    REGISTER_KERNEL(pad, CPU, F32, pad_kernel);
    REGISTER_KERNEL(pad, CPU, F64, pad_kernel);
    REGISTER_KERNEL(pad, CPU, C32, pad_kernel);
    REGISTER_KERNEL(pad, CPU, C64, pad_kernel);

    // ============================================================================
    // roll: roll elements along axis
    // ============================================================================

    template<typename T>
    static void roll_impl_1d(const Array& x, Array& out, int shift) {
        const T* src = x.data<T>();
        T* dst = out.data<T>();
        int64_t n = x.numel();

        // Normalize shift to [0, n-1]
        shift = ((shift % n) + n) % n;
        if (shift == 0) {
            std::memcpy(dst, src, n * sizeof(T));
            return;
        }

#pragma omp parallel for
        for (int64_t i = 0; i < n; ++i) {
            int64_t src_idx = (i - shift + n) % n;
            dst[i] = src[src_idx];
        }
    }

    template<typename T>
    static void roll_impl_nd(const Array& x, Array& out, int shift, int axis) {
        const T* src = x.data<T>();
        T* dst = out.data<T>();

        const Shape& shape = x.shape();
        int ndim = shape.ndim();

        // Precompute strides
        std::vector<int64_t> strides(ndim);
        strides[ndim - 1] = 1;
        for (int d = ndim - 2; d >= 0; --d) {
            strides[d] = strides[d + 1] * shape.dim(d + 1);
        }

        std::vector<int64_t> dims = shape.dims();
        int64_t axis_size = dims[axis];
        int64_t axis_stride = strides[axis];

        // Normalize shift
        shift = ((shift % axis_size) + axis_size) % axis_size;
        if (shift == 0) {
            std::memcpy(dst, src, x.numel() * sizeof(T));
            return;
        }

        int64_t total = x.numel();
        int64_t block_size = axis_size * axis_stride;
        int64_t num_blocks = total / block_size;

#pragma omp parallel for collapse(2)
        for (int64_t block = 0; block < num_blocks; ++block) {
            for (int64_t i = 0; i < axis_size; ++i) {
                int64_t src_base = block * block_size + i * axis_stride;
                int64_t dst_base = block * block_size + ((i + shift) % axis_size) * axis_stride;
                // Copy entire segment of length axis_stride
                for (int64_t j = 0; j < axis_stride; ++j) {
                    dst[dst_base + j] = src[src_base + j];
                }
            }
        }
    }

    // Macro to generate roll cases
#define ROLL_ND_CASE(dtype, T) \
    case DType::dtype: \
        roll_impl_nd<T>(x, out, shift, axis); \
        break

#define ROLL_1D_CASE(dtype, T) \
    case DType::dtype: \
        roll_impl_1d<T>(flat, rolled, shift); \
        break

    static OpArgs roll_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        int shift = std::any_cast<int>(args[1]);
        int axis = std::any_cast<int>(args[2]);

        Array out;

        if (axis == -1) {
            // Flatten, roll, then reshape back
            Array flat = x.reshape(Shape({ x.numel() }));
            if (!flat.is_contiguous()) {
                flat = flat.contiguous();
            }
            Array rolled(flat.shape(), x.dtype(), x.place());

            switch (x.dtype()) {
                ROLL_1D_CASE(BOOL, bool);
                ROLL_1D_CASE(U8, uint8_t);
                ROLL_1D_CASE(I8, int8_t);
                ROLL_1D_CASE(I16, int16_t);
                ROLL_1D_CASE(I32, int32_t);
                ROLL_1D_CASE(I64, int64_t);
                ROLL_1D_CASE(U16, uint16_t);
                ROLL_1D_CASE(U32, uint32_t);
                ROLL_1D_CASE(U64, uint64_t);
                ROLL_1D_CASE(F16, uint16_t);
                ROLL_1D_CASE(BF16, uint16_t);
                ROLL_1D_CASE(F32, float);
                ROLL_1D_CASE(F64, double);
                ROLL_1D_CASE(C32, std::complex<float>);
                ROLL_1D_CASE(C64, std::complex<double>);
            default: INS_THROW("roll: unsupported dtype");
            }
            out = rolled.reshape(x.shape());
        }
        else {
            out = Array(x.shape(), x.dtype(), x.place());
            switch (x.dtype()) {
                ROLL_ND_CASE(BOOL, bool);
                ROLL_ND_CASE(U8, uint8_t);
                ROLL_ND_CASE(I8, int8_t);
                ROLL_ND_CASE(I16, int16_t);
                ROLL_ND_CASE(I32, int32_t);
                ROLL_ND_CASE(I64, int64_t);
                ROLL_ND_CASE(U16, uint16_t);
                ROLL_ND_CASE(U32, uint32_t);
                ROLL_ND_CASE(U64, uint64_t);
                ROLL_ND_CASE(F16, uint16_t);
                ROLL_ND_CASE(BF16, uint16_t);
                ROLL_ND_CASE(F32, float);
                ROLL_ND_CASE(F64, double);
                ROLL_ND_CASE(C32, std::complex<float>);
                ROLL_ND_CASE(C64, std::complex<double>);
            default: INS_THROW("roll: unsupported dtype");
            }
        }
        return { out };
    }

#undef ROLL_ND_CASE
#undef ROLL_1D_CASE

    // Register roll for all types
    REGISTER_KERNEL(roll, CPU, BOOL, roll_kernel);
    REGISTER_KERNEL(roll, CPU, U8, roll_kernel);
    REGISTER_KERNEL(roll, CPU, I8, roll_kernel);
    REGISTER_KERNEL(roll, CPU, I16, roll_kernel);
    REGISTER_KERNEL(roll, CPU, I32, roll_kernel);
    REGISTER_KERNEL(roll, CPU, I64, roll_kernel);
    REGISTER_KERNEL(roll, CPU, U16, roll_kernel);
    REGISTER_KERNEL(roll, CPU, U32, roll_kernel);
    REGISTER_KERNEL(roll, CPU, U64, roll_kernel);
    REGISTER_KERNEL(roll, CPU, F16, roll_kernel);
    REGISTER_KERNEL(roll, CPU, BF16, roll_kernel);
    REGISTER_KERNEL(roll, CPU, F32, roll_kernel);
    REGISTER_KERNEL(roll, CPU, F64, roll_kernel);
    REGISTER_KERNEL(roll, CPU, C32, roll_kernel);
    REGISTER_KERNEL(roll, CPU, C64, roll_kernel);


    // ============================================================================
    // diag: extract diagonal or create diagonal array
    // ============================================================================

    template<typename T>
    static void diag_extract_impl(const Array& x, Array& out, int k) {
        const T* src = x.data<T>();
        T* dst = out.data<T>();

        const Shape& shape = x.shape();
        int64_t rows = shape.dim(0);
        int64_t cols = shape.dim(1);

        int64_t offset = 0;
        if (k >= 0) {
            offset = k * rows;
        }
        else {
            offset = (-k);
        }

        int64_t diag_len = 0;
        if (k >= 0) {
            diag_len = std::min(rows, cols - k);
        }
        else {
            diag_len = std::min(rows + k, cols);
        }

        for (int64_t i = 0; i < diag_len; ++i) {
            int64_t src_idx;
            if (k >= 0) {
                src_idx = i * cols + (i + k);
            }
            else {
                src_idx = (i - k) * cols + i;
            }
            dst[i] = src[src_idx];
        }
    }

    template<typename T>
    static void diag_construct_impl(const Array& x, Array& out, int k) {
        const T* src = x.data<T>();
        T* dst = out.data<T>();

        int64_t n = x.numel();
        int64_t rows = out.shape().dim(0);
        int64_t cols = out.shape().dim(1);

        // Initialize to zero
#pragma omp parallel for
        for (int64_t i = 0; i < rows * cols; ++i) {
            dst[i] = T(0);
        }

        // Set diagonal
        for (int64_t i = 0; i < n; ++i) {
            int64_t j = i + k;
            if (j >= 0 && j < cols) {
                dst[i * cols + j] = src[i];
            }
        }
    }

    static OpArgs diag_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        int k = std::any_cast<int>(args[1]);

        const Shape& shape = x.shape();
        Array out;

        if (shape.ndim() == 1) {
            // Construct diagonal matrix from 1D array
            int64_t n = x.numel();
            int64_t size = n + std::abs(k);
            Shape out_shape({ size, size });
            out = Array(out_shape, x.dtype(), x.place());

            switch (x.dtype()) {
            case DType::BOOL: diag_construct_impl<bool>(x, out, k); break;
            case DType::U8:   diag_construct_impl<uint8_t>(x, out, k); break;
            case DType::I8:   diag_construct_impl<int8_t>(x, out, k); break;
            case DType::I16:  diag_construct_impl<int16_t>(x, out, k); break;
            case DType::I32:  diag_construct_impl<int32_t>(x, out, k); break;
            case DType::I64:  diag_construct_impl<int64_t>(x, out, k); break;
            case DType::U16:  diag_construct_impl<uint16_t>(x, out, k); break;
            case DType::U32:  diag_construct_impl<uint32_t>(x, out, k); break;
            case DType::U64:  diag_construct_impl<uint64_t>(x, out, k); break;
            case DType::F16:  diag_construct_impl<uint16_t>(x, out, k); break;
            case DType::BF16: diag_construct_impl<uint16_t>(x, out, k); break;
            case DType::F32:  diag_construct_impl<float>(x, out, k); break;
            case DType::F64:  diag_construct_impl<double>(x, out, k); break;
            case DType::C32:  diag_construct_impl<std::complex<float>>(x, out, k); break;
            case DType::C64:  diag_construct_impl<std::complex<double>>(x, out, k); break;
            default: INS_THROW("diag: unsupported dtype");
            }
        }
        else if (shape.ndim() == 2) {
            // Extract diagonal from 2D array
            int64_t rows = shape.dim(0);
            int64_t cols = shape.dim(1);
            int64_t diag_len = 0;
            if (k >= 0) {
                diag_len = std::min(rows, cols - k);
            }
            else {
                diag_len = std::min(rows + k, cols);
            }
            Shape out_shape({ diag_len });
            out = Array(out_shape, x.dtype(), x.place());

            switch (x.dtype()) {
            case DType::BOOL: diag_extract_impl<bool>(x, out, k); break;
            case DType::U8:   diag_extract_impl<uint8_t>(x, out, k); break;
            case DType::I8:   diag_extract_impl<int8_t>(x, out, k); break;
            case DType::I16:  diag_extract_impl<int16_t>(x, out, k); break;
            case DType::I32:  diag_extract_impl<int32_t>(x, out, k); break;
            case DType::I64:  diag_extract_impl<int64_t>(x, out, k); break;
            case DType::U16:  diag_extract_impl<uint16_t>(x, out, k); break;
            case DType::U32:  diag_extract_impl<uint32_t>(x, out, k); break;
            case DType::U64:  diag_extract_impl<uint64_t>(x, out, k); break;
            case DType::F16:  diag_extract_impl<uint16_t>(x, out, k); break;
            case DType::BF16: diag_extract_impl<uint16_t>(x, out, k); break;
            case DType::F32:  diag_extract_impl<float>(x, out, k); break;
            case DType::F64:  diag_extract_impl<double>(x, out, k); break;
            case DType::C32:  diag_extract_impl<std::complex<float>>(x, out, k); break;
            case DType::C64:  diag_extract_impl<std::complex<double>>(x, out, k); break;
            default: INS_THROW("diag: unsupported dtype");
            }
        }
        else {
            INS_THROW("diag: input must be 1D or 2D");
        }

        return { out };
    }

    REGISTER_KERNEL(diag, CPU, BOOL, diag_kernel);
    REGISTER_KERNEL(diag, CPU, U8, diag_kernel);
    REGISTER_KERNEL(diag, CPU, I8, diag_kernel);
    REGISTER_KERNEL(diag, CPU, I16, diag_kernel);
    REGISTER_KERNEL(diag, CPU, I32, diag_kernel);
    REGISTER_KERNEL(diag, CPU, I64, diag_kernel);
    REGISTER_KERNEL(diag, CPU, U16, diag_kernel);
    REGISTER_KERNEL(diag, CPU, U32, diag_kernel);
    REGISTER_KERNEL(diag, CPU, U64, diag_kernel);
    REGISTER_KERNEL(diag, CPU, F16, diag_kernel);
    REGISTER_KERNEL(diag, CPU, BF16, diag_kernel);
    REGISTER_KERNEL(diag, CPU, F32, diag_kernel);
    REGISTER_KERNEL(diag, CPU, F64, diag_kernel);
    REGISTER_KERNEL(diag, CPU, C32, diag_kernel);
    REGISTER_KERNEL(diag, CPU, C64, diag_kernel);

    // ============================================================================
    // tril: lower triangle of an array
    // ============================================================================

    template<typename T>
    static void tril_impl(const Array& x, Array& out, int k) {
        const T* src = x.data<T>();
        T* dst = out.data<T>();

        const Shape& shape = x.shape();
        int64_t rows = shape.dim(0);
        int64_t cols = shape.dim(1);

#pragma omp parallel for collapse(2)
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                if (j <= i + k) {
                    dst[i * cols + j] = src[i * cols + j];
                }
                else {
                    dst[i * cols + j] = T(0);
                }
            }
        }
    }

    static OpArgs tril_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        int k = std::any_cast<int>(args[1]);

        Array out(x.shape(), x.dtype(), x.place());

        switch (x.dtype()) {
        case DType::BOOL: tril_impl<bool>(x, out, k); break;
        case DType::U8:   tril_impl<uint8_t>(x, out, k); break;
        case DType::I8:   tril_impl<int8_t>(x, out, k); break;
        case DType::I16:  tril_impl<int16_t>(x, out, k); break;
        case DType::I32:  tril_impl<int32_t>(x, out, k); break;
        case DType::I64:  tril_impl<int64_t>(x, out, k); break;
        case DType::U16:  tril_impl<uint16_t>(x, out, k); break;
        case DType::U32:  tril_impl<uint32_t>(x, out, k); break;
        case DType::U64:  tril_impl<uint64_t>(x, out, k); break;
        case DType::F16:  tril_impl<uint16_t>(x, out, k); break;
        case DType::BF16: tril_impl<uint16_t>(x, out, k); break;
        case DType::F32:  tril_impl<float>(x, out, k); break;
        case DType::F64:  tril_impl<double>(x, out, k); break;
        case DType::C32:  tril_impl<std::complex<float>>(x, out, k); break;
        case DType::C64:  tril_impl<std::complex<double>>(x, out, k); break;
        default: INS_THROW("tril: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(tril, CPU, BOOL, tril_kernel);
    REGISTER_KERNEL(tril, CPU, U8, tril_kernel);
    REGISTER_KERNEL(tril, CPU, I8, tril_kernel);
    REGISTER_KERNEL(tril, CPU, I16, tril_kernel);
    REGISTER_KERNEL(tril, CPU, I32, tril_kernel);
    REGISTER_KERNEL(tril, CPU, I64, tril_kernel);
    REGISTER_KERNEL(tril, CPU, U16, tril_kernel);
    REGISTER_KERNEL(tril, CPU, U32, tril_kernel);
    REGISTER_KERNEL(tril, CPU, U64, tril_kernel);
    REGISTER_KERNEL(tril, CPU, F16, tril_kernel);
    REGISTER_KERNEL(tril, CPU, BF16, tril_kernel);
    REGISTER_KERNEL(tril, CPU, F32, tril_kernel);
    REGISTER_KERNEL(tril, CPU, F64, tril_kernel);
    REGISTER_KERNEL(tril, CPU, C32, tril_kernel);
    REGISTER_KERNEL(tril, CPU, C64, tril_kernel);

    // ============================================================================
    // triu: upper triangle of an array
    // ============================================================================

    template<typename T>
    static void triu_impl(const Array& x, Array& out, int k) {
        const T* src = x.data<T>();
        T* dst = out.data<T>();

        const Shape& shape = x.shape();
        int64_t rows = shape.dim(0);
        int64_t cols = shape.dim(1);

#pragma omp parallel for collapse(2)
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                if (j >= i + k) {
                    dst[i * cols + j] = src[i * cols + j];
                }
                else {
                    dst[i * cols + j] = T(0);
                }
            }
        }
    }

    static OpArgs triu_kernel(const OpArgs& args) {
        const Array& x = std::any_cast<const Array&>(args[0]);
        int k = std::any_cast<int>(args[1]);

        Array out(x.shape(), x.dtype(), x.place());

        switch (x.dtype()) {
        case DType::BOOL: triu_impl<bool>(x, out, k); break;
        case DType::U8:   triu_impl<uint8_t>(x, out, k); break;
        case DType::I8:   triu_impl<int8_t>(x, out, k); break;
        case DType::I16:  triu_impl<int16_t>(x, out, k); break;
        case DType::I32:  triu_impl<int32_t>(x, out, k); break;
        case DType::I64:  triu_impl<int64_t>(x, out, k); break;
        case DType::U16:  triu_impl<uint16_t>(x, out, k); break;
        case DType::U32:  triu_impl<uint32_t>(x, out, k); break;
        case DType::U64:  triu_impl<uint64_t>(x, out, k); break;
        case DType::F16:  triu_impl<uint16_t>(x, out, k); break;
        case DType::BF16: triu_impl<uint16_t>(x, out, k); break;
        case DType::F32:  triu_impl<float>(x, out, k); break;
        case DType::F64:  triu_impl<double>(x, out, k); break;
        case DType::C32:  triu_impl<std::complex<float>>(x, out, k); break;
        case DType::C64:  triu_impl<std::complex<double>>(x, out, k); break;
        default: INS_THROW("triu: unsupported dtype");
        }
        return { out };
    }

    REGISTER_KERNEL(triu, CPU, BOOL, triu_kernel);
    REGISTER_KERNEL(triu, CPU, U8, triu_kernel);
    REGISTER_KERNEL(triu, CPU, I8, triu_kernel);
    REGISTER_KERNEL(triu, CPU, I16, triu_kernel);
    REGISTER_KERNEL(triu, CPU, I32, triu_kernel);
    REGISTER_KERNEL(triu, CPU, I64, triu_kernel);
    REGISTER_KERNEL(triu, CPU, U16, triu_kernel);
    REGISTER_KERNEL(triu, CPU, U32, triu_kernel);
    REGISTER_KERNEL(triu, CPU, U64, triu_kernel);
    REGISTER_KERNEL(triu, CPU, F16, triu_kernel);
    REGISTER_KERNEL(triu, CPU, BF16, triu_kernel);
    REGISTER_KERNEL(triu, CPU, F32, triu_kernel);
    REGISTER_KERNEL(triu, CPU, F64, triu_kernel);
    REGISTER_KERNEL(triu, CPU, C32, triu_kernel);
    REGISTER_KERNEL(triu, CPU, C64, triu_kernel);

    // ============================================================================
    // contiguous_copy: convert non-contiguous array to contiguous memory layout
    // ============================================================================

    static OpArgs contiguous_copy_kernel(const OpArgs& args) {
        const Array& out = std::any_cast<const Array&>(args[0]);
        const Array& in = std::any_cast<const Array&>(args[1]);

        Array& mutable_out = const_cast<Array&>(out);

        // Fast path: already contiguous
        if (in.is_contiguous()) {
            std::memcpy(mutable_out.data(), in.data(), in.numel() * dtype_size(in.dtype()));
            return { mutable_out };
        }

        size_t elem_size = dtype_size(in.dtype());
        int64_t total = in.numel();

        char* dst_base = static_cast<char*>(mutable_out.data());
        // Note: in.data() already includes in.offset(), so we don't add it again
        const char* src_base = static_cast<const char*>(in.data());

        const Shape& shape = in.shape();
        const Strides& src_strides = in.strides();
        int ndim = shape.ndim();
        const std::vector<int64_t> dims = shape.dims();

        // Precompute dimension strides for linear index conversion
        std::vector<int64_t> dim_strides(ndim);
        if (ndim > 0) {
            dim_strides[ndim - 1] = 1;
            for (int d = ndim - 2; d >= 0; --d) {
                dim_strides[d] = dim_strides[d + 1] * dims[d + 1];
            }
        }

        for (int64_t linear = 0; linear < total; ++linear) {
            // Convert linear index to multi-dimensional indices
            std::vector<int64_t> indices(ndim);
            int64_t remaining = linear;
            for (int d = 0; d < ndim; ++d) {
                indices[d] = remaining / dim_strides[d];
                remaining %= dim_strides[d];
            }

            // Compute source byte offset
            // src_base already includes in.offset(), so start from 0
            int64_t src_byte_offset = 0;
            for (int d = 0; d < ndim; ++d) {
                src_byte_offset += indices[d] * src_strides[d] * elem_size;
            }

            // Copy element
            std::memcpy(dst_base + linear * elem_size,
                src_base + src_byte_offset,
                elem_size);
        }

        return { mutable_out };
    }

    // Register for all types including FP8
    REGISTER_KERNEL(contiguous_copy, CPU, BOOL, contiguous_copy_kernel);
    REGISTER_KERNEL(contiguous_copy, CPU, U8, contiguous_copy_kernel);
    REGISTER_KERNEL(contiguous_copy, CPU, I8, contiguous_copy_kernel);
    REGISTER_KERNEL(contiguous_copy, CPU, I16, contiguous_copy_kernel);
    REGISTER_KERNEL(contiguous_copy, CPU, I32, contiguous_copy_kernel);
    REGISTER_KERNEL(contiguous_copy, CPU, I64, contiguous_copy_kernel);
    REGISTER_KERNEL(contiguous_copy, CPU, U16, contiguous_copy_kernel);
    REGISTER_KERNEL(contiguous_copy, CPU, U32, contiguous_copy_kernel);
    REGISTER_KERNEL(contiguous_copy, CPU, U64, contiguous_copy_kernel);
    REGISTER_KERNEL(contiguous_copy, CPU, F16, contiguous_copy_kernel);
    REGISTER_KERNEL(contiguous_copy, CPU, BF16, contiguous_copy_kernel);
    REGISTER_KERNEL(contiguous_copy, CPU, F32, contiguous_copy_kernel);
    REGISTER_KERNEL(contiguous_copy, CPU, F64, contiguous_copy_kernel);
    REGISTER_KERNEL(contiguous_copy, CPU, C32, contiguous_copy_kernel);
    REGISTER_KERNEL(contiguous_copy, CPU, C64, contiguous_copy_kernel);
    REGISTER_KERNEL(contiguous_copy, CPU, F8_E4M3, contiguous_copy_kernel);
    REGISTER_KERNEL(contiguous_copy, CPU, F8_E5M2, contiguous_copy_kernel);

    // ============================================================================
    // Module registration
    // ============================================================================

    REGISTER_MODULE(manipulation, CPU);

} // namespace ins::cpu