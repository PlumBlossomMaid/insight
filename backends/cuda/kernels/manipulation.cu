// backends/cuda/kernels/manipulation.cu
#include "insight/plugin/op_registry.h"
#include "insight/core/array.h"
#include <cuda_runtime.h>
#include <cstring>

namespace ins {
    namespace gpu {

        // ============================================================================
        // contiguous_copy_kernel
        // ============================================================================

        template<typename T>
        __global__ void contiguous_copy_kernel(
            T* dst, const T* src,
            const int64_t* shape, const int64_t* src_strides,
            const int64_t* dst_strides, int ndim, int64_t total) {

            int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
            if (linear >= total) return;

            int64_t indices[8];
            int64_t remaining = linear;
            for (int d = ndim - 1; d >= 0; --d) {
                indices[d] = remaining % shape[d];
                remaining /= shape[d];
            }

            int64_t src_offset = 0;
            for (int d = 0; d < ndim; ++d) {
                src_offset += indices[d] * src_strides[d];
            }

            dst[linear] = src[src_offset];
        }

        // ============================================================================
        // concat_kernel
        // ============================================================================

        template<typename T>
        __global__ void concat_kernel(
            const T* const* inputs, T* output,
            const int64_t* in_shapes, const int64_t* out_dims,
            int num_inputs, int ndim, int axis, int64_t total) {

            int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
            if (linear >= total) return;

            int64_t out_coord[8];
            int64_t remaining = linear;
            for (int d = ndim - 1; d >= 0; --d) {
                out_coord[d] = remaining % out_dims[d];
                remaining /= out_dims[d];
            }

            int input_id = -1;
            int64_t offset = 0;
            for (int i = 0; i < num_inputs; ++i) {
                int64_t dim_size = in_shapes[i * ndim + axis];
                if (out_coord[axis] < offset + dim_size) {
                    input_id = i;
                    break;
                }
                offset += dim_size;
            }

            if (input_id == -1) return;

            int64_t in_coord[8];
            for (int d = 0; d < ndim; ++d) {
                if (d == axis) {
                    in_coord[d] = out_coord[d] - offset;
                }
                else {
                    in_coord[d] = out_coord[d];
                }
            }

            int64_t in_offset = 0;
            int64_t stride = 1;
            const int64_t* in_shape = &in_shapes[input_id * ndim];
            for (int d = ndim - 1; d >= 0; --d) {
                in_offset += in_coord[d] * stride;
                stride *= in_shape[d];
            }

            output[linear] = inputs[input_id][in_offset];
        }

        // ============================================================================
        // repeat_kernel
        // ============================================================================

        template<typename T>
        __global__ void repeat_kernel(
            T* dst, const T* src,
            const int64_t* src_shape, const int64_t* dst_shape,
            const int64_t* repeats, int ndim, int64_t total) {

            int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
            if (linear >= total) return;

            int64_t out_coord[8];
            int64_t remaining = linear;
            for (int d = ndim - 1; d >= 0; --d) {
                out_coord[d] = remaining % dst_shape[d];
                remaining /= dst_shape[d];
            }

            int64_t in_coord[8];
            for (int d = 0; d < ndim; ++d) {
                in_coord[d] = out_coord[d] / repeats[d];
            }

            int64_t in_offset = 0;
            int64_t stride = 1;
            for (int d = ndim - 1; d >= 0; --d) {
                in_offset += in_coord[d] * stride;
                stride *= src_shape[d];
            }

            dst[linear] = src[in_offset];
        }

        // ============================================================================
        // tile_kernel
        // ============================================================================
        template<typename T>
        __global__ void tile_kernel(
            T* dst, const T* src,
            const int64_t* in_dims, const int64_t* out_dims,
            const int64_t* reps, int ndim, int64_t total) {

            int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
            if (linear >= total) return;

            // Convert linear to output coordinates
            int64_t out_coord[8];
            int64_t remaining = linear;
            for (int d = ndim - 1; d >= 0; --d) {
                out_coord[d] = remaining % out_dims[d];
                remaining /= out_dims[d];
            }

            // Compute input coordinates (tile: out_coord % in_dims)
            int64_t in_coord[8];
            for (int d = 0; d < ndim; ++d) {
                in_coord[d] = out_coord[d] % in_dims[d];
            }

            // Compute input offset
            int64_t in_offset = 0;
            int64_t stride = 1;
            for (int d = ndim - 1; d >= 0; --d) {
                in_offset += in_coord[d] * stride;
                stride *= in_dims[d];
            }

            dst[linear] = src[in_offset];
        }

        // ============================================================================
        // pad_kernel
        // ============================================================================

        template<typename T>
        __global__ void pad_kernel(
            T* dst, const T* src,
            const int64_t* src_dims, const int64_t* dst_dims,
            const int64_t* start, int ndim, int64_t total, T pad_value) {

            int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
            if (linear >= total) return;

            int64_t dst_coord[8];
            int64_t remaining = linear;
            for (int d = ndim - 1; d >= 0; --d) {
                dst_coord[d] = remaining % dst_dims[d];
                remaining /= dst_dims[d];
            }

            bool in_padding = false;
            int64_t src_coord[8];
            for (int d = 0; d < ndim; ++d) {
                src_coord[d] = dst_coord[d] - start[d];
                if (src_coord[d] < 0 || src_coord[d] >= src_dims[d]) {
                    in_padding = true;
                    break;
                }
            }

            if (in_padding) {
                dst[linear] = pad_value;
                return;
            }

            int64_t src_offset = 0;
            int64_t stride = 1;
            for (int d = ndim - 1; d >= 0; --d) {
                src_offset += src_coord[d] * stride;
                stride *= src_dims[d];
            }

            dst[linear] = src[src_offset];
        }

        // ============================================================================
        // flip_kernel
        // ============================================================================

        template<typename T>
        __global__ void flip_kernel_single_axis(
            const T* src, T* dst,
            const int64_t* dims, const int64_t* strides,
            int ndim, int axis, int64_t total) {

            int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
            if (linear >= total) return;

            // Convert linear index to coordinates
            int64_t indices[8];
            int64_t remaining = linear;
            for (int d = ndim - 1; d >= 0; --d) {
                indices[d] = remaining % dims[d];
                remaining /= dims[d];
            }

            // Flip the target axis
            indices[axis] = dims[axis] - 1 - indices[axis];

            // Compute source offset using strides
            int64_t src_offset = 0;
            for (int d = 0; d < ndim; ++d) {
                src_offset += indices[d] * strides[d];
            }

            dst[linear] = src[src_offset];
        }

        // ============================================================================
        // roll_kernel
        // ============================================================================

        template<typename T>
        __global__ void roll_kernel(
            T* dst, const T* src,
            const int64_t* shape, const int64_t* strides,
            int shift, int axis, int ndim, int64_t total) {

            int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
            if (linear >= total) return;

            int64_t coord[8];
            int64_t remaining = linear;
            for (int d = ndim - 1; d >= 0; --d) {
                coord[d] = remaining % shape[d];
                remaining /= shape[d];
            }

            int64_t src_coord[8];
            for (int d = 0; d < ndim; ++d) {
                src_coord[d] = coord[d];
            }
            int ax = axis;
            if (ax < 0) ax += ndim;
            int64_t dim_size = shape[ax];
            src_coord[ax] = (coord[ax] - shift) % dim_size;
            if (src_coord[ax] < 0) src_coord[ax] += dim_size;

            int64_t src_offset = 0;
            int64_t stride = 1;
            for (int d = ndim - 1; d >= 0; --d) {
                src_offset += src_coord[d] * stride;
                stride *= shape[d];
            }

            dst[linear] = src[src_offset];
        }

        // ============================================================================
        // Wrappers and registrations
        // ============================================================================

        // contiguous_copy wrapper
        static OpArgs contiguous_copy_wrapper(const OpArgs& args) {
            const Array& out = std::any_cast<const Array&>(args[0]);
            const Array& in = std::any_cast<const Array&>(args[1]);
            Array& mutable_out = const_cast<Array&>(out);

            if (in.is_contiguous()) {
                std::memcpy(mutable_out.data(), in.data(), in.numel() * dtype_size(in.dtype()));
                return { mutable_out };
            }

            size_t elem_size = dtype_size(in.dtype());
            int64_t total = in.numel();
            int ndim = in.shape().ndim();

            std::vector<int64_t> shape_vec = in.shape().dims();
            std::vector<int64_t> src_strides_vec = in.strides().data();
            std::vector<int64_t> dst_strides_vec(ndim);
            if (ndim > 0) {
                dst_strides_vec[ndim - 1] = 1;
                for (int i = ndim - 2; i >= 0; --i) {
                    dst_strides_vec[i] = dst_strides_vec[i + 1] * shape_vec[i + 1];
                }
            }

            int64_t* d_shape = nullptr;
            int64_t* d_src_strides = nullptr;
            int64_t* d_dst_strides = nullptr;
            cudaMalloc(&d_shape, ndim * sizeof(int64_t));
            cudaMalloc(&d_src_strides, ndim * sizeof(int64_t));
            cudaMalloc(&d_dst_strides, ndim * sizeof(int64_t));
            cudaMemcpy(d_shape, shape_vec.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_src_strides, src_strides_vec.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_dst_strides, dst_strides_vec.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

            dim3 block(256);
            dim3 grid((total + block.x - 1) / block.x);

            if (elem_size == 4) {
                contiguous_copy_kernel<float> << <grid, block >> > (
                    mutable_out.data<float>(), in.data<float>(),
                    d_shape, d_src_strides, d_dst_strides, ndim, total);
            }
            else if (elem_size == 8) {
                if (in.dtype() == DType::C32) {
                    contiguous_copy_kernel<std::complex<float>> << <grid, block >> > (
                        mutable_out.data<std::complex<float>>(), in.data<std::complex<float>>(),
                        d_shape, d_src_strides, d_dst_strides, ndim, total);
                }
                else {
                    contiguous_copy_kernel<double> << <grid, block >> > (
                        mutable_out.data<double>(), in.data<double>(),
                        d_shape, d_src_strides, d_dst_strides, ndim, total);
                }
            }
            else if (elem_size == 16) {
                contiguous_copy_kernel<std::complex<double>> << <grid, block >> > (
                    mutable_out.data<std::complex<double>>(), in.data<std::complex<double>>(),
                    d_shape, d_src_strides, d_dst_strides, ndim, total);
            }
            else if (elem_size == 2) {
                contiguous_copy_kernel<uint16_t> << <grid, block >> > (
                    mutable_out.data<uint16_t>(), in.data<uint16_t>(),
                    d_shape, d_src_strides, d_dst_strides, ndim, total);
            }
            else {
                contiguous_copy_kernel<uint8_t> << <grid, block >> > (
                    mutable_out.data<uint8_t>(), in.data<uint8_t>(),
                    d_shape, d_src_strides, d_dst_strides, ndim, total);
            }

            cudaFree(d_shape);
            cudaFree(d_src_strides);
            cudaFree(d_dst_strides);

            return { mutable_out };
        }

        // concat wrapper
        static OpArgs concat_wrapper(const OpArgs& args) {
            const std::vector<Array>& tensors = std::any_cast<const std::vector<Array>&>(args[0]);
            int axis = std::any_cast<int>(args[1]);
            const Shape& out_shape = std::any_cast<const Shape&>(args[2]);

            DType dtype = tensors[0].dtype();
            Place place = tensors[0].place();
            Array out(out_shape, dtype, place);
            Array& mutable_out = const_cast<Array&>(out);

            int ndim = out_shape.ndim();
            int num_inputs = tensors.size();
            int64_t total = out.numel();

            std::vector<const void*> in_ptrs(num_inputs);
            for (int i = 0; i < num_inputs; ++i) {
                in_ptrs[i] = tensors[i].data();
            }

            std::vector<int64_t> in_shapes(num_inputs * ndim);
            for (int i = 0; i < num_inputs; ++i) {
                for (int d = 0; d < ndim; ++d) {
                    in_shapes[i * ndim + d] = tensors[i].shape().dim(d);
                }
            }
            std::vector<int64_t> out_dims = out_shape.dims();

            void** d_in_ptrs = nullptr;
            int64_t* d_in_shapes = nullptr;
            int64_t* d_out_dims = nullptr;
            cudaMalloc(&d_in_ptrs, num_inputs * sizeof(void*));
            cudaMalloc(&d_in_shapes, in_shapes.size() * sizeof(int64_t));
            cudaMalloc(&d_out_dims, ndim * sizeof(int64_t));
            cudaMemcpy(d_in_ptrs, in_ptrs.data(), num_inputs * sizeof(void*), cudaMemcpyHostToDevice);
            cudaMemcpy(d_in_shapes, in_shapes.data(), in_shapes.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_out_dims, out_dims.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

            dim3 block(256);
            dim3 grid((total + block.x - 1) / block.x);
            size_t elem_size = dtype_size(dtype);

            if (elem_size == 4) {
                concat_kernel<float> << <grid, block >> > (
                    (const float* const*)d_in_ptrs, mutable_out.data<float>(),
                    d_in_shapes, d_out_dims, num_inputs, ndim, axis, total);
            }
            else if (elem_size == 8) {
                if (is_complex(dtype)) {
                    concat_kernel<std::complex<double>> << <grid, block >> > (
                        (const std::complex<double>*const*)d_in_ptrs,
                        mutable_out.data<std::complex<double>>(),
                        d_in_shapes, d_out_dims, num_inputs, ndim, axis, total);
                }
                else {
                    concat_kernel<double> << <grid, block >> > (
                        (const double* const*)d_in_ptrs, mutable_out.data<double>(),
                        d_in_shapes, d_out_dims, num_inputs, ndim, axis, total);
                }
            }
            else if (elem_size == 2) {
                concat_kernel<uint16_t> << <grid, block >> > (
                    (const uint16_t* const*)d_in_ptrs, mutable_out.data<uint16_t>(),
                    d_in_shapes, d_out_dims, num_inputs, ndim, axis, total);
            }
            else {
                concat_kernel<uint8_t> << <grid, block >> > (
                    (const uint8_t* const*)d_in_ptrs, mutable_out.data<uint8_t>(),
                    d_in_shapes, d_out_dims, num_inputs, ndim, axis, total);
            }

            cudaFree(d_in_ptrs);
            cudaFree(d_in_shapes);
            cudaFree(d_out_dims);

            return { mutable_out };
        }

        // repeat wrapper
        static OpArgs repeat_wrapper(const OpArgs& args) {
            const Array& x = std::any_cast<const Array&>(args[0]);
            int repeats = std::any_cast<int>(args[1]);
            int axis = std::any_cast<int>(args[2]);

            // 计算输出形状（与 CPU 一致）
            Shape in_shape = x.shape();
            std::vector<int64_t> out_dims = in_shape.dims();
            out_dims[axis] *= repeats;
            Shape out_shape(out_dims);

            Array out(out_shape, x.dtype(), x.place());
            Array& mutable_out = const_cast<Array&>(out);

            int ndim = in_shape.ndim();
            int64_t total = out.numel();

            // 准备 GPU 数据
            std::vector<int64_t> src_shape = in_shape.dims();
            std::vector<int64_t> dst_shape = out_shape.dims();
            std::vector<int64_t> repeats_vec(ndim, 1);
            repeats_vec[axis] = repeats;

            int64_t* d_src_shape = nullptr;
            int64_t* d_dst_shape = nullptr;
            int64_t* d_repeats = nullptr;
            cudaMalloc(&d_src_shape, ndim * sizeof(int64_t));
            cudaMalloc(&d_dst_shape, ndim * sizeof(int64_t));
            cudaMalloc(&d_repeats, ndim * sizeof(int64_t));
            cudaMemcpy(d_src_shape, src_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_dst_shape, dst_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_repeats, repeats_vec.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

            dim3 block(256);
            dim3 grid((total + block.x - 1) / block.x);
            size_t elem_size = dtype_size(x.dtype());

            // 根据元素大小分发（不是 dtype）
            if (elem_size == 4) {
                repeat_kernel<float> << <grid, block >> > (
                    mutable_out.data<float>(), x.data<float>(),
                    d_src_shape, d_dst_shape, d_repeats, ndim, total);
            }
            else if (elem_size == 8) {
                if (is_complex(x.dtype())) {
                    repeat_kernel<std::complex<double>> << <grid, block >> > (
                        mutable_out.data<std::complex<double>>(), x.data<std::complex<double>>(),
                        d_src_shape, d_dst_shape, d_repeats, ndim, total);
                }
                else {
                    repeat_kernel<double> << <grid, block >> > (
                        mutable_out.data<double>(), x.data<double>(),
                        d_src_shape, d_dst_shape, d_repeats, ndim, total);
                }
            }
            else if (elem_size == 2) {
                repeat_kernel<uint16_t> << <grid, block >> > (
                    mutable_out.data<uint16_t>(), x.data<uint16_t>(),
                    d_src_shape, d_dst_shape, d_repeats, ndim, total);
            }
            else {
                repeat_kernel<uint8_t> << <grid, block >> > (
                    mutable_out.data<uint8_t>(), x.data<uint8_t>(),
                    d_src_shape, d_dst_shape, d_repeats, ndim, total);
            }

            cudaFree(d_src_shape);
            cudaFree(d_dst_shape);
            cudaFree(d_repeats);

            return { mutable_out };
        }

        static OpArgs tile_wrapper(const OpArgs& args) {
            const Array& x = std::any_cast<const Array&>(args[0]);
            const Shape& reps = std::any_cast<const Shape&>(args[1]);

            // Compute output shape (same as CPU)
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
            Array& mutable_out = const_cast<Array&>(out);

            int ndim = out_ndim;
            int64_t total = out.numel();

            // Prepare in_dims (padded with 1 on left)
            std::vector<int64_t> in_dims(ndim, 1);
            for (int i = 0; i < in_ndim; ++i) {
                in_dims[ndim - in_ndim + i] = in_shape.dim(i);
            }

            // Prepare reps (padded with 1 on left)
            std::vector<int64_t> reps_vec(ndim, 1);
            for (int i = 0; i < reps.ndim(); ++i) {
                reps_vec[ndim - reps.ndim() + i] = reps.dim(i);
            }

            int64_t* d_in_dims = nullptr;
            int64_t* d_out_dims = nullptr;
            int64_t* d_reps = nullptr;
            cudaMalloc(&d_in_dims, ndim * sizeof(int64_t));
            cudaMalloc(&d_out_dims, ndim * sizeof(int64_t));
            cudaMalloc(&d_reps, ndim * sizeof(int64_t));
            cudaMemcpy(d_in_dims, in_dims.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_out_dims, out_dims.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_reps, reps_vec.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

            dim3 block(256);
            dim3 grid((total + block.x - 1) / block.x);
            size_t elem_size = dtype_size(x.dtype());

            if (elem_size == 4) {
                tile_kernel<float> << <grid, block >> > (
                    mutable_out.data<float>(), x.data<float>(),
                    d_in_dims, d_out_dims, d_reps, ndim, total);
            }
            else if (elem_size == 8) {
                if (is_complex(x.dtype())) {
                    tile_kernel<std::complex<double>> << <grid, block >> > (
                        mutable_out.data<std::complex<double>>(), x.data<std::complex<double>>(),
                        d_in_dims, d_out_dims, d_reps, ndim, total);
                }
                else {
                    tile_kernel<double> << <grid, block >> > (
                        mutable_out.data<double>(), x.data<double>(),
                        d_in_dims, d_out_dims, d_reps, ndim, total);
                }
            }
            else if (elem_size == 2) {
                tile_kernel<uint16_t> << <grid, block >> > (
                    mutable_out.data<uint16_t>(), x.data<uint16_t>(),
                    d_in_dims, d_out_dims, d_reps, ndim, total);
            }
            else {
                tile_kernel<uint8_t> << <grid, block >> > (
                    mutable_out.data<uint8_t>(), x.data<uint8_t>(),
                    d_in_dims, d_out_dims, d_reps, ndim, total);
            }

            cudaFree(d_in_dims);
            cudaFree(d_out_dims);
            cudaFree(d_reps);

            return { mutable_out };
        }


        // pad wrapper
        static OpArgs pad_wrapper(const OpArgs& args) {
            const Array& x = std::any_cast<const Array&>(args[0]);
            const std::vector<int64_t>& pad_width = std::any_cast<const std::vector<int64_t>&>(args[1]);
            double constant_value = std::any_cast<double>(args[2]);

            int ndim = x.shape().ndim();
            std::vector<int64_t> out_dims(ndim);
            for (int i = 0; i < ndim; ++i) {
                out_dims[i] = x.shape().dim(i) + pad_width[2 * i] + pad_width[2 * i + 1];
            }
            Shape out_shape(out_dims);

            Array out(out_shape, x.dtype(), x.place());
            Array& mutable_out = const_cast<Array&>(out);

            int64_t total = out.numel();

            std::vector<int64_t> src_dims = x.shape().dims();
            std::vector<int64_t> dst_dims = out_shape.dims();
            std::vector<int64_t> start(ndim);
            for (int i = 0; i < ndim; ++i) {
                start[i] = pad_width[2 * i];
            }

            int64_t* d_src_dims = nullptr;
            int64_t* d_dst_dims = nullptr;
            int64_t* d_start = nullptr;
            cudaMalloc(&d_src_dims, ndim * sizeof(int64_t));
            cudaMalloc(&d_dst_dims, ndim * sizeof(int64_t));
            cudaMalloc(&d_start, ndim * sizeof(int64_t));
            cudaMemcpy(d_src_dims, src_dims.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_dst_dims, dst_dims.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_start, start.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

            dim3 block(256);
            dim3 grid((total + block.x - 1) / block.x);
            size_t elem_size = dtype_size(x.dtype());

            if (elem_size == 4) {
                pad_kernel<float> << <grid, block >> > (
                    mutable_out.data<float>(), x.data<float>(),
                    d_src_dims, d_dst_dims, d_start, ndim, total, static_cast<float>(constant_value));
            }
            else if (elem_size == 8) {
                if (is_complex(x.dtype())) {
                    pad_kernel<std::complex<double>> << <grid, block >> > (
                        mutable_out.data<std::complex<double>>(),
                        x.data<std::complex<double>>(),
                        d_src_dims, d_dst_dims, d_start, ndim, total,
                        std::complex<double>(constant_value, 0.0));
                }
                else {
                    pad_kernel<double> << <grid, block >> > (
                        mutable_out.data<double>(), x.data<double>(),
                        d_src_dims, d_dst_dims, d_start, ndim, total, constant_value);
                }
            }
            else if (elem_size == 2) {
                pad_kernel<uint16_t> << <grid, block >> > (
                    mutable_out.data<uint16_t>(), x.data<uint16_t>(),
                    d_src_dims, d_dst_dims, d_start, ndim, total, static_cast<uint16_t>(constant_value));
            }
            else {
                pad_kernel<uint8_t> << <grid, block >> > (
                    mutable_out.data<uint8_t>(), x.data<uint8_t>(),
                    d_src_dims, d_dst_dims, d_start, ndim, total, static_cast<uint8_t>(constant_value));
            }

            cudaFree(d_src_dims);
            cudaFree(d_dst_dims);
            cudaFree(d_start);

            return { mutable_out };
        }

        // flip wrapper
        template<typename T>
        static OpArgs flip_wrapper(const OpArgs& args) {
            const Array& x = std::any_cast<const Array&>(args[0]);
            int axis = std::any_cast<int>(args[1]);

            Array out(x.shape(), x.dtype(), x.place());
            Array& mutable_out = const_cast<Array&>(out);

            int64_t total = out.numel();
            int ndim = x.shape().ndim();

            std::vector<int64_t> dims = x.shape().dims();
            std::vector<int64_t> strides = x.strides().data();

            int64_t* d_dims = nullptr;
            int64_t* d_strides = nullptr;
            cudaMalloc(&d_dims, ndim * sizeof(int64_t));
            cudaMalloc(&d_strides, ndim * sizeof(int64_t));
            cudaMemcpy(d_dims, dims.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_strides, strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

            dim3 block(256);
            dim3 grid((total + block.x - 1) / block.x);

            flip_kernel_single_axis<T> << <grid, block >> > (
                x.data<T>(), mutable_out.data<T>(),
                d_dims, d_strides, ndim, axis, total);

            cudaFree(d_dims);
            cudaFree(d_strides);

            return { mutable_out };
        }

        // roll wrapper
        static OpArgs roll_wrapper(const OpArgs& args) {
            const Array& x = std::any_cast<const Array&>(args[0]);
            int shift = std::any_cast<int>(args[1]);
            int axis = std::any_cast<int>(args[2]);  // -1 表示 flatten

            Array out;

            if (axis == -1) {
                // Flatten, roll, then reshape back
                Array flat = x.reshape(Shape({ x.numel() }));
                if (!flat.is_contiguous()) {
                    flat = flat.contiguous();
                }
                Array rolled(flat.shape(), x.dtype(), x.place());
                Array& mutable_rolled = const_cast<Array&>(rolled);

                int64_t total = flat.numel();
                int ndim = 1;
                std::vector<int64_t> shape_vec = { total };
                std::vector<int64_t> strides_vec = { 1 };

                int64_t* d_shape = nullptr;
                int64_t* d_strides = nullptr;
                cudaMalloc(&d_shape, ndim * sizeof(int64_t));
                cudaMalloc(&d_strides, ndim * sizeof(int64_t));
                cudaMemcpy(d_shape, shape_vec.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
                cudaMemcpy(d_strides, strides_vec.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

                dim3 block(256);
                dim3 grid((total + block.x - 1) / block.x);
                size_t elem_size = dtype_size(x.dtype());
                int norm_shift = ((shift % total) + total) % total;

                // 根据元素大小分发
                if (elem_size == 4) {
                    roll_kernel<float> << <grid, block >> > (
                        mutable_rolled.data<float>(), flat.data<float>(),
                        d_shape, d_strides, norm_shift, 0, ndim, total);
                }
                else if (elem_size == 8) {
                    if (is_complex(x.dtype())) {
                        roll_kernel<std::complex<double>> << <grid, block >> > (
                            mutable_rolled.data<std::complex<double>>(), flat.data<std::complex<double>>(),
                            d_shape, d_strides, norm_shift, 0, ndim, total);
                    }
                    else {
                        roll_kernel<double> << <grid, block >> > (
                            mutable_rolled.data<double>(), flat.data<double>(),
                            d_shape, d_strides, norm_shift, 0, ndim, total);
                    }
                }
                else if (elem_size == 2) {
                    roll_kernel<uint16_t> << <grid, block >> > (
                        mutable_rolled.data<uint16_t>(), flat.data<uint16_t>(),
                        d_shape, d_strides, norm_shift, 0, ndim, total);
                }
                else {
                    roll_kernel<uint8_t> << <grid, block >> > (
                        mutable_rolled.data<uint8_t>(), flat.data<uint8_t>(),
                        d_shape, d_strides, norm_shift, 0, ndim, total);
                }

                cudaFree(d_shape);
                cudaFree(d_strides);

                out = rolled.reshape(x.shape());
            }
            else {
                // Normal ND case
                out = Array(x.shape(), x.dtype(), x.place());
                Array& mutable_out = const_cast<Array&>(out);

                int ndim = x.shape().ndim();
                int64_t total = x.numel();

                std::vector<int64_t> shape_vec = x.shape().dims();
                std::vector<int64_t> strides_vec = x.strides().data();

                int64_t* d_shape = nullptr;
                int64_t* d_strides = nullptr;
                cudaMalloc(&d_shape, ndim * sizeof(int64_t));
                cudaMalloc(&d_strides, ndim * sizeof(int64_t));
                cudaMemcpy(d_shape, shape_vec.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
                cudaMemcpy(d_strides, strides_vec.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

                dim3 block(256);
                dim3 grid((total + block.x - 1) / block.x);
                size_t elem_size = dtype_size(x.dtype());

                // Normalize shift for the axis
                int64_t axis_size = x.shape().dim(axis);
                int norm_shift = ((shift % axis_size) + axis_size) % axis_size;

                if (elem_size == 4) {
                    roll_kernel<float> << <grid, block >> > (
                        mutable_out.data<float>(), x.data<float>(),
                        d_shape, d_strides, norm_shift, axis, ndim, total);
                }
                else if (elem_size == 8) {
                    if (is_complex(x.dtype())) {
                        roll_kernel<std::complex<double>> << <grid, block >> > (
                            mutable_out.data<std::complex<double>>(), x.data<std::complex<double>>(),
                            d_shape, d_strides, norm_shift, axis, ndim, total);
                    }
                    else {
                        roll_kernel<double> << <grid, block >> > (
                            mutable_out.data<double>(), x.data<double>(),
                            d_shape, d_strides, norm_shift, axis, ndim, total);
                    }
                }
                else if (elem_size == 2) {
                    roll_kernel<uint16_t> << <grid, block >> > (
                        mutable_out.data<uint16_t>(), x.data<uint16_t>(),
                        d_shape, d_strides, norm_shift, axis, ndim, total);
                }
                else {
                    roll_kernel<uint8_t> << <grid, block >> > (
                        mutable_out.data<uint8_t>(), x.data<uint8_t>(),
                        d_shape, d_strides, norm_shift, axis, ndim, total);
                }

                cudaFree(d_shape);
                cudaFree(d_strides);
            }

            return { out };
        }

        // ============================================================================
        // diag_kernel - Extract diagonal from 2D matrix or construct diagonal matrix
        // ============================================================================

        // Scalar version for extraction
        template<typename T>
        __global__ void diag_extract_kernel_scalar(
            T* dst, const T* src,
            int64_t rows, int64_t cols, int64_t k, int64_t out_len) {

            int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= out_len) return;

            // src_idx = i * cols + (i + k)
            int64_t row = i;
            int64_t col = i + k;

            if (col >= 0 && col < cols && row < rows) {
                dst[i] = src[row * cols + col];
            }
            else {
                dst[i] = T(0);
            }
        }

        // C32 (complex64) version for extraction - using float2
        __global__ void diag_extract_kernel_c32(
            float2* dst, const float2* src,
            int64_t rows, int64_t cols, int64_t k, int64_t out_len) {

            int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= out_len) return;

            int64_t row = i;
            int64_t col = i + k;

            if (col >= 0 && col < cols && row < rows) {
                dst[i] = src[row * cols + col];
            }
            else {
                dst[i] = make_float2(0.0f, 0.0f);
            }
        }

        // C64 (complex128) version for extraction - using double2
        __global__ void diag_extract_kernel_c64(
            double2* dst, const double2* src,
            int64_t rows, int64_t cols, int64_t k, int64_t out_len) {

            int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= out_len) return;

            int64_t row = i;
            int64_t col = i + k;

            if (col >= 0 && col < cols && row < rows) {
                dst[i] = src[row * cols + col];
            }
            else {
                dst[i] = make_double2(0.0, 0.0);
            }
        }

        // Scalar version for construction
        template<typename T>
        __global__ void diag_construct_kernel_scalar(
            T* dst, const T* src,
            int64_t n, int64_t m, int64_t k, int64_t src_len) {

            int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
            int64_t total = n * m;
            if (linear >= total) return;

            int64_t i = linear / m;
            int64_t j = linear % m;
            int64_t diag_idx = j - k;

            if (diag_idx < 0 || diag_idx >= src_len || diag_idx != i) {
                dst[linear] = T(0);
            }
            else {
                dst[linear] = src[diag_idx];
            }
        }

        // C32 version for construction
        __global__ void diag_construct_kernel_c32(
            float2* dst, const float2* src,
            int64_t n, int64_t m, int64_t k, int64_t src_len) {

            int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
            int64_t total = n * m;
            if (linear >= total) return;

            int64_t i = linear / m;
            int64_t j = linear % m;
            int64_t diag_idx = j - k;

            if (diag_idx < 0 || diag_idx >= src_len || diag_idx != i) {
                dst[linear] = make_float2(0.0f, 0.0f);
            }
            else {
                dst[linear] = src[diag_idx];
            }
        }

        // C64 version for construction
        __global__ void diag_construct_kernel_c64(
            double2* dst, const double2* src,
            int64_t n, int64_t m, int64_t k, int64_t src_len) {

            int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
            int64_t total = n * m;
            if (linear >= total) return;

            int64_t i = linear / m;
            int64_t j = linear % m;
            int64_t diag_idx = j - k;

            if (diag_idx < 0 || diag_idx >= src_len || diag_idx != i) {
                dst[linear] = make_double2(0.0, 0.0);
            }
            else {
                dst[linear] = src[diag_idx];
            }
        }

        // ============================================================================
        // tril_kernel - Lower triangle
        // ============================================================================

        // Scalar version
        template<typename T>
        __global__ void tril_kernel_scalar(
            T* dst, const T* src,
            int64_t rows, int64_t cols, int64_t k) {

            int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
            int64_t total = rows * cols;
            if (linear >= total) return;

            int64_t i = linear / cols;
            int64_t j = linear % cols;

            if (j <= i + k) {
                dst[linear] = src[linear];
            }
            else {
                dst[linear] = T(0);
            }
        }

        // C32 version
        __global__ void tril_kernel_c32(
            float2* dst, const float2* src,
            int64_t rows, int64_t cols, int64_t k) {

            int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
            int64_t total = rows * cols;
            if (linear >= total) return;

            int64_t i = linear / cols;
            int64_t j = linear % cols;

            if (j <= i + k) {
                dst[linear] = src[linear];
            }
            else {
                dst[linear] = make_float2(0.0f, 0.0f);
            }
        }

        // C64 version
        __global__ void tril_kernel_c64(
            double2* dst, const double2* src,
            int64_t rows, int64_t cols, int64_t k) {

            int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
            int64_t total = rows * cols;
            if (linear >= total) return;

            int64_t i = linear / cols;
            int64_t j = linear % cols;

            if (j <= i + k) {
                dst[linear] = src[linear];
            }
            else {
                dst[linear] = make_double2(0.0, 0.0);
            }
        }

        // ============================================================================
        // triu_kernel - Upper triangle
        // ============================================================================

        // Scalar version
        template<typename T>
        __global__ void triu_kernel_scalar(
            T* dst, const T* src,
            int64_t rows, int64_t cols, int64_t k) {

            int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
            int64_t total = rows * cols;
            if (linear >= total) return;

            int64_t i = linear / cols;
            int64_t j = linear % cols;

            if (j >= i + k) {
                dst[linear] = src[linear];
            }
            else {
                dst[linear] = T(0);
            }
        }

        // C32 version
        __global__ void triu_kernel_c32(
            float2* dst, const float2* src,
            int64_t rows, int64_t cols, int64_t k) {

            int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
            int64_t total = rows * cols;
            if (linear >= total) return;

            int64_t i = linear / cols;
            int64_t j = linear % cols;

            if (j >= i + k) {
                dst[linear] = src[linear];
            }
            else {
                dst[linear] = make_float2(0.0f, 0.0f);
            }
        }

        // C64 version
        __global__ void triu_kernel_c64(
            double2* dst, const double2* src,
            int64_t rows, int64_t cols, int64_t k) {

            int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
            int64_t total = rows * cols;
            if (linear >= total) return;

            int64_t i = linear / cols;
            int64_t j = linear % cols;

            if (j >= i + k) {
                dst[linear] = src[linear];
            }
            else {
                dst[linear] = make_double2(0.0, 0.0);
            }
        }

        // ============================================================================
        // diag wrapper
        // ============================================================================

        static OpArgs diag_wrapper(const OpArgs& args) {
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
                Array& mutable_out = const_cast<Array&>(out);

                int64_t total = out.numel();
                int64_t src_len = n;
                int64_t rows = size;
                int64_t cols = size;

                dim3 block(256);
                dim3 grid((total + block.x - 1) / block.x);
                size_t elem_size = dtype_size(x.dtype());

                if (x.dtype() == DType::C32) {
                    diag_construct_kernel_c32 << <grid, block >> > (
                        mutable_out.data<float2>(), x.data<float2>(),
                        rows, cols, k, src_len);
                }
                else if (x.dtype() == DType::C64) {
                    diag_construct_kernel_c64 << <grid, block >> > (
                        mutable_out.data<double2>(), x.data<double2>(),
                        rows, cols, k, src_len);
                }
                else {
                    if (elem_size == 4) {
                        diag_construct_kernel_scalar<float> << <grid, block >> > (
                            mutable_out.data<float>(), x.data<float>(),
                            rows, cols, k, src_len);
                    }
                    else if (elem_size == 8) {
                        diag_construct_kernel_scalar<double> << <grid, block >> > (
                            mutable_out.data<double>(), x.data<double>(),
                            rows, cols, k, src_len);
                    }
                    else if (elem_size == 2) {
                        diag_construct_kernel_scalar<uint16_t> << <grid, block >> > (
                            mutable_out.data<uint16_t>(), x.data<uint16_t>(),
                            rows, cols, k, src_len);
                    }
                    else {
                        diag_construct_kernel_scalar<uint8_t> << <grid, block >> > (
                            mutable_out.data<uint8_t>(), x.data<uint8_t>(),
                            rows, cols, k, src_len);
                    }
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
                Array& mutable_out = const_cast<Array&>(out);

                int64_t out_len = diag_len;
                int64_t total = out_len;

                dim3 block(256);
                dim3 grid((total + block.x - 1) / block.x);
                size_t elem_size = dtype_size(x.dtype());

                if (x.dtype() == DType::C32) {
                    diag_extract_kernel_c32 << <grid, block >> > (
                        mutable_out.data<float2>(), x.data<float2>(),
                        rows, cols, k, out_len);
                }
                else if (x.dtype() == DType::C64) {
                    diag_extract_kernel_c64 << <grid, block >> > (
                        mutable_out.data<double2>(), x.data<double2>(),
                        rows, cols, k, out_len);
                }
                else {
                    if (elem_size == 4) {
                        diag_extract_kernel_scalar<float> << <grid, block >> > (
                            mutable_out.data<float>(), x.data<float>(),
                            rows, cols, k, out_len);
                    }
                    else if (elem_size == 8) {
                        diag_extract_kernel_scalar<double> << <grid, block >> > (
                            mutable_out.data<double>(), x.data<double>(),
                            rows, cols, k, out_len);
                    }
                    else if (elem_size == 2) {
                        diag_extract_kernel_scalar<uint16_t> << <grid, block >> > (
                            mutable_out.data<uint16_t>(), x.data<uint16_t>(),
                            rows, cols, k, out_len);
                    }
                    else {
                        diag_extract_kernel_scalar<uint8_t> << <grid, block >> > (
                            mutable_out.data<uint8_t>(), x.data<uint8_t>(),
                            rows, cols, k, out_len);
                    }
                }
            }
            else {
                INS_THROW("diag: input must be 1D or 2D");
            }

            return { out };
        }

        // ============================================================================
        // tril wrapper
        // ============================================================================

        static OpArgs tril_wrapper(const OpArgs& args) {
            const Array& x = std::any_cast<const Array&>(args[0]);
            int k = std::any_cast<int>(args[1]);

            // 创建输出数组（与 CPU 一致）
            Array out(x.shape(), x.dtype(), x.place());
            Array& mutable_out = const_cast<Array&>(out);

            int64_t rows = x.shape().dim(0);
            int64_t cols = x.shape().dim(1);
            int64_t total = rows * cols;
            size_t elem_size = dtype_size(x.dtype());

            dim3 block(256);
            dim3 grid((total + block.x - 1) / block.x);

            if (x.dtype() == DType::C32) {
                tril_kernel_c32 << <grid, block >> > (
                    mutable_out.data<float2>(), x.data<float2>(),
                    rows, cols, k);
            }
            else if (x.dtype() == DType::C64) {
                tril_kernel_c64 << <grid, block >> > (
                    mutable_out.data<double2>(), x.data<double2>(),
                    rows, cols, k);
            }
            else {
                if (elem_size == 4) {
                    tril_kernel_scalar<float> << <grid, block >> > (
                        mutable_out.data<float>(), x.data<float>(),
                        rows, cols, k);
                }
                else if (elem_size == 8) {
                    tril_kernel_scalar<double> << <grid, block >> > (
                        mutable_out.data<double>(), x.data<double>(),
                        rows, cols, k);
                }
                else if (elem_size == 2) {
                    tril_kernel_scalar<uint16_t> << <grid, block >> > (
                        mutable_out.data<uint16_t>(), x.data<uint16_t>(),
                        rows, cols, k);
                }
                else {
                    tril_kernel_scalar<uint8_t> << <grid, block >> > (
                        mutable_out.data<uint8_t>(), x.data<uint8_t>(),
                        rows, cols, k);
                }
            }

            return { mutable_out };
        }

        // ============================================================================
        // triu wrapper
        // ============================================================================

        static OpArgs triu_wrapper(const OpArgs& args) {
            const Array& x = std::any_cast<const Array&>(args[0]);
            int k = std::any_cast<int>(args[1]);

            // 创建输出数组（与 CPU 一致）
            Array out(x.shape(), x.dtype(), x.place());
            Array& mutable_out = const_cast<Array&>(out);

            int64_t rows = x.shape().dim(0);
            int64_t cols = x.shape().dim(1);
            int64_t total = rows * cols;
            size_t elem_size = dtype_size(x.dtype());

            dim3 block(256);
            dim3 grid((total + block.x - 1) / block.x);

            if (x.dtype() == DType::C32) {
                triu_kernel_c32 << <grid, block >> > (
                    mutable_out.data<float2>(), x.data<float2>(),
                    rows, cols, k);
            }
            else if (x.dtype() == DType::C64) {
                triu_kernel_c64 << <grid, block >> > (
                    mutable_out.data<double2>(), x.data<double2>(),
                    rows, cols, k);
            }
            else {
                if (elem_size == 4) {
                    triu_kernel_scalar<float> << <grid, block >> > (
                        mutable_out.data<float>(), x.data<float>(),
                        rows, cols, k);
                }
                else if (elem_size == 8) {
                    triu_kernel_scalar<double> << <grid, block >> > (
                        mutable_out.data<double>(), x.data<double>(),
                        rows, cols, k);
                }
                else if (elem_size == 2) {
                    triu_kernel_scalar<uint16_t> << <grid, block >> > (
                        mutable_out.data<uint16_t>(), x.data<uint16_t>(),
                        rows, cols, k);
                }
                else {
                    triu_kernel_scalar<uint8_t> << <grid, block >> > (
                        mutable_out.data<uint8_t>(), x.data<uint8_t>(),
                        rows, cols, k);
                }
            }

            return { mutable_out };
        }

    } // namespace gpu
} // namespace ins

// ============================================================================
// Kernel registrations (must be in global namespace)
// ============================================================================

REGISTER_KERNEL(contiguous_copy, GPU, F32, ins::gpu::contiguous_copy_wrapper);
REGISTER_KERNEL(contiguous_copy, GPU, F64, ins::gpu::contiguous_copy_wrapper);
REGISTER_KERNEL(contiguous_copy, GPU, I32, ins::gpu::contiguous_copy_wrapper);
REGISTER_KERNEL(contiguous_copy, GPU, I64, ins::gpu::contiguous_copy_wrapper);
REGISTER_KERNEL(contiguous_copy, GPU, I16, ins::gpu::contiguous_copy_wrapper);
REGISTER_KERNEL(contiguous_copy, GPU, I8, ins::gpu::contiguous_copy_wrapper);
REGISTER_KERNEL(contiguous_copy, GPU, U8, ins::gpu::contiguous_copy_wrapper);
REGISTER_KERNEL(contiguous_copy, GPU, U16, ins::gpu::contiguous_copy_wrapper);
REGISTER_KERNEL(contiguous_copy, GPU, U32, ins::gpu::contiguous_copy_wrapper);
REGISTER_KERNEL(contiguous_copy, GPU, U64, ins::gpu::contiguous_copy_wrapper);
REGISTER_KERNEL(contiguous_copy, GPU, BOOL, ins::gpu::contiguous_copy_wrapper);
REGISTER_KERNEL(contiguous_copy, GPU, F16, ins::gpu::contiguous_copy_wrapper);
REGISTER_KERNEL(contiguous_copy, GPU, BF16, ins::gpu::contiguous_copy_wrapper);
REGISTER_KERNEL(contiguous_copy, GPU, C32, ins::gpu::contiguous_copy_wrapper);
REGISTER_KERNEL(contiguous_copy, GPU, C64, ins::gpu::contiguous_copy_wrapper);
REGISTER_KERNEL(contiguous_copy, GPU, F8_E4M3, ins::gpu::contiguous_copy_wrapper);
REGISTER_KERNEL(contiguous_copy, GPU, F8_E5M2, ins::gpu::contiguous_copy_wrapper);

REGISTER_KERNEL(concat, GPU, BOOL, ins::gpu::concat_wrapper);
REGISTER_KERNEL(concat, GPU, U8, ins::gpu::concat_wrapper);
REGISTER_KERNEL(concat, GPU, I8, ins::gpu::concat_wrapper);
REGISTER_KERNEL(concat, GPU, I16, ins::gpu::concat_wrapper);
REGISTER_KERNEL(concat, GPU, I32, ins::gpu::concat_wrapper);
REGISTER_KERNEL(concat, GPU, I64, ins::gpu::concat_wrapper);
REGISTER_KERNEL(concat, GPU, U16, ins::gpu::concat_wrapper);
REGISTER_KERNEL(concat, GPU, U32, ins::gpu::concat_wrapper);
REGISTER_KERNEL(concat, GPU, U64, ins::gpu::concat_wrapper);
REGISTER_KERNEL(concat, GPU, F16, ins::gpu::concat_wrapper);
REGISTER_KERNEL(concat, GPU, BF16, ins::gpu::concat_wrapper);
REGISTER_KERNEL(concat, GPU, F32, ins::gpu::concat_wrapper);
REGISTER_KERNEL(concat, GPU, F64, ins::gpu::concat_wrapper);
REGISTER_KERNEL(concat, GPU, C32, ins::gpu::concat_wrapper);
REGISTER_KERNEL(concat, GPU, C64, ins::gpu::concat_wrapper);

REGISTER_KERNEL(repeat, GPU, F16, ins::gpu::repeat_wrapper);
REGISTER_KERNEL(repeat, GPU, BF16, ins::gpu::repeat_wrapper);
REGISTER_KERNEL(repeat, GPU, F32, ins::gpu::repeat_wrapper);
REGISTER_KERNEL(repeat, GPU, F64, ins::gpu::repeat_wrapper);
REGISTER_KERNEL(repeat, GPU, I8, ins::gpu::repeat_wrapper);
REGISTER_KERNEL(repeat, GPU, I16, ins::gpu::repeat_wrapper);
REGISTER_KERNEL(repeat, GPU, I32, ins::gpu::repeat_wrapper);
REGISTER_KERNEL(repeat, GPU, I64, ins::gpu::repeat_wrapper);
REGISTER_KERNEL(repeat, GPU, U8, ins::gpu::repeat_wrapper);
REGISTER_KERNEL(repeat, GPU, U16, ins::gpu::repeat_wrapper);
REGISTER_KERNEL(repeat, GPU, U32, ins::gpu::repeat_wrapper);
REGISTER_KERNEL(repeat, GPU, U64, ins::gpu::repeat_wrapper);
REGISTER_KERNEL(repeat, GPU, BOOL, ins::gpu::repeat_wrapper);
REGISTER_KERNEL(repeat, GPU, C32, ins::gpu::repeat_wrapper);
REGISTER_KERNEL(repeat, GPU, C64, ins::gpu::repeat_wrapper);

REGISTER_KERNEL(tile, GPU, BOOL, ins::gpu::tile_wrapper);
REGISTER_KERNEL(tile, GPU, U8, ins::gpu::tile_wrapper);
REGISTER_KERNEL(tile, GPU, I8, ins::gpu::tile_wrapper);
REGISTER_KERNEL(tile, GPU, I16, ins::gpu::tile_wrapper);
REGISTER_KERNEL(tile, GPU, I32, ins::gpu::tile_wrapper);
REGISTER_KERNEL(tile, GPU, I64, ins::gpu::tile_wrapper);
REGISTER_KERNEL(tile, GPU, U16, ins::gpu::tile_wrapper);
REGISTER_KERNEL(tile, GPU, U32, ins::gpu::tile_wrapper);
REGISTER_KERNEL(tile, GPU, U64, ins::gpu::tile_wrapper);
REGISTER_KERNEL(tile, GPU, F16, ins::gpu::tile_wrapper);
REGISTER_KERNEL(tile, GPU, BF16, ins::gpu::tile_wrapper);
REGISTER_KERNEL(tile, GPU, F32, ins::gpu::tile_wrapper);
REGISTER_KERNEL(tile, GPU, F64, ins::gpu::tile_wrapper);
REGISTER_KERNEL(tile, GPU, C32, ins::gpu::tile_wrapper);
REGISTER_KERNEL(tile, GPU, C64, ins::gpu::tile_wrapper);

REGISTER_KERNEL(pad, GPU, BOOL, ins::gpu::pad_wrapper);
REGISTER_KERNEL(pad, GPU, U8, ins::gpu::pad_wrapper);
REGISTER_KERNEL(pad, GPU, I8, ins::gpu::pad_wrapper);
REGISTER_KERNEL(pad, GPU, I16, ins::gpu::pad_wrapper);
REGISTER_KERNEL(pad, GPU, I32, ins::gpu::pad_wrapper);
REGISTER_KERNEL(pad, GPU, I64, ins::gpu::pad_wrapper);
REGISTER_KERNEL(pad, GPU, U16, ins::gpu::pad_wrapper);
REGISTER_KERNEL(pad, GPU, U32, ins::gpu::pad_wrapper);
REGISTER_KERNEL(pad, GPU, U64, ins::gpu::pad_wrapper);
REGISTER_KERNEL(pad, GPU, F16, ins::gpu::pad_wrapper);
REGISTER_KERNEL(pad, GPU, BF16, ins::gpu::pad_wrapper);
REGISTER_KERNEL(pad, GPU, F32, ins::gpu::pad_wrapper);
REGISTER_KERNEL(pad, GPU, F64, ins::gpu::pad_wrapper);
REGISTER_KERNEL(pad, GPU, C32, ins::gpu::pad_wrapper);
REGISTER_KERNEL(pad, GPU, C64, ins::gpu::pad_wrapper);


REGISTER_KERNEL(flip, GPU, BOOL, ins::gpu::flip_wrapper<bool>);
REGISTER_KERNEL(flip, GPU, U8, ins::gpu::flip_wrapper<uint8_t>);
REGISTER_KERNEL(flip, GPU, I8, ins::gpu::flip_wrapper<int8_t>);
REGISTER_KERNEL(flip, GPU, I16, ins::gpu::flip_wrapper<int16_t>);
REGISTER_KERNEL(flip, GPU, I32, ins::gpu::flip_wrapper<int32_t>);
REGISTER_KERNEL(flip, GPU, I64, ins::gpu::flip_wrapper<int64_t>);
REGISTER_KERNEL(flip, GPU, U16, ins::gpu::flip_wrapper<uint16_t>);
REGISTER_KERNEL(flip, GPU, U32, ins::gpu::flip_wrapper<uint32_t>);
REGISTER_KERNEL(flip, GPU, U64, ins::gpu::flip_wrapper<uint64_t>);
REGISTER_KERNEL(flip, GPU, F16, ins::gpu::flip_wrapper<uint16_t>);   // half storage
REGISTER_KERNEL(flip, GPU, BF16, ins::gpu::flip_wrapper<uint16_t>);  // bfloat16 storage
REGISTER_KERNEL(flip, GPU, F32, ins::gpu::flip_wrapper<float>);
REGISTER_KERNEL(flip, GPU, F64, ins::gpu::flip_wrapper<double>);
REGISTER_KERNEL(flip, GPU, C32, ins::gpu::flip_wrapper<std::complex<float>>);
REGISTER_KERNEL(flip, GPU, C64, ins::gpu::flip_wrapper<std::complex<double>>);

REGISTER_KERNEL(roll, GPU, BOOL, ins::gpu::roll_wrapper);
REGISTER_KERNEL(roll, GPU, U8, ins::gpu::roll_wrapper);
REGISTER_KERNEL(roll, GPU, I8, ins::gpu::roll_wrapper);
REGISTER_KERNEL(roll, GPU, I16, ins::gpu::roll_wrapper);
REGISTER_KERNEL(roll, GPU, I32, ins::gpu::roll_wrapper);
REGISTER_KERNEL(roll, GPU, I64, ins::gpu::roll_wrapper);
REGISTER_KERNEL(roll, GPU, U16, ins::gpu::roll_wrapper);
REGISTER_KERNEL(roll, GPU, U32, ins::gpu::roll_wrapper);
REGISTER_KERNEL(roll, GPU, U64, ins::gpu::roll_wrapper);
REGISTER_KERNEL(roll, GPU, F16, ins::gpu::roll_wrapper);
REGISTER_KERNEL(roll, GPU, BF16, ins::gpu::roll_wrapper);
REGISTER_KERNEL(roll, GPU, F32, ins::gpu::roll_wrapper);
REGISTER_KERNEL(roll, GPU, F64, ins::gpu::roll_wrapper);
REGISTER_KERNEL(roll, GPU, C32, ins::gpu::roll_wrapper);
REGISTER_KERNEL(roll, GPU, C64, ins::gpu::roll_wrapper);

REGISTER_KERNEL(diag, GPU, BOOL, ins::gpu::diag_wrapper);
REGISTER_KERNEL(diag, GPU, U8, ins::gpu::diag_wrapper);
REGISTER_KERNEL(diag, GPU, I8, ins::gpu::diag_wrapper);
REGISTER_KERNEL(diag, GPU, I16, ins::gpu::diag_wrapper);
REGISTER_KERNEL(diag, GPU, I32, ins::gpu::diag_wrapper);
REGISTER_KERNEL(diag, GPU, I64, ins::gpu::diag_wrapper);
REGISTER_KERNEL(diag, GPU, U16, ins::gpu::diag_wrapper);
REGISTER_KERNEL(diag, GPU, U32, ins::gpu::diag_wrapper);
REGISTER_KERNEL(diag, GPU, U64, ins::gpu::diag_wrapper);
REGISTER_KERNEL(diag, GPU, F16, ins::gpu::diag_wrapper);
REGISTER_KERNEL(diag, GPU, BF16, ins::gpu::diag_wrapper);
REGISTER_KERNEL(diag, GPU, F32, ins::gpu::diag_wrapper);
REGISTER_KERNEL(diag, GPU, F64, ins::gpu::diag_wrapper);
REGISTER_KERNEL(diag, GPU, C32, ins::gpu::diag_wrapper);
REGISTER_KERNEL(diag, GPU, C64, ins::gpu::diag_wrapper);

REGISTER_KERNEL(tril, GPU, BOOL, ins::gpu::tril_wrapper);
REGISTER_KERNEL(tril, GPU, U8, ins::gpu::tril_wrapper);
REGISTER_KERNEL(tril, GPU, I8, ins::gpu::tril_wrapper);
REGISTER_KERNEL(tril, GPU, I16, ins::gpu::tril_wrapper);
REGISTER_KERNEL(tril, GPU, I32, ins::gpu::tril_wrapper);
REGISTER_KERNEL(tril, GPU, I64, ins::gpu::tril_wrapper);
REGISTER_KERNEL(tril, GPU, U16, ins::gpu::tril_wrapper);
REGISTER_KERNEL(tril, GPU, U32, ins::gpu::tril_wrapper);
REGISTER_KERNEL(tril, GPU, U64, ins::gpu::tril_wrapper);
REGISTER_KERNEL(tril, GPU, F16, ins::gpu::tril_wrapper);
REGISTER_KERNEL(tril, GPU, BF16, ins::gpu::tril_wrapper);
REGISTER_KERNEL(tril, GPU, F32, ins::gpu::tril_wrapper);
REGISTER_KERNEL(tril, GPU, F64, ins::gpu::tril_wrapper);
REGISTER_KERNEL(tril, GPU, C32, ins::gpu::tril_wrapper);
REGISTER_KERNEL(tril, GPU, C64, ins::gpu::tril_wrapper);

REGISTER_KERNEL(triu, GPU, BOOL, ins::gpu::triu_wrapper);
REGISTER_KERNEL(triu, GPU, U8, ins::gpu::triu_wrapper);
REGISTER_KERNEL(triu, GPU, I8, ins::gpu::triu_wrapper);
REGISTER_KERNEL(triu, GPU, I16, ins::gpu::triu_wrapper);
REGISTER_KERNEL(triu, GPU, I32, ins::gpu::triu_wrapper);
REGISTER_KERNEL(triu, GPU, I64, ins::gpu::triu_wrapper);
REGISTER_KERNEL(triu, GPU, U16, ins::gpu::triu_wrapper);
REGISTER_KERNEL(triu, GPU, U32, ins::gpu::triu_wrapper);
REGISTER_KERNEL(triu, GPU, U64, ins::gpu::triu_wrapper);
REGISTER_KERNEL(triu, GPU, F16, ins::gpu::triu_wrapper);
REGISTER_KERNEL(triu, GPU, BF16, ins::gpu::triu_wrapper);
REGISTER_KERNEL(triu, GPU, F32, ins::gpu::triu_wrapper);
REGISTER_KERNEL(triu, GPU, F64, ins::gpu::triu_wrapper);
REGISTER_KERNEL(triu, GPU, C32, ins::gpu::triu_wrapper);
REGISTER_KERNEL(triu, GPU, C64, ins::gpu::triu_wrapper);

REGISTER_MODULE(manipulation, GPU);