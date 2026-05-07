// src/ops/manipulation.cpp
#include "insight/ops/manipulation.h"
#include "insight/ops/elementwise.h"
#include "insight/plugin/op_registry.h"

namespace ins {

    static DeviceKind get_device_kind(const Place& place) {
        return place.is_cpu() ? DeviceKind::CPU : DeviceKind::GPU;
    }

    // ========== Shape manipulation ==========

    Array reshape(const Array& x, const Shape& new_shape) {
        return x.reshape(new_shape);
    }

    Array flatten(const Array& x) {
        return x.reshape(Shape({ x.numel() }));
    }

    Array ravel(const Array& x) {
        // If contiguous, return view; otherwise copy
        if (x.is_contiguous()) {
            return flatten(x);
        }
        else {
            return x.copy().reshape(Shape({ x.numel() }));
        }
    }

    Array squeeze(const Array& x) {
        return x.squeeze();
    }

    Array squeeze(const Array& x, int axis) {
        Shape shape = x.shape();
        int ndim = shape.ndim();
        if (axis < 0) axis += ndim;
        INS_CHECK(axis >= 0 && axis < ndim, "squeeze: axis out of range");
        INS_CHECK(shape.dim(axis) == 1, "squeeze: axis dimension must be 1");

        std::vector<int64_t> new_dims;
        for (int i = 0; i < ndim; ++i) {
            if (i != axis) new_dims.push_back(shape.dim(i));
        }
        return x.reshape(Shape(new_dims));
    }

    Array unsqueeze(const Array& x, int axis) {
        return x.unsqueeze(axis);
    }

    // ========== Transpose ==========

    Array transpose(const Array& x) {
        return x.transpose();
    }

    Array permute(const Array& x, const std::vector<int>& axes) {
        return x.transpose(axes);
    }

    Array swapaxes(const Array& x, int axis1, int axis2) {
        Shape shape = x.shape();
        int ndim = shape.ndim();
        if (axis1 < 0) axis1 += ndim;
        if (axis2 < 0) axis2 += ndim;
        INS_CHECK(axis1 >= 0 && axis1 < ndim, "swapaxes: axis1 out of range");
        INS_CHECK(axis2 >= 0 && axis2 < ndim, "swapaxes: axis2 out of range");

        // 构造 permutation
        std::vector<int> perm(ndim);
        for (int i = 0; i < ndim; ++i) perm[i] = i;
        std::swap(perm[axis1], perm[axis2]);
        return x.transpose(perm);
    }

    Array moveaxis(const Array& x, int source, int destination) {
        Shape shape = x.shape();
        int ndim = shape.ndim();
        if (source < 0) source += ndim;
        if (destination < 0) destination += ndim;
        INS_CHECK(source >= 0 && source < ndim, "moveaxis: source out of range");
        INS_CHECK(destination >= 0 && destination < ndim, "moveaxis: destination out of range");

        if (source == destination) return x;

        std::vector<int> perm;
        for (int i = 0; i < ndim; ++i) {
            if (i != source) perm.push_back(i);
        }
        perm.insert(perm.begin() + destination, source);
        return x.transpose(perm);
    }

    // ========== Flipping and rotating ==========

    Array flip(const Array& x, std::optional<int> axis) {
        if (!axis.has_value()) {
            // Flip all axes
            Array result = x;
            for (int i = 0; i < x.shape().ndim(); ++i) {
                result = flip(result, i);
            }
            return result;
        }
        int ax = axis.value();
        if (ax < 0) ax += x.shape().ndim();
        OpArgs args = { x, ax };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs result = ops()["flip"][dev][x.dtype()](args);
        return std::any_cast<Array>(result[0]);
    }

    Array rot90(const Array& x, int k, const std::vector<int>& axes) {
        INS_CHECK(x.shape().ndim() >= 2, "rot90: requires at least 2 dimensions");
        INS_CHECK(axes.size() == 2, "rot90: axes must have exactly 2 elements");

        int ndim = x.shape().ndim();
        int axis1 = axes[0];
        int axis2 = axes[1];
        if (axis1 < 0) axis1 += ndim;
        if (axis2 < 0) axis2 += ndim;
        INS_CHECK(axis1 >= 0 && axis1 < ndim, "rot90: axis1 out of range");
        INS_CHECK(axis2 >= 0 && axis2 < ndim, "rot90: axis2 out of range");
        INS_CHECK(axis1 != axis2, "rot90: axes must be different");

        // Normalize k to [0, 3]
        int k_mod = ((k % 4) + 4) % 4;
        if (k_mod == 0) return x.copy();

        // Build permutation that swaps axis1 and axis2
        std::vector<int> perm(ndim);
        for (int i = 0; i < ndim; ++i) perm[i] = i;
        perm[axis1] = axis2;
        perm[axis2] = axis1;

        Array result = x;

        for (int i = 0; i < k_mod; ++i) {
            // One 90-degree counter-clockwise rotation:
            // 1. Transpose (swap axis1 and axis2)
            result = result.transpose(perm);
            // 2. Flip along axis2 (the original axis2, now at position axis1)
            result = flip(result, axis1);  // axis1 is the new position of axis2
        }

        return result;
    }

    // ========== Joining ==========

    Array concat(const std::vector<Array>& tensors, int axis) {
        if (tensors.empty()) INS_THROW("concat: no tensors provided");
        if (tensors.size() == 1) return tensors[0];

        const Shape& first_shape = tensors[0].shape();
        int ndim = first_shape.ndim();
        if (axis < 0) axis += ndim;
        INS_CHECK(axis >= 0 && axis < ndim, "concat: axis out of range");

        // Check compatibility
        DType dtype = tensors[0].dtype();
        Place place = tensors[0].place();
        std::vector<int64_t> out_dims = first_shape.dims();
        int64_t concat_size = 0;

        for (const auto& t : tensors) {
            INS_CHECK(t.dtype() == dtype, "concat: dtype mismatch");
            INS_CHECK(t.place() == place, "concat: device mismatch");
            INS_CHECK(t.shape().ndim() == ndim, "concat: dimension mismatch");
            for (int i = 0; i < ndim; ++i) {
                if (i != axis) {
                    INS_CHECK(t.shape().dim(i) == out_dims[i],
                        "concat: shape mismatch at dimension ", i);
                }
            }
            concat_size += t.shape().dim(axis);
        }
        out_dims[axis] = concat_size;
        Shape out_shape(out_dims);

        OpArgs args = { tensors, axis, out_shape };
        DeviceKind dev = get_device_kind(place);
        OpArgs result = ops()["concat"][dev][dtype](args);

        return std::any_cast<Array>(result[0]);
    }

    Array stack(const std::vector<Array>& tensors, int axis) {
        if (tensors.empty()) INS_THROW("stack: no tensors provided");

        const Shape& first_shape = tensors[0].shape();
        int ndim = first_shape.ndim();
        if (axis < 0) axis += ndim + 1;
        INS_CHECK(axis >= 0 && axis <= ndim, "stack: axis out of range");

        // Check all tensors have same shape
        for (const auto& t : tensors) {
            INS_CHECK(t.shape() == first_shape, "stack: shape mismatch");
            INS_CHECK(t.dtype() == tensors[0].dtype(), "stack: dtype mismatch");
            INS_CHECK(t.place() == tensors[0].place(), "stack: device mismatch");
        }

        // First unsqueeze each tensor, then concat
        std::vector<Array> expanded;
        for (const auto& t : tensors) {
            expanded.push_back(unsqueeze(t, axis));
        }
        return concat(expanded, axis);
    }

    // ========== Splitting ==========

    std::vector<Array> split(const Array& x, int indices_or_sections, int axis) {
        Shape shape = x.shape();
        int ndim = shape.ndim();
        if (axis < 0) axis += ndim;
        INS_CHECK(axis >= 0 && axis < ndim, "split: axis out of range");

        int64_t dim_size = shape.dim(axis);
        if (dim_size % indices_or_sections != 0) {
            INS_THROW("split: axis dimension must be divisible by number of splits");
        }

        int64_t split_size = dim_size / indices_or_sections;
        std::vector<int64_t> indices;
        for (int64_t i = split_size; i < dim_size; i += split_size) {
            indices.push_back(i);
        }
        return split(x, indices, axis);
    }

    std::vector<Array> split(const Array& x, const std::vector<int64_t>& indices, int axis) {
        Shape shape = x.shape();
        int ndim = shape.ndim();
        if (axis < 0) axis += ndim;
        INS_CHECK(axis >= 0 && axis < ndim, "split: axis out of range");

        std::vector<Array> result;
        int64_t start = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            int64_t end = indices[i];

            std::vector<Slice> slices(ndim, Slice::all());
            slices[axis] = Slice(start, end);
            Array part = x.slice(slices);
            result.push_back(part);
            start = end;
        }

        // Last part
        std::vector<Slice> slices(ndim, Slice::all());
        slices[axis] = Slice(start, shape.dim(axis));
        Array last_part = x.slice(slices);
        result.push_back(last_part);

        return result;
    }

    // ========== Tiling and Repeating ==========

    Array repeat(const Array& x, int repeats, std::optional<int> axis) {
        if (!axis.has_value()) {
            // Flatten then repeat
            Array flat = ravel(x);
            return repeat(flat, repeats, 0);
        }
        int ax = axis.value();
        if (ax < 0) ax += x.shape().ndim();

        OpArgs args = { x, repeats, ax };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs result = ops()["repeat"][dev][x.dtype()](args);
        return std::any_cast<Array>(result[0]);
    }

    Array repeat(const Array& x, const std::vector<int>& repeats, int axis) {
        // For simplicity, only support scalar repeats for now
        INS_THROW("repeat with vector repeats not yet implemented");
    }

    Array tile(const Array& x, const Shape& reps) {
        OpArgs args = { x, reps };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs result = ops()["tile"][dev][x.dtype()](args);
        return std::any_cast<Array>(result[0]);
    }

    // ========== Padding ==========

    Array pad(const Array& x, const std::vector<int64_t>& pad_width, double constant_value) {
        OpArgs args = { x, pad_width, constant_value };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs result = ops()["pad"][dev][x.dtype()](args);
        return std::any_cast<Array>(result[0]);
    }

    // ========== Rolling ==========

    Array roll(const Array& x, int shift, std::optional<int> axis) {
        OpArgs args = { x, shift };
        if (axis.has_value()) {
            args.emplace_back(axis.value());
        }
        else {
            args.emplace_back(-1);  // -1 means flatten
        }
        DeviceKind dev = get_device_kind(x.place());
        OpArgs result = ops()["roll"][dev][x.dtype()](args);
        return std::any_cast<Array>(result[0]);
    }

    // ========== Diagonal ==========

    Array diag(const Array& x, int k) {
        OpArgs args = { x, k };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs result = ops()["diag"][dev][x.dtype()](args);
        return std::any_cast<Array>(result[0]);
    }

    Array diagonal(const Array& x, int offset, int axis1, int axis2) {
        // For 2D arrays, this is similar to diag
        return diag(x, offset);
    }

    // ========== Triangular ==========

    Array tril(const Array& x, int k) {
        OpArgs args = { x, k };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs result = ops()["tril"][dev][x.dtype()](args);
        return std::any_cast<Array>(result[0]);
    }

    Array triu(const Array& x, int k) {
        OpArgs args = { x, k };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs result = ops()["triu"][dev][x.dtype()](args);
        return std::any_cast<Array>(result[0]);
    }

    // ========== Slicing (Views) ==========

    /// Slice a single dimension
    Array slice(Array& x, int dim, int64_t start, int64_t stop, int64_t step)
    {
        return x.slice(dim, start, stop, step);
    }

    /// Multi-dimensional slice with Slice objects
    Array slice(Array& x, const std::vector<Slice>& slices)
    {
        return x.slice(slices);
    }

    // ========== diff ==========
    Array diff(const Array& x, int n, int axis) {
        INS_CHECK(x.defined(), "diff: input is undefined");
        INS_CHECK(n >= 0, "diff: n must be non-negative");

        if (n == 0) {
            return x.copy();
        }

        int ndim = x.shape().ndim();
        int ax = axis;
        if (ax < 0) ax += ndim;
        INS_CHECK(ax >= 0 && ax < ndim, "diff: axis out of range");

        Array result = x;
        for (int i = 0; i < n; ++i) {
            int64_t axis_size = result.shape().dim(ax);
            INS_CHECK(axis_size > 1, "diff: axis size must be at least 2");

            // front = result[..., 0:-1, ...]
            // back  = result[..., 1:, ...]
            std::vector<Slice> slices_front(ndim, Slice::all());
            std::vector<Slice> slices_back(ndim, Slice::all());
            slices_front[ax] = Slice(0, axis_size - 1);
            slices_back[ax] = Slice(1, axis_size);

            Array front = result.slice(slices_front);
            Array back = result.slice(slices_back);

            result = sub(back, front);
        }

        return result;
    }

} // namespace ins