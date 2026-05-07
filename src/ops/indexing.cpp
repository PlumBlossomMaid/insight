// src/ops/indexing.cpp
#include "insight/core/array.h"
#include "insight/ops/indexing.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/broadcast.h"
#include "insight/ops/creation.h"
#include "insight/ops/reduction.h"
#include "insight/plugin/op_registry.h"
#include "insight/utils/promotion.h"
#include <cmath>
#include <iostream>
#include "insight/io/print.h"
namespace ins {

    static DeviceKind get_device_kind(const Place& place) {
        return place.is_cpu() ? DeviceKind::CPU : DeviceKind::GPU;
    }

    // ========== Helper: prepare indexing by flattening ==========
    static Array prepare_flattened(const Array& x, std::optional<int> axis) {
        if (!axis.has_value()) {
            return x.reshape(Shape({ x.numel() }));
        }
        return x.contiguous();
    }

    static Shape broadcast_shapes_for_indexing(const Shape& x_shape,
        const Shape& idx_shape,
        int axis) {
        int ndim = x_shape.ndim();
        std::vector<int64_t> out_dims(ndim);
        for (int i = 0; i < ndim; ++i) {
            if (i == axis) {
                out_dims[i] = idx_shape.dim(i);
            }
            else {
                int64_t xdim = x_shape.dim(i);
                int64_t idxdim = (i < idx_shape.ndim()) ? idx_shape.dim(i) : 1;
                if (xdim != idxdim && xdim != 1 && idxdim != 1) {
                    INS_THROW("broadcast_shapes_for_indexing: shape mismatch at axis ", i,
                        ": ", xdim, " vs ", idxdim);
                }
                out_dims[i] = std::max(xdim, idxdim);
            }
        }
        return Shape(out_dims);
    }

    // ========== take ==========
    Array take(const Array& x, const Array& indices, std::optional<int> axis) {
        Array prepared = prepare_flattened(x, axis);
        Array idx = indices;
        if (idx.dtype() != DType::I64) {
            idx = idx.to(DType::I64);
        }
        if (idx.place() != x.place()) {
            idx = idx.to(x.place());
        }

        Shape out_shape;
        if (axis.has_value()) {
            std::vector<int64_t> dims = x.shape().dims();
            int ax = axis.value();
            if (ax < 0) ax += x.shape().ndim();
            dims[ax] = idx.numel();
            out_shape = Shape(dims);
        }
        else {
            out_shape = Shape({ idx.numel() });
        }

        Array result(out_shape, x.dtype(), x.place());

        OpArgs args = { result, prepared, idx, axis.value_or(-1), axis.has_value() };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["take"][dev][x.dtype()](args);

        return std::any_cast<Array>(output[0]);
    }

    // ========== take_along_axis ==========
    Array take_along_axis(const Array& x, const Array& indices, int axis) {
        int ndim = x.shape().ndim();
        int ax = axis;
        if (ax < 0) ax += ndim;
        INS_CHECK(ax >= 0 && ax < ndim, "take_along_axis: axis out of range");

        Array idx = indices;
        if (idx.dtype() != DType::I64) {
            idx = idx.to(DType::I64);
        }
        if (idx.place() != x.place()) {
            idx = idx.to(x.place());
        }

        Shape out_shape = broadcast_shapes_for_indexing(x.shape(), idx.shape(), ax);
        Array result(out_shape, x.dtype(), x.place());

        OpArgs args = { result, x, idx, ax };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["take_along_axis"][dev][x.dtype()](args);

        return std::any_cast<Array>(output[0]);
    }

    // ========== put ==========
    Array put(const Array& x, const Array& indices, const Array& values, std::optional<int> axis) {
        Array prepared = prepare_flattened(x, axis);
        Array idx = indices;
        if (idx.dtype() != DType::I64) {
            idx = idx.to(DType::I64);
        }
        if (idx.place() != x.place()) {
            idx = idx.to(x.place());
        }
        Array val = values;
        if (val.numel() == 1) {
            val = broadcast_to(val, Shape({ idx.numel() }));
        }
        INS_CHECK(val.numel() == idx.numel(), "put: values must broadcast to indices shape");

        Array result = prepared.copy();

        OpArgs args = { result, idx, val };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["put"][dev][x.dtype()](args);

        Array res = std::any_cast<Array>(output[0]);
        if (!axis.has_value()) {
            return res.reshape(x.shape());
        }
        return res;
    }

    // ========== put_along_axis ==========
    Array put_along_axis(const Array& x, const Array& indices, const Array& values, int axis) {
        int ndim = x.shape().ndim();
        int ax = axis;
        if (ax < 0) ax += ndim;
        INS_CHECK(ax >= 0 && ax < ndim, "put_along_axis: axis out of range");

        Array idx = indices;
        if (idx.dtype() != DType::I64) {
            idx = idx.to(DType::I64);
        }
        if (idx.place() != x.place()) {
            idx = idx.to(x.place());
        }
        Array val = values;
        if (val.numel() == 1) {
            Shape target_shape = broadcast_shapes_for_indexing(x.shape(), idx.shape(), ax);
            val = broadcast_to(val, target_shape);
        }

        Array result = x.copy();

        OpArgs args = { result, idx, val, ax };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["put_along_axis"][dev][x.dtype()](args);

        return std::any_cast<Array>(output[0]);
    }

    // ========== gather ==========
    Array gather(const Array& x, int dim, const Array& index) {
        return take_along_axis(x, index, dim);
    }

    // ========== scatter ==========
    Array scatter(const Array& x, int dim, const Array& index, const Array& src) {
        return scatter_reduce(x, dim, index, src, "replace");
    }

    Array scatter_add(const Array& x, int dim, const Array& index, const Array& src) {
        return scatter_reduce(x, dim, index, src, "add");
    }

    Array scatter_reduce(const Array& x, int dim, const Array& index, const Array& src,
        const std::string& reduce) {
        int ndim = x.shape().ndim();
        int d = dim;
        if (d < 0) d += ndim;
        INS_CHECK(d >= 0 && d < ndim, "scatter_reduce: dim out of range");

        Array idx = index;
        if (idx.dtype() != DType::I64) {
            idx = idx.to(DType::I64);
        }
        if (idx.place() != x.place()) {
            idx = idx.to(x.place());
        }
        Array src_broadcast = src;
        if (src_broadcast.shape() != idx.shape()) {
            src_broadcast = broadcast_to(src_broadcast, idx.shape());
        }

        Array result = x.copy();

        OpArgs args = { result, idx, src_broadcast, d, reduce };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["scatter_reduce"][dev][x.dtype()](args);

        return std::any_cast<Array>(output[0]);
    }

    // ========== masked_select ==========
    Array masked_select(const Array& x, const Array& mask) {
        Array condition = mask;
        if (condition.dtype() != DType::BOOL) {
            condition = condition.to(DType::BOOL);
        }
        if (condition.place() != x.place()) {
            condition = condition.to(x.place());
        }

        if (condition.shape() != x.shape()) {
            condition = broadcast_to(condition, x.shape());
        }

        Array flattened_cond = condition.reshape(Shape({ condition.numel() }));
        Array cnt = count_nonzero(flattened_cond);
        int64_t count = cnt.item<int64_t>();

        Array result(Shape({ count }), x.dtype(), x.place());

        OpArgs args = { result, x, condition };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["masked_select"][dev][x.dtype()](args);

        return std::any_cast<Array>(output[0]);
    }

    // ========== compress ==========
    Array compress(const Array& x, const Array& condition, std::optional<int> axis) {
        int ndim = x.shape().ndim();
        int ax = axis.value_or(0);
        if (ax < 0) ax += ndim;
        INS_CHECK(ax >= 0 && ax < ndim, "compress: axis out of range");

        Array cond = condition;
        if (cond.dtype() != DType::BOOL) {
            cond = cond.to(DType::BOOL);
        }
        int64_t axis_dim = x.shape().dim(ax);
        if (cond.numel() != axis_dim) {
            INS_THROW("compress: condition length must match axis dimension");
        }

        Array cnt = count_nonzero(cond);
        int64_t keep_count = cnt.item<int64_t>();

        std::vector<int64_t> out_dims = x.shape().dims();
        out_dims[ax] = keep_count;
        Shape out_shape(out_dims);

        Array result(out_shape, x.dtype(), x.place());

        OpArgs args = { result, x, cond, ax };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["compress"][dev][x.dtype()](args);

        return std::any_cast<Array>(output[0]);
    }

    // ========== where ==========
    Array where(const Array& condition, const Array& x, const Array& y) {
        auto broadcasted = broadcast_arrays({ condition, x, y });
        Array cond = broadcasted[0];
        if (cond.dtype() != DType::BOOL) {
            cond = cond.to(DType::BOOL);
        }
        Array X = broadcasted[1];
        Array Y = broadcasted[2];

        Array result(cond.shape(), X.dtype(), X.place());

        OpArgs args = { result, cond, X, Y };
        DeviceKind dev = get_device_kind(condition.place());
        OpArgs output = ops()["where"][dev][X.dtype()](args);

        return std::any_cast<Array>(output[0]);
    }

    // ========== nonzero ==========
    Array nonzero(const Array& x) {
        OpArgs args = { x };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["nonzero"][dev][x.dtype()](args);
        return std::any_cast<Array>(output[0]);
    }

    // ========== flatnonzero ==========
    Array flatnonzero(const Array& x) {
        Array flat_x = x.reshape(Shape({ x.numel() }));
        Array nz = nonzero(flat_x);

        if (nz.numel() == 0) {
            return nz;
        }
        return nz.reshape(Shape({ nz.numel() }));
    }

    // ========== argsort ==========
    Array argsort(const Array& x, int axis, bool descending) {
        int ndim = x.shape().ndim();
        int ax = axis;
        if (ax < 0) ax += ndim;
        INS_CHECK(ax >= 0 && ax < ndim, "argsort: axis out of range");

        Array prepared = x;
        std::vector<int> perm(ndim);
        for (int i = 0; i < ndim; ++i) perm[i] = i;
        if (ax != ndim - 1) {
            std::swap(perm[ax], perm[ndim - 1]);
            prepared = prepared.transpose(perm);
            prepared = prepared.contiguous();
        }

        Shape out_shape = x.shape();
        Array result(out_shape, DType::I64, x.place());

        OpArgs args = { result, prepared, descending };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["argsort"][dev][x.dtype()](args);

        Array res = std::any_cast<Array>(output[0]);
        if (ax != ndim - 1) {
            std::vector<int> inv_perm(ndim);
            for (int i = 0; i < ndim; ++i) inv_perm[perm[i]] = i;
            res = res.transpose(inv_perm);
        }
        return res;
    }

    // ========== sort ==========
    Array sort(const Array& x, int axis, bool descending) {
        Array indices = argsort(x, axis, descending);
        return take_along_axis(x, indices, axis);
    }

    // ========== topk ==========
    std::tuple<Array, Array> topk(const Array& x, int64_t k, int axis,
        bool largest, bool sorted) {
        int ndim = x.shape().ndim();
        int ax = axis;
        if (ax < 0) ax += ndim;
        INS_CHECK(ax >= 0 && ax < ndim, "topk: axis out of range");
        INS_CHECK(k > 0, "topk: k must be positive");

        int64_t axis_size = x.shape().dim(ax);
        if (k > axis_size) k = axis_size;

        Array prepared = x;
        std::vector<int> perm(ndim);
        for (int i = 0; i < ndim; ++i) perm[i] = i;
        if (ax != ndim - 1) {
            std::swap(perm[ax], perm[ndim - 1]);
            prepared = prepared.transpose(perm);
            prepared = prepared.contiguous();
        }

        std::vector<int64_t> out_dims = prepared.shape().dims();
        out_dims.back() = k;
        Shape out_shape(out_dims);

        Array values(out_shape, x.dtype(), x.place());
        Array indices(out_shape, DType::I64, x.place());

        OpArgs args = { values, indices, prepared, k, largest, sorted };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["topk"][dev][x.dtype()](args);

        Array vals = std::any_cast<Array>(output[0]);
        Array idxs = std::any_cast<Array>(output[1]);

        if (ax != ndim - 1) {
            std::vector<int> inv_perm(ndim);
            for (int i = 0; i < ndim; ++i) inv_perm[perm[i]] = i;
            vals = vals.transpose(inv_perm);
            idxs = idxs.transpose(inv_perm);
        }

        return { vals, idxs };
    }

    // ========== searchsorted ==========
    Array searchsorted(const Array& x, const Array& v, const std::string& side,
        std::optional<Array> sorter) {
        INS_CHECK(x.shape().ndim() == 1, "searchsorted: x must be 1D");
        Array sorted_x = x;
        if (sorter.has_value()) {
            sorted_x = take(sorted_x, sorter.value());
        }

        INS_CHECK(sorted_x.place() == v.place() || v.place().is_cpu(),
            "searchsorted: x and v must be on same device or v on CPU");

        Array result(v.shape(), DType::I64, v.place());

        OpArgs args = { result, sorted_x, v, side };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["searchsorted"][dev][x.dtype()](args);

        return std::any_cast<Array>(output[0]);
    }

    // ========== unique ==========
    UniqueResult unique(const Array& x, bool return_indices,
        bool return_inverse, bool return_counts) {
        Array flattened = x.reshape(Shape({ x.numel() }));

        OpArgs args = { flattened, return_indices, return_inverse, return_counts };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["unique"][dev][x.dtype()](args);

        UniqueResult result;
        result.unique = std::any_cast<Array>(output[0]);
        size_t idx = 1;
        if (return_indices && idx < output.size()) {
            result.indices = std::any_cast<Array>(output[idx++]);
        }
        if (return_inverse && idx < output.size()) {
            result.inverse = std::any_cast<Array>(output[idx++]);
        }
        if (return_counts && idx < output.size()) {
            result.counts = std::any_cast<Array>(output[idx++]);
        }
        return result;
    }

    // ========== partition ==========
    Array partition(const Array& x, int64_t kth, int axis) {
        int ndim = x.shape().ndim();
        int ax = axis;
        if (ax < 0) ax += ndim;
        INS_CHECK(ax >= 0 && ax < ndim, "partition: axis out of range");
        INS_CHECK(kth >= 0 && kth < x.shape().dim(ax),
            "partition: kth out of range");

        // Prepare output (same shape as input)
        Array result(x.shape(), x.dtype(), x.place());

        OpArgs args = { result, x, kth, ax };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["partition"][dev][x.dtype()](args);

        return std::any_cast<Array>(output[0]);
    }

    Array argpartition(const Array& x, int64_t kth, int axis) {
        int ndim = x.shape().ndim();
        int ax = axis;
        if (ax < 0) ax += ndim;
        INS_CHECK(ax >= 0 && ax < ndim, "argpartition: axis out of range");
        INS_CHECK(kth >= 0 && kth < x.shape().dim(ax),
            "argpartition: kth out of range");

        // Prepare output (indices, same shape as input)
        Array result(x.shape(), DType::I64, x.place());

        OpArgs args = { result, x, kth, ax };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["argpartition"][dev][x.dtype()](args);

        return std::any_cast<Array>(output[0]);
    }

    // ========== lexsort ==========

    Array lexsort(const Array& keys, int axis) {
        int ndim = keys.shape().ndim();
        int ax = axis;
        if (ax < 0) ax += ndim;

        INS_CHECK(ndim >= 1, "lexsort: keys must be at least 1D");
        INS_CHECK(ax >= 0 && ax < ndim, "lexsort: axis out of range");

        // Step 1: Move target axis to last position
        std::vector<int> perm(ndim);
        for (int i = 0; i < ndim; ++i) perm[i] = i;
        if (ax != ndim - 1) {
            std::swap(perm[ax], perm[ndim - 1]);
        }

        Array transposed = keys.transpose(perm);
        transposed = transposed.contiguous();

        const Shape& trans_shape = transposed.shape();

        // Step 2: Determine batch dimensions (all except last axis)
        int64_t last_dim = trans_shape.dim(ndim - 1);
        int64_t batch_size = 1;
        for (int i = 0; i < ndim - 1; ++i) {
            batch_size *= trans_shape.dim(i);
        }

        // Number of keys = total_size / (batch_size * last_dim)
        int64_t total = transposed.numel();
        int64_t nkeys = total / (batch_size * last_dim);

        // Step 3: Prepare output (same shape as input)
        Array result(keys.shape(), DType::I64, keys.place());

        // Step 4: Call backend
        OpArgs args = { result, transposed, batch_size, last_dim, nkeys };
        DeviceKind dev = get_device_kind(keys.place());
        OpArgs output = ops()["lexsort"][dev][keys.dtype()](args);

        Array res = std::any_cast<Array>(output[0]);

        // Step 5: Transpose back if needed
        if (ax != ndim - 1) {
            std::vector<int> inv_perm(ndim);
            for (int i = 0; i < ndim; ++i) {
                inv_perm[perm[i]] = i;
            }
            res = res.transpose(inv_perm);
        }

        return res;
    }

    // ========== indices ==========
    Array indices(const Shape& shape, bool sparse) {
        int ndim = shape.ndim();
        if (sparse) {
            std::vector<Array> sparse_results;
            for (int d = 0; d < ndim; ++d) {
                Array coord = arange(shape.dim(d), DType::I64);
                sparse_results.push_back(coord);
            }
            // Sparse mode returns a tuple - for now, just return dense
            INS_THROW("indices: sparse mode not yet implemented - use dense mode");
        }

        std::vector<int64_t> out_dims;
        out_dims.push_back(ndim);
        for (int i = 0; i < ndim; ++i) {
            out_dims.push_back(shape.dim(i));
        }
        Shape out_shape(out_dims);

        Array result(out_shape, DType::I64, CPUPlace());

        OpArgs args = { result, shape };
        DeviceKind dev = get_device_kind(CPUPlace());
        OpArgs output = ops()["indices"][dev][DType::I64](args);

        return std::any_cast<Array>(output[0]);
    }

    // ========== ix_ ==========
    std::vector<Array> ix_(const std::vector<Array>& arrays) {
        int n = arrays.size();
        if (n == 0) return {};

        for (const auto& arr : arrays) {
            INS_CHECK(arr.shape().ndim() == 1, "ix_: all inputs must be 1D");
        }

        // Build broadcast shapes: each array gets shape with 1s except its own dimension
        std::vector<Shape> target_shapes(n);
        for (int i = 0; i < n; ++i) {
            std::vector<int64_t> dims(n, 1);
            dims[i] = arrays[i].numel();
            target_shapes[i] = Shape(dims);
        }

        std::vector<Array> result;
        for (int i = 0; i < n; ++i) {
            // Reshape to (len, 1, 1, ...)
            Array reshaped = arrays[i].reshape(target_shapes[i]);
            // Do NOT broadcast further - keep as view
            result.push_back(reshaped);
        }
        return result;
    }

    // ========== interp ==========
    // Helper: extract scalar value from array at given index, converting to double
    static double extract_scalar(const Array& arr, int64_t idx) {
        Array scalar = arr.at(idx);
        if (scalar.dtype() == DType::F32) {
            return static_cast<double>(scalar.item<float>());
        }
        else if (scalar.dtype() == DType::F64) {
            return scalar.item<double>();
        }
        else {
            INS_THROW("extract_scalar: unsupported dtype ", dtype_name(scalar.dtype()));
        }
    }

    // ========== interp ==========
    Array interp(const Array& x, const Array& xp, const Array& fp,
        std::optional<double> left, std::optional<double> right) {
        INS_CHECK(x.defined() && xp.defined() && fp.defined(),
            "interp: inputs are undefined");
        INS_CHECK(xp.shape().ndim() == 1, "interp: xp must be 1D");
        INS_CHECK(fp.shape().ndim() == 1, "interp: fp must be 1D");
        INS_CHECK(xp.numel() == fp.numel(), "interp: xp and fp must have same length");

        // 确保 xp 是递增的
        Array sort_idx = argsort(xp);
        Array sorted_xp = take(xp, sort_idx);
        Array sorted_fp = take(fp, sort_idx);

        // 获取边界值（使用 at() 获取标量，自动处理类型转换）
        double left_val = left.has_value() ? left.value() : extract_scalar(sorted_fp, 0);
        double right_val = right.has_value() ? right.value() : extract_scalar(sorted_fp, -1);
        double xp_min = extract_scalar(sorted_xp, 0);
        double xp_max = extract_scalar(sorted_xp, -1);

        // 查找每个 x 在 xp 中的右区间索引
        Array right_idxs = searchsorted(sorted_xp, x, "right");

        // 左区间索引
        Array left_idxs = sub(right_idxs, ins::Array(1));

        // 处理边界索引（避免 -1）
        Array safe_left = maximum(left_idxs, zeros_like(left_idxs));
        Array safe_right = minimum(right_idxs, full(right_idxs.shape(), sorted_xp.numel() - 1, DType::I64));

        // 取端点值
        Array x_left = take(sorted_xp, safe_left);
        Array x_right = take(sorted_xp, safe_right);
        Array y_left = take(sorted_fp, safe_left);
        Array y_right = take(sorted_fp, safe_right);

        // 线性插值
        Array t = div(sub(x, x_left), sub(x_right, x_left));
        Array interp_vals = add(y_left, mul(t, sub(y_right, y_left)));

        // 边界处理：x <= xp_min 用 left_val，x >= xp_max 用 right_val
        // 使用 broadcast 让标量参与比较
        Array result = where(less_equal(x, full(x.shape(), xp_min, interp_vals.dtype())),
            full(x.shape(), left_val, interp_vals.dtype()),
            interp_vals);
        result = where(greater_equal(x, full(x.shape(), xp_max, interp_vals.dtype())),
            full(x.shape(), right_val, interp_vals.dtype()),
            result);

        return result;
    }

} // namespace ins