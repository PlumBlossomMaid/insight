// src/ops/reduction.cpp
#include "insight/ops/reduction.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/creation.h"
#include "insight/ops/manipulation.h"
#include "insight/plugin/op_registry.h"
#include "insight/utils/promotion.h"
#include <cmath>

namespace ins {

    static DeviceKind get_device_kind(const Place& place) {
        return place.is_cpu() ? DeviceKind::CPU : DeviceKind::GPU;
    }

    // ========== ReductionInfo: prepare reduction by moving axis to last ==========

    struct ReductionInfo {
        bool flatten_all = false;
        int64_t batch_size = 1;
        int64_t reduce_size = 1;
        Shape out_shape;
        std::vector<int> perm;          // permutation to move reduction axis to last
        std::vector<int> inv_perm;      // inverse permutation to restore original order
        int reduced_axis = -1;          // original axis index
        int ndim = 0;
    };

    static ReductionInfo prepare_reduction(const Shape& in_shape, std::optional<int> axis, bool keepdim) {
        ReductionInfo info;
        info.ndim = in_shape.ndim();

        if (!axis.has_value()) {
            info.flatten_all = true;
            info.batch_size = 1;
            info.reduce_size = in_shape.numel();
            if (keepdim) {
                std::vector<int64_t> dims(info.ndim, 1);
                info.out_shape = Shape(dims);
            }
            else {
                info.out_shape = Shape({});
            }
            return info;
        }

        int ax = axis.value();
        if (ax < 0) ax += info.ndim;
        info.reduced_axis = ax;

        // Build permutation: move reduction axis to last position
        info.perm.reserve(info.ndim);
        for (int i = 0; i < info.ndim; ++i) {
            if (i != ax) info.perm.push_back(i);
        }
        info.perm.push_back(ax);

        // Build inverse permutation
        info.inv_perm.resize(info.ndim);
        for (int i = 0; i < info.ndim; ++i) {
            info.inv_perm[info.perm[i]] = i;
        }

        // Compute output shape
        std::vector<int64_t> out_dims;
        for (int i = 0; i < info.ndim; ++i) {
            if (i == ax) {
                if (keepdim) out_dims.push_back(1);
            }
            else {
                out_dims.push_back(in_shape.dim(i));
            }
        }
        info.out_shape = Shape(out_dims);

        // Compute batch size (product of non-reduced dimensions)
        info.batch_size = 1;
        for (int i = 0; i < info.ndim - 1; ++i) {
            info.batch_size *= in_shape.dim(info.perm[i]);
        }
        info.reduce_size = in_shape.dim(ax);

        return info;
    }

    static Array prepare_input(const Array& x, const ReductionInfo& info) {
        if (info.flatten_all) {
            return x.reshape(Shape({ x.numel() }));
        }
        if (info.perm.empty()) {
            return x.contiguous();
        }

        // Check if permutation is identity
        bool is_identity = true;
        for (size_t i = 0; i < info.perm.size(); ++i) {
            if (info.perm[i] != static_cast<int>(i)) {
                is_identity = false;
                break;
            }
        }
        if (is_identity) {
            return x.contiguous();
        }

        Array transposed = x.transpose(info.perm);
        return transposed.contiguous();
    }

    static Array transpose_output(const Array& result, const ReductionInfo& info) {
        if (info.flatten_all || info.perm.empty() || info.inv_perm.empty()) {
            return result;
        }

        int result_ndim = result.shape().ndim();
        if (result_ndim != static_cast<int>(info.inv_perm.size())) {
            return result;
        }

        // Check if inverse permutation is identity
        bool is_identity = true;
        for (int i = 0; i < result_ndim; ++i) {
            if (info.inv_perm[i] != i) {
                is_identity = false;
                break;
            }
        }
        if (is_identity) {
            return result;
        }

        Array transposed = result.transpose(info.inv_perm);
        return transposed.contiguous();
    }

    static Array post_process_keepdim(const Array& result, const ReductionInfo& info, bool keepdim, std::optional<int> axis) {
        if (!keepdim && axis.has_value()) {
            int ax = info.reduced_axis;
            if (ax < result.shape().ndim() && result.shape().dim(ax) == 1) {
                return result.squeeze(ax);
            }
        }
        return result;
    }

    // ========== sum ==========
    Array sum(const Array& x, std::optional<int> axis, bool keepdim) {
        ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
        Array prepared = prepare_input(x, info);

        Shape out_shape = info.out_shape;
        if (info.flatten_all && !keepdim) {
            out_shape = Shape({});
        }

        Array result(out_shape, x.dtype(), x.place());

        OpArgs args = { result, prepared, info.batch_size, info.reduce_size };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["sum"][dev][x.dtype()](args);

        Array res = std::any_cast<Array>(output[0]);
        res = transpose_output(res, info);

        if (keepdim && axis.has_value() && !info.flatten_all) {
            int ax = axis.value();
            int ndim = x.shape().ndim();
            if (ax < 0) ax += ndim;

            if (res.shape().ndim() == ndim) {
                std::vector<int> perm(ndim);
                for (int i = 0; i < ndim; ++i) {
                    if (i == ax) {
                        perm[i] = ndim - 1;
                    }
                    else if (i < ax) {
                        perm[i] = i;
                    }
                    else {
                        perm[i] = i - 1;
                    }
                }
                res = res.transpose(perm);
            }
        }

        return res;
    }

    // ========== mean ==========
    Array mean(const Array& x, std::optional<int> axis, bool keepdim) {
        // 计算 sum
        Array s = sum(x, axis, true);

        int64_t n = axis.has_value() ? x.shape().dim(axis.value()) : x.numel();
        double inv_n = 1.0 / static_cast<double>(n);

        DType out_dtype = is_integer(x.dtype()) ? DType::F64 : x.dtype();
        Array s_cast = (s.dtype() != out_dtype) ? s.to(out_dtype) : s;

        Array divisor = full(s_cast.shape(), inv_n, out_dtype, s_cast.place());

        Array result = mul(s_cast, divisor);

        if (!keepdim && axis.has_value()) {
            int ax = axis.value();
            int ndim = x.shape().ndim();
            if (ax < 0) ax += ndim;
            if (result.shape().ndim() > 0 && result.shape().dim(result.shape().ndim() - 1) == 1) {
                result = result.squeeze(result.shape().ndim() - 1);
            }
        }

        return result;
    }

    // ========== max ==========
    Array max(const Array& x, std::optional<int> axis, bool keepdim) {
        ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
        Array prepared = prepare_input(x, info);

        Shape out_shape = info.out_shape;
        if (info.flatten_all && !keepdim) {
            out_shape = Shape({});
        }

        Array result(out_shape, x.dtype(), x.place());

        OpArgs args = { result, prepared, info.batch_size, info.reduce_size };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["max"][dev][x.dtype()](args);

        Array res = std::any_cast<Array>(output[0]);
        res = transpose_output(res, info);
        return post_process_keepdim(res, info, keepdim, axis);
    }

    // ========== min ==========
    Array min(const Array& x, std::optional<int> axis, bool keepdim) {
        ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
        Array prepared = prepare_input(x, info);

        Shape out_shape = info.out_shape;
        if (info.flatten_all && !keepdim) {
            out_shape = Shape({});
        }

        Array result(out_shape, x.dtype(), x.place());

        OpArgs args = { result, prepared, info.batch_size, info.reduce_size };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["min"][dev][x.dtype()](args);

        Array res = std::any_cast<Array>(output[0]);
        res = transpose_output(res, info);
        return post_process_keepdim(res, info, keepdim, axis);
    }

    // ========== prod ==========
    Array prod(const Array& x, std::optional<int> axis, bool keepdim) {
        ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
        Array prepared = prepare_input(x, info);

        Shape out_shape = info.out_shape;
        if (info.flatten_all && !keepdim) {
            out_shape = Shape({});
        }

        Array result(out_shape, x.dtype(), x.place());

        OpArgs args = { result, prepared, info.batch_size, info.reduce_size };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["prod"][dev][x.dtype()](args);

        Array res = std::any_cast<Array>(output[0]);
        res = transpose_output(res, info);
        return post_process_keepdim(res, info, keepdim, axis);
    }

    // ========== any ==========
    Array any(const Array& x, std::optional<int> axis, bool keepdim) {
        ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
        Array prepared = prepare_input(x, info);

        Shape out_shape = info.out_shape;
        if (info.flatten_all && !keepdim) {
            out_shape = Shape({});
        }

        Array result(out_shape, DType::BOOL, x.place());

        OpArgs args = { result, prepared, info.batch_size, info.reduce_size };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["any"][dev][x.dtype()](args);

        Array res = std::any_cast<Array>(output[0]);
        res = transpose_output(res, info);
        return post_process_keepdim(res, info, keepdim, axis);
    }

    // ========== all ==========
    Array all(const Array& x, std::optional<int> axis, bool keepdim) {
        ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
        Array prepared = prepare_input(x, info);

        Shape out_shape = info.out_shape;
        if (info.flatten_all && !keepdim) {
            out_shape = Shape({});
        }

        Array result(out_shape, DType::BOOL, x.place());

        OpArgs args = { result, prepared, info.batch_size, info.reduce_size };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["all"][dev][x.dtype()](args);

        Array res = std::any_cast<Array>(output[0]);
        res = transpose_output(res, info);
        return post_process_keepdim(res, info, keepdim, axis);
    }

    // ========== argmax ==========
    Array argmax(const Array& x, std::optional<int> axis, bool keepdim) {
        ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
        Array prepared = prepare_input(x, info);

        Shape out_shape = info.out_shape;
        if (info.flatten_all && !keepdim) {
            out_shape = Shape({});
        }

        Array result(out_shape, DType::I64, x.place());

        OpArgs args = { result, prepared, info.batch_size, info.reduce_size };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["argmax"][dev][x.dtype()](args);

        Array res = std::any_cast<Array>(output[0]);
        res = transpose_output(res, info);
        return post_process_keepdim(res, info, keepdim, axis);
    }

    // ========== argmin ==========
    Array argmin(const Array& x, std::optional<int> axis, bool keepdim) {
        ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
        Array prepared = prepare_input(x, info);

        Shape out_shape = info.out_shape;
        if (info.flatten_all && !keepdim) {
            out_shape = Shape({});
        }

        Array result(out_shape, DType::I64, x.place());

        OpArgs args = { result, prepared, info.batch_size, info.reduce_size };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["argmin"][dev][x.dtype()](args);

        Array res = std::any_cast<Array>(output[0]);
        res = transpose_output(res, info);
        return post_process_keepdim(res, info, keepdim, axis);
    }

    // ========== var ==========
    Array var(const Array& x, std::optional<int> axis, bool keepdim, int ddof) {
        if (!axis.has_value()) {
            Array flat = x.reshape(Shape({ x.numel() }));
            return var(flat, 0, keepdim, ddof);
        }

        int ax = axis.value();
        if (ax < 0) ax += x.shape().ndim();

        int64_t n = x.shape().dim(ax);

        // mean with keepdim
        Array m = mean(x, ax, true);

        // (x - mean)^2
        Array centered = sub(x, m);
        Array squared = square(centered);

        // sum along axis
        Array sum_sq = sum(squared, ax, keepdim);

        // divide by (n - ddof)
        double divisor = static_cast<double>(n - ddof);
        if (divisor <= 0.0) divisor = 1.0;
        Array div_arr = full(sum_sq.shape(), divisor, sum_sq.dtype(), sum_sq.place());

        return div(sum_sq, div_arr);
    }

    // ========== std ==========
    Array std(const Array& x, std::optional<int> axis, bool keepdim, int ddof) {
        Array v = var(x, axis, keepdim, ddof);
        return sqrt(v);
    }

    // ========== sem ==========
    Array sem(const Array& x, std::optional<int> axis, bool keepdim, int ddof) {
        Array s = std(x, axis, true, ddof);
        int64_t n = axis.has_value() ? x.shape().dim(axis.value()) : x.numel();
        double inv_sqrt_n = 1.0 / std::sqrt(static_cast<double>(n));
        Array factor = full(s.shape(), inv_sqrt_n, s.dtype(), s.place());
        Array result = mul(s, factor);
        if (!keepdim && axis.has_value()) {
            int ax = axis.value();
            int ndim = x.shape().ndim();
            if (ax < 0) ax += ndim;
            if (ax < result.shape().ndim() && result.shape().dim(ax) == 1) {
                result = result.squeeze(ax);
            }
        }
        return result;
    }

    // ========== count_nonzero ==========
    Array count_nonzero(const Array& x, std::optional<int> axis, bool keepdim) {
        ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
        Array prepared = prepare_input(x, info);

        Shape out_shape = info.out_shape;
        if (info.flatten_all && !keepdim) {
            out_shape = Shape({});
        }

        Array result(out_shape, DType::I64, x.place());

        OpArgs args = { result, prepared, info.batch_size, info.reduce_size };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["count_nonzero"][dev][x.dtype()](args);

        Array res = std::any_cast<Array>(output[0]);
        res = transpose_output(res, info);
        return post_process_keepdim(res, info, keepdim, axis);
    }

    // ========== cumsum ==========

    Array cumsum(const Array& x, int axis, DType dtype) {
        int ndim = x.shape().ndim();
        int ax = axis;
        if (ax < 0) ax += ndim;
        DType out_dtype = (dtype != DType::F64) ? dtype : promote_types(x.dtype(), DType::F64);
        Array result(x.shape(), out_dtype, x.place());
        OpArgs args = { result, x, ax };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["cumsum"][dev][x.dtype()](args);
        return std::any_cast<Array>(output[0]);
    }

    // ========== cumprod ==========
    Array cumprod(const Array& x, int axis, DType dtype) {
        int ndim = x.shape().ndim();
        int ax = axis;
        if (ax < 0) ax += ndim;
        DType out_dtype = (dtype != DType::F64) ? dtype : promote_types(x.dtype(), DType::F64);
        Array result(x.shape(), out_dtype, x.place());
        OpArgs args = { result, x, ax };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["cumprod"][dev][x.dtype()](args);
        return std::any_cast<Array>(output[0]);
    }

    // ========== cummax ==========
    Array cummax(const Array& x, int axis) {
        int ndim = x.shape().ndim();
        int ax = axis;
        if (ax < 0) ax += ndim;
        Array prepared = x.contiguous();
        Array result(prepared.shape(), prepared.dtype(), prepared.place());
        OpArgs args = { result, prepared, ax };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["cummax"][dev][x.dtype()](args);
        return std::any_cast<Array>(output[0]);
    }

    // ========== cummin ==========
    Array cummin(const Array& x, int axis) {
        int ndim = x.shape().ndim();
        int ax = axis;
        if (ax < 0) ax += ndim;
        Array prepared = x.contiguous();
        Array result(prepared.shape(), prepared.dtype(), prepared.place());
        OpArgs args = { result, prepared, ax };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["cummin"][dev][x.dtype()](args);
        return std::any_cast<Array>(output[0]);
    }

    // ========== median ==========
    Array median(const Array& x, std::optional<int> axis, bool keepdim) {
        return quantile(x, 0.5, axis, keepdim);
    }

    // ========== quantile (scalar) ==========
    Array quantile(const Array& x, double q, std::optional<int> axis, bool keepdim) {

        ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
        Array prepared = prepare_input(x, info);

        if (!prepared.is_contiguous()) {
            prepared = prepared.contiguous();
        }

        Shape out_shape = info.out_shape;
        if (info.flatten_all && !keepdim) {
            out_shape = Shape({});
        }

        Array result(out_shape, DType::F64, x.place());

        OpArgs args = { result, prepared, info.batch_size, info.reduce_size, q };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["quantile"][dev][x.dtype()](args);

        Array res = std::any_cast<Array>(output[0]);
        res = transpose_output(res, info);
        return post_process_keepdim(res, info, keepdim, axis);
    }

    // ========== quantile (array) ==========
    Array quantile(const Array& x, const Array& q, std::optional<int> axis, bool keepdim) {
        int64_t n_q = q.numel();
        if (n_q == 1) {
            double q_val;
            switch (q.dtype()) {
            case DType::F32: q_val = q.item<float>(); break;
            case DType::F64: q_val = q.item<double>(); break;
            default: INS_THROW("quantile: q must be float32 or float64, got ", dtype_name(q.dtype()));
            }
            return quantile(x, q_val, axis, keepdim);
        }

        // Multiple quantiles: compute each and stack
        std::vector<Array> results;
        for (int64_t i = 0; i < n_q; ++i) {
            Array q_i = q.slice(0, i, i + 1);
            double q_val;
            switch (q.dtype()) {
            case DType::F32: q_val = q_i.item<float>(); break;
            case DType::F64: q_val = q_i.item<double>(); break;
            default: INS_THROW("quantile: q must be float32 or float64, got ", dtype_name(q.dtype()));
            }
            results.push_back(quantile(x, q_val, axis, keepdim));
        }
        return stack(results, 0);
    }

    // ========== nansum ==========
    Array nansum(const Array& x, std::optional<int> axis, bool keepdim) {
        ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
        Array prepared = prepare_input(x, info);

        Shape out_shape = info.out_shape;
        if (info.flatten_all && !keepdim) {
            out_shape = Shape({});
        }

        Array sum_out(out_shape, x.dtype(), x.place());
        Array count_out(out_shape, DType::I64, x.place());

        OpArgs args = { sum_out, count_out, prepared, info.batch_size, info.reduce_size };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["nansum"][dev][x.dtype()](args);

        Array sum_result = std::any_cast<Array>(output[0]);
        Array count_result = std::any_cast<Array>(output[1]);

        sum_result = transpose_output(sum_result, info);
        return post_process_keepdim(sum_result, info, keepdim, axis);
    }

    // ========== nanmean ==========
    Array nanmean(const Array& x, std::optional<int> axis, bool keepdim) {
        ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
        Array prepared = prepare_input(x, info);

        Shape out_shape = info.out_shape;
        if (info.flatten_all && !keepdim) {
            out_shape = Shape({});
        }
        Array sum_out(out_shape, x.dtype(), x.place());
        Array count_out(out_shape, DType::I64, x.place());
        OpArgs args = { sum_out, count_out, prepared, info.batch_size, info.reduce_size };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["nansum"][dev][x.dtype()](args);
        Array sum_arr = std::any_cast<Array>(output[0]);
        Array count_arr = std::any_cast<Array>(output[1]);
        Array count_float = count_arr.to(x.dtype());
        Array result = div(sum_arr, count_float);
        result = transpose_output(result, info);
        result = post_process_keepdim(result, info, keepdim, axis);

        return result;
    }

    // ========== nanmax ==========
    Array nanmax(const Array& x, std::optional<int> axis, bool keepdim) {
        ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
        Array prepared = prepare_input(x, info);

        Shape out_shape = info.out_shape;
        if (info.flatten_all && !keepdim) {
            out_shape = Shape({});
        }

        Array result(out_shape, x.dtype(), x.place());

        OpArgs args = { result, prepared, info.batch_size, info.reduce_size };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["nanmax"][dev][x.dtype()](args);

        Array res = std::any_cast<Array>(output[0]);
        res = transpose_output(res, info);
        return post_process_keepdim(res, info, keepdim, axis);
    }

    // ========== nanmin ==========
    Array nanmin(const Array& x, std::optional<int> axis, bool keepdim) {
        ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
        Array prepared = prepare_input(x, info);

        Shape out_shape = info.out_shape;
        if (info.flatten_all && !keepdim) {
            out_shape = Shape({});
        }

        Array result(out_shape, x.dtype(), x.place());

        OpArgs args = { result, prepared, info.batch_size, info.reduce_size };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["nanmin"][dev][x.dtype()](args);

        Array res = std::any_cast<Array>(output[0]);
        res = transpose_output(res, info);
        return post_process_keepdim(res, info, keepdim, axis);
    }

    // ========== nanvar ==========
    Array nanvar(const Array& x, std::optional<int> axis, bool keepdim, int ddof) {
        ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
        Array prepared = prepare_input(x, info);

        Shape out_shape = info.out_shape;
        if (info.flatten_all && !keepdim) {
            out_shape = Shape({});
        }

        Array result(out_shape, x.dtype(), x.place());

        OpArgs args = { result, prepared, info.batch_size, info.reduce_size, ddof };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["nanvar"][dev][x.dtype()](args);

        Array res = std::any_cast<Array>(output[0]);
        res = transpose_output(res, info);
        return post_process_keepdim(res, info, keepdim, axis);
    }

    // ========== nanstd ==========
    Array nanstd(const Array& x, std::optional<int> axis, bool keepdim, int ddof) {
        Array v = nanvar(x, axis, keepdim, ddof);
        return sqrt(v);
    }

    // ========== nanmedian ==========
    Array nanmedian(const Array& x, std::optional<int> axis, bool keepdim) {
        return nanquantile(x, 0.5, axis, keepdim);
    }

    // ========== nanquantile (scalar) ==========
    Array nanquantile(const Array& x, double q, std::optional<int> axis, bool keepdim) {
        ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
        Array prepared = prepare_input(x, info);

        Shape out_shape = info.out_shape;
        if (info.flatten_all && !keepdim) {
            out_shape = Shape({});
        }

        Array result(out_shape, x.dtype(), x.place());

        OpArgs args = { result, prepared, info.batch_size, info.reduce_size, q };
        DeviceKind dev = get_device_kind(x.place());
        OpArgs output = ops()["nanquantile"][dev][x.dtype()](args);

        Array res = std::any_cast<Array>(output[0]);
        res = transpose_output(res, info);
        return post_process_keepdim(res, info, keepdim, axis);
    }

    // ========== nanquantile (array) ==========
    Array nanquantile(const Array& x, const Array& q, std::optional<int> axis, bool keepdim) {
        int64_t n_q = q.numel();
        if (n_q == 1) {
            double q_val;
            switch (q.dtype()) {
            case DType::F32: q_val = q.item<float>(); break;
            case DType::F64: q_val = q.item<double>(); break;
            default: INS_THROW("nanquantile: q must be float32 or float64, got ", dtype_name(q.dtype()));
            }
            return nanquantile(x, q_val, axis, keepdim);
        }

        // Multiple quantiles: compute each and stack
        std::vector<Array> results;
        for (int64_t i = 0; i < n_q; ++i) {
            Array q_i = q.slice(0, i, i + 1);
            double q_val;
            switch (q.dtype()) {
            case DType::F32: q_val = q_i.item<float>(); break;
            case DType::F64: q_val = q_i.item<double>(); break;
            default: INS_THROW("nanquantile: q must be float32 or float64, got ", dtype_name(q.dtype()));
            }
            results.push_back(nanquantile(x, q_val, axis, keepdim));
        }
        return stack(results, 0);
    }

    // ========== percentile ==========
    Array percentile(const Array& x, double q, std::optional<int> axis, bool keepdim) {
        return quantile(x, q / 100.0, axis, keepdim);
    }

} // namespace ins