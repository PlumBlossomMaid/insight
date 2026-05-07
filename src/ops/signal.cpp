// src/ops/signal.cpp
#include "insight/ops/signal.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/indexing.h"
#include "insight/ops/creation.h"
#include "insight/ops/reduction.h"
#include "insight/ops/manipulation.h"
#include "insight/ops/operator.h"
#include "insight/ops/fft.h"
#include <cmath>
#include <limits>
#include <numbers>

namespace ins {

    Array unwrap(const Array& p, int axis, double discont, double period) {
        INS_CHECK(p.defined(), "unwrap: input is undefined");
        INS_CHECK(p.dtype() == DType::F32 || p.dtype() == DType::F64,
            "unwrap: only float32/64 supported");

        // 标量直接返回
        if (p.numel() == 1) {
            return p.copy();
        }

        int ndim = p.shape().ndim();
        int ax = axis;
        if (ax < 0) ax += ndim;
        INS_CHECK(ax >= 0 && ax < ndim, "unwrap: axis out of range");

        // 如果 axis 维度小于 2，无法计算差分，直接返回原数组
        int64_t axis_size = p.shape().dim(ax);
        if (axis_size < 2) {
            return p.copy();
        }

        // 计算差分
        Array diff_p = diff(p, 1, ax);

        // 检测跳变
        Array jumps = greater_than(abs(diff_p), full(diff_p.shape(), discont, diff_p.dtype()));

        // 修正符号
        Array correction_step = where(jumps,
            where(greater_than(diff_p, full(diff_p.shape(), discont, diff_p.dtype())),
                full(diff_p.shape(), -period, diff_p.dtype()),
                full(diff_p.shape(), period, diff_p.dtype())),
            zeros_like(diff_p));

        // 累积修正量
        Array cumulative = cumsum(correction_step, ax);

        // 在前面补 0
        std::vector<int64_t> pad_width(ndim * 2, 0);
        pad_width[ax * 2] = 1;
        pad_width[ax * 2 + 1] = 0;

        Array cumulative_padded = pad(cumulative, pad_width, 0.0);
        Array result = add(p, cumulative_padded);

        return result;
    }

    Array sinc(const Array& x) {
        INS_CHECK(x.defined(), "sinc: input is undefined");
        INS_CHECK(is_floating_point(x.dtype()),
            "sinc: only floating point types supported, got ", dtype_name(x.dtype()));

        // sinc(x) = sin(πx) / (πx)
		const double pi = std::numbers::pi;

        // Convert to working dtype (F64 for integer inputs, preserve F32)
        DType working_dtype = (x.dtype() == DType::F32) ? DType::F32 : DType::F64;
        Array x_work = (x.dtype() == working_dtype) ? x : x.to(working_dtype);

        // Compute πx
        Array pi_x = x_work * pi;

        // Compute numerator = sin(πx)
        Array numerator = sin(pi_x);

        // Compute denominator = πx
        Array denominator = pi_x;

        // Create mask for x ≈ 0 (where sin(πx) ≈ πx)
        Array eps = full(x_work.shape(), 1e-8, working_dtype, x_work.place());
        Array is_zero = less_equal(abs(pi_x), eps);

        // For x ≈ 0, sinc(x) ≈ 1
        Array one = full(x_work.shape(), 1.0, working_dtype, x_work.place());
        Array sinc_val = div(numerator, denominator);
        Array result = where(is_zero, one, sinc_val);

        // Convert back to original dtype if needed
        if (result.dtype() != x.dtype()) {
            result = result.to(x.dtype());
        }

        return result;
    }

    Array convolve(const Array& a, const Array& v, const std::string& mode) {
        INS_CHECK(a.defined() && v.defined(), "convolve: inputs are undefined");
        INS_CHECK(a.shape().ndim() == 1 && v.shape().ndim() == 1,
            "convolve: only 1D tensors supported");
        INS_CHECK(a.dtype() == v.dtype(),
            "convolve: inputs must have same dtype");
        INS_CHECK(is_floating_point(a.dtype()),
            "convolve: only floating point types supported");
        INS_CHECK(mode == "full" || mode == "same" || mode == "valid",
            "convolve: mode must be 'full', 'same', or 'valid'");

        int64_t n = a.numel();
        int64_t m = v.numel();
        int64_t conv_len = n + m - 1;

        // Use FFT for efficiency
        int64_t fft_len = fft::next_fast_len(conv_len);

        // Convert to working dtype (F64 for better precision)
        DType working_dtype = (a.dtype() == DType::F32) ? DType::F32 : DType::F64;
        Array a_work = (a.dtype() == working_dtype) ? a : a.to(working_dtype);
        Array v_work = (v.dtype() == working_dtype) ? v : v.to(working_dtype);

        // Real FFT (rfft) -> complex
        Array A = fft::rfft(a_work, fft_len);
        Array V = fft::rfft(v_work, fft_len);

        // Complex multiplication
        Array conv_cpx = mul(A, V);

        // Inverse real FFT
        Array conv = fft::irfft(conv_cpx, conv_len);

        // Crop according to mode
        if (mode == "same") {
            int64_t start = (m - 1) / 2;
            conv = slice(conv, { 0 }, { start }, { start + n });
        }
        else if (mode == "valid") {
            int64_t start = m - 1;
            conv = slice(conv, { 0 }, { start }, { start + n - m + 1 });
        }
        // "full" remains as is

        // Convert back to original dtype if needed
        if (conv.dtype() != a.dtype()) {
            conv = conv.to(a.dtype());
        }

        return conv;
    }

} // namespace ins