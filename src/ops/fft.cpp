// src/ops/fft.cpp
#include "insight/ops/fft.h"
#include "insight/ops/creation.h"
#include "insight/ops/complex.h"
#include "insight/ops/manipulation.h"
#include "insight/ops/elementwise.h"
#include "insight/core/slice.h"
#include "insight/plugin/op_registry.h"
#include <cmath>

namespace ins {
    namespace fft {

        static DeviceKind get_device_kind(const Place& place) {
            return place.is_cpu() ? DeviceKind::CPU : DeviceKind::GPU;
        }

        struct FFTPlan {
            int64_t fft_len;           // FFT length after padding/truncation
            int64_t batch_size;        // Number of 1D FFTs to perform
            std::vector<int> inv_perm; // Inverse permutation for result
            bool input_is_complex;
            bool output_is_complex;
            bool inverse;              // true for ifft/irfft, false for fft/rfft
            bool real_input;           // true for rfft/irfft
            int fft_ndim;              // Number of dimensions excluding complex dim
            int transformed_axis;      // Original axis being transformed
            DType dtype;               // F32 or F64
            std::string norm;
        };

        FFTPlan prepare_fft(const Array& x, int n, int axis, bool inverse, bool real_input,
            const std::string& norm) {
            FFTPlan plan;
            plan.inverse = inverse;
            plan.real_input = real_input;
            plan.norm = norm;

            if (is_complex(x)) {
                plan.dtype = x.dtype();  // C32 or C64
                plan.input_is_complex = true;
            }
            else {
                plan.dtype = (x.dtype() == DType::F64) ? DType::F64 : DType::F32;
                plan.input_is_complex = false;
            }

            int ndim = x.shape().ndim();
            plan.fft_ndim = ndim;

            if (plan.input_is_complex && plan.dtype != DType::C32 && plan.dtype != DType::C64) {
                INS_THROW("prepare_fft: complex input must have C32 or C64 dtype");
            }

            plan.transformed_axis = axis;
            if (plan.transformed_axis < 0) plan.transformed_axis += plan.fft_ndim;
            INS_CHECK(plan.transformed_axis >= 0 && plan.transformed_axis < plan.fft_ndim,
                "prepare_fft: axis out of range");

            int64_t current_len = x.shape().dim(plan.transformed_axis);
            plan.fft_len = (n > 0) ? n : current_len;

            INS_CHECK(plan.fft_len > 0, "prepare_fft: fft_len must be positive");

            plan.inv_perm.clear();
            if (plan.transformed_axis != plan.fft_ndim - 1) {
                std::vector<int> perm;
                for (int i = 0; i < plan.fft_ndim; ++i) {
                    if (i != plan.transformed_axis) perm.push_back(i);
                }
                perm.push_back(plan.transformed_axis);

                plan.inv_perm.resize(ndim);
                for (size_t i = 0; i < perm.size(); ++i) {
                    plan.inv_perm[perm[i]] = static_cast<int>(i);
                }
            }

            plan.batch_size = 1;
            for (int i = 0; i < plan.fft_ndim; ++i) {
                if (i != plan.transformed_axis) {
                    plan.batch_size *= x.shape().dim(i);
                }
            }

            INS_CHECK(plan.batch_size > 0, "prepare_fft: batch_size must be positive");

            if (real_input) {
                plan.output_is_complex = !inverse;
            }
            else {
                plan.output_is_complex = plan.input_is_complex != inverse;
            }

            return plan;
        }

        static Array prepare_input(const Array& x, const FFTPlan& plan, int n) {
            Array result = x;

            if (!result.is_contiguous()) {
                result = result.contiguous();
            }

            // Transpose if needed
            if (!plan.inv_perm.empty()) {
                std::vector<int> perm(plan.inv_perm.size());
                for (size_t i = 0; i < plan.inv_perm.size(); ++i) {
                    perm[plan.inv_perm[i]] = static_cast<int>(i);
                }
                result = result.transpose(perm);
                result = result.contiguous();
            }

            // Convert to working dtype if needed
            if (!is_complex(result)) {
                if (result.dtype() != plan.dtype) {
                    result = result.to(plan.dtype);
                }
            }

            // Pad or truncate on the transformed axis (now last dimension)
            int last_axis_idx = result.shape().ndim() - 1;
            int64_t current_len = result.shape().dim(last_axis_idx);

            // For irfft, n is the complex input length; for rfft, n is the real input length
            if (n > 0 && n != current_len) {
                if (n > current_len) {
                    // Pad with zeros
                    std::vector<int64_t> pad_width(2 * result.shape().ndim(), 0);
                    pad_width[2 * last_axis_idx + 1] = n - current_len;
                    result = pad(result, pad_width, 0.0);
                }
                else {
                    // Truncate
                    result = slice(result, { last_axis_idx }, { 0 }, { n });
                }
                result = result.contiguous();
            }

            return result;
        }

        static Array create_output(const FFTPlan& plan, const Array& input, int n) {
            std::vector<int64_t> out_dims;
            int ndim = input.shape().ndim();

            if (plan.real_input) {
                if (plan.inverse) {
                    // irfft: complex -> real
                    for (int i = 0; i < ndim - 1; ++i) {
                        out_dims.push_back(input.shape().dim(i));
                    }
                    int64_t out_len;
                    if (n > 0) {
                        out_len = n;
                    }
                    else {
                        int64_t in_complex_len = input.shape().dim(ndim - 1);
                        out_len = (in_complex_len - 1) * 2;
                    }
                    out_dims.push_back(out_len);
                    DType out_dtype = (input.dtype() == DType::C32) ? DType::F32 : DType::F64;
                    Shape out_shape(out_dims);
                    return Array(out_shape, out_dtype, input.place());
                }
                else {
                    // rfft: real -> complex
                    for (int i = 0; i < ndim; ++i) {
                        out_dims.push_back(input.shape().dim(i));
                    }
                    int64_t out_len = plan.fft_len / 2 + 1;
                    out_dims.back() = out_len;
                    DType out_dtype = (plan.dtype == DType::F32) ? DType::C32 : DType::C64;
                    Shape out_shape(out_dims);
                    return Array(out_shape, out_dtype, input.place());
                }
            }
            else {
                // Standard complex FFT (C2C)
                for (int i = 0; i < ndim; ++i) {
                    out_dims.push_back(input.shape().dim(i));
                }
                Shape out_shape(out_dims);
                DType out_dtype = plan.dtype;
                return Array(out_shape, out_dtype, input.place());
            }
        }

        static void apply_norm(Array& result, int64_t fft_len, const std::string& norm, bool inverse) {
            double factor = 1.0;
            if (norm == "backward") {
                if (inverse) factor = 1.0 / fft_len;
            }
            else if (norm == "forward") {
                if (!inverse) factor = 1.0 / fft_len;
            }
            else if (norm == "ortho") {
                factor = 1.0 / std::sqrt(static_cast<double>(fft_len));
            }
            else {
                return;
            }

            if (std::abs(factor - 1.0) > 1e-12) {
                Array scalar = full(Shape({ 1 }), factor, result.dtype(), result.place());
                result = mul(result, scalar);
            }
        }

        // ============================================================================
        // Public API
        // ============================================================================

        Array fft(const Array& x, int n, int axis, const std::string& norm) {
            // If input is real, convert to complex first
            if (!is_complex(x)) {
                Array x_complex = to_complex(x);
                return fft(x_complex, n, axis, norm);
            }

            INS_CHECK(x.defined(), "fft: input is undefined");

            FFTPlan plan = prepare_fft(x, n, axis, false, false, norm);
            Array input = prepare_input(x, plan, n);
            Array result = create_output(plan, input, n);

            OpArgs args = { result, input, plan.fft_len, plan.batch_size, plan.inverse, plan.real_input, plan.norm };
            DeviceKind dev = get_device_kind(x.place());
            OpArgs output = ops()["fft"][dev][plan.dtype](args);

            Array out = std::any_cast<Array>(output[0]);

            if (!plan.inv_perm.empty()) {
                out = out.transpose(plan.inv_perm);
                out = out.contiguous();
            }

            apply_norm(out, plan.fft_len, norm, false);
            return out;
        }

        Array ifft(const Array& x, int n, int axis, const std::string& norm) {
            if (!is_complex(x)) {
                Array x_complex = to_complex(x);
                return ifft(x_complex, n, axis, norm);
            }

            INS_CHECK(x.defined(), "ifft: input is undefined");
            INS_CHECK(is_complex(x), "ifft: input must be complex");

            FFTPlan plan = prepare_fft(x, n, axis, true, false, norm);
            Array input = prepare_input(x, plan, n);
            Array result = create_output(plan, input, n);

            OpArgs args = { result, input, plan.fft_len, plan.batch_size, plan.inverse, plan.real_input, plan.norm };
            DeviceKind dev = get_device_kind(x.place());
            OpArgs output = ops()["fft"][dev][plan.dtype](args);

            Array out = std::any_cast<Array>(output[0]);

            if (!plan.inv_perm.empty()) {
                out = out.transpose(plan.inv_perm);
                out = out.contiguous();
            }

            apply_norm(out, plan.fft_len, norm, true);
            return out;
        }

        Array rfft(const Array& x, int n, int axis, const std::string& norm) {
            INS_CHECK(x.defined(), "rfft: input is undefined");
            INS_CHECK(!is_complex(x), "rfft: input must be real");

            FFTPlan plan = prepare_fft(x, n, axis, false, true, norm);
            Array input = prepare_input(x, plan, n);
            Array result = create_output(plan, input, n);

            OpArgs args = { result, input, plan.fft_len, plan.batch_size, plan.inverse, plan.real_input, plan.norm };
            DeviceKind dev = get_device_kind(x.place());
            OpArgs output = ops()["rfft"][dev][plan.dtype](args);

            Array out = std::any_cast<Array>(output[0]);

            if (!plan.inv_perm.empty()) {
                out = out.transpose(plan.inv_perm);
                out = out.contiguous();
            }

            apply_norm(out, plan.fft_len, norm, false);
            return out;
        }

        Array irfft(const Array& x, int n, int axis, const std::string& norm) {
            INS_CHECK(x.defined(), "irfft: input is undefined");
            INS_CHECK(is_complex(x), "irfft: input must be complex");

            FFTPlan plan = prepare_fft(x, n, axis, true, true, norm);

            // Compute expected real output length
            int64_t out_len;
            if (n > 0) {
                out_len = n;
            }
            else {
                int64_t in_len = x.shape().dim(axis);
                out_len = (in_len - 1) * 2;
            }

            // Compute expected complex input length for hermitian symmetry
            int64_t expected_input_len = out_len / 2 + 1;

            Array input = prepare_input(x, plan, expected_input_len);
            Array result = create_output(plan, input, n);

            OpArgs args = { result, input, out_len, plan.batch_size, plan.inverse, plan.real_input, plan.norm };
            DeviceKind dev = get_device_kind(x.place());
            OpArgs output = ops()["irfft"][dev][plan.dtype](args);

            Array out = std::any_cast<Array>(output[0]);

            if (!plan.inv_perm.empty()) {
                out = out.transpose(plan.inv_perm);
                out = out.contiguous();
            }

            // Apply normalization using real output length
            apply_norm(out, out_len, norm, true);
            return out;
        }

        // Hermitian FFT: hfft = irfft (complex Hermitian -> real)
        Array hfft(const Array& x, int n, int axis, const std::string& norm) {
            return irfft(x, n, axis, norm);
        }

        // Inverse Hermitian FFT: ihfft = rfft (real -> complex Hermitian)
        Array ihfft(const Array& x, int n, int axis, const std::string& norm) {
            return rfft(x, n, axis, norm);
        }

        // 2D and N-D versions call fftn/ifftn
        Array fft2(const Array& x, const std::vector<int64_t>& s,
            const std::vector<int>& axes, const std::string& norm) {
            if (!is_complex(x)) {
                INS_THROW("fft2: input must be complex (dtype C32 or C64). "
                    "For real input, use rfft2() instead.");
            }

            std::vector<int> target_axes = axes;
            if (target_axes.empty()) target_axes = { -2, -1 };
            return fftn(x, s, target_axes, norm);
        }

        Array ifft2(const Array& x, const std::vector<int64_t>& s,
            const std::vector<int>& axes, const std::string& norm) {
            if (!is_complex(x)) {
                INS_THROW("ifft2: input must be complex (dtype C32 or C64). "
                    "For real output, use irfft2() instead.");
            }

            std::vector<int> target_axes = axes;
            if (target_axes.empty()) target_axes = { -2, -1 };
            return ifftn(x, s, target_axes, norm);
        }

        Array rfft2(const Array& x, const std::vector<int64_t>& s,
            const std::vector<int>& axes, const std::string& norm) {
            if (is_complex(x)) {
                INS_THROW("rfft2: input must be real. For complex input, use fft2() instead.");
            }

            std::vector<int> target_axes = axes;
            if (target_axes.empty()) target_axes = { -2, -1 };
            return rfftn(x, s, target_axes, norm);
        }

        Array irfft2(const Array& x, const std::vector<int64_t>& s,
            const std::vector<int>& axes, const std::string& norm) {
            if (!is_complex(x)) {
                INS_THROW("irfft2: input must be complex (dtype C32 or C64). "
                    "For real output, use rfft2() instead.");
            }

            std::vector<int> target_axes = axes;
            if (target_axes.empty()) target_axes = { -2, -1 };

            return irfftn(x, s, target_axes, norm);
        }

        Array hfft2(const Array& x, const std::vector<int64_t>& s,
            const std::vector<int>& axes, const std::string& norm) {
            return irfft2(x, s, axes, norm);
        }

        Array ihfft2(const Array& x, const std::vector<int64_t>& s,
            const std::vector<int>& axes, const std::string& norm) {
            return rfft2(x, s, axes, norm);
        }

        Array fftn(const Array& x, const std::vector<int64_t>& s,
            const std::vector<int>& axes, const std::string& norm) {
            INS_CHECK(x.defined(), "fftn: input is undefined");

            std::vector<int> target_axes = axes;
            int ndim = x.shape().ndim();
            int data_ndim = ndim;

            if (target_axes.empty()) {
                for (int i = 0; i < data_ndim; ++i) {
                    target_axes.push_back(i);
                }
            }

            Array result = x;
            if (!s.empty()) {
                INS_CHECK(s.size() == target_axes.size(),
                    "fftn: s must have same length as axes");
                for (size_t i = 0; i < target_axes.size(); ++i) {
                    int64_t target_len = s[i];
                    int axis = target_axes[i];
                    if (target_len > 0) {
                        int64_t current_len = result.shape().dim(axis);
                        if (target_len < current_len) {
                            result = slice(result, { axis }, { 0 }, { target_len });
                        }
                        else if (target_len > current_len) {
                            std::vector<int64_t> pad_width(2 * result.shape().ndim(), 0);
                            pad_width[2 * axis + 1] = target_len - current_len;
                            result = pad(result, pad_width, 0.0);
                        }
                    }
                }
                result = result.contiguous();
            }

            for (int ax : target_axes) {
                result = fft(result, -1, ax, norm);
                result = result.contiguous();
            }

            return result;
        }

        Array ifftn(const Array& x, const std::vector<int64_t>& s,
            const std::vector<int>& axes, const std::string& norm) {
            INS_CHECK(x.defined(), "ifftn: input is undefined");

            std::vector<int> target_axes = axes;
            int ndim = x.shape().ndim();
            int data_ndim = ndim;

            if (target_axes.empty()) {
                for (int i = 0; i < data_ndim; ++i) {
                    target_axes.push_back(i);
                }
            }

            Array result = x;
            if (!s.empty()) {
                INS_CHECK(s.size() == target_axes.size(),
                    "ifftn: s must have same length as axes");
                for (size_t i = 0; i < target_axes.size(); ++i) {
                    int64_t target_len = s[i];
                    int axis = target_axes[i];
                    if (target_len > 0) {
                        int64_t current_len = result.shape().dim(axis);
                        if (target_len < current_len) {
                            result = slice(result, { axis }, { 0 }, { target_len });
                        }
                        else if (target_len > current_len) {
                            std::vector<int64_t> pad_width(2 * result.shape().ndim(), 0);
                            pad_width[2 * axis + 1] = target_len - current_len;
                            result = pad(result, pad_width, 0.0);
                        }
                    }
                }
                result = result.contiguous();
            }

            for (int ax : target_axes) {
                result = ifft(result, -1, ax, norm);
                result = result.contiguous();
            }

            return result;
        }

        Array rfftn(const Array& x, const std::vector<int64_t>& s,
            const std::vector<int>& axes, const std::string& norm) {
            INS_CHECK(x.defined(), "rfftn: input is undefined");
            INS_CHECK(!is_complex(x), "rfftn: input must be real");

            std::vector<int> target_axes = axes;
            int ndim = x.shape().ndim();
            if (target_axes.empty()) {
                for (int i = 0; i < ndim; ++i) {
                    target_axes.push_back(i);
                }
            }

            Array result = x;
            if (!s.empty()) {
                INS_CHECK(s.size() == target_axes.size(),
                    "rfftn: s must have same length as axes");
                for (size_t i = 0; i < target_axes.size(); ++i) {
                    int64_t target_len = s[i];
                    int axis = target_axes[i];
                    if (target_len > 0) {
                        int64_t current_len = result.shape().dim(axis);
                        if (target_len < current_len) {
                            result = slice(result, { axis }, { 0 }, { target_len });
                        }
                        else if (target_len > current_len) {
                            std::vector<int64_t> pad_width(2 * result.shape().ndim(), 0);
                            pad_width[2 * axis + 1] = target_len - current_len;
                            result = pad(result, pad_width, 0.0);
                        }
                    }
                }
                result = result.contiguous();
            }

            int last_axis = target_axes.back();
            result = rfft(result, -1, last_axis, norm);
            result = result.contiguous();

            for (int i = static_cast<int>(target_axes.size()) - 2; i >= 0; --i) {
                result = fft(result, -1, target_axes[i], norm);
                result = result.contiguous();
            }

            return result;
        }

        Array irfftn(const Array& x, const std::vector<int64_t>& s,
            const std::vector<int>& axes, const std::string& norm) {
            INS_CHECK(x.defined(), "irfftn: input is undefined");
            INS_CHECK(is_complex(x), "irfftn: input must be complex");

            int ndim = x.shape().ndim();
            int data_ndim = ndim;

            std::vector<int> target_axes = axes;
            if (target_axes.empty()) {
                for (int i = 0; i < data_ndim; ++i) {
                    target_axes.push_back(i);
                }
            }

            // Build expected output shape
            std::vector<int64_t> out_shape_dims;
            for (int i = 0; i < data_ndim; ++i) {
                if (s.size() > static_cast<size_t>(i)) {
                    out_shape_dims.push_back(s[i]);
                }
                else {
                    out_shape_dims.push_back(x.shape().dim(i));
                }
            }
            Shape out_shape(out_shape_dims);

            Array result = x;

            // Apply ifft on all axes except the last
            for (size_t i = 0; i < target_axes.size() - 1; ++i) {
                int ax = target_axes[i];
                result = ifft(result, -1, ax, norm);
                result = result.contiguous();
            }

            // Apply irfft on the last axis, using the specified output length from s
            int last_axis = target_axes.back();
            int64_t out_len = (s.size() > static_cast<size_t>(last_axis)) ? s[last_axis] : -1;
            result = irfft(result, out_len, last_axis, norm);

            return result;
        }

        Array hfftn(const Array& x, const std::vector<int64_t>& s,
            const std::vector<int>& axes, const std::string& norm) {
            return irfftn(x, s, axes, norm);
        }

        Array ihfftn(const Array& x, const std::vector<int64_t>& s,
            const std::vector<int>& axes, const std::string& norm) {
            return rfftn(x, s, axes, norm);
        }

        // ============================================================================
        // Helper functions
        // ============================================================================

        Array fftfreq(int64_t n, double d) {
            INS_CHECK(n > 0, "fftfreq: n must be positive");
            INS_CHECK(d > 0, "fftfreq: d must be positive");

            std::vector<double> data(n);
            double inv = 1.0 / (d * n);
            int64_t mid = (n + 1) / 2;

            for (int64_t i = 0; i < mid; ++i) {
                data[i] = i * inv;
            }
            for (int64_t i = mid; i < n; ++i) {
                data[i] = (i - n) * inv;
            }

            return to_array(data, Shape({ n }));
        }

        Array rfftfreq(int64_t n, double d) {
            INS_CHECK(n > 0, "rfftfreq: n must be positive");
            INS_CHECK(d > 0, "rfftfreq: d must be positive");

            int64_t len = n / 2 + 1;
            std::vector<double> data(len);
            double inv = 1.0 / (d * n);

            for (int64_t i = 0; i < len; ++i) {
                data[i] = i * inv;
            }

            return to_array(data, Shape({ len }));
        }

        Array fftshift(const Array& x, int axis) {
            INS_CHECK(x.defined(), "fftshift: input is undefined");

            int ndim = x.shape().ndim();
            if (axis == -1) {
                // Shift all axes
                Array result = x;
                for (int i = 0; i < ndim; ++i) {
                    result = fftshift(result, i);
                }
                return result;
            }

            int ax = axis;
            if (ax < 0) ax += ndim;
            INS_CHECK(ax >= 0 && ax < ndim, "fftshift: axis out of range");

            int64_t n = x.shape().dim(ax);
            int64_t mid = n / 2;
            auto last = x.slice(ax, mid, n);
            auto first = x.slice(ax, 0, mid);

            Array result = concat({ last, first }, ax);
            return result;
        }

        Array ifftshift(const Array& x, int axis) {
            INS_CHECK(x.defined(), "ifftshift: input is undefined");

            int ndim = x.shape().ndim();
            if (axis == -1) {
                Array result = x;
                for (int i = 0; i < ndim; ++i) {
                    result = ifftshift(result, i);
                }
                return result;
            }

            int ax = axis;
            if (ax < 0) ax += ndim;
            INS_CHECK(ax >= 0 && ax < ndim, "ifftshift: axis out of range");

            int64_t n = x.shape().dim(ax);
            int64_t mid = (n + 1) / 2;  // Ceiling division for odd lengths

            auto last = x.slice(ax, mid, n);
            auto first = x.slice(ax, 0, mid);

            return concat({ last, first }, ax);
        }

        int next_fast_len(int target) {
            if (target <= 0) return 1;

            const int primes[] = { 2, 3, 5, 7 };

            std::function<int(int, int)> impl = [&](int n, int idx) -> int {
                if (n == 1) return 1;
                if (idx == 4) return INT_MAX;

                int p = primes[idx];
                int branch1 = impl((n + p - 1) / p, idx);
                if (branch1 != INT_MAX) branch1 *= p;
                int branch2 = impl(n, idx + 1);

                return std::min(branch1, branch2);
                };

            return impl(target, 0);
        }

    } // namespace fft
} // namespace ins