// src/ops/math.cpp
#include "insight/ops/elementwise.h"
#include "insight/plugin/op_registry.h"
#include "insight/ops/broadcast.h"
#include "insight/utils/promotion.h"
#include <iostream>

namespace ins {

    static DeviceKind get_device_kind(const Place& place) {
        return place.is_cpu() ? DeviceKind::CPU : DeviceKind::GPU;
    }

    // Helper macro for binary operations
    #define DEFINE_BINARY_OP(op_name) \
    Array op_name(const Array& a, const Array& b) { \
        DType out_dtype = promote_types(a.dtype(), b.dtype()); \
        Array a1 = (a.dtype() == out_dtype) ? a : a.to(out_dtype); \
        Array b1 = (b.dtype() == out_dtype) ? b : b.to(out_dtype); \
        Place target_place = promote_places(a1.place(), b1.place()); \
        if (a1.place() != target_place) a1 = a1.to(target_place); \
        if (b1.place() != target_place) b1 = b1.to(target_place); \
        if (a1.shape() != b1.shape()) { \
            auto bc = broadcast_arrays({a1, b1}); \
            a1 = bc[0]; \
            b1 = bc[1]; \
        } \
        OpArgs args = {a1, b1}; \
        DeviceKind dev = get_device_kind(target_place); \
        \
        /* Check if kernel exists before calling */ \
        if (!ops().has_kernel(#op_name, dev, out_dtype)) { \
            std::string err_msg = "Kernel not found: " #op_name " for device "; \
            err_msg += (dev == DeviceKind::CPU ? "CPU" : "GPU"); \
            err_msg += " with dtype "; \
            err_msg += dtype_name(out_dtype); \
            err_msg += ". Available kernels for " #op_name " on this device: "; \
            auto available = ops().list_kernels(#op_name, dev); \
            for (size_t i = 0; i < available.size(); ++i) { \
                if (i > 0) err_msg += ", "; \
                err_msg += dtype_name(available[i]); \
            } \
            INS_THROW(err_msg); \
        } \
        \
        OpArgs result = ops()[#op_name][dev][out_dtype](args); \
        return std::any_cast<Array>(result[0]); \
    }

// Arithmetic operations
    DEFINE_BINARY_OP(add);
    DEFINE_BINARY_OP(sub);
    DEFINE_BINARY_OP(mul);
    DEFINE_BINARY_OP(div);

    // Power operation
    Array pow(const Array& a, const Array& b) {
        // pow requires float types, promote to float if needed
        DType out_dtype = promote_types(a.dtype(), b.dtype());
        if (!is_floating_point(out_dtype) && !is_complex(out_dtype)) {
            out_dtype = DType::F32;  // Promote integer to float32
        }
        Array a1 = (a.dtype() == out_dtype) ? a : a.to(out_dtype);
        Array b1 = (b.dtype() == out_dtype) ? b : b.to(out_dtype);
        Place target_place = promote_places(a1.place(), b1.place());
        if (a1.place() != target_place) a1 = a1.to(target_place);
        if (b1.place() != target_place) b1 = b1.to(target_place);
        if (a1.shape() != b1.shape()) {
            auto bc = broadcast_arrays({ a1, b1 });
            a1 = bc[0];
            b1 = bc[1];
        }
        if (!a1.is_contiguous()) a1 = a1.contiguous();
        if (!b1.is_contiguous()) b1 = b1.contiguous();
        OpArgs args = { a1, b1 };
        DeviceKind dev = get_device_kind(target_place);
        OpArgs result = ops()["pow"][dev][out_dtype](args);
        return std::any_cast<Array>(result[0]);
    }

    // Modulo operation
    DEFINE_BINARY_OP(mod)

        // Comparison operations (return bool)
#define DEFINE_CMP_OP(op_name) \
    Array op_name(const Array& a, const Array& b) { \
        DType out_dtype = promote_types(a.dtype(), b.dtype()); \
        Array a1 = (a.dtype() == out_dtype) ? a : a.to(out_dtype); \
        Array b1 = (b.dtype() == out_dtype) ? b : b.to(out_dtype); \
        Place target_place = promote_places(a1.place(), b1.place()); \
        if (a1.place() != target_place) a1 = a1.to(target_place); \
        if (b1.place() != target_place) b1 = b1.to(target_place); \
        if (a1.shape() != b1.shape()) { \
            auto bc = broadcast_arrays({a1, b1}); \
            a1 = bc[0]; \
            b1 = bc[1]; \
        } \
        if (!a1.is_contiguous()) a1 = a1.contiguous(); \
        if (!b1.is_contiguous()) b1 = b1.contiguous(); \
        OpArgs args = {a1, b1}; \
        DeviceKind dev = get_device_kind(target_place); \
        /* Comparison returns bool, but we use the same dtype for dispatch */ \
        OpArgs result = ops()[#op_name][dev][out_dtype](args); \
        return std::any_cast<Array>(result[0]); \
    }

        DEFINE_CMP_OP(equal)
        DEFINE_CMP_OP(not_equal)
        DEFINE_CMP_OP(greater)
        DEFINE_CMP_OP(less)
        DEFINE_CMP_OP(greater_equal)
        DEFINE_CMP_OP(less_equal)

        Array greater_than(const Array& a, const Array& b) { return greater(a, b); }
        Array less_than(const Array& a, const Array& b) { return less(a, b); }

        // Logical operations
        Array logical_and(const Array& a, const Array& b) {
        // Convert to bool first
        Array a1 = (a.dtype() == DType::BOOL) ? a : a.to(DType::BOOL);
        Array b1 = (b.dtype() == DType::BOOL) ? b : b.to(DType::BOOL);
        Place target_place = promote_places(a1.place(), b1.place());
        if (a1.place() != target_place) a1 = a1.to(target_place);
        if (b1.place() != target_place) b1 = b1.to(target_place);
        if (a1.shape() != b1.shape()) {
            auto bc = broadcast_arrays({ a1, b1 });
            a1 = bc[0];
            b1 = bc[1];
        }
        if (!a1.is_contiguous()) a1 = a1.contiguous();
        if (!b1.is_contiguous()) b1 = b1.contiguous();
        OpArgs args = { a1, b1 };
        DeviceKind dev = get_device_kind(target_place);
        OpArgs result = ops()["logical_and"][dev][DType::BOOL](args);
        return std::any_cast<Array>(result[0]);
    }

    Array logical_or(const Array& a, const Array& b) {
        Array a1 = (a.dtype() == DType::BOOL) ? a : a.to(DType::BOOL);
        Array b1 = (b.dtype() == DType::BOOL) ? b : b.to(DType::BOOL);
        Place target_place = promote_places(a1.place(), b1.place());
        if (a1.place() != target_place) a1 = a1.to(target_place);
        if (b1.place() != target_place) b1 = b1.to(target_place);
        if (a1.shape() != b1.shape()) {
            auto bc = broadcast_arrays({ a1, b1 });
            a1 = bc[0];
            b1 = bc[1];
        }
        if (!a1.is_contiguous()) a1 = a1.contiguous();
        if (!b1.is_contiguous()) b1 = b1.contiguous();
        OpArgs args = { a1, b1 };
        DeviceKind dev = get_device_kind(target_place);
        OpArgs result = ops()["logical_or"][dev][DType::BOOL](args);
        return std::any_cast<Array>(result[0]);
    }

    Array logical_xor(const Array& a, const Array& b) {
        Array a1 = (a.dtype() == DType::BOOL) ? a : a.to(DType::BOOL);
        Array b1 = (b.dtype() == DType::BOOL) ? b : b.to(DType::BOOL);
        Place target_place = promote_places(a1.place(), b1.place());
        if (a1.place() != target_place) a1 = a1.to(target_place);
        if (b1.place() != target_place) b1 = b1.to(target_place);
        if (a1.shape() != b1.shape()) {
            auto bc = broadcast_arrays({ a1, b1 });
            a1 = bc[0];
            b1 = bc[1];
        }
        if (!a1.is_contiguous()) a1 = a1.contiguous();
        if (!b1.is_contiguous()) b1 = b1.contiguous();
        OpArgs args = { a1, b1 };
        DeviceKind dev = get_device_kind(target_place);
        OpArgs result = ops()["logical_xor"][dev][DType::BOOL](args);
        return std::any_cast<Array>(result[0]);
    }

    // Bitwise operations (only for integer types)
#define DEFINE_BITWISE_OP(op_name) \
    Array op_name(const Array& a, const Array& b) { \
        DType out_dtype = promote_types(a.dtype(), b.dtype()); \
        Array a1 = (a.dtype() == out_dtype) ? a : a.to(out_dtype); \
        Array b1 = (b.dtype() == out_dtype) ? b : b.to(out_dtype); \
        Place target_place = promote_places(a1.place(), b1.place()); \
        if (a1.place() != target_place) a1 = a1.to(target_place); \
        if (b1.place() != target_place) b1 = b1.to(target_place); \
        if (a1.shape() != b1.shape()) { \
            auto bc = broadcast_arrays({a1, b1}); \
            a1 = bc[0]; \
            b1 = bc[1]; \
        } \
        if (!a1.is_contiguous()) a1 = a1.contiguous(); \
        if (!b1.is_contiguous()) b1 = b1.contiguous(); \
        OpArgs args = {a1, b1}; \
        DeviceKind dev = get_device_kind(target_place); \
        OpArgs result = ops()[#op_name][dev][out_dtype](args); \
        return std::any_cast<Array>(result[0]); \
    }

    DEFINE_BITWISE_OP(bitwise_and)
    DEFINE_BITWISE_OP(bitwise_or)
    DEFINE_BITWISE_OP(bitwise_xor)
    DEFINE_BITWISE_OP(bitwise_left_shift)
    DEFINE_BITWISE_OP(bitwise_right_shift)

    // Maximum and Minimum
    DEFINE_BINARY_OP(maximum)
    DEFINE_BINARY_OP(minimum)

} // namespace ins