// src/ops/unary.cpp
#include "insight/ops/elementwise.h"
#include "insight/plugin/op_registry.h"
#include "insight/utils/promotion.h"

namespace ins {

    static DeviceKind get_device_kind(const Place& place) {
        return place.is_cpu() ? DeviceKind::CPU : DeviceKind::GPU;
    }

    // Helper macro for unary operations (no type promotion needed)
    #define DEFINE_UNARY_OP(op_name) \
    Array op_name(const Array& x) { \
        INS_CHECK(x.defined(), #op_name ": input is undefined"); \
        OpArgs args = {x}; \
        DeviceKind dev = get_device_kind(x.place()); \
        DType dtype = x.dtype(); \
        const auto& device_layer = ops()[#op_name]; \
        const auto& dtype_layer = device_layer[dev]; \
        const auto& kernel = dtype_layer[dtype]; \
        OpArgs result = kernel(args); \
        return std::any_cast<Array>(result[0]); \
    }

    // ============================================================================
    // Basic math operations
    // ============================================================================

        DEFINE_UNARY_OP(abs)
        DEFINE_UNARY_OP(negative)
        DEFINE_UNARY_OP(square)

        // ============================================================================
        // Exponential and logarithmic
        // ============================================================================

        DEFINE_UNARY_OP(exp)
        DEFINE_UNARY_OP(exp2)
        DEFINE_UNARY_OP(expm1)
        DEFINE_UNARY_OP(log)
        DEFINE_UNARY_OP(log2)
        DEFINE_UNARY_OP(log10)
        DEFINE_UNARY_OP(log1p)

        // ============================================================================
        // Power and root
        // ============================================================================

        DEFINE_UNARY_OP(sqrt)
        DEFINE_UNARY_OP(cbrt)
        DEFINE_UNARY_OP(reciprocal)

        // ============================================================================
        // Trigonometric
        // ============================================================================

        DEFINE_UNARY_OP(sin)
        DEFINE_UNARY_OP(cos)
        DEFINE_UNARY_OP(tan)
        DEFINE_UNARY_OP(asin)
        DEFINE_UNARY_OP(acos)
        DEFINE_UNARY_OP(atan)

        // ============================================================================
        // Hyperbolic
        // ============================================================================

        DEFINE_UNARY_OP(sinh)
        DEFINE_UNARY_OP(cosh)
        DEFINE_UNARY_OP(tanh)
        DEFINE_UNARY_OP(asinh)
        DEFINE_UNARY_OP(acosh)
        DEFINE_UNARY_OP(atanh)

        // ============================================================================
        // Rounding
        // ============================================================================

        DEFINE_UNARY_OP(floor)
        DEFINE_UNARY_OP(ceil)
        DEFINE_UNARY_OP(trunc)
        DEFINE_UNARY_OP(rint)

        // ============================================================================
        // Sign
        // ============================================================================

        DEFINE_UNARY_OP(sign)

        // ============================================================================
        // Complex
        // ============================================================================

        Array conj(const Array& x) {
        INS_CHECK(x.defined(), "conj: input is undefined");
        if (is_complex(x.dtype())) {
            if (!x.is_contiguous()) {
                Array cont = x.contiguous();
                return conj(cont);
            }
            OpArgs args = { x };
            DeviceKind dev = get_device_kind(x.place());
            DType dtype = x.dtype();
            OpArgs result = ops()["conj"][dev][dtype](args);
            return std::any_cast<Array>(result[0]);
        }
        // For real numbers, conjugate is identity
        return x;
    }

    // ============================================================================
    // Degree/radian conversion (can be implemented as unary ops)
    // ============================================================================

    DEFINE_UNARY_OP(deg2rad)
    DEFINE_UNARY_OP(rad2deg)

    // ============================================================================
    // Logical and bitwise
    // ============================================================================

    DEFINE_UNARY_OP(logical_not)
    DEFINE_UNARY_OP(bitwise_not)

} // namespace ins