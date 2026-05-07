// insight/utils/promotion.cpp
#include "insight/utils/promotion.h"
#include "insight/core/exception.h"
#include <unordered_map>
#include <vector>

namespace ins {

    // Allowed promotion table
    static const std::unordered_map<DType, std::vector<DType>> allowed_promotions = {
        {DType::BOOL, {DType::U8, DType::I8, DType::I16, DType::I32, DType::I64,
                       DType::F16, DType::BF16, DType::F32, DType::F64}},
        {DType::U8,   {DType::I16, DType::I32, DType::I64,
                       DType::F16, DType::BF16, DType::F32, DType::F64}},
        {DType::I8,   {DType::I16, DType::I32, DType::I64,
                       DType::F16, DType::BF16, DType::F32, DType::F64}},
        {DType::I16,  {DType::I32, DType::I64,
                       DType::F16, DType::BF16, DType::F32, DType::F64}},
        {DType::I32,  {DType::I64, DType::F32, DType::F64}},
        {DType::I64,  {DType::F32, DType::F64}},
        {DType::F16,  {DType::F32, DType::F64}},
        {DType::BF16, {DType::F32, DType::F64}},
        {DType::F32,  {DType::F64, DType::C32, DType::C64}},
        {DType::F64,  {DType::C64}},
        {DType::C32,  {DType::C64}},
        {DType::C64,  {}},  // Highest complex
        // F8 types cannot promote to anything
        // U16, U32, U64 cannot promote (no integer promotion chain defined)
    };

    // Helper: get priority index from enum (order is the enum value)
    static int priority_index(DType dtype) {
        int idx = static_cast<int>(dtype);
        INS_CHECK(idx > 0 && idx < static_cast<int>(DType::DTYPE_COUNT),
            "Invalid dtype for promotion: ", idx);
        return idx;
    }

    bool can_promote(DType from, DType to) {
        auto it = allowed_promotions.find(from);
        if (it == allowed_promotions.end()) return false;
        for (auto d : it->second) {
            if (d == to) return true;
        }
        return false;
    }

    DType promote_types(DType a, DType b) {
        if (a == b) return a;

        int pa = priority_index(a);
        int pb = priority_index(b);

        // Higher priority (larger index) wins
        DType higher = (pa > pb) ? a : b;
        DType lower = (pa > pb) ? b : a;

        // Check if promotion is allowed
        INS_CHECK(can_promote(lower, higher),
            "Cannot promote from ", dtype_name(lower), " to ", dtype_name(higher));

        return higher;
    }

    Place promote_places(const Place& a, const Place& b) {
        if (a.is_gpu() || b.is_gpu()) {
            return Place::GPU(0);
        }
        return Place::CPU(0);
    }

} // namespace ins