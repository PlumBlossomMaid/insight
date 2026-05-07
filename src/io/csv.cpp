// src/io/csv.cpp
#include "insight/io/csv.h"
#include "insight/plugin/op_registry.h"
#include "insight/core/dtype.h"
#include <fstream>
#include <vector>
#include <string>

namespace ins {

    // List of data types to export (excludes UNKNOWN and DTYPE_COUNT)
    static const std::vector<DType> dt_order = {
        // Core types (fully supported)
        DType::BOOL,
        DType::U8, DType::I8, DType::I16, DType::I32, DType::I64,
        DType::U16, DType::U32, DType::U64,
        DType::F32, DType::F64,
        DType::C32, DType::C64,

        // Partially supported (storage only for now)
        DType::F16, DType::BF16,
        DType::F8_E4M3, DType::F8_E5M2
    };

    // List of operators to export (all registered operators)
    static std::vector<std::string> get_all_operators() {
        return ops().get_operator_names();
    }

    static bool is_kernel_registered(const std::string& op_name, DeviceKind device, DType dtype) {
        return ops().has_kernel(op_name, device, dtype);
    }

    void export_support_csv(const std::string& filename, DeviceKind device) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            INS_THROW("Failed to open file: ", filename);
        }

        // Write header row
        file << "Operator";
        for (DType dt : dt_order) {
            file << "," << dtype_name(dt);
        }
        file << "\n";

        // Write data rows
        auto operators = get_all_operators();
        for (const auto& op_name : operators) {
            file << op_name;
            for (DType dt : dt_order) {
                bool supported = is_kernel_registered(op_name, device, dt);
                file << "," << (supported ? "1" : "0");
            }
            file << "\n";
        }

        file.close();
    }

} // namespace ins