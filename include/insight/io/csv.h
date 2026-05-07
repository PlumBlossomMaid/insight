// insight/io/csv.h
#pragma once
#include <string>
#include "insight/core/place.h"

namespace ins {

	/**
	 * @brief Export operator support matrix to CSV file.
	 *
	 * This function enumerates all registered operators and checks which
	 * data types are supported for the specified device. The output CSV
	 * can be used for documentation or CI validation.
	 *
	 * @param filename Output file path (e.g., "cpu_support.csv")
	 * @param device DeviceKind::CPU or DeviceKind::GPU
	 *
	 * Example output format (CPU):
	 *   Operator,BOOL,U8,I8,I16,I32,I64,U16,U32,U64,F16,BF16,F32,F64,C32,C64
	 *   abs,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0
	 *   add,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1
	 */
	void export_support_csv(const std::string& filename, DeviceKind device);

} // namespace ins