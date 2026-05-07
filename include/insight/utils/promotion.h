// insight/utils/promotion.h
#pragma once
#include "insight/core/dtype.h"
#include "insight/core/place.h"

namespace ins {

	/**
	 * @brief Check if a type can be promoted to another.
	 * @param from Source data type
	 * @param to Target data type
	 * @return true if promotion is allowed
	 */
	bool can_promote(DType from, DType to);

	/**
	 * @brief Promote two data types to a common type.
	 *
	 * Rules follow NumPy promotion semantics:
	 * - BOOL < U8 < I8 < I16 < I32 < I64 < F16 < BF16 < F32 < F64 < C32 < C64
	 * - Promotion always goes to the higher priority type
	 * - Invalid promotions (e.g., bool -> complex) throw exception
	 *
	 * @param a First data type
	 * @param b Second data type
	 * @return Promoted data type
	 * @throws ins::Exception if promotion is not allowed
	 */
	DType promote_types(DType a, DType b);

	/**
	 * @brief Promote two places to a common device.
	 *
	 * Rules:
	 * - If either is GPU, result is GPU (device 0)
	 * - Otherwise result is CPU (device 0)
	 *
	 * @param a First place
	 * @param b Second place
	 * @return Promoted place
	 */
	Place promote_places(const Place& a, const Place& b);

} // namespace ins