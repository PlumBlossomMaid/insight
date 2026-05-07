// insight/ops/broadcast.h
#pragma once
#include <vector>
#include "insight/core/array.h"

namespace ins {

	/**
	 * @brief Compute broadcasted shape from two shapes.
	 *
	 * Follows NumPy broadcasting rules. Shapes are compared from the trailing dimension.
	 *
	 * @param a First shape
	 * @param b Second shape
	 * @return Broadcasted shape
	 * @throws Exception if shapes are not broadcastable
	 */
	Shape broadcast_shape(const Shape& a, const Shape& b);

	/**
	 * @brief Broadcast a single array to target shape.
	 *
	 * Returns a view (zero-copy) with strides set to 0 for broadcasted dimensions.
	 * The original array and the returned view share the same underlying storage.
	 *
	 * @param x Input array
	 * @param target_shape Target shape
	 * @return Broadcasted view of x
	 * @throws Exception if broadcast is not possible
	 */
	Array broadcast_to(const Array& x, const Shape& target_shape);

	/**
	 * @brief Broadcast multiple arrays to a common shape.
	 *
	 * Returns a vector of views (zero-copy) all with the same shape.
	 * The common shape is computed by broadcasting all input shapes.
	 *
	 * @param tensors Vector of input arrays
	 * @return Vector of broadcasted views (same order as input)
	 * @throws Exception if inputs are empty or shapes are not broadcastable
	 */
	std::vector<Array> broadcast_arrays(const std::vector<Array>& tensors);

} // namespace ins