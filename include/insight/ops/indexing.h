// include/insight/ops/indexing.h
#pragma once
#include "insight/core/array.h"
#include <optional>
#include <tuple>
#include <string>

namespace ins {

    // ============================================================================
    // Forward declarations for return types
    // ============================================================================

    struct UniqueResult;

    // ============================================================================
    // Take / Put
    // ============================================================================

    /**
     * @brief Take elements from an array along an axis.
     *
     * @param x Input array
     * @param indices Indices (1D) to take
     * @param axis Axis to take from (nullopt means flattened)
     * @return Array of taken elements
     */
    Array take(const Array& x, const Array& indices, std::optional<int> axis = std::nullopt);

    /**
     * @brief Take elements along an axis using broadcasting indices.
     *
     * @param x Input array
     * @param indices Indices array (must broadcast to x.shape except on axis)
     * @param axis Axis to take from
     * @return Array with same shape as indices
     */
    Array take_along_axis(const Array& x, const Array& indices, int axis);

    /**
     * @brief Put values into an array at given indices.
     *
     * @param x Input array
     * @param indices Indices (1D) to put at
     * @param values Values to put (broadcastable to indices shape)
     * @param axis Axis to put along (nullopt means flattened)
     * @return Array with values placed
     */
    Array put(const Array& x, const Array& indices, const Array& values,
        std::optional<int> axis = std::nullopt);

    /**
     * @brief Put values along an axis using broadcasting indices.
     *
     * @param x Input array
     * @param indices Indices array
     * @param values Values array
     * @param axis Axis to put along
     * @return Array with values placed
     */
    Array put_along_axis(const Array& x, const Array& indices, const Array& values, int axis);

    // ============================================================================
    // Gather / Scatter (PyTorch style)
    // ============================================================================

    /**
     * @brief Gather values along an axis using indices.
     *
     * @param x Input array
     * @param dim Axis to gather from
     * @param index Indices array (same shape as output)
     * @return Gathered array (same shape as index)
     */
    Array gather(const Array& x, int dim, const Array& index);

    /**
     * @brief Scatter values into an array at given indices.
     *
     * @param x Input array (output shape is same as x)
     * @param dim Axis to scatter to
     * @param index Indices array (same shape as src)
     * @param src Source values
     * @return Array with scattered values
     */
    Array scatter(const Array& x, int dim, const Array& index, const Array& src);

    /**
     * @brief Scatter with addition (accumulate).
     *
     * @param x Input array
     * @param dim Axis to scatter to
     * @param index Indices array
     * @param src Source values
     * @return Array with accumulated values
     */
    Array scatter_add(const Array& x, int dim, const Array& index, const Array& src);

    /**
     * @brief Scatter with reduction.
     *
     * @param x Input array
     * @param dim Axis to scatter to
     * @param index Indices array
     * @param src Source values
     * @param reduce Reduction mode: "add", "mul", "max", "min", "replace"
     * @return Array with reduced values
     */
    Array scatter_reduce(const Array& x, int dim, const Array& index, const Array& src,
        const std::string& reduce = "replace");

    // ============================================================================
    // Masked operations
    // ============================================================================

    /**
     * @brief Select elements where mask is true.
     *
     * @param x Input array
     * @param mask Boolean mask (must be broadcastable to x.shape)
     * @return 1D array of selected elements
     */
    Array masked_select(const Array& x, const Array& mask);

    /**
     * @brief Compress array along an axis using a condition array.
     *
     * @param x Input array
     * @param condition Boolean 1D array (length must match axis dimension)
     * @param axis Axis to compress (default: 0)
     * @return Compressed array
     */
    Array compress(const Array& x, const Array& condition, std::optional<int> axis = std::nullopt);

    /**
     * @brief Extract elements where condition is true (same as masked_select).
     */
    inline Array extract(const Array& x, const Array& condition) {
        return masked_select(x, condition);
    }

    // ============================================================================
    // Non-zero
    // ============================================================================

    /**
     * @brief Return indices of non-zero elements.
     *
     * @param x Input array
     * @return Array of shape (ndim, n) where n is number of non-zero elements
     */
    Array nonzero(const Array& x);

    /**
     * @brief Return flattened indices of non-zero elements.
     *
     * @param x Input array
     * @return 1D array of indices
     */
    Array flatnonzero(const Array& x);

    // ============================================================================
    // Where
    // ============================================================================

    /**
     * @brief Return elements chosen from x or y depending on condition.
     *
     * @param condition Boolean array
     * @param x Values where condition is true
     * @param y Values where condition is false
     * @return Array of same shape as broadcasted
     */
    Array where(const Array& condition, const Array& x, const Array& y);

    /**
     * @brief Return indices where condition is true (single argument).
     *
     * @param condition Boolean array
     * @return Array of indices (same as nonzero)
     */
    inline Array where(const Array& condition) {
        return nonzero(condition);
    }

    // ============================================================================
    // Sorting
    // ============================================================================

    /**
     * @brief Return indices that would sort the array.
     *
     * @param x Input array
     * @param axis Axis to sort along (-1 for last)
     * @param descending If true, descending order
     * @return Array of indices (int64)
     */
    Array argsort(const Array& x, int axis = -1, bool descending = false);

    /**
     * @brief Return sorted array along an axis.
     *
     * @param x Input array
     * @param axis Axis to sort along (-1 for last)
     * @param descending If true, descending order
     * @return Sorted array
     */
    Array sort(const Array& x, int axis = -1, bool descending = false);

    /**
     * @brief Return top k largest/smallest values and their indices.
     *
     * @param x Input array
     * @param k Number of elements to return
     * @param axis Axis to reduce (-1 for last)
     * @param largest If true, return largest; else smallest
     * @param sorted If true, return sorted results
     * @return Tuple of (values, indices)
     */
    std::tuple<Array, Array> topk(const Array& x, int64_t k, int axis = -1,
        bool largest = true, bool sorted = true);

    // ============================================================================
    // Searching
    // ============================================================================

    /**
     * @brief Find indices where elements should be inserted to maintain order.
     *
     * @param x Sorted input array (1D)
     * @param v Values to search for
     * @param side 'left' or 'right' insertion side
     * @param sorter Optional indices from argsort (for unsorted x)
     * @return Array of insertion indices (same shape as v)
     */
    Array searchsorted(const Array& x, const Array& v, const std::string& side = "left",
        std::optional<Array> sorter = std::nullopt);

    // ============================================================================
    // Unique
    // ============================================================================

    /**
     * @brief Result of unique operation.
     */
    struct UniqueResult {
        Array unique;      ///< Unique elements
        Array indices;     ///< First occurrence indices (if requested)
        Array inverse;     ///< Indices to reconstruct input (if requested)
        Array counts;      ///< Counts of each unique element (if requested)
    };

    /**
     * @brief Return unique elements of an array.
     *
     * @param x Input array
     * @param return_indices If true, return indices of first occurrence
     * @param return_inverse If true, return indices to reconstruct input
     * @param return_counts If true, return counts of each unique element
     * @return UniqueResult struct
     */
    UniqueResult unique(const Array& x,
        bool return_indices = false,
        bool return_inverse = false,
        bool return_counts = false);

    /**
     * @brief Return unique elements and their counts.
     */
    inline std::pair<Array, Array> unique_counts(const Array& x) {
        UniqueResult res = unique(x, false, false, true);
        return { res.unique, res.counts };
    }

    // ============================================================================
    // Partitioning
    // ============================================================================

    /**
     * @brief Return a partitioned array (kth-smallest element at position kth).
     *
     * @param x Input array
     * @param kth Element index to partition around
     * @param axis Axis to partition along
     * @return Partitioned array
     */
    Array partition(const Array& x, int64_t kth, int axis = -1);

    /**
     * @brief Return indices that would partition the array.
     *
     * @param x Input array
     * @param kth Element index to partition around
     * @param axis Axis to partition along
     * @return Indices array
     */
    Array argpartition(const Array& x, int64_t kth, int axis = -1);

    // ============================================================================
    // Multi-key sorting
    // ============================================================================

    /**
     * @brief Perform indirect stable sort using a sequence of keys.
     *
     * @param keys Array of keys (stacked along axis 0)
     * @param axis Axis to sort along (must be 0)
     * @return Indices that would sort the keys
     */
    Array lexsort(const Array& keys, int axis = -1);

    // ============================================================================
    // Grid indices
    // ============================================================================

    /**
     * @brief Return indices representing a grid.
     *
     * Returns an array where the first dimension represents the dimension index,
     * and the remaining dimensions are the grid coordinates.
     *
     * Example:
     * @code
     * Array idx = indices({2, 3});
     * // idx.shape() = (2, 2, 3)
     * // idx[0] = [[0,0,0], [1,1,1]]
     * // idx[1] = [[0,1,2], [0,1,2]]
     * @endcode
     *
     * @param shape Shape of the grid
     * @param sparse If true, return a list of 1D arrays (return type changes)
     * @return Dense grid indices array of shape (ndim, ...)
     */
    Array indices(const Shape& shape, bool sparse = false);

    /**
     * @brief Return coordinate arrays from index vectors (like numpy.ix_).
     *
     * Takes 1D arrays and returns broadcasted versions that can be used for
     * advanced indexing.
     *
     * Example:
     * @code
     * auto grids = ix_({to_array({0,2}), to_array({1,3,5})});
     * // grids[0] shape (2,1) -> [[0],[2]]
     * // grids[1] shape (1,3) -> [[1,3,5]]
     * @endcode
     *
     * @param arrays List of 1D arrays
     * @return Broadcasted arrays (each with ndim = len(arrays))
     */
    std::vector<Array> ix_(const std::vector<Array>& arrays);

    // ============================================================================
    // Interp
    // ============================================================================

    /**
     * @brief One-dimensional linear interpolation.
     *
     * Returns the one-dimensional linear interpolation of a function for
     * given data points (xp, fp). Similar to numpy.interp.
     *
     * @param x The x-coordinates at which to evaluate the interpolated values
     * @param xp The x-coordinates of the data points (must be increasing)
     * @param fp The y-coordinates of the data points (same length as xp)
     * @param left Value to return for x < xp[0] (default: fp[0])
     * @param right Value to return for x > xp[-1] (default: fp[-1])
     * @return Interpolated values, same shape as x
     *
     * Example:
     * @code
     * Array xp = to_array({1.0, 2.0, 3.0});
     * Array fp = to_array({2.0, 4.0, 6.0});
     * Array x = to_array({1.5, 2.5});
     * Array y = interp(x, xp, fp);  // [3.0, 5.0]
     * @endcode
     */
    Array interp(const Array& x, const Array& xp, const Array& fp,
        std::optional<double> left = std::nullopt,
        std::optional<double> right = std::nullopt);

} // namespace ins