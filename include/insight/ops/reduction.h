// include/insight/ops/reduction.h
#pragma once

#include <optional>
#include "insight/core/array.h"

namespace ins {

    // ========== Basic Reduction ==========

    /**
     * @brief Sum of array elements along specified axis.
     *
     * @param x Input array
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @return Array containing sum values
     */
    Array sum(const Array& x, std::optional<int> axis = std::nullopt, bool keepdim = false);

    /**
     * @brief Mean of array elements along specified axis.
     *
     * @param x Input array
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @return Array containing mean values
     */
    Array mean(const Array& x, std::optional<int> axis = std::nullopt, bool keepdim = false);

    /**
     * @brief Maximum of array elements along specified axis.
     *
     * @param x Input array
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @return Array containing maximum values
     */
    Array max(const Array& x, std::optional<int> axis = std::nullopt, bool keepdim = false);

    /**
     * @brief Minimum of array elements along specified axis.
     *
     * @param x Input array
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @return Array containing minimum values
     */
    Array min(const Array& x, std::optional<int> axis = std::nullopt, bool keepdim = false);

    /**
     * @brief Product of array elements along specified axis.
     *
     * @param x Input array
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @return Array containing product values
     */
    Array prod(const Array& x, std::optional<int> axis = std::nullopt, bool keepdim = false);

    /**
     * @brief Logical OR of array elements along specified axis.
     * Returns true if any element is non-zero.
     *
     * @param x Input array
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @return Array containing boolean values
     */
    Array any(const Array& x, std::optional<int> axis = std::nullopt, bool keepdim = false);

    /**
     * @brief Logical AND of array elements along specified axis.
     * Returns true only if all elements are non-zero.
     *
     * @param x Input array
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @return Array containing boolean values
     */
    Array all(const Array& x, std::optional<int> axis = std::nullopt, bool keepdim = false);

    // ========== Positional Reduction ==========

    /**
     * @brief Returns indices of maximum values along specified axis.
     *
     * @param x Input array
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @return Array of indices (dtype: int64)
     */
    Array argmax(const Array& x, std::optional<int> axis = std::nullopt, bool keepdim = false);

    /**
     * @brief Returns indices of minimum values along specified axis.
     *
     * @param x Input array
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @return Array of indices (dtype: int64)
     */
    Array argmin(const Array& x, std::optional<int> axis = std::nullopt, bool keepdim = false);

    // ========== Statistical Reduction ==========

    /**
     * @brief Variance of array elements along specified axis.
     *
     * @param x Input array
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @param ddof Delta degrees of freedom (0 for population, 1 for sample)
     * @return Array containing variance values
     */
    Array var(const Array& x, std::optional<int> axis = std::nullopt, bool keepdim = false, int ddof = 0);

    /**
     * @brief Standard deviation of array elements along specified axis.
     *
     * @param x Input array
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @param ddof Delta degrees of freedom (0 for population, 1 for sample)
     * @return Array containing standard deviation values
     */
    Array std(const Array& x, std::optional<int> axis = std::nullopt, bool keepdim = false, int ddof = 0);

    /**
     * @brief Standard error of the mean along specified axis.
     *
     * @param x Input array
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @param ddof Delta degrees of freedom (0 for population, 1 for sample)
     * @return Array containing standard error values
     */
    Array sem(const Array& x, std::optional<int> axis = std::nullopt, bool keepdim = false, int ddof = 0);

    // ========== Count ==========

    /**
     * @brief Count number of non-zero elements along specified axis.
     *
     * @param x Input array
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @return Array of counts (dtype: int64)
     */
    Array count_nonzero(const Array& x, std::optional<int> axis = std::nullopt, bool keepdim = false);

    // ========== Cumulative ==========

    /**
     * @brief Cumulative sum along specified axis.
     *
     * @param x Input array
     * @param axis Axis to accumulate (0 <= axis < ndim)
     * @param dtype Output data type (default: float64 for integer inputs)
     * @return Array with same shape as input
     */
    Array cumsum(const Array& x, int axis, DType dtype = DType::F64);

    /**
     * @brief Cumulative product along specified axis.
     *
     * @param x Input array
     * @param axis Axis to accumulate (0 <= axis < ndim)
     * @param dtype Output data type (default: float64 for integer inputs)
     * @return Array with same shape as input
     */
    Array cumprod(const Array& x, int axis, DType dtype = DType::F64);

    /**
     * @brief Cumulative maximum along specified axis.
     *
     * @param x Input array
     * @param axis Axis to accumulate (0 <= axis < ndim)
     * @return Array with same shape as input
     */
    Array cummax(const Array& x, int axis);

    /**
     * @brief Cumulative minimum along specified axis.
     *
     * @param x Input array
     * @param axis Axis to accumulate (0 <= axis < ndim)
     * @return Array with same shape as input
     */
    Array cummin(const Array& x, int axis);

    // ========== Quantile ==========

    /**
     * @brief Median value along specified axis.
     *
     * @param x Input array
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @return Array containing median values
     */
    Array median(const Array& x, std::optional<int> axis = std::nullopt, bool keepdim = false);

    /**
     * @brief Quantile values along specified axis (q in [0, 1]).
     *
     * @param x Input array
     * @param q Quantile (single value, 0 <= q <= 1)
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @return Array containing quantile values
     */
    Array quantile(const Array& x, double q, std::optional<int> axis = std::nullopt, bool keepdim = false);

    /**
     * @brief Quantile values along specified axis (multiple quantiles).
     *
     * @param x Input array
     * @param q Array of quantiles (all values in [0, 1])
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @return Array of shape (len(q),) + output_shape
     */
    Array quantile(const Array& x, const Array& q, std::optional<int> axis = std::nullopt, bool keepdim = false);

    /**
     * @brief Percentile values along specified axis (q in [0, 100]).
     *
     * @param x Input array
     * @param q Percentile (single value, 0 <= q <= 100)
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @return Array containing percentile values
     */
    Array percentile(const Array& x, double q, std::optional<int> axis, bool keepdim);

    // ========== NaN-safe Reduction ==========

    /**
     * @brief Sum of array elements, ignoring NaN values.
     *
     * @param x Input array
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @return Array containing sum values (NaN if all elements are NaN)
     */
    Array nansum(const Array& x, std::optional<int> axis = std::nullopt, bool keepdim = false);

    /**
     * @brief Mean of array elements, ignoring NaN values.
     *
     * @param x Input array
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @return Array containing mean values (NaN if all elements are NaN)
     */
    Array nanmean(const Array& x, std::optional<int> axis = std::nullopt, bool keepdim = false);

    /**
     * @brief Maximum of array elements, ignoring NaN values.
     *
     * @param x Input array
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @return Array containing maximum values (NaN if all elements are NaN)
     */
    Array nanmax(const Array& x, std::optional<int> axis = std::nullopt, bool keepdim = false);

    /**
     * @brief Minimum of array elements, ignoring NaN values.
     *
     * @param x Input array
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @return Array containing minimum values (NaN if all elements are NaN)
     */
    Array nanmin(const Array& x, std::optional<int> axis = std::nullopt, bool keepdim = false);

    /**
     * @brief Variance of array elements, ignoring NaN values.
     *
     * @param x Input array
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @param ddof Delta degrees of freedom (0 for population, 1 for sample)
     * @return Array containing variance values (NaN if all elements are NaN)
     */
    Array nanvar(const Array& x, std::optional<int> axis = std::nullopt, bool keepdim = false, int ddof = 0);

    /**
     * @brief Standard deviation of array elements, ignoring NaN values.
     *
     * @param x Input array
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @param ddof Delta degrees of freedom (0 for population, 1 for sample)
     * @return Array containing standard deviation values (NaN if all elements are NaN)
     */
    Array nanstd(const Array& x, std::optional<int> axis = std::nullopt, bool keepdim = false, int ddof = 0);

    /**
     * @brief Median of array elements, ignoring NaN values.
     *
     * @param x Input array
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @return Array containing median values (NaN if all elements are NaN)
     */
    Array nanmedian(const Array& x, std::optional<int> axis = std::nullopt, bool keepdim = false);

    /**
     * @brief Quantile values along specified axis, ignoring NaN values.
     *
     * @param x Input array
     * @param q Quantile (single value, 0 <= q <= 1)
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @return Array containing quantile values (NaN if all elements are NaN)
     */
    Array nanquantile(const Array& x, double q, std::optional<int> axis = std::nullopt, bool keepdim = false);

    /**
     * @brief Quantile values along specified axis, ignoring NaN values.
     *
     * @param x Input array
     * @param q Array of quantiles (all values in [0, 1])
     * @param axis Axis to reduce (nullopt means flatten)
     * @param keepdim If true, keep reduced dimensions as size 1
     * @return Array of shape (len(q),) + output_shape
     */
    Array nanquantile(const Array& x, const Array& q, std::optional<int> axis = std::nullopt, bool keepdim = false);

} // namespace ins