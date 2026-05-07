// insight/ops/manipulation.h
#pragma once
#include <vector>
#include <optional>
#include "insight/core/array.h"

namespace ins {

    // ========== Shape manipulation (views, zero-copy) ==========

    /**
     * @brief Reshape array to new shape (view if possible).
     */
    Array reshape(const Array& x, const Shape& new_shape);

    /**
     * @brief Flatten array to 1D (copy).
     */
    Array flatten(const Array& x);

    /**
     * @brief Flatten array to 1D (may return view if contiguous).
     */
    Array ravel(const Array& x);

    /**
     * @brief Remove dimensions of size 1 (view).
     */
    Array squeeze(const Array& x);
    Array squeeze(const Array& x, int axis);

    /**
     * @brief Add a dimension of size 1 (view).
     */
    Array unsqueeze(const Array& x, int axis);

    /**
     * @brief Expand dimensions (alias for unsqueeze).
     */
    inline Array expand_dims(const Array& x, int axis) { return unsqueeze(x, axis); }

    // ========== Transpose and axis manipulation (views) ==========

    /**
     * @brief Transpose last two dimensions (view).
     */
    Array transpose(const Array& x);

    /**
     * @brief Permute dimensions arbitrarily (view).
     */
    Array permute(const Array& x, const std::vector<int>& axes);

    /**
     * @brief Swap two axes (view).
     */
    Array swapaxes(const Array& x, int axis1, int axis2);

    /**
     * @brief Move axis to new position (view).
     */
    Array moveaxis(const Array& x, int source, int destination);

    // ========== Flipping and rotating (views) ==========

    /**
     * @brief Reverse order of elements along given axis (view).
     */
    Array flip(const Array& x, std::optional<int> axis = std::nullopt);

    /**
     * @brief Flip left-right (axis=1).
     */
    inline Array fliplr(const Array& x) { return flip(x, 1); }

    /**
     * @brief Flip up-down (axis=0).
     */
    inline Array flipud(const Array& x) { return flip(x, 0); }

    /**
     * @brief Rotate 2D array by 90 degrees (view).
     */
    Array rot90(const Array& x, int k = 1, const std::vector<int>& axes = { 0, 1 });

    // ========== Joining (copy) ==========

    /**
     * @brief Concatenate arrays along an existing axis.
     */
    Array concat(const std::vector<Array>& tensors, int axis = 0);

    /**
     * @brief Stack arrays along a new axis.
     */
    Array stack(const std::vector<Array>& tensors, int axis = 0);

    /**
     * @brief Stack arrays vertically (axis=0).
     */
    inline Array vstack(const std::vector<Array>& tensors) { return concat(tensors, 0); }

    /**
     * @brief Stack arrays horizontally (axis=1).
     */
    inline Array hstack(const std::vector<Array>& tensors) { return concat(tensors, 1); }

    // ========== Splitting (views) ==========

    /**
     * @brief Split array into multiple sub-arrays (views).
     */
    std::vector<Array> split(const Array& x, int indices_or_sections, int axis = 0);
    std::vector<Array> split(const Array& x, const std::vector<int64_t>& indices, int axis = 0);

    /**
     * @brief Split vertically (axis=0).
     */
    inline std::vector<Array> vsplit(const Array& x, int indices_or_sections) {
        return split(x, indices_or_sections, 0);
    }

    /**
     * @brief Split horizontally (axis=1).
     */
    inline std::vector<Array> hsplit(const Array& x, int indices_or_sections) {
        return split(x, indices_or_sections, 1);
    }

    // ========== Tiling and repeating (copy) ==========

    /**
     * @brief Repeat elements of an array.
     */
    Array repeat(const Array& x, int repeats, std::optional<int> axis = std::nullopt);
    Array repeat(const Array& x, const std::vector<int>& repeats, int axis);

    /**
     * @brief Tile an array by repeating it along specified axes.
     */
    Array tile(const Array& x, const Shape& reps);

    // ========== Padding (copy) ==========

    /**
     * @brief Pad an array with constant values.
     */
    Array pad(const Array& x, const std::vector<int64_t>& pad_width, double constant_value = 0.0);

    // ========== Rolling (copy) ==========

    /**
     * @brief Roll array elements along given axis.
     */
    Array roll(const Array& x, int shift, std::optional<int> axis = std::nullopt);

    // ========== Diagonal (copy) ==========

    /**
     * @brief Extract diagonal or construct diagonal array.
     */
    Array diag(const Array& x, int k = 0);

    /**
     * @brief Extract diagonal elements.
     */
    Array diagonal(const Array& x, int offset = 0, int axis1 = 0, int axis2 = 1);

    // ========== Triangular (copy) ==========

    /**
     * @brief Lower triangle of an array.
     */
    Array tril(const Array& x, int k = 0);

    /**
     * @brief Upper triangle of an array.
     */
    Array triu(const Array& x, int k = 0);

    // ========== Slicing (Views) ==========

    /// Slice a single dimension
    Array slice(Array& x, int dim, int64_t start, int64_t stop, int64_t step = 1);

    /// Multi-dimensional slice with Slice objects
    Array slice(Array& x, const std::vector<Slice>& slices);

    // ========== Diff ==========

    /**
     * @brief Calculate the n-th order difference along given axis.
     *
     * Computes the n-th order forward differences. The first order difference
     * is given by out[i] = x[i+1] - x[i] along the given axis.
     *
     * @param x Input array
     * @param n Order of difference (default: 1)
     * @param axis Axis along which to compute differences (default: -1)
     * @return Array of differences (shape is reduced by n along the axis)
     *
     * Example:
     * @code
     * Array a = to_array({1, 2, 4, 7});
     * Array d = diff(a);           // [1, 2, 3]
     * @endcode
     */
    Array diff(const Array& x, int n = 1, int axis = -1);

} // namespace ins