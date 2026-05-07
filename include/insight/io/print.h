// insight/io/print.h
#pragma once
#include <ostream>
#include "insight/core/array.h"

namespace ins {

    /**
     * @brief Print options for array formatting.
     *
     * Aligns with Paddle's set_printoptions behavior.
     */
    struct PrintOptions {
        int precision = 8;      ///< Number of digits for floating point
        int threshold = 1000;   ///< Total elements threshold for summary
        int edgeitems = 3;      ///< Number of items at beginning/end in summary
        int linewidth = 80;     ///< Maximum line width (not yet implemented)
        bool sci_mode = false;  ///< Use scientific notation
		std::string prefix = "Array"; ///< Prefix for each Array
    };

    /**
     * @brief Get current print options.
     * @return Reference to thread-local print options
     */
    PrintOptions& get_print_options();

    /**
     * @brief Set printing options.
     *
     * @param precision Number of digits for floating point (default: 8)
     * @param threshold Total elements threshold for summary (default: 1000)
     * @param edgeitems Number of items at beginning/end in summary (default: 3)
     * @param linewidth Maximum line width (default: 80)
     * @param sci_mode Use scientific notation (default: false)
     */
    void set_printoptions(int precision = -1, int threshold = -1,
        int edgeitems = -1, int linewidth = -1,
        bool sci_mode = false);

    /**
     * @brief Convert array to string representation.
     *
     * Format follows Paddle's tensor printing style:
     *   Array(shape=[2, 3], dtype=float32, place=cpu,
     *        [[0.1, 0.2, 0.3],
     *         [0.4, 0.5, 0.6]])
     *
     * @param arr Input array (will be copied to CPU if needed)
     * @return Formatted string
     */
    std::string to_string(const Array& arr, int indent = get_print_options().prefix.size() + 1);

    /**
     * @brief Output array to stream.
     *
     * @param os Output stream
     * @param arr Array to print
     * @return Reference to the output stream
     */
    std::ostream& operator<<(std::ostream& os, const Array& arr);

} // namespace ins