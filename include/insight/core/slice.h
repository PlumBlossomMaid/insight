// insight/core/slice.h
#pragma once
#include <cstdint>
#include <optional>
#include <string>

namespace ins {

    /**
     * @brief Slice descriptor for array indexing.
     *
     * Follows NumPy/Python slice semantics.
     *
     * Examples:
     *   Slice()                    ->  all elements (::)
     *   Slice(5)                   ->  from index 5 to end (5:)
     *   Slice(2, 5)                ->  from 2 to 4 (2:5)
     *   Slice(2, 5, 2)             ->  from 2 to 4, step 2 (2:5:2)
     *   Slice(-5, -2)              ->  negative indices (count from end)
     *   Slice(std::nullopt, 5)     ->  from start to 4 (:5)
     *   Slice(5, std::nullopt)     ->  from 5 to end (5:)
     *   Slice(std::nullopt, std::nullopt, -1) ->  full reverse (::-1)
     */
    struct Slice {
        std::optional<int64_t> start;
        std::optional<int64_t> stop;
        int64_t step = 1;

        // Constructors
        Slice() = default;
        Slice(int64_t start, int64_t stop, int64_t step = 1);

        // Constructor with optional values (for parser)
        Slice(std::optional<int64_t> start, std::optional<int64_t> stop, int64_t step = 1)
            : start(start), stop(stop), step(step) {
        }

        // Named constructors
        static Slice all();

        // Query
        bool is_all() const;
        bool is_reverse() const { return step < 0; }

        // Normalize with dimension size (returns normalized start, stop, step, and length)
        void normalize(int64_t dim_size, int64_t& start_out, int64_t& stop_out, int64_t& step_out) const;

        // Compute slice result: new size and offset in elements
        int64_t size(int64_t dim_size) const;
        int64_t offset(int64_t dim_size, int64_t base_stride) const;

        // For chained slicing: apply slice on top of another slice
        Slice compose(const Slice& inner) const;

        // String conversion
        std::string to_string() const;
    };

    // Parse slice string (e.g., "::2", "1:10:2", "::-1")
    // Rules:
    //   - Leading/trailing spaces are ignored
    //   - Empty part (after trim) means omitted (std::nullopt)
    //   - Non-empty part must be a valid integer
    //   - At most 2 colons (i.e., at most 3 parts)
    Slice parse_slice(const std::string& spec);

} // namespace ins