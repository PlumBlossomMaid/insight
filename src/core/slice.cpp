// src/core/slice.cpp
#include "insight/core/slice.h"
#include "insight/core/exception.h"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <vector>

namespace ins {

    // ========== Helper functions ==========

    static inline std::string trim(const std::string& s) {
        size_t start = 0, end = s.size();
        while (start < end && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
        while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
        return s.substr(start, end - start);
    }

    static inline std::vector<std::string> split(const std::string& s, char delim) {
        std::vector<std::string> result;
        std::string cur;
        for (char c : s) {
            if (c == delim) {
                result.push_back(cur);
                cur.clear();
            }
            else {
                cur.push_back(c);
            }
        }
        result.push_back(cur);
        return result;
    }

    static inline int64_t parse_int(const std::string& s, int part_idx) {
        try {
            size_t pos;
            int64_t val = std::stoll(s, &pos);
            if (pos != s.size()) {
                INS_THROW("Invalid slice: extra characters in part ", part_idx, ": ", s);
            }
            return val;
        }
        catch (const std::exception&) {
            INS_THROW("Invalid slice: cannot parse part ", part_idx, ": ", s);
        }
    }

    // ========== Slice constructors ==========

    Slice::Slice(int64_t start, int64_t stop, int64_t step)
        : start(start), stop(stop), step(step) {
    }

    Slice Slice::all() {
        return Slice();
    }

    // ========== Slice query ==========

    bool Slice::is_all() const {
        return !start.has_value() && !stop.has_value() && step == 1;
    }

    // ========== Slice normalization ==========

    void Slice::normalize(int64_t dim_size, int64_t& start_out, int64_t& stop_out, int64_t& step_out) const {
        step_out = step;
        if (step_out == 0) {
            INS_THROW("Slice: step cannot be zero");
        }

        // Normalize start
        if (start.has_value()) {
            start_out = start.value();
            if (start_out < 0) {
                start_out += dim_size;
            }
            start_out = std::clamp(start_out, 0LL, dim_size - 1);
        }
        else {
            start_out = (step_out > 0) ? 0 : dim_size - 1;
        }

        // Normalize stop
        if (stop.has_value()) {
            stop_out = stop.value();
            if (stop_out < 0) {
                stop_out += dim_size;
            }
            if (step_out > 0) {
                stop_out = std::clamp(stop_out, 0LL, dim_size);
            }
            else {
                stop_out = std::clamp(stop_out, -1LL, dim_size - 1);
            }
        }
        else {
            stop_out = (step_out > 0) ? dim_size : -1;
        }
    }

    int64_t Slice::size(int64_t dim_size) const {
        int64_t s, e, st;
        normalize(dim_size, s, e, st);
        if (st > 0) {
            return (e - s + st - 1) / st;
        }
        else {
            return (s - e + (-st) - 1) / (-st);
        }
    }

    int64_t Slice::offset(int64_t dim_size, int64_t base_stride) const {
        int64_t s, e, st;
        normalize(dim_size, s, e, st);
        return s * base_stride;
    }

    // ========== Slice composition ==========

    Slice Slice::compose(const Slice& inner) const {
        // Compose two slices: outer(inner(x))
        // This is used for chained indexing, e.g., arr[outer][inner]
        // Note: This is a simplified version assuming contiguous indexing.
        // For full composition with strides, more complex computation is needed.

        Slice result;

        // Compose start
        if (inner.start.has_value()) {
            result.start = inner.start;
        }
        else if (start.has_value()) {
            result.start = start;
        }

        // Compose stop (simplified - for full implementation, need to compute based on sizes)
        if (inner.stop.has_value()) {
            result.stop = inner.stop;
        }
        else if (stop.has_value()) {
            result.stop = stop;
        }

        // Compose step
        result.step = step * inner.step;

        if (result.step == 0) {
            INS_THROW("Slice composition: step cannot be zero");
        }

        return result;
    }

    // ========== String conversion ==========

    std::string Slice::to_string() const {
        std::ostringstream oss;

        auto format_opt = [&](std::optional<int64_t> val) {
            if (val.has_value()) {
                oss << val.value();
            }
            };

        format_opt(start);
        oss << ":";
        format_opt(stop);
        if (step != 1) {
            oss << ":";
            oss << step;
        }

        return oss.str();
    }

    // ========== Slice parser ==========

    Slice parse_slice(const std::string& spec) {
        auto parts = split(spec, ':');

        if (parts.size() > 3) {
            INS_THROW("Invalid slice: too many colons (max 3), got ", parts.size());
        }

        // Pad to 3 parts
        while (parts.size() < 3) {
            parts.push_back("");
        }

        auto parse_part = [&](const std::string& part, int idx) -> std::optional<int64_t> {
            std::string trimmed = trim(part);
            if (trimmed.empty()) {
                return std::nullopt;
            }
            return parse_int(trimmed, idx);
            };

        std::optional<int64_t> start = parse_part(parts[0], 0);
        std::optional<int64_t> stop = parse_part(parts[1], 1);
        std::optional<int64_t> step = parse_part(parts[2], 2);

        if (step.has_value() && step.value() == 0) {
            INS_THROW("Invalid slice: step cannot be zero");
        }

        return Slice(start, stop, step.value_or(1));
    }

} // namespace ins