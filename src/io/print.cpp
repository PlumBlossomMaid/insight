// src/io/print.cpp
#include "insight/io/print.h"
#include "insight/core/exception.h"
#include <iomanip>
#include <sstream>
#include <cmath>
#include <vector>
#include <complex>
#include <algorithm>
#include <thread>

namespace ins {

    namespace {

        thread_local PrintOptions g_print_opts;

    } // anonymous namespace

    PrintOptions& get_print_options() {
        return g_print_opts;
    }

    void set_printoptions(int precision, int threshold, int edgeitems,
        int linewidth, bool sci_mode) {
        if (precision > 0) g_print_opts.precision = precision;
        if (threshold > 0) g_print_opts.threshold = threshold;
        if (edgeitems > 0) g_print_opts.edgeitems = edgeitems;
        if (linewidth > 0) g_print_opts.linewidth = linewidth;
        g_print_opts.sci_mode = sci_mode;
    }

    namespace {

        // Format a single scalar value
        template<typename T>
        std::string format_scalar(T value) {
            std::ostringstream oss;
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                if (g_print_opts.sci_mode) {
                    oss << std::scientific << std::setprecision(g_print_opts.precision) << value;
                }
                else {
                    std::ostringstream tmp;
                    tmp << std::fixed << std::setprecision(g_print_opts.precision) << value;
                    std::string str = tmp.str();
                    // Only trim trailing zeros when using default precision (8)
                    if (g_print_opts.precision == 8) {
                        while (str.size() > 1 && str.back() == '0') {
                            str.pop_back();
                        }
                        if (str.back() == '.') {
                            // Keep as is, e.g., "1."
                        }
                    }
                    oss << str;
                }
            }
            else if constexpr (std::is_same_v<T, bool>) {
                oss << (value ? "true" : "false");
            }
            else if constexpr (std::is_same_v<T, std::complex<float>> ||
                std::is_same_v<T, std::complex<double>>) {
                auto re = value.real();
                auto im = value.imag();
                int prec = g_print_opts.precision;
                if (g_print_opts.sci_mode) {
                    if (im >= 0) {
                        oss << "(" << std::scientific << std::setprecision(prec) << re
                            << "+" << std::scientific << std::setprecision(prec) << im << "j)";
                    }
                    else {
                        oss << "(" << std::scientific << std::setprecision(prec) << re
                            << std::scientific << std::setprecision(prec) << im << "j)";
                    }
                }
                else {
                    if (im >= 0) {
                        oss << "(" << std::fixed << std::setprecision(prec) << re
                            << "+" << std::fixed << std::setprecision(prec) << im << "j)";
                    }
                    else {
                        oss << "(" << std::fixed << std::setprecision(prec) << re
                            << std::fixed << std::setprecision(prec) << im << "j)";
                    }
                }
            }
            else {
                oss << value;
            }
            return oss.str();
        }

        // Format a 1D array (row)
        template<typename T>
        std::string format_row(const T* data, int64_t size) {
            std::ostringstream oss;
            oss << "[";

            if (size > 2 * g_print_opts.edgeitems) {
                // First part
                for (int i = 0; i < g_print_opts.edgeitems; ++i) {
                    if (i > 0) oss << ", ";
                    oss << format_scalar(data[i]);
                }
                oss << ", ..., ";
                // Last part
                for (int i = size - g_print_opts.edgeitems; i < size; ++i) {
                    if (i > size - g_print_opts.edgeitems) oss << ", ";
                    oss << format_scalar(data[i]);
                }
            }
            else {
                for (int64_t i = 0; i < size; ++i) {
                    if (i > 0) oss << ", ";
                    oss << format_scalar(data[i]);
                }
            }
            oss << "]";
            return oss.str();
        }

        // Recursively format array
        template<typename T>
        std::string format_array_recursive(const T* data, const Shape& shape,
            int dim, int indent, bool summary) {
            int ndim = shape.ndim();

            if (ndim == 0) {
                return format_scalar(*data);
            }

            int64_t dim_size = shape.dim(dim);

            if (dim == ndim - 1) {
                return format_row(data, dim_size);
            }

            // Calculate stride for this dimension
            int64_t stride = 1;
            for (int d = dim + 1; d < ndim; ++d) {
                stride *= shape.dim(d);
            }

            bool dim_summary = summary && (dim_size > 2 * g_print_opts.edgeitems);
            int64_t actual_size = dim_summary ? 2 * g_print_opts.edgeitems : dim_size;

            // Collect all elements
            std::vector<std::string> elements;
            for (int64_t i = 0; i < actual_size; ++i) {
                int64_t src_idx = i;
                if (dim_summary && i >= g_print_opts.edgeitems) {
                    src_idx = dim_size - (2 * g_print_opts.edgeitems - i);
                }
                elements.push_back(format_array_recursive(
                    data + src_idx * stride, shape, dim + 1, indent + 1, summary));
            }

            if (dim_summary && dim_size > 2 * g_print_opts.edgeitems) {
                if (g_print_opts.edgeitems > 0) {
                    elements.insert(elements.begin() + g_print_opts.edgeitems, "...");
                }
            }

            // Join with separator: ",\n" + spaces (indent)
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < elements.size(); ++i) {
                if (i > 0) {
                    oss << ",";
                    // 只有最外层且维度 >= 3 时，才添加空行
                    if (dim == 0 && ndim >= 3) {
                        oss << "\n\n";
                    }
                    else {
                        oss << "\n";
                    }
                    oss << std::string(indent + 1, ' ');
                }
                oss << elements[i];
            }
            oss << "]";
            return oss.str();
        }

    } // anonymous namespace

    std::string to_string(const Array& arr, int indent) {
        INS_CHECK(arr.defined(), "Cannot print undefined array");
		auto place = arr.place();
        // Copy to CPU and make contiguous
        Array cpu = arr.to(CPUPlace());
        if (!cpu.is_contiguous()) {
            cpu = cpu.contiguous();
        }

        Shape shape = cpu.shape();
        DType dtype = cpu.dtype();
        int64_t total_elements = cpu.numel();
        bool summary = (total_elements > g_print_opts.threshold);

        std::string data_str;

        switch (dtype) {
        case DType::BOOL: {
            const bool* data = cpu.data<bool>();
            data_str = format_array_recursive(data, shape, 0, indent, summary);
            break;
        }
        case DType::U8: {
            const uint8_t* data = cpu.data<uint8_t>();
            data_str = format_array_recursive(data, shape, 0, indent, summary);
            break;
        }
        case DType::I8: {
            const int8_t* data = cpu.data<int8_t>();
            data_str = format_array_recursive(data, shape, 0, indent, summary);
            break;
        }
        case DType::I16: {
            const int16_t* data = cpu.data<int16_t>();
            data_str = format_array_recursive(data, shape, 0, indent, summary);
            break;
        }
        case DType::I32: {
            const int32_t* data = cpu.data<int32_t>();
            data_str = format_array_recursive(data, shape, 0, indent, summary);
            break;
        }
        case DType::I64: {
            const int64_t* data = cpu.data<int64_t>();
            data_str = format_array_recursive(data, shape, 0, indent, summary);
            break;
        }
        case DType::U16: {
            const uint16_t* data = cpu.data<uint16_t>();
            data_str = format_array_recursive(data, shape, 0, indent, summary);
            break;
        }
        case DType::U32: {
            const uint32_t* data = cpu.data<uint32_t>();
            data_str = format_array_recursive(data, shape, 0, indent, summary);
            break;
        }
        case DType::U64: {
            const uint64_t* data = cpu.data<uint64_t>();
            data_str = format_array_recursive(data, shape, 0, indent, summary);
            break;
        }
        case DType::F32: {
            const float* data = cpu.data<float>();
            data_str = format_array_recursive(data, shape, 0, indent, summary);
            break;
        }
        case DType::F64: {
            const double* data = cpu.data<double>();
            data_str = format_array_recursive(data, shape, 0, indent, summary);
            break;
        }
        case DType::C32: {
            const std::complex<float>* data = cpu.data<std::complex<float>>();
            data_str = format_array_recursive(data, shape, 0, indent, summary);
            break;
        }
        case DType::C64: {
            const std::complex<double>* data = cpu.data<std::complex<double>>();
            data_str = format_array_recursive(data, shape, 0, indent, summary);
            break;
        }
        default:
            INS_THROW("to_string: unsupported dtype ", static_cast<int>(dtype));
        }

        // Build final string
        std::ostringstream oss;
        oss << g_print_opts.prefix << "(shape=" << shape
            << ", dtype=" << dtype
            << ", place=" << place << ",\n";
        oss << std::string(indent, ' ')
            << data_str << ")";

        return oss.str();
    }

    std::ostream& operator<<(std::ostream& os, const Array& arr) {
        os << to_string(arr);
        return os;
    }

} // namespace ins