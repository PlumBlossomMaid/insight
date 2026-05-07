// src/core/array.cpp
#include "insight/ops/reduction.h"
#include "insight/core/array.h"
#include "insight/core/exception.h"
#include "insight/plugin/op_registry.h"
#include "array_impl.h"
#include <cstring>
#include <numeric>
#include <iostream>
#ifdef INSIGHT_WITH_OPENMP
#include <omp.h>
#endif

namespace ins {

    static gpu::Device* get_gpu_device() {
        static std::unique_ptr<gpu::Device> device = []() -> std::unique_ptr<gpu::Device> {
            auto* factory = insight_create_device_factory();
            if (factory && factory->is_available()) {
                return factory->create_device();
            }
            return nullptr;
            }();
        return device.get();
    }

    // ========== Constructors ==========

    Array::Array() : impl_(nullptr) {}

    Array::Array(const Shape& shape, DType dtype, const Place& place)
        : impl_(std::make_shared<ArrayImpl>()) {
        impl_->shape = shape;
        impl_->dtype = dtype;
        impl_->place = place;
        impl_->update_strides_from_shape();
        impl_->allocate_storage();
        impl_->validate();

        // Initialize to zero
        if (impl_->storage && impl_->nbytes() > 0) {
            if (place.is_cpu()) {
                std::memset(impl_->storage.get(), 0, impl_->nbytes());
            }
            else {
                auto* device = get_gpu_device();
                INS_CHECK(device != nullptr, "GPU not available, got device = ", device);
                device->device_memory_set(place.device_id(), impl_->storage.get(), 0, impl_->nbytes());
            }
        }
    }

    // Boolean
    Array::Array(bool value)
        : Array(Shape({}), DType::BOOL, Place::CPU()) {
        *data<bool>() = value;
        *this = this->to(ins::get_device());
    }

    // Unsigned integers
    Array::Array(uint8_t value)
        : Array(Shape({}), DType::U8, Place::CPU()) {
        *data<uint8_t>() = value;
        *this = this->to(ins::get_device());
    }

    // Signed integers (all sizes)
    Array::Array(int8_t value)
        : Array(Shape({}), DType::I8, Place::CPU()) {
        *data<int8_t>() = value;
        *this = this->to(ins::get_device());
    }

    Array::Array(int16_t value)
        : Array(Shape({}), DType::I16, Place::CPU()) {
        *data<int16_t>() = value;
        *this = this->to(ins::get_device());
    }

    Array::Array(int32_t value)
        : Array(Shape({}), DType::I32, Place::CPU()) {
        *data<int32_t>() = value;
        *this = this->to(ins::get_device());
    }

    Array::Array(int64_t value)
        : Array(Shape({}), DType::I64, Place::CPU()) {
        *data<int64_t>() = value;
        *this = this->to(ins::get_device());
    }

    // Unsigned integers
    Array::Array(uint16_t value)
        : Array(Shape({}), DType::U16, Place::CPU()) {
        *data<uint16_t>() = value;
        *this = this->to(ins::get_device());
    }

    Array::Array(uint32_t value)
        : Array(Shape({}), DType::U32, Place::CPU()) {
        *data<uint32_t>() = value;
        *this = this->to(ins::get_device());
    }

    Array::Array(uint64_t value)
        : Array(Shape({}), DType::U64, Place::CPU()) {
        *data<uint64_t>() = value;
        *this = this->to(ins::get_device());
    }

    // Floating point
    Array::Array(float value)
        : Array(Shape({}), DType::F32, Place::CPU()) {
        *data<float>() = value;
        *this = this->to(ins::get_device());
    }

    Array::Array(double value)
        : Array(Shape({}), DType::F64, Place::CPU()) {
        *data<double>() = value;
        *this = this->to(ins::get_device());
    }

    // Complex
    Array::Array(std::complex<float> value)
        : Array(Shape({}), DType::C32, Place::CPU()) {
        *data<std::complex<float>>() = value;
        *this = this->to(ins::get_device());
    }

    Array::Array(std::complex<double> value)
        : Array(Shape({}), DType::C64, Place::CPU()) {
        *data<std::complex<double>>() = value;
        *this = this->to(ins::get_device());
    }

    Array::Array(std::shared_ptr<ArrayImpl> parent_impl,
        const Shape& shape, const Strides& strides, int64_t offset)
        : impl_(std::make_shared<ArrayImpl>()) {

        impl_->storage = parent_impl->storage;
        impl_->dtype = parent_impl->dtype;
        impl_->place = parent_impl->place;

        impl_->shape = shape;
        impl_->strides = strides;
        impl_->offset = offset;
        impl_->is_view = true;
    }

    Array::~Array() = default;

    // ========== Assignment ==========

    Array& Array::operator=(const Array& other) {
        if (this != &other) {
            Array temp(other);
            std::swap(impl_, temp.impl_);
        }
        return *this;
    }

    Array& Array::operator=(Array&& other) noexcept {
        if (this != &other) {
            impl_ = std::move(other.impl_);
            other.impl_ = nullptr;
        }
        return *this;
    }

    // ========== State ==========

    bool Array::defined() const {
        return impl_ != nullptr && impl_->storage != nullptr;
    }

    // ========== Metadata ==========

    Shape Array::shape() const {
        INS_CHECK(defined(), "Array is not initialized");
        return impl_->shape;
    }

    DType Array::dtype() const {
        INS_CHECK(defined(), "Array is not initialized");
        return impl_->dtype;
    }

    Place Array::place() const {
        INS_CHECK(defined(), "Array is not initialized");
        return impl_->place;
    }

    int64_t Array::numel() const {
        INS_CHECK(defined(), "Array is not initialized");
        return impl_->shape.numel();
    }

    // ========== Memory Layout ==========

    bool Array::is_contiguous() const {
        INS_CHECK(defined(), "Array is not initialized");
        if (numel() == 1) {
            return true;
        }
        return impl_->strides.is_contiguous(impl_->shape);
    }

    Array Array::contiguous() const {
        INS_CHECK(defined(), "Cannot call contiguous on undefined array");

        if (is_contiguous()) {
            return *this;
        }
        if (numel() == 1) {
            return *this;
        }

        // Create result array (same device, contiguous)
        Array result(shape(), dtype(), place());

        // Dispatch via kernel registry
        OpArgs args = { result, *this };
        DeviceKind dev = place().is_cpu() ? DeviceKind::CPU : DeviceKind::GPU;
        OpArgs output = ops()["contiguous_copy"][dev][dtype()](args);

        return std::any_cast<Array>(output[0]);
    }

    // ========== Data Access ==========

    void* Array::data() {
        INS_CHECK(defined(), "Array is not initialized");
        char* base = static_cast<char*>(impl_->storage.get());
        return base + impl_->offset * dtype_size(dtype());
    }

    const void* Array::data() const {
        INS_CHECK(defined(), "Array is not initialized");
        const char* base = static_cast<const char*>(impl_->storage.get());
        return base + impl_->offset * dtype_size(dtype());
    }

    // ========== Element Access ==========

    Array Array::at(const std::vector<int64_t>& indices) const {
        INS_CHECK(defined(), "Array is not initialized");
        INS_CHECK(indices.size() == static_cast<size_t>(shape().ndim()),
            "at(): index count mismatch. Expected ", shape().ndim(),
            ", got ", indices.size());

        // Compute offset using strides, with negative index support
        int64_t elem_offset = impl_->offset;
        for (size_t i = 0; i < indices.size(); ++i) {
            int64_t idx = indices[i];
            int64_t dim_size = shape().dim(static_cast<int>(i));

            if (idx < 0) {
                idx += dim_size;
            }

            INS_CHECK(idx >= 0 && idx < dim_size,
                "at(): index [", i, "] = ", indices[i],
                " out of range [", -dim_size, ", ", dim_size - 1, "]");
            elem_offset += idx * impl_->strides[i];
        }

        // Return a zero-dimensional array (scalar) as a view
        Shape scalar_shape({});  // 0-dimensional
        Strides scalar_strides;   // empty strides
        return Array(impl_, scalar_shape, scalar_strides, elem_offset);
    }

    // ========== Slicing (Views) ==========

    Array Array::slice(int dim, int64_t start, int64_t stop, int64_t step) const {
        INS_CHECK(defined(), "Cannot slice undefined array");
        INS_CHECK(dim >= 0 && dim < shape().ndim(),
            "slice(): dimension index out of range");

        std::vector<Slice> slices(shape().ndim(), Slice::all());
        slices[dim] = Slice(start, stop, step);
        return slice(slices);
    }

    Array Array::slice(const std::vector<Slice>& slices) const {
        INS_CHECK(defined(), "Cannot slice undefined array");
        INS_CHECK(slices.size() == static_cast<size_t>(shape().ndim()),
            "slice(): number of slices must match number of dimensions");

        // Check contiguity
        if (!is_contiguous()) {
            INS_THROW("slice(): non-contiguous array not yet supported");
        }

        std::vector<int64_t> new_dims;
        std::vector<int64_t> new_strides_vec;
        int64_t new_offset = impl_->offset;

        for (size_t i = 0; i < slices.size(); ++i) {
            const Slice& s = slices[i];
            int64_t dim_size = shape().dim(static_cast<int>(i));
            int64_t base_stride = impl_->strides[i];

            // Normalize slice (handles negative indices)
            int64_t start, stop, step;
            s.normalize(dim_size, start, stop, step);

            // Compute new dimension size
            int64_t new_dim = 0;
            if (step > 0) {
                new_dim = (stop - start + step - 1) / step;
            }
            else {
                new_dim = (start - stop + (-step) - 1) / (-step);
            }
            new_dims.push_back(new_dim);

            // New stride = step * base_stride
            new_strides_vec.push_back(step * base_stride);

            // New offset = start * base_stride
            new_offset += start * base_stride;
        }

        Shape new_shape(new_dims);
        Strides new_strides(new_strides_vec);

        return Array(impl_, new_shape, new_strides, new_offset);
    }
    
    Array Array::operator[](const Slice& slice) const {
        // Assume slicing the first dimension (common in 1D/2D cases)
        return this->slice(0, slice.start.value_or(0),
            slice.stop.value_or(shape().dim(0)),
            slice.step);
    }

    Array Array::operator[](const std::string& spec) const {
        if (spec.find(':') != std::string::npos) {
            Slice s = parse_slice(spec);
            return slice({ s });
        }

        if (spec.empty()) {
            INS_THROW("Invalid index: empty string");
        }

        bool is_number = true;
        for (size_t i = 0; i < spec.size(); ++i) {
            char c = spec[i];
            if (i == 0 && c == '-') continue;
            if (!std::isdigit(static_cast<unsigned char>(c))) {
                is_number = false;
                break;
            }
        }

        if (is_number) {
            int64_t idx = std::stoll(spec);
            if (idx < 0) {
                idx += shape().ndim();
            }
            INS_CHECK(idx >= 0 && idx < shape().ndim(),
                "Index out of range: ", idx, " for dimension with size ", shape().ndim());
            return at(idx);
        }

        INS_THROW("Invalid index/slice syntax: ", spec);
    }

    // ========== View Operations ==========

    Array Array::reshape(const Shape& new_shape) const {
        INS_CHECK(defined(), "Cannot reshape undefined array");
        INS_CHECK(new_shape.numel() == numel(),
            "reshape(): shape.numel() mismatch. Expected ", numel(),
            ", got ", new_shape.numel());

        if (!is_contiguous()) {
            // Auto-contiguous for user convenience
            Array cont = contiguous();
            return cont.reshape(new_shape);
        }

        Strides new_strides(new_shape);
        return Array(impl_, new_shape, new_strides, impl_->offset);
    }

    Array Array::transpose() const {
        INS_CHECK(defined(), "Cannot transpose undefined array");
        INS_CHECK(shape().ndim() >= 2,
            "transpose(): requires at least 2 dimensions, got ", shape().ndim());

        // Check contiguity
        if (!is_contiguous()) {
            INS_THROW("transpose(): non-contiguous array not yet supported. "
                "Call contiguous() first.");
        }

        int ndim = shape().ndim();

        // Swap last two dimensions in shape
        std::vector<int64_t> new_dims = shape().dims();
        std::swap(new_dims[ndim - 1], new_dims[ndim - 2]);
        Shape new_shape(new_dims);

        // Swap last two dimensions in strides
        std::vector<int64_t> new_strides_vec = impl_->strides.data();
        std::swap(new_strides_vec[ndim - 1], new_strides_vec[ndim - 2]);
        Strides new_strides(new_strides_vec);

        // Create view
        return Array(impl_, new_shape, new_strides, impl_->offset);
    }

    Array Array::transpose(const std::vector<int>& perm) const {
        INS_CHECK(defined(), "transpose: input is undefined");
        int ndim = shape().ndim();
        INS_CHECK(perm.size() == static_cast<size_t>(ndim),
            "transpose(): perm size must match number of dimensions");

        std::vector<int64_t> new_dims(ndim);
        for (int i = 0; i < ndim; ++i) {
            new_dims[i] = shape().dim(perm[i]);
        }
        Shape new_shape(new_dims);

        std::vector<int64_t> new_strides_vec(ndim);
        const std::vector<int64_t>& old_strides = impl_->strides.data();
        for (int i = 0; i < ndim; ++i) {
            new_strides_vec[i] = old_strides[perm[i]];
        }
        Strides new_strides(new_strides_vec);

        return Array(impl_, new_shape, new_strides, impl_->offset);
    }


    Array Array::squeeze(std::optional<int> axis) const {
        INS_CHECK(defined(), "Cannot squeeze undefined array");

        if (axis.has_value()) {
            int ax = axis.value();
            int nd = shape().ndim();
            if (ax < 0) ax += nd;
            INS_CHECK(ax >= 0 && ax < nd, "squeeze: axis out of range");
            INS_CHECK(shape().dim(ax) == 1, "squeeze: axis ", ax, " has size ", shape().dim(ax), ", cannot squeeze");

            std::vector<int64_t> new_dims;
            for (int i = 0; i < nd; ++i) {
                if (i != ax) {
                    new_dims.push_back(shape().dim(i));
                }
            }
            return reshape(Shape(new_dims));
        }
        else {
            // Remove all dimensions of size 1
            std::vector<int64_t> new_dims;
            for (int i = 0; i < shape().ndim(); ++i) {
                if (shape().dim(i) != 1) {
                    new_dims.push_back(shape().dim(i));
                }
            }
            if (new_dims.size() == static_cast<size_t>(shape().ndim())) {
                return *this;
            }
            return reshape(Shape(new_dims));
        }
    }

    Array Array::view(const Shape& new_shape) const {
        INS_CHECK(defined(), "view: array is undefined");
        INS_CHECK(new_shape.numel() == numel(),
            "view: shape.numel() mismatch. Expected ", numel(),
            ", got ", new_shape.numel());
        INS_CHECK(is_contiguous(), "view: only contiguous arrays can be viewed");

        Array result;
        result.impl_ = impl_;  // Share storage
        result.impl_->shape = new_shape;
        // strides will be recomputed from new_shape (contiguous)
        // offset remains the same
        result.impl_->update_strides_from_shape();
        result.impl_->is_view = true;
        result.impl_->validate();
        return result;
    }

    Array Array::view(DType new_dtype) const {
        INS_CHECK(defined(), "view: array is undefined");
        INS_CHECK(dtype_size(dtype()) == dtype_size(new_dtype),
            "view: dtype size mismatch. Original size=", dtype_size(dtype()),
            ", target size=", dtype_size(new_dtype));
        INS_CHECK(is_contiguous(), "view: only contiguous arrays can be viewed");

        Array result;
        result.impl_ = impl_;  // Share storage
        result.impl_->dtype = new_dtype;
        // shape and strides remain the same
        result.impl_->is_view = true;
        result.impl_->validate();
        return result;
    }

    Array Array::view(const Shape& new_shape, DType new_dtype) const {
        INS_CHECK(defined(), "view: array is undefined");

        size_t original_bytes = numel() * dtype_size(dtype());
        size_t new_bytes = new_shape.numel() * dtype_size(new_dtype);
        INS_CHECK(original_bytes == new_bytes,
            "view: total bytes mismatch. Original=", original_bytes,
            ", new=", new_bytes);
        INS_CHECK(is_contiguous(), "view: only contiguous arrays can be viewed");

        Array result;
        result.impl_ = std::make_shared<ArrayImpl>();
        result.impl_->storage = impl_->storage;
        result.impl_->shape = new_shape;
        result.impl_->dtype = new_dtype;
        result.impl_->strides = Strides(new_shape);
        result.impl_->offset = 0;
        result.impl_->is_view = true;
        result.impl_->place = impl_->place;
        result.impl_->validate();
        return result;
    }

    Array Array::unsqueeze(int dim) const {
        INS_CHECK(defined(), "Cannot unsqueeze undefined array");
        int nd = shape().ndim();
        if (dim < 0) dim += nd + 1;
        INS_CHECK(dim >= 0 && dim <= nd, "unsqueeze(): dim out of range, got ", dim);

        std::vector<int64_t> new_dims = shape().dims();
        new_dims.insert(new_dims.begin() + dim, 1);
        return reshape(Shape(new_dims));
    }

    // ========== Device/Type Conversion ==========

    Array Array::to(const Place& target) const {
        INS_CHECK(defined(), "Cannot convert undefined array");

        if (place() == target) {
            return *this;
        }

        // CPU to CPU
        if (place().is_cpu() && target.is_cpu()) {
            Array result(shape(), dtype(), target);
            std::memcpy(result.data(), data(), numel() * dtype_size(dtype()));
            return result;
        }

        // Get GPU device (for GPU-related transfers)
        auto* device = get_gpu_device();

        // CPU to GPU
        if (place().is_cpu() && target.is_gpu()) {
            Array result(shape(), dtype(), target);
            INS_CHECK(device != nullptr, "GPU not available, got device = ", device);
            device->memory_copy_h2d(target.device_id(), result.data(), data(),
                numel() * dtype_size(dtype()));
            return result;
        }

        // GPU to CPU
        if (place().is_gpu() && target.is_cpu()) {
            Array result(shape(), dtype(), target);
            INS_CHECK(device != nullptr, "GPU not available, got device = ", device);
            device->memory_copy_d2h(place().device_id(), result.data(), data(),
                numel() * dtype_size(dtype()));
            return result;
        }

        // GPU to GPU (different device)
        if (place().is_gpu() && target.is_gpu()) {
            Array result(shape(), dtype(), target);
            INS_CHECK(device != nullptr, "GPU not available, got device = ", device);
            device->memory_copy_p2p(target.device_id(), place().device_id(),
                result.data(), data(), numel() * dtype_size(dtype()));
            return result;
        }

        INS_THROW("Unsupported device conversion: from ", place(), " to ", target);
    }

    Array Array::to(DType target_dtype) const {
        INS_CHECK(defined(), "Array::to: array is not initialized"); 

        if (dtype() == target_dtype) {
            return *this; 
        }

        // Call cast operator
        OpArgs args = { *this, target_dtype }; 
        DeviceKind dev = place().is_cpu() ? DeviceKind::CPU : DeviceKind::GPU; 
        auto res = ops()["cast"][dev][dtype()](args)[0]; 
        auto result = std::any_cast<Array>(res); 

        if (!result.is_contiguous()) {
            result = result.contiguous(); 
        }

        return result; 
    }

    Array Array::to(const Place& target, DType target_dtype) const {
        return to(target).to(target_dtype);
    }

    Array Array::to(const Array& other) const {
        return to(other.place(), other.dtype());
    }

    // ========== Copy ==========

    Array Array::copy() const {
        INS_CHECK(defined(), "Cannot copy undefined array");

        Array result(shape(), dtype(), place());
        size_t bytes = numel() * dtype_size(dtype());

        if (place().is_cpu()) {
            std::memcpy(result.data(), data(), bytes);
        }
        else {
            auto* device = get_gpu_device();
            device->memory_copy_d2d(place().device_id(), result.data(), data(), bytes);
        }

        return result;
    }

    // ========== Advanced Accessors ==========

    const Strides Array::strides() const { return impl_->strides; }
    int64_t Array::offset() const { return impl_->offset; }

    // ========== Boolean conversion ==========

    Array::operator bool() const {
        if(!defined())
			return false;
        INS_CHECK(numel() == 1,
            "The truth value of an array with more than one element is ambiguous. "
            "Use .any() or .all() for boolean reduction.");

        switch (dtype()) {
        case DType::BOOL:
            return item<bool>();
        case DType::F32:
            return item<float>() != 0.0f;
        case DType::F64:
            return item<double>() != 0.0;
        case DType::I32:
            return item<int32_t>() != 0;
        case DType::I64:
            return item<int64_t>() != 0;
        case DType::U8:
            return item<uint8_t>() != 0;
        case DType::I8:
            return item<int8_t>() != 0;
        case DType::I16:
            return item<int16_t>() != 0;
        case DType::U16:
            return item<uint16_t>() != 0;
        case DType::U32:
            return item<uint32_t>() != 0;
        case DType::U64:
            return item<uint64_t>() != 0;
        default:
            INS_THROW("operator bool(): unsupported dtype ", dtype_name(dtype()));
        }
    }

    bool Array::any() const {
        Array result = ins::any(*this, std::nullopt, false);
        if (result.numel() != 1) {
            INS_THROW("any(): internal error - reduction should return scalar");
        }
        return result.item<bool>();
    }

    bool Array::all() const {
        Array result = ins::all(*this, std::nullopt, false);
        if (result.numel() != 1) {
            INS_THROW("all(): internal error - reduction should return scalar");
        }
        return result.item<bool>();
    }

} // namespace ins