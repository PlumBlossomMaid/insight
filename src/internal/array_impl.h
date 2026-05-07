// src/core/array_impl.h
#pragma once
#include "insight/core/shape.h"
#include "insight/core/dtype.h"
#include "insight/core/place.h"
#include "insight/core/strides.h"
#include "insight/plugin/device_ext.h"

// Forward declarations for GPU functions
extern "C" ins::gpu::DeviceFactory* insight_create_device_factory();

namespace ins
{
    class ArrayImpl {
    public:
        Shape shape;
        Strides strides;
        DType dtype;
        Place place;
        std::shared_ptr<void> storage;
        int64_t offset = 0;  // offset in elements
        bool is_view = false;
        size_t nbytes() const { return shape.numel() * dtype_size(dtype); }

        void allocate_storage() {
            size_t bytes = nbytes();
            if (bytes == 0) return;

            if (place.is_cpu()) {
                void* ptr = std::malloc(bytes);
                INS_CHECK(ptr != nullptr, "Failed to allocate ", bytes, " bytes on CPU");
                storage = std::shared_ptr<void>(ptr, std::free);
            }
            else {
                // GPU: lazy initialization of device
                static std::unique_ptr<gpu::Device> gpu_device = []() -> std::unique_ptr<gpu::Device> {
                    auto* factory = insight_create_device_factory();
                    if (factory && factory->is_available()) {
                        return factory->create_device();
                    }
                    return nullptr;
                    }();

                INS_CHECK(gpu_device != nullptr, "GPU not available for allocation");

                void* ptr;
                gpu_device->device_memory_allocate(place.device_id(), &ptr, bytes);
                INS_CHECK(ptr != nullptr, "Failed to allocate ", bytes, " bytes on GPU ", place.device_id());

                // Capture pointer to the unique_ptr's content, not the unique_ptr itself
                gpu::Device* device_ptr = gpu_device.get();
                int device_id = place.device_id();
                storage = std::shared_ptr<void>(ptr, [device_ptr, device_id](void* p) {
                    if (p && device_ptr) {
                        device_ptr->device_memory_deallocate(device_id, p, 0);
                    }
                    });
            }
        }

        void update_strides_from_shape() {
            strides = Strides(shape);
            offset = 0;
        }

        void validate() const {
            if (is_view) {
                INS_CHECK(storage != nullptr, "View has no storage");
                INS_CHECK(offset >= 0, "Negative offset");
            }
            else {
                INS_CHECK(offset == 0, "Non-zero offset on non-view array");
            }
        }
    };
}