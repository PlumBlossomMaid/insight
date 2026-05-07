// ============================================================================
// backends/cuda/device/cuda_factory.h
// ============================================================================
#pragma once

#include "insight/plugin/device_ext.h"
#include <memory>

namespace ins {
    namespace gpu {

        class CUDADeviceFactory : public DeviceFactory {
        public:
            const char* device_type() const override;
            const char* sub_device_type() const override;
            std::unique_ptr<Device> create_device() override;
            bool is_available() const override;
        };

    } // namespace gpu
} // namespace ins