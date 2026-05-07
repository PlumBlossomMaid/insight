// ============================================================================
// backends/cuda/device/cuda_factory.cpp
// ============================================================================
#include "cuda_factory.h"
#include "cuda_device.h"
#include <cuda_runtime.h>

namespace ins {
    namespace gpu {

        const char* CUDADeviceFactory::device_type() const {
            return "CUDA";
        }

        const char* CUDADeviceFactory::sub_device_type() const {
            return "V1.0";
        }

        std::unique_ptr<Device> CUDADeviceFactory::create_device() {
            return std::make_unique<CUDADevice>();
        }

        bool CUDADeviceFactory::is_available() const {
            int count;
            cudaError_t err = cudaGetDeviceCount(&count);
            return (err == cudaSuccess && count > 0);
        }

    } // namespace gpu
} // namespace ins

// ============================================================================
// Plugin Entry Point (for dynamic loading)
// ============================================================================
// For static linking, this function can be omitted and registration done
// via static initialization in DeviceManager.
// For dynamic loading, uncomment the following:

static ins::gpu::DeviceFactory* g_gpu_factory = nullptr;

extern "C" ins::gpu::DeviceFactory* insight_create_device_factory() {
	if (g_gpu_factory == nullptr) {
        g_gpu_factory = new ins::gpu::CUDADeviceFactory();
    }
    return g_gpu_factory;
}

extern "C" void insight_destroy_device_factory(ins::gpu::DeviceFactory* factory) {
    delete factory;
}
