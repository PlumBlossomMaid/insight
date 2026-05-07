// backends/cuda/gpu_init.cpp
#include "insight/plugin/device_ext.h"
#include "insight/plugin/op_registry.h"
#include "device/cuda_device.h"
#include "device/cuda_factory.h"
#include <cuda_runtime.h>

extern "C" {
    USE_MODULE(cast, GPU);
    USE_MODULE(elementwise, GPU);
    USE_MODULE(manipulation, GPU);
    USE_MODULE(unary, GPU);
    USE_MODULE(random, GPU);
    USE_MODULE(creation, GPU);
    USE_MODULE(reduction, GPU);
    USE_MODULE(indexing, GPU);
    USE_MODULE(fft, GPU);
    USE_MODULE(linalg, GPU);
}


namespace ins::gpu {

    // ============================================================================
    // Initialization Entry Point
    // ============================================================================

    extern "C" DeviceFactory* init_gpu() {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            return nullptr;  // No CUDA devices
        }

        return insight_create_device_factory();
    }

} // namespace ins::gpu