// src/init.cpp
#include "insight/init.h"
#include "insight/plugin/device_ext.h"
#include "backends/cpu/init.h"

namespace ins {

    static bool g_initialized = false;

    void init() {
        if (g_initialized) return;

        // Initialize all CPU backend kernels
        cpu::init_cpu();

        // GPU backend
        ins::gpu::DeviceFactory* gpu_factory = insight_create_device_factory();
        if (gpu_factory->is_available())
        {
            gpu::init_gpu();
        }
        else
        {
            // Clean up if not available
			insight_destroy_device_factory(gpu_factory);
        }

        g_initialized = true;
    }

    bool is_initialized() {
        return g_initialized;
    }

} // namespace ins