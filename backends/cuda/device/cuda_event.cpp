// ============================================================================
// backends/cuda/device/cuda_event.cpp
// ============================================================================
#include "cuda_event.h"

namespace ins {
    namespace gpu {

        CUDAEvent::CUDAEvent() {
            cudaEventCreate(&event_);
        }

        CUDAEvent::~CUDAEvent() {
            cudaEventDestroy(event_);
        }

    } // namespace gpu
} // namespace ins