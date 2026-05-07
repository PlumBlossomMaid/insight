// ============================================================================
// backends/cuda/device/cuda_event.h
// ============================================================================
#pragma once

#include "insight/plugin/device_ext.h"
#include <cuda_runtime.h>

namespace ins {
    namespace gpu {

        class CUDAEvent : public Event {
        public:
            CUDAEvent();
            ~CUDAEvent();

            cudaEvent_t get() const { return event_; }

        private:
            cudaEvent_t event_;
        };

    } // namespace gpu
} // namespace ins