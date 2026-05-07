// ============================================================================
// backends/cuda/device/cuda_stream.h
// ============================================================================
#pragma once

#include "insight/plugin/device_ext.h"
#include <cuda_runtime.h>

namespace ins {
    namespace gpu {

        class CUDAStream : public Stream {
        public:
            explicit CUDAStream(cudaStream_t stream);
            ~CUDAStream();

            cudaStream_t get() const { return stream_; }

        private:
            cudaStream_t stream_;
        };

    } // namespace gpu
} // namespace ins