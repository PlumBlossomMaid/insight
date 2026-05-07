// ============================================================================
// backends/cuda/device/cuda_stream.cpp
// ============================================================================
#include "cuda_stream.h"

namespace ins {
	namespace gpu {

		CUDAStream::CUDAStream(cudaStream_t stream) : stream_(stream) {}

		CUDAStream::~CUDAStream() {}

	} // namespace gpu
} // namespace ins