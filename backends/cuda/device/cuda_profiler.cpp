// ============================================================================
// backends/cuda/device/cuda_profiler.cpp
// ============================================================================
#include "cuda_profiler.h"

namespace ins {
    namespace gpu {

        CUDAProfiler::CUDAProfiler(int device_id, const std::string& name)
            : device_id_(device_id), name_(name), current_depth_(0), is_recording_(false) {
            cudaSetDevice(device_id_);
        }

        CUDAProfiler::~CUDAProfiler() {
            reset();
        }

        void CUDAProfiler::start() {
            if (is_recording_) return;
            is_recording_ = true;
            reset();
        }

        void CUDAProfiler::stop() {
            if (!is_recording_) return;
            is_recording_ = false;
        }

        void CUDAProfiler::reset() {
            for (auto& event : events_) {
                if (event.start) cudaEventDestroy(event.start);
                if (event.end) cudaEventDestroy(event.end);
            }
            events_.clear();
            current_depth_ = 0;
        }

        void CUDAProfiler::begin_event(const char* name) {
            if (!is_recording_) return;
            EventRecord record;
            record.name = name;
            record.depth = current_depth_++;
            cudaEventCreate(&record.start);
            cudaEventCreate(&record.end);
            cudaEventRecord(record.start, nullptr);
            events_.push_back(std::move(record));
        }

        void CUDAProfiler::end_event() {
            if (!is_recording_) return;
            if (events_.empty()) return;
            auto& last = events_.back();
            cudaEventRecord(last.end, nullptr);
            current_depth_--;
        }

        void CUDAProfiler::collect_trace_data(uint64_t start_ns) {
            for (auto& event : events_) {
                cudaEventSynchronize(event.end);
            }
            // Framework will handle export
        }

    } // namespace gpu
} // namespace ins