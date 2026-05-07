// ============================================================================
// backends/cuda/device/cuda_profiler.h
// ============================================================================
#pragma once

#include "insight/plugin/device_ext.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace ins {
    namespace gpu {

        class CUDAProfiler : public Profiler {
        public:
            CUDAProfiler(int device_id, const std::string& name);
            ~CUDAProfiler();

            void start() override;
            void stop() override;
            void reset() override;
            void begin_event(const char* name) override;
            void end_event() override;
            void collect_trace_data(uint64_t start_ns) override;

        private:
            struct EventRecord {
                std::string name;
                cudaEvent_t start;
                cudaEvent_t end;
                int depth;
            };

            int device_id_;
            std::string name_;
            std::vector<EventRecord> events_;
            int current_depth_;
            bool is_recording_;
        };

    } // namespace gpu
} // namespace ins