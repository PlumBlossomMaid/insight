// ============================================================================
// backends/cuda/device/cuda_device.h
// ============================================================================
#pragma once

#include "insight/plugin/device_ext.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <string>
#include <array>

namespace ins {
    namespace gpu {

        class CUDAStream;
        class CUDAEvent;
        class CUDAProfiler;

        class CUDADevice : public virtual StreamSupport,
            public virtual AsyncCopySupport,
            public virtual DeviceInfoSupport,
            public virtual ProfilerSupport {
        public:
            CUDADevice();
            ~CUDADevice();

            // Device Lifecycle
            void initialize() override;
            void finalize() override;
            void init_device(int device_id) override;
            void deinit_device(int device_id) override;

            // Device Management
            void set_device(int device_id) override;
            int get_device() const override;
            void synchronize_device(int device_id) override;
            size_t get_device_count() const override;
            void get_device_list(size_t* devices) const override;

            // Memory Management
            void device_memory_allocate(int device_id, void** ptr, size_t size) override;
            void device_memory_deallocate(int device_id, void* ptr, size_t size) override;
            void host_memory_allocate(int device_id, void** ptr, size_t size) override;
            void host_memory_deallocate(int device_id, void* ptr, size_t size) override;
            void memory_copy_h2d(int device_id, void* dst, const void* src, size_t size) override;
            void memory_copy_d2h(int device_id, void* dst, const void* src, size_t size) override;
            void memory_copy_d2d(int device_id, void* dst, const void* src, size_t size) override;
            void memory_copy_p2p(int dst_device, int src_device,
                void* dst, const void* src, size_t size) override;
            void device_memory_set(int device_id, void* ptr, unsigned char value, size_t size) override;
            void device_memory_stats(int device_id, size_t* total, size_t* free) override;

            // Async Memory Operations
            void async_memory_copy_h2d(int device_id, Stream* stream,
                void* dst, const void* src, size_t size) override;
            void async_memory_copy_d2h(int device_id, Stream* stream,
                void* dst, const void* src, size_t size) override;
            void async_memory_copy_d2d(int device_id, Stream* stream,
                void* dst, const void* src, size_t size) override;
            void async_memory_copy_p2p(int dst_device, int src_device, Stream* stream,
                void* dst, const void* src, size_t size) override;

            // Stream Management
            Stream* create_stream(int device_id) override;
            void destroy_stream(int device_id, Stream* stream) override;
            bool query_stream(int device_id, Stream* stream) override;
            void synchronize_stream(int device_id, Stream* stream) override;
            void stream_add_callback(int device_id, Stream* stream,
                void (*callback)(int, Stream*, void*, int*),
                void* user_data) override;
            void stream_wait_event(int device_id, Stream* stream, Event* event) override;

            // Events
            Event* create_event(int device_id) override;
            void destroy_event(int device_id, Event* event) override;
            void record_event(int device_id, Stream* stream, Event* event) override;
            bool query_event(int device_id, Event* event) override;
            void synchronize_event(int device_id, Event* event) override;
            float elapsed_time(Event* start, Event* end) override;

            // Device Information
            size_t get_compute_capability(int device_id) const override;
            size_t get_runtime_version(int device_id) const override;
            size_t get_driver_version(int device_id) const override;

            // Extended Device Information
            size_t get_multi_process(int device_id) const override;
            size_t get_max_threads_per_mp(int device_id) const override;
            size_t get_max_threads_per_block(int device_id) const override;
            std::array<size_t, 3> get_max_grid_dim_size(int device_id) const override;

            // Profiler
            Profiler* create_profiler(int device_id, const std::string& name) override;
            void destroy_profiler(Profiler* profiler) override;

        private:
            cudaDeviceProp get_device_properties(int device_id) const;
            int current_device_;
            std::vector<int> initialized_devices_;
        };

    } // namespace gpu
} // namespace ins