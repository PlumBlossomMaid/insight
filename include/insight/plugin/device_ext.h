// ============================================================================
// include/insight/plugin/device_ext.h
// ============================================================================
/**
 * @file device_ext.h
 * @brief Insight GPU Device Interface (C++ ABI)
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <array>

#include "insight/core/dtype.h"
#include "insight/core/exception.h"

namespace ins {
    namespace gpu {

        class Stream;
        class Event;
        class Profiler;

        // ============================================================================
        // Base Device Interface (Required)
        // ============================================================================

        class Device {
        public:
            virtual ~Device() = default;

            // Device Lifecycle (Optional)
            virtual void initialize() {}
            virtual void finalize() {}
            virtual void init_device(int /*device_id*/) {}
            virtual void deinit_device(int /*device_id*/) {}

            // Device Management (Required)
            virtual void set_device(int device_id) = 0;
            virtual int get_device() const = 0;
            virtual void synchronize_device(int device_id) = 0;
            virtual size_t get_device_count() const = 0;
            virtual void get_device_list(size_t* devices) const = 0;

            // Memory Management (Required)
            virtual void device_memory_allocate(int device_id, void** ptr, size_t size) = 0;
            virtual void device_memory_deallocate(int device_id, void* ptr, size_t size) = 0;
            virtual void memory_copy_h2d(int device_id, void* dst, const void* src, size_t size) = 0;
            virtual void memory_copy_d2h(int device_id, void* dst, const void* src, size_t size) = 0;
            virtual void memory_copy_d2d(int device_id, void* dst, const void* src, size_t size) = 0;

            // Optional Memory Operations (with default implementation)
            virtual void host_memory_allocate(int device_id, void** ptr, size_t size) {
                *ptr = std::malloc(size);
            }
            virtual void host_memory_deallocate(int device_id, void* ptr, size_t size) {
                std::free(ptr);
            }
            virtual void memory_copy_p2p(int dst_device, int src_device,
                void* dst, const void* src, size_t size) {
                void* tmp = std::malloc(size);
                memory_copy_d2h(src_device, tmp, src, size);
                memory_copy_h2d(dst_device, dst, tmp, size);
                std::free(tmp);
            }
            virtual void device_memory_set(int device_id, void* ptr, unsigned char value, size_t size) {
                unsigned char* p = static_cast<unsigned char*>(ptr);
                for (size_t i = 0; i < size; ++i) p[i] = value;
            }
            virtual void device_memory_stats(int device_id, size_t* total, size_t* free) {
                *total = 0;
                *free = 0;
            }

            // Async Memory Operations (Optional)
            virtual void async_memory_copy_h2d(int /*device_id*/, Stream* /*stream*/,
                void* dst, const void* src, size_t size) {
                memory_copy_h2d(0, dst, src, size);
            }
            virtual void async_memory_copy_d2h(int /*device_id*/, Stream* /*stream*/,
                void* dst, const void* src, size_t size) {
                memory_copy_d2h(0, dst, src, size);
            }
            virtual void async_memory_copy_d2d(int /*device_id*/, Stream* /*stream*/,
                void* dst, const void* src, size_t size) {
                memory_copy_d2d(0, dst, src, size);
            }
            virtual void async_memory_copy_p2p(int dst_device, int src_device, Stream* /*stream*/,
                void* dst, const void* src, size_t size) {
                memory_copy_p2p(dst_device, src_device, dst, src, size);
            }

            // Events (Required)
            virtual Event* create_event(int device_id) = 0;
            virtual void destroy_event(int device_id, Event* event) = 0;
            virtual void record_event(int device_id, Stream* stream, Event* event) = 0;
            virtual bool query_event(int device_id, Event* event) = 0;
            virtual void synchronize_event(int device_id, Event* event) = 0;
            virtual float elapsed_time(Event* start, Event* end) = 0;

            // Device Information (Required)
            virtual size_t get_compute_capability(int device_id) const = 0;
            virtual size_t get_runtime_version(int device_id) const = 0;
            virtual size_t get_driver_version(int device_id) const = 0;
        };

        // ============================================================================
        // Optional Feature Interfaces (must use virtual inheritance)
        // ============================================================================

        class StreamSupport : public virtual Device {
        public:
            virtual Stream* create_stream(int device_id) = 0;
            virtual void destroy_stream(int device_id, Stream* stream) = 0;
            virtual bool query_stream(int device_id, Stream* stream) = 0;
            virtual void synchronize_stream(int device_id, Stream* stream) = 0;
            virtual void stream_add_callback(int device_id, Stream* stream,
                void (*callback)(int, Stream*, void*, int*),
                void* user_data) = 0;
            virtual void stream_wait_event(int device_id, Stream* stream, Event* event) = 0;
        };

        class AsyncCopySupport : public virtual Device {
        public:
            virtual void async_memory_copy_h2d(int device_id, Stream* stream,
                void* dst, const void* src, size_t size) = 0;
            virtual void async_memory_copy_d2h(int device_id, Stream* stream,
                void* dst, const void* src, size_t size) = 0;
            virtual void async_memory_copy_d2d(int device_id, Stream* stream,
                void* dst, const void* src, size_t size) = 0;
            virtual void async_memory_copy_p2p(int dst_device, int src_device, Stream* stream,
                void* dst, const void* src, size_t size) = 0;
        };

        class DeviceInfoSupport : public virtual Device {
        public:
            virtual size_t get_multi_process(int device_id) const = 0;
            virtual size_t get_max_threads_per_mp(int device_id) const = 0;
            virtual size_t get_max_threads_per_block(int device_id) const = 0;
            virtual std::array<size_t, 3> get_max_grid_dim_size(int device_id) const = 0;
        };

        class ProfilerSupport : public virtual Device {
        public:
            virtual Profiler* create_profiler(int device_id, const std::string& name) = 0;
            virtual void destroy_profiler(Profiler* profiler) = 0;
        };

        // ============================================================================
        // Base Implementations
        // ============================================================================

        class Stream {
        public:
            virtual ~Stream() = default;
        };

        class Event {
        public:
            virtual ~Event() = default;
        };

        class Profiler {
        public:
            virtual ~Profiler() = default;
            virtual void start() = 0;
            virtual void stop() = 0;
            virtual void reset() = 0;
            virtual void begin_event(const char* name) = 0;
            virtual void end_event() = 0;
            virtual void collect_trace_data(uint64_t start_ns) = 0;
        };

        // ============================================================================
        // Device Factory
        // ============================================================================

        class DeviceFactory {
        public:
            virtual ~DeviceFactory() = default;
            virtual const char* device_type() const = 0;
            virtual const char* sub_device_type() const = 0;
            virtual std::unique_ptr<Device> create_device() = 0;
            virtual bool is_available() const = 0;
        };

        extern "C" ins::gpu::DeviceFactory* init_gpu();

    } // namespace gpu
} // namespace ins

extern "C" ins::gpu::DeviceFactory* insight_create_device_factory();

extern "C" void insight_destroy_device_factory(ins::gpu::DeviceFactory* factory);