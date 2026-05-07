// ============================================================================
// backends/cuda/device/cuda_device.cpp
// ============================================================================
#include "cuda_device.h"
#include "cuda_stream.h"
#include "cuda_event.h"
#include "cuda_profiler.h"
#include "insight/core/exception.h"
#include <algorithm>
#include <cstring>

namespace ins {
    namespace gpu {

        CUDADevice::CUDADevice() : current_device_(-1) {}

        CUDADevice::~CUDADevice() { finalize(); }

        void CUDADevice::initialize() {}

        void CUDADevice::finalize() {
            initialized_devices_.clear();
            current_device_ = -1;
        }

        void CUDADevice::init_device(int device_id) {
            cudaError_t err = cudaSetDevice(device_id);
            if (err != cudaSuccess) {
                INS_THROW("Failed to set CUDA device for initialization: ", cudaGetErrorString(err));
            }
            initialized_devices_.push_back(device_id);
        }

        void CUDADevice::deinit_device(int device_id) {
            auto it = std::find(initialized_devices_.begin(), initialized_devices_.end(), device_id);
            if (it != initialized_devices_.end()) {
                initialized_devices_.erase(it);
            }
            if (current_device_ == device_id) {
                current_device_ = -1;
            }
        }

        void CUDADevice::set_device(int device_id) {
            if (current_device_ == device_id) return;
            cudaError_t err = cudaSetDevice(device_id);
            if (err != cudaSuccess) {
                INS_THROW("Failed to set CUDA device ", device_id, ": ", cudaGetErrorString(err));
            }
            current_device_ = device_id;
        }

        int CUDADevice::get_device() const {
            int device;
            cudaError_t err = cudaGetDevice(&device);
            if (err != cudaSuccess) {
                INS_THROW("Failed to get current CUDA device: ", cudaGetErrorString(err));
            }
            return device;
        }

        void CUDADevice::synchronize_device(int device_id) {
            int original_device = get_device();
            set_device(device_id);
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                INS_THROW("Failed to synchronize CUDA device ", device_id, ": ", cudaGetErrorString(err));
            }
            set_device(original_device);
        }

        size_t CUDADevice::get_device_count() const {
            int count;
            cudaError_t err = cudaGetDeviceCount(&count);
            if (err != cudaSuccess) {
                return 0;
            }
            return static_cast<size_t>(count);
        }

        void CUDADevice::get_device_list(size_t* devices) const {
            size_t count = get_device_count();
            for (size_t i = 0; i < count; ++i) {
                devices[i] = i;
            }
        }

        void CUDADevice::device_memory_allocate(int device_id, void** ptr, size_t size) {
            int original_device = get_device();
            set_device(device_id);
            cudaError_t err = cudaMalloc(ptr, size);
            if (err != cudaSuccess) {
                INS_THROW("cudaMalloc failed for device ", device_id,
                    " (size=", size, "): ", cudaGetErrorString(err));
            }
            set_device(original_device);
        }

        void CUDADevice::device_memory_deallocate(int device_id, void* ptr, size_t size) {
            (void)size;
            int original_device = get_device();
            set_device(device_id);
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess) {
                INS_THROW("cudaFree failed for device ", device_id, ": ", cudaGetErrorString(err));
            }
            set_device(original_device);
        }

        void CUDADevice::host_memory_allocate(int device_id, void** ptr, size_t size) {
            int original_device = get_device();
            set_device(device_id);
            cudaError_t err = cudaHostAlloc(ptr, size, cudaHostAllocDefault);
            if (err != cudaSuccess) {
                INS_THROW("cudaHostAlloc failed for device ", device_id,
                    " (size=", size, "): ", cudaGetErrorString(err));
            }
            set_device(original_device);
        }

        void CUDADevice::host_memory_deallocate(int device_id, void* ptr, size_t size) {
            (void)size;
            int original_device = get_device();
            set_device(device_id);
            cudaError_t err = cudaFreeHost(ptr);
            if (err != cudaSuccess) {
                INS_THROW("cudaFreeHost failed for device ", device_id, ": ", cudaGetErrorString(err));
            }
            set_device(original_device);
        }

        void CUDADevice::memory_copy_h2d(int device_id, void* dst, const void* src, size_t size) {
            int original_device = get_device();
            set_device(device_id);
            cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                INS_THROW("cudaMemcpy H2D failed for device ", device_id,
                    " (size=", size, "): ", cudaGetErrorString(err));
            }
            set_device(original_device);
        }

        void CUDADevice::memory_copy_d2h(int device_id, void* dst, const void* src, size_t size) {
            int original_device = get_device();
            set_device(device_id);
            cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                INS_THROW("cudaMemcpy D2H failed for device ", device_id,
                    " (size=", size, "): ", cudaGetErrorString(err));
            }
            set_device(original_device);
        }

        void CUDADevice::memory_copy_d2d(int device_id, void* dst, const void* src, size_t size) {
            int original_device = get_device();
            set_device(device_id);
            cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                INS_THROW("cudaMemcpy D2D failed for device ", device_id,
                    " (size=", size, "): ", cudaGetErrorString(err));
            }
            set_device(original_device);
        }

        void CUDADevice::memory_copy_p2p(int dst_device, int src_device,
            void* dst, const void* src, size_t size) {
            int original_device = get_device();
            int can_access = 0;
            cudaError_t err = cudaDeviceCanAccessPeer(&can_access, dst_device, src_device);
            if (err != cudaSuccess) {
                INS_THROW("Failed to check peer access capability between device ",
                    src_device, " and ", dst_device, ": ", cudaGetErrorString(err));
            }
            if (can_access) {
                set_device(dst_device);
                err = cudaDeviceEnablePeerAccess(src_device, 0);
                if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                    INS_THROW("Failed to enable peer access from device ",
                        dst_device, " to ", src_device, ": ", cudaGetErrorString(err));
                }
                err = cudaMemcpyPeer(dst, dst_device, src, src_device, size);
                if (err != cudaSuccess) {
                    INS_THROW("cudaMemcpyPeer failed from device ", src_device,
                        " to ", dst_device, " (size=", size, "): ", cudaGetErrorString(err));
                }
            }
            else {
                void* tmp = std::malloc(size);
                memory_copy_d2h(src_device, tmp, src, size);
                memory_copy_h2d(dst_device, dst, tmp, size);
                std::free(tmp);
            }
            set_device(original_device);
        }

        void CUDADevice::device_memory_set(int device_id, void* ptr, unsigned char value, size_t size) {
            int original_device = get_device();
            set_device(device_id);
            cudaError_t err = cudaMemset(ptr, value, size);
            if (err != cudaSuccess) {
                INS_THROW("cudaMemset failed for device ", device_id,
                    " (size=", size, "): ", cudaGetErrorString(err));
            }
            set_device(original_device);
        }

        void CUDADevice::device_memory_stats(int device_id, size_t* total, size_t* free) {
            int original_device = get_device();
            set_device(device_id);
            size_t free_bytes, total_bytes;
            cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
            if (err != cudaSuccess) {
                INS_THROW("cudaMemGetInfo failed for device ", device_id, ": ", cudaGetErrorString(err));
            }
            *total = total_bytes;
            *free = free_bytes;
            set_device(original_device);
        }

        void CUDADevice::async_memory_copy_h2d(int device_id, Stream* stream,
            void* dst, const void* src, size_t size) {
            int original_device = get_device();
            set_device(device_id);
            cudaStream_t cuda_stream = (stream) ? static_cast<CUDAStream*>(stream)->get() : nullptr;
            cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, cuda_stream);
            if (err != cudaSuccess) {
                INS_THROW("cudaMemcpyAsync H2D failed for device ", device_id,
                    " (size=", size, "): ", cudaGetErrorString(err));
            }
            set_device(original_device);
        }

        void CUDADevice::async_memory_copy_d2h(int device_id, Stream* stream,
            void* dst, const void* src, size_t size) {
            int original_device = get_device();
            set_device(device_id);
            cudaStream_t cuda_stream = (stream) ? static_cast<CUDAStream*>(stream)->get() : nullptr;
            cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, cuda_stream);
            if (err != cudaSuccess) {
                INS_THROW("cudaMemcpyAsync D2H failed for device ", device_id,
                    " (size=", size, "): ", cudaGetErrorString(err));
            }
            set_device(original_device);
        }

        void CUDADevice::async_memory_copy_d2d(int device_id, Stream* stream,
            void* dst, const void* src, size_t size) {
            int original_device = get_device();
            set_device(device_id);
            cudaStream_t cuda_stream = (stream) ? static_cast<CUDAStream*>(stream)->get() : nullptr;
            cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, cuda_stream);
            if (err != cudaSuccess) {
                INS_THROW("cudaMemcpyAsync D2D failed for device ", device_id,
                    " (size=", size, "): ", cudaGetErrorString(err));
            }
            set_device(original_device);
        }

        void CUDADevice::async_memory_copy_p2p(int dst_device, int src_device, Stream* stream,
            void* dst, const void* src, size_t size) {
            int original_device = get_device();
            int can_access = 0;
            cudaError_t err = cudaDeviceCanAccessPeer(&can_access, dst_device, src_device);
            if (err != cudaSuccess) {
                INS_THROW("Failed to check peer access capability between device ",
                    src_device, " and ", dst_device, ": ", cudaGetErrorString(err));
            }
            if (can_access) {
                set_device(dst_device);
                err = cudaDeviceEnablePeerAccess(src_device, 0);
                if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                    INS_THROW("Failed to enable peer access from device ",
                        dst_device, " to ", src_device, ": ", cudaGetErrorString(err));
                }
                cudaStream_t cuda_stream = (stream) ? static_cast<CUDAStream*>(stream)->get() : nullptr;
                err = cudaMemcpyPeerAsync(dst, dst_device, src, src_device, size, cuda_stream);
                if (err != cudaSuccess) {
                    INS_THROW("cudaMemcpyPeerAsync failed from device ", src_device,
                        " to ", dst_device, " (size=", size, "): ", cudaGetErrorString(err));
                }
            }
            else {
                void* tmp = std::malloc(size);
                async_memory_copy_d2h(src_device, stream, tmp, src, size);
                if (stream) {
                    synchronize_stream(src_device, stream);
                }
                else {
                    synchronize_device(src_device);
                }
                async_memory_copy_h2d(dst_device, stream, dst, tmp, size);
                std::free(tmp);
            }
            set_device(original_device);
        }

        Stream* CUDADevice::create_stream(int device_id) {
            int original_device = get_device();
            set_device(device_id);
            cudaStream_t stream;
            cudaError_t err = cudaStreamCreate(&stream);
            if (err != cudaSuccess) {
                INS_THROW("cudaStreamCreate failed for device ", device_id, ": ", cudaGetErrorString(err));
            }
            set_device(original_device);
            return new CUDAStream(stream);
        }

        void CUDADevice::destroy_stream(int device_id, Stream* stream) {
            if (!stream) return;
            int original_device = get_device();
            set_device(device_id);
            cudaStream_t cuda_stream = static_cast<CUDAStream*>(stream)->get();
            cudaError_t err = cudaStreamDestroy(cuda_stream);
            if (err != cudaSuccess) {
                INS_THROW("cudaStreamDestroy failed for device ", device_id, ": ", cudaGetErrorString(err));
            }
            delete stream;
            set_device(original_device);
        }

        bool CUDADevice::query_stream(int device_id, Stream* stream) {
            if (!stream) return true;
            int original_device = get_device();
            set_device(device_id);
            cudaStream_t cuda_stream = static_cast<CUDAStream*>(stream)->get();
            cudaError_t err = cudaStreamQuery(cuda_stream);
            set_device(original_device);
            return (err == cudaSuccess);
        }

        void CUDADevice::synchronize_stream(int device_id, Stream* stream) {
            if (!stream) {
                synchronize_device(device_id);
                return;
            }
            int original_device = get_device();
            set_device(device_id);
            cudaStream_t cuda_stream = static_cast<CUDAStream*>(stream)->get();
            cudaError_t err = cudaStreamSynchronize(cuda_stream);
            if (err != cudaSuccess) {
                INS_THROW("cudaStreamSynchronize failed for device ", device_id,
                    ": ", cudaGetErrorString(err));
            }
            set_device(original_device);
        }

        void CUDADevice::stream_add_callback(int device_id, Stream* stream,
            void (*callback)(int, Stream*, void*, int*),
            void* user_data) {
            int original_device = get_device();
            set_device(device_id);
            cudaStream_t cuda_stream = (stream) ? static_cast<CUDAStream*>(stream)->get() : nullptr;
            struct CallbackData {
                int device_id;
                Stream* stream;
                void* user_data;
                void (*callback)(int, Stream*, void*, int*);
            };
            auto* data = new CallbackData{ device_id, stream, user_data, callback };
            cudaError_t err = cudaStreamAddCallback(
                cuda_stream,
                [](cudaStream_t, cudaError_t, void* data_ptr) {
                    auto* cb_data = static_cast<CallbackData*>(data_ptr);
                    int status = 0;
                    cb_data->callback(cb_data->device_id, cb_data->stream,
                        cb_data->user_data, &status);
                    delete cb_data;
                },
                data, 0);
            if (err != cudaSuccess) {
                delete data;
                INS_THROW("cudaStreamAddCallback failed for device ", device_id,
                    ": ", cudaGetErrorString(err));
            }
            set_device(original_device);
        }

        void CUDADevice::stream_wait_event(int device_id, Stream* stream, Event* event) {
            if (!event) return;
            int original_device = get_device();
            set_device(device_id);
            cudaStream_t cuda_stream = (stream) ? static_cast<CUDAStream*>(stream)->get() : nullptr;
            cudaEvent_t cuda_event = static_cast<CUDAEvent*>(event)->get();
            cudaError_t err = cudaStreamWaitEvent(cuda_stream, cuda_event, 0);
            if (err != cudaSuccess) {
                INS_THROW("cudaStreamWaitEvent failed for device ", device_id,
                    ": ", cudaGetErrorString(err));
            }
            set_device(original_device);
        }

        Event* CUDADevice::create_event(int device_id) {
            int original_device = get_device();
            set_device(device_id);
            auto* event = new CUDAEvent();
            set_device(original_device);
            return event;
        }

        void CUDADevice::destroy_event(int device_id, Event* event) {
            if (!event) return;
            int original_device = get_device();
            set_device(device_id);
            delete event;
            set_device(original_device);
        }

        void CUDADevice::record_event(int device_id, Stream* stream, Event* event) {
            if (!event) return;
            int original_device = get_device();
            set_device(device_id);
            cudaStream_t cuda_stream = (stream) ? static_cast<CUDAStream*>(stream)->get() : nullptr;
            cudaEvent_t cuda_event = static_cast<CUDAEvent*>(event)->get();
            cudaError_t err = cudaEventRecord(cuda_event, cuda_stream);
            if (err != cudaSuccess) {
                INS_THROW("cudaEventRecord failed for device ", device_id,
                    ": ", cudaGetErrorString(err));
            }
            set_device(original_device);
        }

        bool CUDADevice::query_event(int device_id, Event* event) {
            if (!event) return true;
            int original_device = get_device();
            set_device(device_id);
            cudaEvent_t cuda_event = static_cast<CUDAEvent*>(event)->get();
            cudaError_t err = cudaEventQuery(cuda_event);
            set_device(original_device);
            return (err == cudaSuccess);
        }

        void CUDADevice::synchronize_event(int device_id, Event* event) {
            if (!event) return;
            int original_device = get_device();
            set_device(device_id);
            cudaEvent_t cuda_event = static_cast<CUDAEvent*>(event)->get();
            cudaError_t err = cudaEventSynchronize(cuda_event);
            if (err != cudaSuccess) {
                INS_THROW("cudaEventSynchronize failed for device ", device_id,
                    ": ", cudaGetErrorString(err));
            }
            set_device(original_device);
        }

        float CUDADevice::elapsed_time(Event* start, Event* end) {
            if (!start || !end) return 0.0f;
            cudaEvent_t cuda_start = static_cast<CUDAEvent*>(start)->get();
            cudaEvent_t cuda_end = static_cast<CUDAEvent*>(end)->get();
            float ms;
            cudaError_t err = cudaEventElapsedTime(&ms, cuda_start, cuda_end);
            if (err != cudaSuccess) {
                INS_THROW("cudaEventElapsedTime failed: ", cudaGetErrorString(err));
            }
            return ms;
        }

        cudaDeviceProp CUDADevice::get_device_properties(int device_id) const {
            int original_device = get_device();
            const_cast<CUDADevice*>(this)->set_device(device_id);
            cudaDeviceProp prop;
            cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
            if (err != cudaSuccess) {
                INS_THROW("cudaGetDeviceProperties failed for device ", device_id,
                    ": ", cudaGetErrorString(err));
            }
            const_cast<CUDADevice*>(this)->set_device(original_device);
            return prop;
        }

        size_t CUDADevice::get_compute_capability(int device_id) const {
            auto prop = get_device_properties(device_id);
            return static_cast<size_t>(prop.major * 10 + prop.minor);
        }

        size_t CUDADevice::get_runtime_version(int device_id) const {
            (void)device_id;
            int version;
            cudaError_t err = cudaRuntimeGetVersion(&version);
            if (err != cudaSuccess) return 0;
            return static_cast<size_t>(version);
        }

        size_t CUDADevice::get_driver_version(int device_id) const {
            (void)device_id;
            int version;
            cudaError_t err = cudaDriverGetVersion(&version);
            if (err != cudaSuccess) return 0;
            return static_cast<size_t>(version);
        }

        size_t CUDADevice::get_multi_process(int device_id) const {
            auto prop = get_device_properties(device_id);
            return static_cast<size_t>(prop.multiProcessorCount);
        }

        size_t CUDADevice::get_max_threads_per_mp(int device_id) const {
            auto prop = get_device_properties(device_id);
            return static_cast<size_t>(prop.maxThreadsPerMultiProcessor);
        }

        size_t CUDADevice::get_max_threads_per_block(int device_id) const {
            auto prop = get_device_properties(device_id);
            return static_cast<size_t>(prop.maxThreadsPerBlock);
        }

        std::array<size_t, 3> CUDADevice::get_max_grid_dim_size(int device_id) const {
            auto prop = get_device_properties(device_id);
            return {
                static_cast<size_t>(prop.maxGridSize[0]),
                static_cast<size_t>(prop.maxGridSize[1]),
                static_cast<size_t>(prop.maxGridSize[2])
            };
        }

        Profiler* CUDADevice::create_profiler(int device_id, const std::string& name) {
            int original_device = get_device();
            set_device(device_id);
            auto* profiler = new CUDAProfiler(device_id, name);
            set_device(original_device);
            return profiler;
        }

        void CUDADevice::destroy_profiler(Profiler* profiler) {
            delete profiler;
        }

    } // namespace gpu
} // namespace ins