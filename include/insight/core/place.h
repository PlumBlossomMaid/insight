// insight/core/place.h
#pragma once
#include <cstdint>
#include <ostream>

namespace ins {

    /**
     * @brief Device kind enumeration.
     */
    enum class DeviceKind {
        CPU,    ///< Host CPU
        GPU,     ///< Accelerator device (CUDA, ROCm, Ascend, etc.)
        DEVICE_COUNT
    };

    /**
     * @brief Device placement descriptor.
     *
     * Examples:
     *   Place()         -> CPU:0 (default)
     *   Place::CPU()    -> CPU:0
     *   Place::CPU(1)   -> CPU:1
     *   Place::GPU()    -> GPU:0 (active backend)
     *   Place::GPU(1)   -> GPU:1
     */
    class Place {
    public:
        Place() = default;
        Place(DeviceKind kind, int device_id = 0);

        static Place CPU(int device_id = 0);
        static Place GPU(int device_id = 0);

        DeviceKind kind() const { return kind_; }
        int device_id() const { return device_id_; }

        bool is_cpu() const { return kind_ == DeviceKind::CPU; }
        bool is_gpu() const { return kind_ == DeviceKind::GPU; }

        bool operator==(const Place& other) const;
        bool operator!=(const Place& other) const;

        std::string to_string() const;

    private:
        DeviceKind kind_ = DeviceKind::CPU;
        int device_id_ = 0;
    };

    inline ins::Place CPUPlace(int id = 0)
    {
        return Place(DeviceKind::CPU, id);
    }

    inline ins::Place GPUPlace(int id = 0)
    {
        return Place(DeviceKind::GPU, id);
    }

    /**
     * @brief Get the current default device.
     * @return Current default device
     */
    Place get_device();

    /**
     * @brief Set the global default device.
     * @param place Default device to use for new arrays
     */
    void set_device(const Place& place);

    std::ostream& operator<<(std::ostream& os, const Place& place);

} // namespace ins