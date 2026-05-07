// src/core/place.cpp
#include "insight/core/place.h"
#include "insight/core/exception.h"
#include <sstream>

namespace ins {

    Place::Place(DeviceKind kind, int device_id)
        : kind_(kind), device_id_(device_id) {
        INS_CHECK(device_id >= 0, "Place: device_id must be non-negative, got ", device_id);
    }

    Place Place::CPU(int device_id) {
        return Place(DeviceKind::CPU, device_id);
    }

    Place Place::GPU(int device_id) {
        return Place(DeviceKind::GPU, device_id);
    }

    bool Place::operator==(const Place& other) const {
        return kind_ == other.kind_ && device_id_ == other.device_id_;
    }

    bool Place::operator!=(const Place& other) const {
        return !(*this == other);
    }

    std::string Place::to_string() const {
        std::ostringstream os;
        os << *this;
        return os.str();
    }

    std::ostream& operator<<(std::ostream& os, const Place& place) {
        if (place.is_cpu()) {
            os << "cpu";
        }
        else {
            os << "gpu:" << place.device_id();
        }
        return os;
    }

    namespace {
        thread_local Place g_default_device = CPUPlace();
    } // anonymous namespace

    Place get_device() {
        return g_default_device;
    }

    void set_device(const Place& place) {
        // 加入gpu之后继续丰富逻辑
        g_default_device = place;
    }


} // namespace ins