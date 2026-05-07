// insight/plugin/op_registry.h
#pragma once
#include <array>
#include <unordered_map>
#include <string>
#include <functional>
#include <any>
#include <vector>
#include "insight/core/dtype.h"
#include "insight/core/place.h"
#include "insight/core/exception.h"

namespace ins {

    using OpArgs = std::vector<std::any>;
    using OpKernel = std::function<OpArgs(const OpArgs&)>;

    /**
     * @brief Type-safe layer for data type dispatch.
     *
     * Provides bounds-checked access to kernels indexed by DType.
     */
    class DTypeLayer {
    public:
        /**
         * @brief Access kernel by data type (non-const, for registration).
         * @param dtype Data type enumerator
         * @return Reference to the kernel (may be empty)
         */
        OpKernel& operator[](DType dtype) {
            size_t idx = static_cast<size_t>(dtype);
            INS_CHECK(idx < static_cast<size_t>(ins::DType::DTYPE_COUNT), "DType index out of range: ", idx);
            return kernels_[idx];
        }

        /**
         * @brief Access kernel by data type (const, for invocation).
         * @param dtype Data type enumerator
         * @return Reference to the kernel
         * @throws Exception if kernel is not registered
         */
        const OpKernel& operator[](DType dtype) const {
            size_t idx = static_cast<size_t>(dtype);
            INS_CHECK(idx < static_cast<size_t>(ins::DType::DTYPE_COUNT), "DType index out of range: ", idx);
            const auto& kernel = kernels_[idx];
            INS_CHECK(static_cast<bool>(kernel),
                "Kernel not registered for dtype: ", ins::dtype_name(dtype));
            return kernel;
        }

    private:
        std::array<OpKernel, (size_t)ins::DType::DTYPE_COUNT> kernels_;
    };

    /**
     * @brief Type-safe layer for device dispatch.
     *
     * Provides bounds-checked access to DTypeLayer indexed by DeviceKind.
     */
    class DeviceLayer {
    public:
        /**
         * @brief Access DTypeLayer by device type (non-const, for registration).
         * @param device Device kind enumerator
         * @return Reference to the DTypeLayer
         */
        DTypeLayer& operator[](DeviceKind device) {
            size_t idx = static_cast<size_t>(device);
            INS_CHECK(idx < static_cast<size_t>(ins::DeviceKind::DEVICE_COUNT), "Device index out of range: ", idx);
            return devices_[idx];
        }

        /**
         * @brief Access DTypeLayer by device type (const, for invocation).
         * @param device Device kind enumerator
         * @return Reference to the DTypeLayer
         */
        const DTypeLayer& operator[](DeviceKind device) const {
            size_t idx = static_cast<size_t>(device);
            INS_CHECK(idx < static_cast<size_t>(ins::DeviceKind::DEVICE_COUNT), "Device index out of range: ", idx);
            return devices_[idx];
        }

    private:
        std::array<DTypeLayer, static_cast<size_t>(ins::DeviceKind::DEVICE_COUNT)> devices_;
    };

    /**
     * @brief Global operator registry.
     *
     * Usage:
     * @code
     *   // Register a kernel
     *   ops()["add"][DeviceKind::CPU][DType::F32] = add_kernel;
     *
     *   // Invoke a kernel
     *   OpArgs result = ops()["add"][DeviceKind::CPU][DType::F32](args);
     * @endcode
     */
    class OpRegistry {
    public:
        /**
         * @brief Access DeviceLayer by operator name.
         * @param op_name Operator name (e.g., "add", "matmul")
         * @return Reference to the DeviceLayer for this operator
         */
        DeviceLayer& operator[](const std::string& op_name) {
            auto it = kernels_.find(op_name);
            if (it == kernels_.end()) {
                it = kernels_.emplace(op_name, DeviceLayer{}).first;
            }
            return it->second;
        }

        /**
         * @brief Get all registered operator names.
         * @return Vector of operator names
         */
        std::vector<std::string> get_operator_names() const {
            std::vector<std::string> names;
            names.reserve(kernels_.size());
            for (const auto& pair : kernels_) {
                names.push_back(pair.first);
            }
            return names;
        }

        /**
         * @brief Check if a kernel is registered.
         * @param op_name Operator name
         * @param device Device kind
         * @param dtype Data type
         * @return true if kernel exists and is callable
         */
        bool has_kernel(const std::string& op_name, DeviceKind device, DType dtype) const {

            try {
                auto it = kernels_.find(op_name);
                if (it == kernels_.end()) return false;

                const auto& device_layer = it->second;
                size_t dev_idx = static_cast<size_t>(device);
                if (dev_idx >= static_cast<size_t>(ins::DeviceKind::DEVICE_COUNT)) return false;

                const auto& dtype_layer = device_layer[static_cast<ins::DeviceKind>(dev_idx)];
                size_t dt_idx = static_cast<size_t>(dtype);
                if (dt_idx >= static_cast<size_t>(ins::DType::DTYPE_COUNT)) return false;
                return static_cast<bool>(dtype_layer[static_cast<ins::DType>(dt_idx)]);
            }
            catch (const ins::Exception&) {
                return false;
            }
        }

        /**
		* @brief List all registered data types for a given operator and device.
        * @param op_name Operator name
        * @param dev Device kind
        * @return Vector of registered data types
        */
        std::vector<DType> list_kernels(const std::string& op_name, DeviceKind dev) const {
            std::vector<DType> result;
            auto op_it = kernels_.find(op_name);
            if (op_it == kernels_.end()) return result;

            const auto& dtype_layer = op_it->second[dev];

            // Iterate over all DType values and check if registered
            for (int i = 0; i < static_cast<int>(DType::DTYPE_COUNT); ++i) {
                DType dt = static_cast<DType>(i);
                try {
                    dtype_layer[dt];
                    result.push_back(dt);
                }
                catch (const std::exception&) {
                    // Not registered, skip
                }
            }
            return result;
        }

    private:
        std::unordered_map<std::string, DeviceLayer> kernels_;
    };

    /**
     * @brief Global access point to the operator registry.
     * @return Reference to the singleton OpRegistry
     */
    inline OpRegistry& ops() {
        static OpRegistry registry;
        return registry;
    }

} // namespace ins

/**
 * @brief Register a kernel for an operator.
 *
 * Usage:
 * @code
 *   REGISTER_KERNEL(add, CPU, F32, add_kernel);
 * @endcode
 *
 * This macro:
 *   1. Registers the kernel function into the global registry
 *   2. Generates a Touch function to prevent linker optimization
 *
 * @param op_name Operator name (without quotes, will be stringified)
 * @param device DeviceKind enumerator (CPU or GPU)
 * @param dtype DType enumerator (F32, F64, etc.)
 * @param kernel_func Kernel function to register (signature: OpArgs(const OpArgs&))
 */
#define REGISTER_KERNEL(op_name, device, dtype, kernel_func) \
    static bool _register_##op_name##_##device##_##dtype = []() { \
        ::ins::ops()[#op_name][::ins::DeviceKind::device][::ins::DType::dtype] = kernel_func; \
        return true; \
    }(); \
    void _touch_##op_name##_##device##_##dtype() { \
        (void)_register_##op_name##_##device##_##dtype; \
    }

 /**
  * @brief Force linking of a kernel.
  *
  * Use this macro in a .cpp file that is guaranteed to be linked
  * (e.g., the main file or a dedicated register.cpp) to prevent
  * the linker from optimizing away the kernel registration.
  *
  * Usage:
  * @code
  *   USE_KERNEL(add, CPU, F32);
  * @endcode
  *
  * @param op_name Operator name
  * @param device DeviceKind enumerator
  * @param dtype DType enumerator
  */
#define USE_KERNEL(op_name, device, dtype) \
    extern void _touch_##op_name##_##device##_##dtype(); \
    static int _use_##op_name##_##device##_##dtype = (_touch_##op_name##_##device##_##dtype(), 0)

  /**
   * @brief Register an operator module and generate a touch function.
   *
   * This macro should be placed at the end of the module's .cpp file (e.g., add.cpp).
   * It generates a extern "C" function that can be called to prevent the linker
   * from optimizing away the static registrations in this module.
   *
   * Usage:
   * @code
   *   // At the end of backends/cpu/add.cpp
   *   REGISTER_MODULE(add, CPU);
   * @endcode
   *
   * @param op_name Operator name (without quotes)
   * @param device DeviceKind enumerator (CPU or GPU)
   */
#define REGISTER_MODULE(op_name, device) \
    extern "C" void _touch_##op_name##_##device() {}

   /**
    * @brief Force linking of an operator module.
    *
    * This macro should be used in a .cpp file that is guaranteed to be linked
    * (e.g., the main file or a dedicated register.cpp). It forces the linker
    * to include the entire module, ensuring all static registrations are executed.
    *
    * Usage:
    * @code
    *   // In main.cpp or register.cpp
    *   USE_MODULE(add, CPU);
    *   USE_MODULE(sub, CPU);
    * @endcode
    *
    * @param op_name Operator name (without quotes)
    * @param device DeviceKind enumerator (CPU or GPU)
    */
#define USE_MODULE(op_name, device) \
    void _touch_##op_name##_##device(); \
    static int _use_##op_name##_##device = (_touch_##op_name##_##device(), 0)