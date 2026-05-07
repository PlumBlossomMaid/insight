// backends/cpu/init.h
#pragma once
#include "insight/plugin/op_registry.h"

namespace ins::cpu {

    /**
     * @brief Register all CPU backend kernels.
     *
     * Call this function during initialization to ensure all CPU kernels
     * are linked and registered.
     */
    extern "C" void init_cpu() {
        USE_MODULE(elementwise, CPU);
        USE_MODULE(cast, CPU);
        USE_MODULE(unary, CPU);
        USE_MODULE(creation, CPU);
        USE_MODULE(manipulation, CPU);
        USE_MODULE(random, CPU);
        USE_MODULE(reduction, CPU);
        USE_MODULE(indexing, CPU);
        USE_MODULE(linalg, CPU);
        USE_MODULE(fft, CPU);
    }

} // namespace ins::cpu