// insight/utils/features.cpp
#include "insight/utils/features.h"

namespace ins {

    bool is_compiled_with_openblas() {
#ifdef INSIGHT_USE_OPENBLAS
        return true;
#else
        return false;
#endif
    }

    bool is_compiled_with_fftw3() {
#ifdef INSIGHT_USE_FFTW3
        return true;
#else
        return false;
#endif
    }

    bool is_compiled_with_lapack() {
#ifdef INSIGHT_USE_LAPACK
        return true;
#else
        return false;
#endif
    }

    bool is_compiled_with_thrust() {
#ifdef INSIGHT_USE_THRUST
        return true;
#else
        return false;
#endif
    }

} // namespace ins