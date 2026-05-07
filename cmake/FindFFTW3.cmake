# cmake/FindFFTW3.cmake
find_path(FFTW3_INCLUDE_DIR
    NAMES fftw3.h
    PATHS
        /usr/include
        /usr/local/include
        /opt/fftw3/include
        $ENV{FFTW3_HOME}/include
        ${CMAKE_PREFIX_PATH}/include
)

find_library(FFTW3_LIBRARY
    NAMES fftw3 libfftw3
    PATHS
        /usr/lib
        /usr/local/lib
        /opt/fftw3/lib
        $ENV{FFTW3_HOME}/lib
        ${CMAKE_PREFIX_PATH}/lib
)

find_library(FFTW3f_LIBRARY
    NAMES fftw3f libfftw3f
    PATHS
        /usr/lib
        /usr/local/lib
        /opt/fftw3/lib
        $ENV{FFTW3_HOME}/lib
        ${CMAKE_PREFIX_PATH}/lib
)

if(FFTW3_INCLUDE_DIR AND FFTW3_LIBRARY AND FFTW3f_LIBRARY)
    set(FFTW3_FOUND TRUE)
    set(FFTW3_LIBRARIES ${FFTW3_LIBRARY} ${FFTW3f_LIBRARY})
    set(FFTW3_INCLUDE_DIRS ${FFTW3_INCLUDE_DIR})
    message(STATUS "Found FFTW3: ${FFTW3_LIBRARY} and ${FFTW3f_LIBRARY}")
else()
    message(STATUS "FFTW3 not found. Looked in:")
    message(STATUS "  Include: ${FFTW3_INCLUDE_DIR}")
    message(STATUS "  Double library: ${FFTW3_LIBRARY}")
    message(STATUS "  Float library: ${FFTW3f_LIBRARY}")
endif()

mark_as_advanced(FFTW3_INCLUDE_DIR FFTW3_LIBRARY FFTW3f_LIBRARY)