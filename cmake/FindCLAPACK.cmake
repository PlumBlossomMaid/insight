# cmake/FindCLAPACK.cmake
# Find CLAPACK (C version of LAPACK) installation

find_path(CLAPACK_INCLUDE_DIR
    NAMES clapack.h
    PATHS
        /usr/include
        /usr/local/include
        /opt/clapack/include
        $ENV{CLAPACK_HOME}/include
        ${CMAKE_PREFIX_PATH}/include
)

# CLAPACK library name varies:
# - Linux: libclapack.a, liblapack.a
# - Windows: lapack.lib, clapack.lib
find_library(CLAPACK_LIBRARY
    NAMES 
        clapack      # Linux / some builds
        libclapack   # Linux
        lapack       # Windows (vcpkg provides lapack.lib)
    PATHS
        /usr/lib
        /usr/local/lib
        /opt/clapack/lib
        $ENV{CLAPACK_HOME}/lib
        ${CMAKE_PREFIX_PATH}/lib
)

# Also find f2c library (required by CLAPACK on some systems)
find_library(F2C_LIBRARY
    NAMES f2c libf2c
    PATHS
        /usr/lib
        /usr/local/lib
        /opt/clapack/lib
        $ENV{CLAPACK_HOME}/lib
        ${CMAKE_PREFIX_PATH}/lib
)

if(CLAPACK_INCLUDE_DIR AND CLAPACK_LIBRARY)
    set(CLAPACK_FOUND TRUE)
    set(CLAPACK_LIBRARIES ${CLAPACK_LIBRARY})
    
    if(F2C_LIBRARY)
        list(APPEND CLAPACK_LIBRARIES ${F2C_LIBRARY})
    endif()
    
    set(CLAPACK_INCLUDE_DIRS ${CLAPACK_INCLUDE_DIR})
    
    # CLAPACK uses column-major; -DCBLAS enables row-major support
    set(CLAPACK_DEFINITIONS -DCBLAS)
    
    message(STATUS "Found CLAPACK: ${CLAPACK_LIBRARY}")
    message(STATUS "  Include: ${CLAPACK_INCLUDE_DIR}")
else()
    set(CLAPACK_FOUND FALSE)
    message(STATUS "CLAPACK not found")
endif()

mark_as_advanced(CLAPACK_INCLUDE_DIR CLAPACK_LIBRARY F2C_LIBRARY)