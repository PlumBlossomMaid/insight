# cmake/FindOpenBLAS.cmake
find_path(OpenBLAS_INCLUDE_DIR
    NAMES cblas.h
    PATHS
        /usr/include
        /usr/local/include
        /opt/OpenBLAS/include
        $ENV{OPENBLAS_HOME}/include
        ${CMAKE_PREFIX_PATH}/include
    PATH_SUFFIXES
        openblas
)

find_library(OpenBLAS_LIBRARY
    NAMES openblas libopenblas
    PATHS
        /usr/lib
        /usr/local/lib
        /opt/OpenBLAS/lib
        $ENV{OPENBLAS_HOME}/lib
        ${CMAKE_PREFIX_PATH}/lib
)

if(OpenBLAS_INCLUDE_DIR AND OpenBLAS_LIBRARY)
    set(OpenBLAS_FOUND TRUE)
    set(OpenBLAS_LIBRARIES ${OpenBLAS_LIBRARY})
    set(OpenBLAS_INCLUDE_DIRS ${OpenBLAS_INCLUDE_DIR})
    message(STATUS "Found OpenBLAS: ${OpenBLAS_LIBRARY}")
    message(STATUS "OpenBLAS include: ${OpenBLAS_INCLUDE_DIR}")
endif()

mark_as_advanced(OpenBLAS_INCLUDE_DIR OpenBLAS_LIBRARY)