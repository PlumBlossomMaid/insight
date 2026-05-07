# CMakeLists.txt
cmake_minimum_required(VERSION 3.15)

# ========== 设置策略以消除警告 ==========
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 NEW)
endif()
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()

project(insight VERSION 1.0.0 LANGUAGES CXX CUDA)

# ========== 全局 CUDA 架构设置 ==========
if(INSIGHT_WITH_CUDA AND NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "52;61;70;75;80;86;89" CACHE STRING "CUDA architectures")
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 地址消毒器（调试用）- 只对 C++ 生效
#if(MSVC)
#    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/fsanitize=address>)
#    add_link_options($<$<COMPILE_LANGUAGE:CXX>:/fsanitize=address>)
#else()
#    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address -g>)
#    add_link_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>)
#endif()

# Options
option(INSIGHT_BUILD_TESTS "Build unit tests" ON)
option(INSIGHT_WITH_CUDA "Enable CUDA backend" ON)

# Add custom CMake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Include auto-fetch function
include(FetchDependency)

# ========== Directory Definitions ==========
set(INSIGHT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(INSIGHT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INSIGHT_INTERNAL_DIR ${INSIGHT_SRC_DIR}/internal)
set(INSIGHT_UTILS_DIR ${INSIGHT_SRC_DIR}/utils)

# ========== Source Files Collection ==========
file(GLOB_RECURSE INSIGHT_SOURCES
    ${INSIGHT_SRC_DIR}/*/*.cpp
)

# ========== GoogleTest (for unit testing) ==========
if(INSIGHT_BUILD_TESTS)
    auto_fetch_dependency(googletest
        "https://github.com/google/googletest.git"
        "v1.15.0"
    )
endif()

# ========== Core Library ==========
add_library(insight_core STATIC
    ${INSIGHT_SOURCES}
    src/init.cpp
)

# ========== OpenMP ==========
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(insight_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(insight_core PRIVATE ${OpenMP_CXX_FLAGS})
    target_compile_definitions(insight_core PUBLIC INSIGHT_WITH_OPENMP)
endif()

# ========== Thrust ==========
option(INSIGHT_USE_THRUST "Use Thrust for GPU algorithms (sort, scan, etc.)" ON)

# ========== OpenBLAS ==========
option(INSIGHT_USE_OPENBLAS "Use OpenBLAS for linear algebra" ON)
if(INSIGHT_USE_OPENBLAS)
    find_package(OpenBLAS)
    if(OpenBLAS_FOUND)
        target_link_libraries(insight_core PRIVATE ${OpenBLAS_LIBRARIES})
        target_compile_definitions(insight_core PUBLIC INSIGHT_USE_OPENBLAS)
        message(STATUS "OpenBLAS found and enabled")
    else()
        message(WARNING "OpenBLAS not found, disabling INSIGHT_USE_OPENBLAS")
        set(INSIGHT_USE_OPENBLAS OFF)
    endif()
else()
    message(STATUS "OpenBLAS support disabled")
endif()

# ========== Linear Algebra (CLAPACK) ==========
option(INSIGHT_USE_LAPACK "Use LAPACK for linear algebra" ON)
if(INSIGHT_USE_LAPACK)
    find_package(CLAPACK)
    if(CLAPACK_FOUND)
        target_link_libraries(insight_core PUBLIC ${CLAPACK_LIBRARIES})
        target_include_directories(insight_core PUBLIC ${CLAPACK_INCLUDE_DIRS})
        target_compile_definitions(insight_core PUBLIC INSIGHT_USE_LAPACK ${CLAPACK_DEFINITIONS})
        message(STATUS "CLAPACK enabled for linear algebra")
    else()
        message(WARNING "CLAPACK not found, disabling INSIGHT_USE_LAPACK")
        set(INSIGHT_USE_LAPACK OFF)
    endif()
endif()

# ========== FFTW3 ==========
option(INSIGHT_USE_FFTW3 "Use FFTW3 for FFT operations" ON)
if(INSIGHT_USE_FFTW3)
    find_package(FFTW3)
    if(FFTW3_FOUND)
        target_link_libraries(insight_core PRIVATE ${FFTW3_LIBRARIES})
        target_compile_definitions(insight_core PUBLIC INSIGHT_USE_FFTW3)
        message(STATUS "FFTW3 found and enabled")
    else()
        message(WARNING "FFTW3 not found, disabling INSIGHT_USE_FFTW3")
        set(INSIGHT_USE_FFTW3 OFF)
    endif()
else()
    message(STATUS "FFTW3 support disabled")
endif()

# ========== Include Directories ==========
target_include_directories(insight_core
    PUBLIC
        ${INSIGHT_INCLUDE_DIR}
    PRIVATE
        ${INSIGHT_INTERNAL_DIR}
        ${INSIGHT_UTILS_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# ========== CPU Backend ==========
add_subdirectory(backends/cpu)
target_link_libraries(insight_core PRIVATE insight_cpu_backend)

# ========== CUDA Backend ==========
if(INSIGHT_WITH_CUDA)
    if(INSIGHT_USE_THRUST)
        set(INSIGHT_CUDA_USE_THRUST ON)
        target_compile_definitions(insight_core PUBLIC INSIGHT_USE_THRUST)
    else()
        set(INSIGHT_CUDA_USE_THRUST OFF)
    endif()
    add_subdirectory(backends/cuda)
    target_link_libraries(insight_core PRIVATE insight_cuda_backend)
    target_compile_definitions(insight_core PUBLIC INSIGHT_WITH_CUDA)
else()
    message(STATUS "CUDA backend disabled by user option")
endif()

# ========== Demo Executable ==========
add_executable(insight_demo
    insight.cpp
)

target_link_libraries(insight_demo PRIVATE insight_core)

target_include_directories(insight_demo
    PRIVATE
        ${INSIGHT_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# ========== Unit Tests ==========
if(INSIGHT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ========== Output Settings ==========
set_target_properties(insight_demo PROPERTIES
    OUTPUT_NAME "insight"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ========== Configuration Summary ==========
message(STATUS "")
message(STATUS "Insight Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "  OpenMP:    ${OpenMP_CXX_FOUND}")
message(STATUS "  OpenBLAS:  ${INSIGHT_USE_OPENBLAS}")
message(STATUS "  FFTW3:     ${INSIGHT_USE_FFTW3}")
message(STATUS "  CUDA:      ${INSIGHT_WITH_CUDA}")
message(STATUS "  Thrust:    ${INSIGHT_USE_THRUST}")
message(STATUS "  Tests:     ${INSIGHT_BUILD_TESTS}")
message(STATUS "")