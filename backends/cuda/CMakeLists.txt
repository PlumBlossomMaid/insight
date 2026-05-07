# backends/cuda/CMakeLists.txt

enable_language(CUDA)

file(GLOB KERNEL_SOURCES "kernels/*.cu")
file(GLOB DEVICE_SOURCES "device/*.cpp")

add_library(insight_cuda_backend STATIC
    ${KERNEL_SOURCES}
    ${DEVICE_SOURCES}
    gpu_init.cpp
)

target_include_directories(insight_cuda_backend
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(insight_cuda_backend
    cudart
    cufft
    cublas
    cusolver
)

# Thrust 控制
if(INSIGHT_CUDA_USE_THRUST)
    target_compile_definitions(insight_cuda_backend PRIVATE INSIGHT_USE_THRUST)
    message(STATUS "CUDA backend: Thrust enabled")
else()
    message(STATUS "CUDA backend: Thrust disabled (pure CUDA fallback)")
endif()

message(STATUS "CUDA backend configured")