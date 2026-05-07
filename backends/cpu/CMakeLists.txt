# backends/cpu/CMakeLists.txt

# 添加当前目录到包含路径
set(CPU_BACKEND_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# 自动收集所有 .cpp 文件
file(GLOB CPU_KERNEL_SOURCES "*.cpp")

add_library(insight_cpu_backend STATIC ${CPU_KERNEL_SOURCES})

# 添加包含目录：让 insight_core 和 insight_tests 能包含 backends/cpu/init.h
target_include_directories(insight_cpu_backend
    PUBLIC
        ${CPU_BACKEND_INCLUDE_DIR}  # 暴露给链接它的目标
    PRIVATE
        ${INSIGHT_INCLUDE_DIR}
)

target_link_libraries(insight_cpu_backend
    insight_core
)