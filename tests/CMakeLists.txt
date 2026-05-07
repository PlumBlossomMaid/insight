# tests/CMakeLists.txt

# 收集目录下所有测试源文件
file(GLOB CPU_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cpu/*.cpp")
file(GLOB CUDA_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cpp")

add_executable(insight_tests_cpu ${CPU_TEST_SOURCES})
add_executable(insight_tests_cuda ${CUDA_TEST_SOURCES})

target_link_libraries(insight_tests_cpu
    insight_core
    insight_cpu_backend
    GTest::gtest_main
)

target_link_libraries(insight_tests_cuda
    insight_core
    insight_cpu_backend
    insight_cuda_backend
    GTest::gtest_main
)

target_include_directories(insight_tests_cpu
    PRIVATE
        ${INSIGHT_INCLUDE_DIR}
)

# 自动发现并注册测试
include(GoogleTest)
gtest_discover_tests(insight_tests_cpu)