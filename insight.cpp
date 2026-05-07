// tests/benchmark/bench_core.cpp
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include "insight/insight.h"

using namespace ins;
using namespace std::chrono;

static double elapsed_ms(high_resolution_clock::time_point start) {
    auto end = high_resolution_clock::now();
    return duration_cast<microseconds>(end - start).count() / 1000.0;
}

#define BENCH(op, a, b) do { \
    auto start = high_resolution_clock::now(); \
    for (int i = 0; i < 100; ++i) { \
        volatile auto r = op(a, b); \
        device->synchronize_device(GPUPlace(0).device_id()); \
    } \
    std::cout << #op << ": " << elapsed_ms(start) << " ms" << std::endl; \
} while(0)

#define BENCH_r(op, a) do { \
    auto start = high_resolution_clock::now(); \
    for (int i = 0; i < 100; ++i) { \
        volatile auto r = op(a); \
        device->synchronize_device(GPUPlace(0).device_id()); \
    } \
    std::cout << #op << ": " << elapsed_ms(start) << " ms" << std::endl; \
} while(0)

int main() {
    ins::init();
    ins::gpu::DeviceFactory* gpu_factory = insight_create_device_factory();
    auto device = gpu_factory->create_device();
    
    int64_t N = 20;
    std::cout << "=== Benchmark: " << N << " elements ===" << std::endl;

    // CPU
    set_device(CPUPlace());
    Array cpu_a = rand({ N }, DType::F32, CPUPlace());
    Array cpu_b = rand({ N }, DType::F32, CPUPlace());

    // GPU
    set_device(GPUPlace(0));
    Array gpu_a = rand({ N }, DType::F32, GPUPlace(0));
    Array gpu_b = rand({ N }, DType::F32, GPUPlace(0));

    std::cout << "\n--- Elementwise ---" << std::endl;
    BENCH(add, cpu_a, cpu_b);
    BENCH(add, gpu_a, gpu_b);

    std::cout << "\n--- Reduction ---" << std::endl;
    BENCH_r(sum, cpu_a);
    BENCH_r(sum, gpu_a);

    return 0;
}