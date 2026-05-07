// launch_config.h
#ifndef LAUNCH_CONFIG_H
#define LAUNCH_CONFIG_H
#include <cuda_runtime.h>

class LaunchConfig {
public:
    dim3 blocks;
    dim3 threads;
    int shmSize;
    cudaStream_t stream;

    // 默认构造函数
    LaunchConfig()
        : blocks(1, 1, 1), threads(256, 1, 1), shmSize(0), stream(nullptr) {
    }

    // 自动计算 blocks（一维）
    LaunchConfig(int n, cudaStream_t stream = nullptr)
        : threads(256, 1, 1), shmSize(0), stream(stream) {
        if (n <= 0) {
            blocks = dim3(0, 1, 1);
        }
        else {
            int num_blocks = (n + threads.x - 1) / threads.x;
            blocks = dim3(num_blocks, 1, 1);
        }
    }

    // 指定 threads（一维）
    LaunchConfig(int n, int threads_per_block, cudaStream_t stream = nullptr)
        : threads(threads_per_block, 1, 1), shmSize(0), stream(stream) {
        if (threads_per_block <= 0) {
            throw std::invalid_argument("threads_per_block must be positive");
        }
        if (n <= 0) {
            blocks = dim3(0, 1, 1);
        }
        else {
            int num_blocks = (n + threads_per_block - 1) / threads_per_block;
            blocks = dim3(num_blocks, 1, 1);
        }
    }

    // 指定 threads 和 shared memory（一维）
    LaunchConfig(int n, int threads_per_block, int shm_size, cudaStream_t stream = nullptr)
        : threads(threads_per_block, 1, 1), shmSize(shm_size), stream(stream) {
        if (threads_per_block <= 0) {
            throw std::invalid_argument("threads_per_block must be positive");
        }
        if (n <= 0) {
            blocks = dim3(0, 1, 1);
        }
        else {
            int num_blocks = (n + threads_per_block - 1) / threads_per_block;
            blocks = dim3(num_blocks, 1, 1);
        }
    }

    // 新增：二维构造函数
    LaunchConfig(dim3 grid, dim3 block, int shm_size = 0, cudaStream_t stream = nullptr)
        : blocks(grid), threads(block), shmSize(shm_size), stream(stream) {
    }

    // 兼容老代码的 set 方法（一维）
    void set(int blocks, int threads, int shmSize, cudaStream_t stream) {
        if (threads <= 0) {
            throw std::invalid_argument("threads must be positive");
        }
        this->blocks = dim3(blocks, 1, 1);
        this->threads = dim3(threads, 1, 1);
        this->shmSize = shmSize;
        this->stream = stream;
    }

    // 新增：二维 set 方法
    void set(dim3 blocks, dim3 threads, int shmSize, cudaStream_t stream) {
        this->blocks = blocks;
        this->threads = threads;
        this->shmSize = shmSize;
        this->stream = stream;
    }
};

#endif // LAUNCH_CONFIG_H