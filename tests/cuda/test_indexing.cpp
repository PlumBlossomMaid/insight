// tests/cuda/test_indexing.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "insight/insight.h"
#include "insight/ops/indexing.h"

using namespace ins;

class IndexingTestGPU : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ins::init();
        set_device(ins::GPUPlace(0));
    }
};

// ========== Helper functions ==========

static Array arange_1d(int64_t n, float start = 0.0f, float step = 1.0f) {
    std::vector<float> data(n);
    for (int64_t i = 0; i < n; ++i) {
        data[i] = start + i * step;
    }
    Array cpu_arr = to_array(data, Shape({ n }));
    return cpu_arr.to(GPUPlace(0));
}

static Array arange_2d(int rows, int cols, float start = 0.0f, float step = 1.0f) {
    std::vector<float> data(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        data[i] = start + i * step;
    }
    Array cpu_arr = to_array(data, Shape({ rows, cols }));
    return cpu_arr.to(GPUPlace(0));
}

template<typename T>
T gpu_item(const Array& gpu_arr) {
    Array cpu_arr = gpu_arr.to(CPUPlace());
    return cpu_arr.item<T>();
}

template<typename T>
void expect_equal_gpu(const Array& gpu_arr, const std::vector<T>& expected, T tol = 1e-5) {
    Array cpu_arr = gpu_arr.to(CPUPlace());
    ASSERT_EQ(cpu_arr.numel(), static_cast<int64_t>(expected.size()));
    const T* data = cpu_arr.data<T>();
    for (size_t i = 0; i < expected.size(); ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            EXPECT_NEAR(data[i], expected[i], tol);
        }
        else {
            EXPECT_EQ(data[i], expected[i]);
        }
    }
}

// ========== take tests ==========

TEST_F(IndexingTestGPU, TakeFlattened) {
    Array x = arange_2d(3, 4);
    Array idx_cpu = to_array(std::vector<int64_t>{0, 2, 3, 7, 10});
    Array idx = idx_cpu.to(GPUPlace(0));
    Array t = take(x, idx);

    EXPECT_EQ(t.numel(), 5);
    std::vector<float> expected = { 0.0f, 2.0f, 3.0f, 7.0f, 10.0f };
    expect_equal_gpu<float>(t, expected);
}

TEST_F(IndexingTestGPU, TakeAlongAxis) {
    Array x = arange_2d(3, 4);
    Array idx_cpu = to_array(std::vector<int64_t>{0, 2, 1, 0, 2});
    Array idx = idx_cpu.to(GPUPlace(0));
    Array t = take(x, idx, 0);

    EXPECT_EQ(t.shape(), Shape({ 5, 4 }));
    std::vector<float> expected = {
        0.0f, 1.0f, 2.0f, 3.0f,
        8.0f, 9.0f, 10.0f, 11.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        0.0f, 1.0f, 2.0f, 3.0f,
        8.0f, 9.0f, 10.0f, 11.0f
    };
    expect_equal_gpu<float>(t, expected);
}

TEST_F(IndexingTestGPU, TakeAxis0) {
    Array x = arange_2d(3, 4);
    Array idx_cpu = to_array(std::vector<int64_t>{0, 2, 1, 0, 2});
    Array idx = idx_cpu.to(GPUPlace(0));
    Array t = take(x, idx, 0);

    std::vector<float> expected = {
        0.0f, 1.0f, 2.0f, 3.0f,
        8.0f, 9.0f, 10.0f, 11.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        0.0f, 1.0f, 2.0f, 3.0f,
        8.0f, 9.0f, 10.0f, 11.0f
    };
    expect_equal_gpu<float>(t, expected);
}

// ========== take_along_axis tests ==========

TEST_F(IndexingTestGPU, TakeAlongAxis0) {
    Array x = arange_2d(3, 4);
    std::vector<int64_t> idx_data = { 2,2,2,2, 1,1,1,1, 0,0,0,0 };
    Array idx_cpu = to_array(idx_data, Shape({ 3, 4 }));
    Array idx = idx_cpu.to(GPUPlace(0));
    Array t = take_along_axis(x, idx, 0);

    EXPECT_EQ(t.shape(), Shape({ 3, 4 }));
    std::vector<float> expected = {
        8.0f, 9.0f, 10.0f, 11.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        0.0f, 1.0f, 2.0f, 3.0f
    };
    expect_equal_gpu<float>(t, expected);
}

// ========== put tests ==========

TEST_F(IndexingTestGPU, PutFlattened) {
    Array x = arange_2d(3, 4);
    Array idx_cpu = to_array(std::vector<int64_t>{0, 5, 10});
    Array idx = idx_cpu.to(GPUPlace(0));
    Array val_cpu = to_array(std::vector<float>{99.0f, 88.0f, 77.0f});
    Array val = val_cpu.to(GPUPlace(0));
    Array p = put(x, idx, val);

    std::vector<float> expected = {
        99.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 88.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 77.0f, 11.0f
    };
    expect_equal_gpu<float>(p, expected);
}

// ========== put_along_axis tests ==========

TEST_F(IndexingTestGPU, PutAlongAxis0) {
    Array x = arange_2d(3, 4);
    std::vector<int64_t> idx_data = { 2, 1, 0 };
    Array idx_cpu = to_array(idx_data, Shape({ 3, 1 }));
    Array idx = idx_cpu.to(GPUPlace(0));
    Array val_cpu = to_array(std::vector<float>{99.0f, 88.0f, 77.0f});
    Array val = val_cpu.to(GPUPlace(0));
    Array p = put_along_axis(x, idx, val, 0);

    std::vector<float> expected = {
        77.0f, 1.0f, 2.0f, 3.0f,
        88.0f, 5.0f, 6.0f, 7.0f,
        99.0f, 9.0f, 10.0f, 11.0f
    };
    expect_equal_gpu<float>(p, expected);
}

// ========== gather / scatter tests ==========

TEST_F(IndexingTestGPU, Gather) {
    Array x = arange_2d(3, 4);
    std::vector<int64_t> idx_data = { 2,2,2,2, 1,1,1,1, 0,0,0,0 };
    Array idx_cpu = to_array(idx_data, Shape({ 3, 4 }));
    Array idx = idx_cpu.to(GPUPlace(0));
    Array g = gather(x, 0, idx);

    EXPECT_EQ(g.shape(), Shape({ 3, 4 }));
    std::vector<float> expected = {
        8.0f, 9.0f, 10.0f, 11.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        0.0f, 1.0f, 2.0f, 3.0f
    };
    expect_equal_gpu<float>(g, expected);
}

// ========== masked_select / compress tests ==========

TEST_F(IndexingTestGPU, MaskedSelect) {
    Array x = arange_2d(3, 4);
    std::vector<bool> mask_data = {
        true, false, true, false,
        false, true, false, true,
        true, false, true, false
    };
    Array mask_cpu = to_array(mask_data, Shape({ 3, 4 }));
    Array mask = mask_cpu.to(GPUPlace(0));
    Array m = masked_select(x, mask);

    EXPECT_EQ(m.numel(), 6);
    std::vector<float> expected = { 0.0f, 2.0f, 5.0f, 7.0f, 8.0f, 10.0f };
    expect_equal_gpu<float>(m, expected);
}

TEST_F(IndexingTestGPU, Compress) {
    Array x = arange_2d(3, 4);
    std::vector<bool> cond_data = { true, false, true };
    Array cond_cpu = to_array(cond_data);
    Array cond = cond_cpu.to(GPUPlace(0));
    Array c = compress(x, cond, 0);

    EXPECT_EQ(c.shape(), Shape({ 2, 4 }));
    std::vector<float> expected = {
        0.0f, 1.0f, 2.0f, 3.0f,
        8.0f, 9.0f, 10.0f, 11.0f
    };
    expect_equal_gpu<float>(c, expected);
}

// ========== where tests ==========

TEST_F(IndexingTestGPU, Where3Arg) {
    Array cond_cpu = to_array(std::vector<bool>{true, false, true, false});
    Array cond = cond_cpu.to(GPUPlace(0));
    Array x = arange_1d(4, 1.0f, 1.0f);
    Array y = arange_1d(4, 10.0f, 1.0f);
    Array w = where(cond, x, y);

    std::vector<float> expected = { 1.0f, 11.0f, 3.0f, 13.0f };
    expect_equal_gpu<float>(w, expected);
}

TEST_F(IndexingTestGPU, Where2Arg) {
    Array cond_cpu = to_array(std::vector<bool>{true, false, true, false});
    Array cond = cond_cpu.to(GPUPlace(0));
    Array nz = where(cond);
    EXPECT_EQ(nz.numel(), 2);
    std::vector<int64_t> expected = { 0, 2 };
    expect_equal_gpu<int64_t>(nz, expected);
}

// ========== nonzero / flatnonzero tests ==========

TEST_F(IndexingTestGPU, Nonzero) {
    std::vector<float> data = { 0, 1, 0, 2, 0, 3, 0, 4 };
    Array x_cpu = to_array(data).reshape(Shape({ 2, 4 }));
    Array x = x_cpu.to(GPUPlace(0));
    Array nz = nonzero(x);

    EXPECT_EQ(nz.shape().dim(0), 2);
    EXPECT_EQ(nz.shape().dim(1), 4);
    std::vector<int64_t> expected = { 0, 0, 1, 1, 1, 3, 1, 3 };
    expect_equal_gpu<int64_t>(nz, expected);
}

TEST_F(IndexingTestGPU, Flatnonzero) {
    std::vector<float> data = { 0, 1, 0, 2, 0, 3, 0, 4 };
    Array x_cpu = to_array(data).reshape(Shape({ 2, 4 }));
    Array x = x_cpu.to(GPUPlace(0));
    Array fnz = flatnonzero(x);

    EXPECT_EQ(fnz.numel(), 4);
    std::vector<int64_t> expected = { 1, 3, 5, 7 };
    expect_equal_gpu<int64_t>(fnz, expected);
}

// ========== argsort / sort tests ==========

TEST_F(IndexingTestGPU, Argsort) {
    std::vector<float> data = { 3, 1, 4, 1, 5, 9, 2, 6 };
    Array x_cpu = to_array(data).reshape(Shape({ 2, 4 }));
    Array x = x_cpu.to(GPUPlace(0));
    Array idx = argsort(x, 1);

    std::vector<int64_t> expected = { 1, 3, 0, 2, 2, 0, 3, 1 };
    expect_equal_gpu<int64_t>(idx, expected);
}

TEST_F(IndexingTestGPU, Sort) {
    std::vector<float> data = { 3, 1, 4, 1, 5, 9, 2, 6 };
    Array x_cpu = to_array(data).reshape(Shape({ 2, 4 }));
    Array x = x_cpu.to(GPUPlace(0));
    Array s = sort(x, 1);

    std::vector<float> expected = { 1.0f, 1.0f, 3.0f, 4.0f, 2.0f, 5.0f, 6.0f, 9.0f };
    expect_equal_gpu<float>(s, expected);
}

// ========== topk tests ==========

TEST_F(IndexingTestGPU, TopkLargest) {
    std::vector<float> data = { 3, 1, 4, 1, 5, 9, 2, 6 };
    Array x_cpu = to_array(data).reshape(Shape({ 2, 4 }));
    Array x = x_cpu.to(GPUPlace(0));
    auto [vals, idxs] = topk(x, 3, 1, true, true);

    std::vector<float> expected_vals = { 4.0f, 3.0f, 1.0f, 9.0f, 6.0f, 5.0f };
    std::vector<int64_t> expected_idxs = { 2, 0, 1, 1, 3, 0 };
    expect_equal_gpu<float>(vals, expected_vals);
    expect_equal_gpu<int64_t>(idxs, expected_idxs);
}

// ========== searchsorted tests ==========

TEST_F(IndexingTestGPU, SearchsortedLeft) {
    Array x_cpu = to_array(std::vector<float>{1, 2, 2, 3, 3, 3, 4, 5});
    Array x = x_cpu.to(GPUPlace(0));
    Array v_cpu = to_array(std::vector<float>{2, 3, 4, 6});
    Array v = v_cpu.to(GPUPlace(0));
    Array left = searchsorted(x, v, "left");

    std::vector<int64_t> expected = { 1, 3, 6, 8 };
    expect_equal_gpu<int64_t>(left, expected);
}

TEST_F(IndexingTestGPU, SearchsortedRight) {
    Array x_cpu = to_array(std::vector<float>{1, 2, 2, 3, 3, 3, 4, 5});
    Array x = x_cpu.to(GPUPlace(0));
    Array v_cpu = to_array(std::vector<float>{2, 3, 4, 6});
    Array v = v_cpu.to(GPUPlace(0));
    Array right = searchsorted(x, v, "right");

    std::vector<int64_t> expected = { 3, 6, 7, 8 };
    expect_equal_gpu<int64_t>(right, expected);
}

// ========== unique tests ==========
TEST_F(IndexingTestGPU, UniqueBasic) {
    std::vector<float> data = { 3, 1, 2, 2, 3, 1, 4, 2 }; ;
    Array x_cpu = to_array(data, ins::CPUPlace()); ;
    Array x = x_cpu.to(GPUPlace(0)); ;
    UniqueResult u_cpu = unique(x_cpu); ;
    UniqueResult u = unique(x); ;

    auto aligned = u_cpu.unique == u.unique;
    aligned = ins::all(aligned);

    EXPECT_TRUE(aligned.item<bool>()); 
    EXPECT_EQ(u.unique.numel(), 4); 
    std::vector<float> expected = { 1, 2, 3, 4 }; 
    expect_equal_gpu<float>(u.unique, expected); 
}

TEST_F(IndexingTestGPU, UniqueWithCounts) {
    std::vector<float> data = { 3, 1, 2, 2, 3, 1, 4, 2 };
    Array x_cpu = to_array(data);
    Array x = x_cpu.to(GPUPlace(0));
    UniqueResult u = unique(x, true, true, true);

    std::vector<int64_t> expected = { 2, 3, 2, 1 };
    expect_equal_gpu<int64_t>(u.counts, expected);
}

// ========== lexsort tests ==========

TEST_F(IndexingTestGPU, Lexsort) {
    std::vector<int> key0 = { 1, 1, 0, 0 };
    std::vector<int> key1 = { 0, 1, 0, 1 };
    std::vector<int> key2 = { 25, 30, 20, 35 };

    std::vector<int> stacked(3 * 4);
    for (int i = 0; i < 4; ++i) {
        stacked[0 * 4 + i] = key0[i];
        stacked[1 * 4 + i] = key1[i];
        stacked[2 * 4 + i] = key2[i];
    }
    Array keys_cpu = to_array(stacked, Shape({ 3, 4 }));
    Array keys_gpu = keys_cpu.to(GPUPlace(0));
    Array idx_cpu = lexsort(keys_cpu, 1);
    Array idx_gpu = lexsort(keys_gpu, 1);

    // Verify that CPU and GPU results are the same
    EXPECT_TRUE(ins::all(idx_cpu == idx_gpu).item<bool>());

    // 展平后对比
    std::vector<int64_t> expected = { 2, 3, 0, 1,  0, 2, 1, 3,  2, 0, 1, 3 };
    expect_equal_gpu<int64_t>(idx_gpu.reshape(Shape({ 12 })), expected);
}

// ========== indices tests ==========

TEST_F(IndexingTestGPU, Indices2D) {
    Array idx = indices({ 2, 3 });

    EXPECT_EQ(idx.shape(), Shape({ 2, 2, 3 }));
    std::vector<int64_t> expected = {
        0, 0, 0, 1, 1, 1,
        0, 1, 2, 0, 1, 2
    };
    expect_equal_gpu<int64_t>(idx, expected);
}

// ========== ix_ tests ==========

TEST_F(IndexingTestGPU, Ix_) {
    std::vector<Array> grids = ix_({
        to_array(std::vector<int64_t>{0, 2}),
        to_array(std::vector<int64_t>{1, 3, 5})
        });

    EXPECT_EQ(grids.size(), 2);
    EXPECT_EQ(grids[0].shape(), Shape({ 2, 1 }));
    EXPECT_EQ(grids[1].shape(), Shape({ 1, 3 }));

    expect_equal_gpu<int64_t>(grids[0], { 0, 2 });
    expect_equal_gpu<int64_t>(grids[1], { 1, 3, 5 });
}

// ========== partition tests ==========

TEST_F(IndexingTestGPU, Partition1D) {
    std::vector<float> data = { 3, 1, 4, 1, 5, 9, 2, 6 };
    Array x_cpu = to_array(data);
    Array x = x_cpu.to(GPUPlace(0));
    Array p = partition(x, 3);

    Array cpu_p = p.to(CPUPlace());
    const float* p_data = cpu_p.data<float>();
    float kth_val = p_data[3];

    int less_count = 0;
    for (int i = 0; i < 8; ++i) {
        if (p_data[i] < kth_val) less_count++;
    }
    EXPECT_EQ(less_count, 3);
}

TEST_F(IndexingTestGPU, Partition2D) {
    std::vector<float> data = { 5, 2, 8, 1, 9, 3, 7, 4, 6, 0, 2, 5 };
    Array x_cpu = to_array(data, Shape({ 3, 4 }));
    Array x = x_cpu.to(GPUPlace(0));
    Array p = partition(x, 2, 1);

    Array cpu_p = p.to(CPUPlace());
    const float* p_data = cpu_p.data<float>();
    for (int row = 0; row < 3; ++row) {
        float kth_val = p_data[row * 4 + 2];
        int less_count = 0;
        for (int col = 0; col < 4; ++col) {
            if (p_data[row * 4 + col] < kth_val) less_count++;
        }
        EXPECT_EQ(less_count, 2);
    }
}

// ========== argpartition tests ==========

TEST_F(IndexingTestGPU, Argpartition1D) {
    std::vector<float> data = { 3, 1, 4, 1, 5, 9, 2, 6 };
    Array x_cpu = to_array(data,ins::CPUPlace());
    Array x_gpu = x_cpu.to(GPUPlace(0));
    Array ap_gpu = argpartition(x_gpu, 3);
    Array ap_cpu = argpartition(x_cpu, 3);

    // Verify that CPU and GPU results are the same
    EXPECT_TRUE(ins::all(ap_gpu == ap_cpu).item<bool>());

    Array cpu_ap = ap_gpu.to(CPUPlace());
    const int64_t* ap_data = cpu_ap.data<int64_t>();
    std::vector<bool> seen(8, false);
    for (int i = 0; i < 8; ++i) {
        EXPECT_GE(ap_data[i], 0);
        EXPECT_LT(ap_data[i], 8);
        seen[ap_data[i]] = true;
    }
    for (int i = 0; i < 8; ++i) {
        EXPECT_TRUE(seen[i]);
    }

    Array values_gpu = take(x_gpu, ap_gpu);
    Array values_cpu = take(x_cpu, ap_cpu);

    EXPECT_TRUE(ins::all(values_gpu == values_cpu).item<bool>());

    auto cpu_values = values_cpu.to(CPUPlace());
    auto val_data = cpu_values.data<float>();
    auto kth_val = val_data[3];

    auto less_count = 0;
    for (int i = 0; i < 8; ++i) {
        if (val_data[i] < kth_val) less_count++;
    }
    EXPECT_EQ(less_count, 3);

    cpu_values = values_gpu.to(CPUPlace());
    val_data = cpu_values.data<float>();
    kth_val = val_data[3];

    less_count = 0;
    for (int i = 0; i < 8; ++i) {
        if (val_data[i] < kth_val) less_count++;
    }
    EXPECT_EQ(less_count, 3);
}

TEST_F(IndexingTestGPU, Argpartition2D) {
    std::vector<float> data = { 5, 2, 8, 1, 9, 3, 7, 4, 6, 0, 2, 5 };
    Array x_cpu = to_array(data, Shape({ 3, 4 }));
    Array x = x_cpu.to(GPUPlace(0));
    Array ap = argpartition(x, 2, 1);

    Array values = take_along_axis(x, ap, 1);
    Array cpu_values = values.to(CPUPlace());
    const float* val_data = cpu_values.data<float>();
    for (int row = 0; row < 3; ++row) {
        float kth_val = val_data[row * 4 + 2];
        int less_count = 0;
        for (int col = 0; col < 4; ++col) {
            if (val_data[row * 4 + col] < kth_val) less_count++;
        }
        EXPECT_EQ(less_count, 2);
    }
}

// ========== interp tests ==========

TEST_F(IndexingTestGPU, InterpBasic) {
    Array xp_cpu = to_array({ 1.0, 2.0, 3.0, 4.0 });
    Array xp = xp_cpu.to(GPUPlace(0));
    Array fp_cpu = to_array({ 2.0, 4.0, 6.0, 8.0 });
    Array fp = fp_cpu.to(GPUPlace(0));
    Array x_cpu = to_array({ 1.5, 2.5, 3.5 });
    Array x = x_cpu.to(GPUPlace(0));
    Array y = interp(x, xp, fp);
    EXPECT_EQ(y.shape(), Shape({ 3 }));
    std::vector<double> expected = { 3.0, 5.0, 7.0 };
    expect_equal_gpu<double>(y, expected, 1e-6);
}

TEST_F(IndexingTestGPU, InterpNonUniform) {
    Array xp_cpu = to_array({ 1.0, 2.0, 4.0, 8.0 });
    Array xp = xp_cpu.to(GPUPlace(0));
    Array fp_cpu = to_array({ 10.0, 20.0, 40.0, 80.0 });
    Array fp = fp_cpu.to(GPUPlace(0));
    Array x_cpu = to_array({ 3.0, 5.0, 6.0 });
    Array x = x_cpu.to(GPUPlace(0));
    Array y = interp(x, xp, fp);
    std::vector<double> expected = { 30.0, 50.0, 60.0 };
    expect_equal_gpu<double>(y, expected, 1e-6);
}

TEST_F(IndexingTestGPU, InterpBoundaryDefault) {
    Array xp_cpu = to_array({ 10.0, 20.0, 30.0, 40.0 });
    Array xp = xp_cpu.to(GPUPlace(0));
    Array fp_cpu = to_array({ 100.0, 200.0, 300.0, 400.0 });
    Array fp = fp_cpu.to(GPUPlace(0));
    Array x_cpu = to_array({ 5.0, 15.0, 25.0, 35.0, 45.0 });
    Array x = x_cpu.to(GPUPlace(0));
    Array y = interp(x, xp, fp);
    std::vector<double> expected = { 100.0, 150.0, 250.0, 350.0, 400.0 };
    expect_equal_gpu<double>(y, expected, 1e-6);
}

TEST_F(IndexingTestGPU, InterpBoundaryCustom) {
    Array xp_cpu = to_array({ 10.0, 20.0, 30.0, 40.0 });
    Array xp = xp_cpu.to(GPUPlace(0));
    Array fp_cpu = to_array({ 100.0, 200.0, 300.0, 400.0 });
    Array fp = fp_cpu.to(GPUPlace(0));
    Array x_cpu = to_array({ 5.0, 15.0, 25.0, 35.0, 45.0 });
    Array x = x_cpu.to(GPUPlace(0));
    Array y = interp(x, xp, fp, 0.0, 999.0);
    std::vector<double> expected = { 0.0, 150.0, 250.0, 350.0, 999.0 };
    expect_equal_gpu<double>(y, expected, 1e-6);
}

TEST_F(IndexingTestGPU, InterpScalar) {
    Array xp_cpu = to_array({ 0.0, 1.0, 2.0, 3.0, 4.0 });
    Array xp = xp_cpu.to(GPUPlace(0));
    Array fp_cpu = to_array({ 0.0, 1.0, 4.0, 9.0, 16.0 });
    Array fp = fp_cpu.to(GPUPlace(0));
    Array x_cpu = to_array({ 2.5 });
    Array x = x_cpu.to(GPUPlace(0));
    Array y = interp(x, xp, fp);
    EXPECT_NEAR(gpu_item<double>(y), 6.5, 1e-6);
}

TEST_F(IndexingTestGPU, InterpUnsortedXp) {
    Array xp_cpu = to_array({ 4.0, 1.0, 3.0, 2.0 });
    Array xp = xp_cpu.to(GPUPlace(0));
    Array fp_cpu = to_array({ 40.0, 10.0, 30.0, 20.0 });
    Array fp = fp_cpu.to(GPUPlace(0));
    Array x_cpu = to_array({ 1.5, 2.5, 3.5 });
    Array x = x_cpu.to(GPUPlace(0));
    Array y = interp(x, xp, fp);
    std::vector<double> expected = { 15.0, 25.0, 35.0 };
    expect_equal_gpu<double>(y, expected, 1e-6);
}