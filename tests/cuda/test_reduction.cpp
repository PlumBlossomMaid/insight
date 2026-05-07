// tests/cuda/test_reduction.cu
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "insight/insight.h"

using namespace ins;

class ReductionTestGPU : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ins::init();
        set_device(ins::GPUPlace(0));
    }
};

// ========== Helper: Create test array ==========
static Array arange_2d(int rows, int cols, float start = 0.0f, float step = 1.0f) {
    std::vector<float> data(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        data[i] = start + i * step;
    }
    Array cpu_arr = to_array(data, Shape({ rows, cols }));
    return cpu_arr.to(GPUPlace(0));
}

static Array arange_3d(int d0, int d1, int d2, float start = 0.0f, float step = 1.0f) {
    std::vector<float> data(d0 * d1 * d2);
    for (int i = 0; i < d0 * d1 * d2; ++i) {
        data[i] = start + i * step;
    }
    Array cpu_arr = to_array(data, Shape({ d0, d1, d2 }));
    return cpu_arr.to(GPUPlace(0));
}

static Array ones_2d(int rows, int cols) {
    std::vector<float> data(rows * cols, 1.0f);
    Array cpu_arr = to_array(data, Shape({ rows, cols }));
    return cpu_arr.to(GPUPlace(0));
}

static Array zeros_2d(int rows, int cols) {
    std::vector<float> data(rows * cols, 0.0f);
    Array cpu_arr = to_array(data, Shape({ rows, cols }));
    return cpu_arr.to(GPUPlace(0));
}

// Helper to get item from GPU array
template<typename T>
T gpu_item(const Array& gpu_arr) {
    Array cpu_arr = gpu_arr.to(CPUPlace());
    return cpu_arr.item<T>();
}

// Helper to compare GPU array data with expected vector
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

// ========== Basic Reduction (2D) ==========

TEST_F(ReductionTestGPU, Sum2D) {
    Array x = arange_2d(3, 4);  // 3 rows, 4 cols: 0..11

    // Sum all (flatten)
    Array s = sum(x);
    EXPECT_NEAR(gpu_item<float>(s), 66.0f, 1e-5);

    // Sum along axis=0 (columns)
    Array s0 = sum(x, 0);
    EXPECT_EQ(s0.shape(), Shape({ 4 }));
    std::vector<float> expected_s0(4);
    for (int j = 0; j < 4; ++j) {
        expected_s0[j] = static_cast<float>(j + (j + 4) + (j + 8));
    }
    expect_equal_gpu<float>(s0, expected_s0);

    // Sum along axis=1 (rows)
    Array s1 = sum(x, 1);
    EXPECT_EQ(s1.shape(), Shape({ 3 }));
    std::vector<float> expected_s1 = {
        static_cast<float>(0 + 1 + 2 + 3),
        static_cast<float>(4 + 5 + 6 + 7),
        static_cast<float>(8 + 9 + 10 + 11)
    };
    expect_equal_gpu<float>(s1, expected_s1);

    // Sum with keepdim
    Array s1_keep = sum(x, 1, true);
    EXPECT_EQ(s1_keep.shape(), Shape({ 3, 1 }));
    std::vector<float> expected_keep = { 6.0f, 22.0f, 38.0f };
    expect_equal_gpu<float>(s1_keep, expected_keep);
}

TEST_F(ReductionTestGPU, Mean2D) {
    Array x = arange_2d(3, 4);

    Array m = mean(x);
    EXPECT_NEAR(gpu_item<float>(m), 5.5f, 1e-5);

    Array m0 = mean(x, 0);
    std::vector<float> expected_m0(4);
    for (int j = 0; j < 4; ++j) {
        expected_m0[j] = (j + (j + 4) + (j + 8)) / 3.0f;
    }
    expect_equal_gpu<float>(m0, expected_m0);

    Array m1 = mean(x, 1);
    std::vector<float> expected_m1 = {
        (0.0f + 1.0f + 2.0f + 3.0f) / 4.0f,
        (4.0f + 5.0f + 6.0f + 7.0f) / 4.0f,
        (8.0f + 9.0f + 10.0f + 11.0f) / 4.0f
    };
    expect_equal_gpu<float>(m1, expected_m1);
}

TEST_F(ReductionTestGPU, MaxMin2D) {
    Array x = arange_2d(3, 4);

    EXPECT_NEAR(gpu_item<float>(max(x)), 11.0f, 1e-5);
    EXPECT_NEAR(gpu_item<float>(min(x)), 0.0f, 1e-5);

    Array max0 = max(x, 0);
    std::vector<float> expected_max0 = { 8.0f, 9.0f, 10.0f, 11.0f };
    expect_equal_gpu<float>(max0, expected_max0);

    Array min0 = min(x, 0);
    std::vector<float> expected_min0 = { 0.0f, 1.0f, 2.0f, 3.0f };
    expect_equal_gpu<float>(min0, expected_min0);
}

TEST_F(ReductionTestGPU, Prod2D) {
    Array x = arange_2d(2, 3, 1.0f, 1.0f);  // [1,2,3; 4,5,6]

    EXPECT_NEAR(gpu_item<float>(prod(x)), 720.0f, 1e-5);

    Array p0 = prod(x, 0);
    std::vector<float> expected_p0 = { 1.0f * 4.0f, 2.0f * 5.0f, 3.0f * 6.0f };
    expect_equal_gpu<float>(p0, expected_p0);

    Array p1 = prod(x, 1);
    std::vector<float> expected_p1 = { 1.0f * 2.0f * 3.0f, 4.0f * 5.0f * 6.0f };
    expect_equal_gpu<float>(p1, expected_p1);
}

TEST_F(ReductionTestGPU, AnyAll2D) {
    Array x = arange_2d(3, 4);
    Array zeros = zeros_2d(3, 4);

    EXPECT_TRUE(gpu_item<bool>(any(x)));
    EXPECT_FALSE(gpu_item<bool>(any(zeros)));
    EXPECT_FALSE(gpu_item<bool>(all(x)));

    Array ones = ones_2d(3, 4);
    EXPECT_TRUE(gpu_item<bool>(all(ones)));

    Array any0 = any(x, 0);
    std::vector<bool> expected_any0 = { true, true, true, true };
    expect_equal_gpu<bool>(any0, expected_any0);
}

// ========== ArgMax/ArgMin ==========

TEST_F(ReductionTestGPU, ArgMaxArgMin2D) {
    Array x = arange_2d(3, 4);  // [[0,1,2,3],[4,5,6,7],[8,9,10,11]]

    Array amax0 = argmax(x, 0);
    std::vector<int64_t> expected_amax0 = { 2, 2, 2, 2 };
    expect_equal_gpu<int64_t>(amax0, expected_amax0);

    Array amax1 = argmax(x, 1);
    std::vector<int64_t> expected_amax1 = { 3, 3, 3 };
    expect_equal_gpu<int64_t>(amax1, expected_amax1);

    Array amin0 = argmin(x, 0);
    std::vector<int64_t> expected_amin0 = { 0, 0, 0, 0 };
    expect_equal_gpu<int64_t>(amin0, expected_amin0);

    Array amin1 = argmin(x, 1);
    std::vector<int64_t> expected_amin1 = { 0, 0, 0 };
    expect_equal_gpu<int64_t>(amin1, expected_amin1);
}

// ========== Var/Std ==========

TEST_F(ReductionTestGPU, VarStd2D) {
    Array x = arange_2d(2, 4, 0.0f, 1.0f);  // [0,1,2,3; 4,5,6,7]

    Array var0 = var(x, std::nullopt, false, 0);
    EXPECT_NEAR(gpu_item<float>(var0), 5.25f, 1e-5);

    Array var1 = var(x, std::nullopt, false, 1);
    EXPECT_NEAR(gpu_item<float>(var1), 6.0f, 1e-5);

    Array std0 = ins::std(x);
    EXPECT_NEAR(gpu_item<float>(std0), std::sqrt(5.25f), 1e-5);

    Array var_axis0 = var(x, 0, false, 1);
    std::vector<float> expected_var = { 8.0f, 8.0f, 8.0f, 8.0f };
    expect_equal_gpu<float>(var_axis0, expected_var);
}

// ========== Cumulative Operations ==========

TEST_F(ReductionTestGPU, CumSumCumProd2D) {
    Array x = arange_2d(2, 5, 1.0f, 1.0f);  // [1,2,3,4,5; 6,7,8,9,10]

    Array cs = cumsum(x, 1);
    std::vector<double> expected_cs = { 1.0, 3.0, 6.0, 10.0, 15.0, 6.0, 13.0, 21.0, 30.0, 40.0 };
    expect_equal_gpu<double>(cs, expected_cs, 1e-8);

    Array cp = cumprod(x, 1, DType::F64);
    std::vector<double> expected_cp = { 1.0, 2.0, 6.0, 24.0, 120.0, 6.0, 42.0, 336.0, 3024.0, 30240.0 };
    expect_equal_gpu<double>(cp, expected_cp, 1e-8);
}

TEST_F(ReductionTestGPU, CumMaxCumMin2D) {
    Array x_cpu = to_array({ 3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f }, Shape({ 2, 4 }));
    Array x = x_cpu.to(GPUPlace(0));

    Array cmax = cummax(x, 1);
    std::vector<float> expected_cmax = { 3.0f, 3.0f, 4.0f, 4.0f, 5.0f, 9.0f, 9.0f, 9.0f };
    expect_equal_gpu<float>(cmax, expected_cmax);

    Array cmin = cummin(x, 1);
    std::vector<float> expected_cmin = { 3.0f, 1.0f, 1.0f, 1.0f, 5.0f, 5.0f, 2.0f, 2.0f };
    expect_equal_gpu<float>(cmin, expected_cmin);
}

// ========== Median/Quantile ==========

TEST_F(ReductionTestGPU, Median2D) {
    std::vector<float> data1 = { 1.0f, 3.0f, 2.0f, 5.0f, 4.0f };
    Array x1_cpu = to_array(data1);
    Array x1 = x1_cpu.to(GPUPlace(0));
    Array m1 = median(x1);
    std::vector<double> expected_m1 = { 3.0 };
    expect_equal_gpu<double>(m1, expected_m1, 1e-8);

    std::vector<float> data2 = { 1.0f, 3.0f, 2.0f, 4.0f };
    Array x2_cpu = to_array(data2);
    Array x2 = x2_cpu.to(GPUPlace(0));
    Array m2 = median(x2);
    std::vector<double> expected_m2 = { 2.5 };
    expect_equal_gpu<double>(m2, expected_m2, 1e-8);

    Array x3 = arange_2d(3, 4);
    Array med = median(x3, 0);
    std::vector<double> expected_med = { 4.0, 5.0, 6.0, 7.0 };
    expect_equal_gpu<double>(med, expected_med, 1e-8);
}

TEST_F(ReductionTestGPU, Quantile2D) {
    if (!is_compiled_with_thrust()) {
        GTEST_SKIP() << "Thrust not available, quantile uses CPU fallback";
    }
    std::vector<float> data = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
    Array x_cpu = to_array(data).reshape(Shape({ 8 }));
    Array x = x_cpu.to(GPUPlace(0));

    Array q25 = quantile(x, 0.25);
    std::vector<double> expected_q25 = { 2.75 };
    expect_equal_gpu<double>(q25, expected_q25, 1e-8);

    Array q50 = median(x);
    std::vector<double> expected_q50 = { 4.5 };
    expect_equal_gpu<double>(q50, expected_q50, 1e-8);

    Array q75 = quantile(x, 0.75);
    std::vector<double> expected_q75 = { 6.25 };
    expect_equal_gpu<double>(q75, expected_q75, 1e-8);
}

// ========== Count Nonzero ==========

TEST_F(ReductionTestGPU, CountNonzero) {
    std::vector<float> data = { 0.0f, 1.0f, 0.0f, 2.0f, 3.0f, 0.0f };
    Array x_cpu = to_array(data).reshape(Shape({ 2, 3 }));
    Array x = x_cpu.to(GPUPlace(0));

    Array cnt = count_nonzero(x);
    EXPECT_EQ(gpu_item<int64_t>(cnt), 3);

    Array cnt0 = count_nonzero(x, 0);
    std::vector<int64_t> expected_cnt0 = { 1, 2, 0 };
    expect_equal_gpu<int64_t>(cnt0, expected_cnt0);

    Array cnt1 = count_nonzero(x, 1);
    std::vector<int64_t> expected_cnt1 = { 1, 2 };
    expect_equal_gpu<int64_t>(cnt1, expected_cnt1);
}

// ========== NaN-safe Operations ==========

TEST_F(ReductionTestGPU, NanSumNanMean) {
    std::vector<float> data = { 1.0f, 2.0f, std::nanf(""), 4.0f, 5.0f };
    Array x_cpu = to_array(data, Shape({ 5 }));
    Array x = x_cpu.to(GPUPlace(0));

    Array s = nansum(x);
    EXPECT_NEAR(gpu_item<float>(s), 12.0f, 1e-5);

    Array m = nanmean(x);
    EXPECT_NEAR(gpu_item<float>(m), 3.0f, 1e-5);
}

TEST_F(ReductionTestGPU, NanMaxNanMin) {
    std::vector<float> data = { 1.0f, std::nanf(""), 3.0f, std::nanf(""), 5.0f };
    Array x_cpu = to_array(data, Shape({ 5 }));
    Array x = x_cpu.to(GPUPlace(0));

    EXPECT_NEAR(gpu_item<float>(nanmax(x)), 5.0f, 1e-5);
    EXPECT_NEAR(gpu_item<float>(nanmin(x)), 1.0f, 1e-5);
}

TEST_F(ReductionTestGPU, NanVarNanStd) {
    std::vector<float> data = { 1.0f, std::nanf(""), 3.0f, 5.0f, std::nanf("") };
    Array x_cpu = to_array(data, Shape({ 5 }));
    Array x = x_cpu.to(GPUPlace(0));

    Array v = nanvar(x, std::nullopt, false, 1);
    EXPECT_NEAR(gpu_item<float>(v), 4.0f, 1e-5);

    Array s = nanstd(x, std::nullopt, false, 1);
    EXPECT_NEAR(gpu_item<float>(s), 2.0f, 1e-5);

    Array v_pop = nanvar(x, std::nullopt, false, 0);
    EXPECT_NEAR(gpu_item<float>(v_pop), 8.0f / 3.0f, 1e-5);

    Array s_pop = nanstd(x);
    EXPECT_NEAR(gpu_item<float>(s_pop), std::sqrt(8.0f / 3.0f), 1e-5);
}

TEST_F(ReductionTestGPU, NanMedianNanQuantile) {
    if (!is_compiled_with_thrust()) {
        GTEST_SKIP() << "Thrust not available, quantile uses CPU fallback";
    }
    std::vector<float> data = { 1.0f, std::nanf(""), 3.0f, std::nanf(""), 5.0f };
    Array x_cpu = to_array(data, Shape({ 5 }));
    Array x = x_cpu.to(GPUPlace(0));

    EXPECT_NEAR(gpu_item<float>(nanmedian(x)), 3.0, 1e-5);
    EXPECT_NEAR(gpu_item<float>(nanquantile(x, 0.25)), 2.0, 1e-5);
}

// ========== 3D Tensors ==========

TEST_F(ReductionTestGPU, Sum3D) {
    Array x = arange_3d(2, 3, 4);  // values 0..23

    Array s0 = sum(x, 0);
    EXPECT_EQ(s0.shape(), Shape({ 3, 4 }));
    std::vector<float> expected_s0(12);
    for (int i = 0; i < 12; ++i) {
        expected_s0[i] = static_cast<float>(i + (i + 12));
    }
    expect_equal_gpu<float>(s0, expected_s0);

    Array s1 = sum(x, 1);
    EXPECT_EQ(s1.shape(), Shape({ 2, 4 }));

    Array s2 = sum(x, 2);
    EXPECT_EQ(s2.shape(), Shape({ 2, 3 }));
    std::vector<float> expected_s2(6);
    for (int i = 0; i < 6; ++i) {
        expected_s2[i] = static_cast<float>(i * 4 + (i * 4 + 1) + (i * 4 + 2) + (i * 4 + 3));
    }
    expect_equal_gpu<float>(s2, expected_s2);
}

TEST_F(ReductionTestGPU, CumSum3D) {
    Array x = arange_3d(2, 2, 3);  // slice0: [0,1,2; 3,4,5]; slice1: [6,7,8; 9,10,11]

    Array cs = cumsum(x, 2);
    std::vector<double> expected_cs = { 0.0, 1.0, 3.0, 3.0, 7.0, 12.0, 6.0, 13.0, 21.0, 9.0, 19.0, 30.0 };
    expect_equal_gpu<double>(cs, expected_cs, 1e-8);
}

// ========== Keepdim Flag ==========

TEST_F(ReductionTestGPU, KeepdimFlag) {
    Array x = arange_2d(3, 4);

    Array s_no_keep = sum(x, 0, false);
    EXPECT_EQ(s_no_keep.shape(), Shape({ 4 }));

    Array s_keep = sum(x, 0, true);
    EXPECT_EQ(s_keep.shape(), Shape({ 1, 4 }));

    Array m_no_keep = mean(x, 1, false);
    EXPECT_EQ(m_no_keep.shape(), Shape({ 3 }));

    Array m_keep = mean(x, 1, true);
    EXPECT_EQ(m_keep.shape(), Shape({ 3, 1 }));
}

// ========== Dtype Consistency ==========

TEST_F(ReductionTestGPU, DtypeConsistency) {
    std::vector<int> data = { 1, 2, 3, 4, 5 };
    Array x_cpu = to_array(data, Shape({ 5 }));
    Array x = x_cpu.to(GPUPlace(0));

    Array s = sum(x);
    EXPECT_EQ(s.dtype(), DType::I32);
    EXPECT_EQ(gpu_item<int32_t>(s), 15);

    Array m = mean(x);
    EXPECT_EQ(m.dtype(), DType::F64);
    EXPECT_NEAR(gpu_item<double>(m), 3.0, 1e-5);
}