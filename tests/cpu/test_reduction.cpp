// tests/cpu/test_reduction.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "insight/insight.h"

using namespace ins;

class ReductionTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ins::init();
    }
};

// ========== Helper: Create test array ==========
static Array arange_2d(int rows, int cols, float start = 0.0f, float step = 1.0f) {
    std::vector<float> data(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        data[i] = start + i * step;
    }
    return to_array(data, Shape({ rows, cols }));
}

static Array arange_3d(int d0, int d1, int d2, float start = 0.0f, float step = 1.0f) {
    std::vector<float> data(d0 * d1 * d2);
    for (int i = 0; i < d0 * d1 * d2; ++i) {
        data[i] = start + i * step;
    }
    return to_array(data, Shape({ d0, d1, d2 }));
}

static Array ones_2d(int rows, int cols) {
    std::vector<float> data(rows * cols, 1.0f);
    return to_array(data, Shape({ rows, cols }));
}

static Array zeros_2d(int rows, int cols) {
    std::vector<float> data(rows * cols, 0.0f);
    return to_array(data, Shape({ rows, cols }));
}

// ========== Basic Reduction (2D) ==========

TEST_F(ReductionTest, Sum2D) {
    Array x = arange_2d(3, 4);  // 3 rows, 4 cols: 0..11

    // Sum all (flatten)
    Array s = sum(x);
    EXPECT_NEAR(s.item<float>(), 66.0f, 1e-5);

    // Sum along axis=0 (columns)
    Array s0 = sum(x, 0);
    EXPECT_EQ(s0.shape(), Shape({ 4 }));
    const float* data = s0.data<float>();
    for (int j = 0; j < 4; ++j) {
        float expected = static_cast<float>(j) + (j + 4) + (j + 8);
        EXPECT_NEAR(data[j], expected, 1e-5);
    }

    // Sum along axis=1 (rows)
    Array s1 = sum(x, 1);
    EXPECT_EQ(s1.shape(), Shape({ 3 }));
    const float* data1 = s1.data<float>();
    EXPECT_NEAR(data1[0], 0 + 1 + 2 + 3, 1e-5);
    EXPECT_NEAR(data1[1], 4 + 5 + 6 + 7, 1e-5);
    EXPECT_NEAR(data1[2], 8 + 9 + 10 + 11, 1e-5);

    // Sum with keepdim
    Array s1_keep = sum(x, 1, true);
    EXPECT_EQ(s1_keep.shape(), Shape({ 3, 1 }));
    const float* data1_keep = s1_keep.data<float>();
    EXPECT_NEAR(data1_keep[0], 6, 1e-5);
    EXPECT_NEAR(data1_keep[1], 22, 1e-5);
    EXPECT_NEAR(data1_keep[2], 38, 1e-5);
}

TEST_F(ReductionTest, Mean2D) {
    Array x = arange_2d(3, 4);

    Array m = mean(x);
    EXPECT_NEAR(m.item<float>(), 5.5f, 1e-5);

    Array m0 = mean(x, 0);
    const float* data = m0.data<float>();
    for (int j = 0; j < 4; ++j) {
        float expected = (j + (j + 4) + (j + 8)) / 3.0f;
        EXPECT_NEAR(data[j], expected, 1e-5);
    }

    Array m1 = mean(x, 1);
    const float* data1 = m1.data<float>();
    EXPECT_NEAR(data1[0], (0 + 1 + 2 + 3) / 4.0f, 1e-5);
}

TEST_F(ReductionTest, MaxMin2D) {
    Array x = arange_2d(3, 4);

    EXPECT_NEAR(max(x).item<float>(), 11.0f, 1e-5);
    EXPECT_NEAR(min(x).item<float>(), 0.0f, 1e-5);

    Array max0 = max(x, 0);
    const float* max_data = max0.data<float>();
    EXPECT_NEAR(max_data[0], 8.0f, 1e-5);
    EXPECT_NEAR(max_data[3], 11.0f, 1e-5);

    Array min0 = min(x, 0);
    const float* min_data = min0.data<float>();
    EXPECT_NEAR(min_data[0], 0.0f, 1e-5);
}

TEST_F(ReductionTest, Prod2D) {
    Array x = arange_2d(2, 3, 1.0f, 1.0f);  // [1,2,3; 4,5,6]

    EXPECT_NEAR(prod(x).item<float>(), 720.0f, 1e-5);

    Array p0 = prod(x, 0);
    const float* data = p0.data<float>();
    EXPECT_NEAR(data[0], 1 * 4, 1e-5);
    EXPECT_NEAR(data[1], 2 * 5, 1e-5);
    EXPECT_NEAR(data[2], 3 * 6, 1e-5);

    Array p1 = prod(x, 1);
    const float* data1 = p1.data<float>();
    EXPECT_NEAR(data1[0], 1 * 2 * 3, 1e-5);
    EXPECT_NEAR(data1[1], 4 * 5 * 6, 1e-5);
}

TEST_F(ReductionTest, AnyAll2D) {
    Array x = arange_2d(3, 4);
    Array zeros = zeros_2d(3, 4);

    EXPECT_TRUE(any(x).item<bool>());
    EXPECT_FALSE(any(zeros).item<bool>());
    EXPECT_FALSE(all(x).item<bool>());

    Array ones = ones_2d(3, 4);
    EXPECT_TRUE(all(ones).item<bool>());

    Array any0 = any(x, 0);
    const bool* any_data = any0.data<bool>();
    for (int j = 0; j < 4; ++j) {
        EXPECT_TRUE(any_data[j]);
    }
}

// ========== ArgMax/ArgMin ==========

TEST_F(ReductionTest, ArgMaxArgMin2D) {
    Array x = arange_2d(3, 4);  // [[0,1,2,3],[4,5,6,7],[8,9,10,11]]

    Array amax0 = argmax(x, 0);
    const int64_t* amax_data = amax0.data<int64_t>();
    for (int j = 0; j < 4; ++j) {
        EXPECT_EQ(amax_data[j], 2);
    }

    Array amax1 = argmax(x, 1);
    const int64_t* amax1_data = amax1.data<int64_t>();
    EXPECT_EQ(amax1_data[0], 3);
    EXPECT_EQ(amax1_data[1], 3);
    EXPECT_EQ(amax1_data[2], 3);

    Array amin0 = argmin(x, 0);
    const int64_t* amin_data = amin0.data<int64_t>();
    for (int j = 0; j < 4; ++j) {
        EXPECT_EQ(amin_data[j], 0);
    }

    Array amin1 = argmin(x, 1);
    const int64_t* amin1_data = amin1.data<int64_t>();
    EXPECT_EQ(amin1_data[0], 0);
    EXPECT_EQ(amin1_data[1], 0);
    EXPECT_EQ(amin1_data[2], 0);
}

// ========== Var/Std ==========

TEST_F(ReductionTest, VarStd2D) {
    Array x = arange_2d(2, 4, 0.0f, 1.0f);  // [0,1,2,3; 4,5,6,7]

    Array var0 = var(x, std::nullopt, false, 0);
    EXPECT_NEAR(var0.item<float>(), 5.25f, 1e-5);

    Array var1 = var(x, std::nullopt, false, 1);
    EXPECT_NEAR(var1.item<float>(), 6.0f, 1e-5);

    Array std0 = ins::std(x);
    EXPECT_NEAR(std0.item<float>(), std::sqrt(5.25f), 1e-5);

    Array var_axis0 = var(x, 0, false, 1);
    const float* var_data = var_axis0.data<float>();
    for (int j = 0; j < 4; ++j) {
        EXPECT_NEAR(var_data[j], 8.0f, 1e-5);
    }
}

// ========== Cumulative Operations ==========

TEST_F(ReductionTest, CumSumCumProd2D) {
    Array x = arange_2d(2, 5, 1.0f, 1.0f);  // [1,2,3,4,5; 6,7,8,9,10]

    Array cs = cumsum(x, 1);
    const double* cs_data = cs.data<double>();
    EXPECT_NEAR(cs_data[0], 1.0, 1e-8);
    EXPECT_NEAR(cs_data[1], 3.0, 1e-8);
    EXPECT_NEAR(cs_data[2], 6.0, 1e-8);
    EXPECT_NEAR(cs_data[3], 10.0, 1e-8);
    EXPECT_NEAR(cs_data[4], 15.0, 1e-8);
    EXPECT_NEAR(cs_data[5], 6.0, 1e-8);
    EXPECT_NEAR(cs_data[6], 13.0, 1e-8);
    EXPECT_NEAR(cs_data[7], 21.0, 1e-8);
    EXPECT_NEAR(cs_data[8], 30.0, 1e-8);
    EXPECT_NEAR(cs_data[9], 40.0, 1e-8);

    Array cp = cumprod(x, 1, DType::F64);
    const double* cp_data = cp.data<double>();
    EXPECT_NEAR(cp_data[0], 1.0, 1e-8);
    EXPECT_NEAR(cp_data[1], 2.0, 1e-8);
    EXPECT_NEAR(cp_data[2], 6.0, 1e-8);
    EXPECT_NEAR(cp_data[3], 24.0, 1e-8);
    EXPECT_NEAR(cp_data[4], 120.0, 1e-8);
}

TEST_F(ReductionTest, CumMaxCumMin2D) {
    Array x = to_array({ 3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f }, Shape({ 2, 4 }));

    Array cmax = cummax(x, 1);
    const float* cmax_data = cmax.data<float>();
    EXPECT_NEAR(cmax_data[0], 3.0f, 1e-5);
    EXPECT_NEAR(cmax_data[1], 3.0f, 1e-5);
    EXPECT_NEAR(cmax_data[2], 4.0f, 1e-5);
    EXPECT_NEAR(cmax_data[3], 4.0f, 1e-5);
    EXPECT_NEAR(cmax_data[4], 5.0f, 1e-5);
    EXPECT_NEAR(cmax_data[5], 9.0f, 1e-5);
    EXPECT_NEAR(cmax_data[6], 9.0f, 1e-5);
    EXPECT_NEAR(cmax_data[7], 9.0f, 1e-5);

    Array cmin = cummin(x, 1);
    const float* cmin_data = cmin.data<float>();
    EXPECT_NEAR(cmin_data[0], 3.0f, 1e-5);
    EXPECT_NEAR(cmin_data[1], 1.0f, 1e-5);
    EXPECT_NEAR(cmin_data[2], 1.0f, 1e-5);
    EXPECT_NEAR(cmin_data[3], 1.0f, 1e-5);
}

// ========== Median/Quantile ==========

TEST_F(ReductionTest, Median2D) {
    std::vector<float> data1 = { 1.0f, 3.0f, 2.0f, 5.0f, 4.0f };
    Array x1 = to_array(data1);
    Array m1 = median(x1);
    const double* m1_data = m1.data<double>();
    EXPECT_NEAR(m1_data[0], 3.0, 1e-8);

    std::vector<float> data2 = { 1.0f, 3.0f, 2.0f, 4.0f };
    Array x2 = to_array(data2);
    Array m2 = median(x2);
    const double* m2_data = m2.data<double>();
    EXPECT_NEAR(m2_data[0], 2.5, 1e-8);

    Array x3 = arange_2d(3, 4);
    Array med = median(x3, 0);
    const double* med_data = med.data<double>();
    EXPECT_NEAR(med_data[0], 4.0, 1e-8);
    EXPECT_NEAR(med_data[1], 5.0, 1e-8);
    EXPECT_NEAR(med_data[2], 6.0, 1e-8);
    EXPECT_NEAR(med_data[3], 7.0, 1e-8);
}

TEST_F(ReductionTest, Quantile2D) {
    std::vector<float> data = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
    Array x = to_array(data).reshape(Shape({ 8 }));

    Array q25 = quantile(x, 0.25);
    const double* q25_data = q25.data<double>();
    EXPECT_NEAR(q25_data[0], 2.75, 1e-8);

    Array q50 = median(x);
    const double* q50_data = q50.data<double>();
    EXPECT_NEAR(q50_data[0], 4.5, 1e-8);

    Array q75 = quantile(x, 0.75);
    const double* q75_data = q75.data<double>();
    EXPECT_NEAR(q75_data[0], 6.25, 1e-8);
}

// ========== Count Nonzero ==========

TEST_F(ReductionTest, CountNonzero) {
    std::vector<float> data = { 0.0f, 1.0f, 0.0f, 2.0f, 3.0f, 0.0f };
    Array x = to_array(data).reshape(Shape({ 2, 3 }));

    Array cnt = count_nonzero(x);
    EXPECT_EQ(cnt.item<int64_t>(), 3);

    Array cnt0 = count_nonzero(x, 0);
    const int64_t* cnt0_data = cnt0.data<int64_t>();
    EXPECT_EQ(cnt0_data[0], 1);
    EXPECT_EQ(cnt0_data[1], 2);
    EXPECT_EQ(cnt0_data[2], 0);

    Array cnt1 = count_nonzero(x, 1);
    const int64_t* cnt1_data = cnt1.data<int64_t>();
    EXPECT_EQ(cnt1_data[0], 1);
    EXPECT_EQ(cnt1_data[1], 2);
}

// ========== NaN-safe Operations ==========

TEST_F(ReductionTest, NanSumNanMean) {
    std::vector<float> data = { 1.0f, 2.0f, std::nanf(""), 4.0f, 5.0f };
    Array x = to_array(data, Shape({ 5 }));

    Array s = nansum(x);
    EXPECT_NEAR(s.item<float>(), 12.0f, 1e-5);

    Array m = nanmean(x);
    EXPECT_NEAR(m.item<float>(), 3.0f, 1e-5);
}

TEST_F(ReductionTest, NanMaxNanMin) {
    std::vector<float> data = { 1.0f, std::nanf(""), 3.0f, std::nanf(""), 5.0f };
    Array x = to_array(data, Shape({ 5 }));

    EXPECT_NEAR(nanmax(x).item<float>(), 5.0f, 1e-5);
    EXPECT_NEAR(nanmin(x).item<float>(), 1.0f, 1e-5);
}

TEST_F(ReductionTest, NanVarNanStd) {
    std::vector<float> data = { 1.0f, std::nanf(""), 3.0f, 5.0f, std::nanf("") };
    Array x = to_array(data, Shape({ 5 }));

    Array v = nanvar(x, std::nullopt, false, 1);
    EXPECT_NEAR(v.item<float>(), 4.0f, 1e-5);

    Array s = nanstd(x, std::nullopt, false, 1);
    EXPECT_NEAR(s.item<float>(), 2.0f, 1e-5);

    Array v_pop = nanvar(x, std::nullopt, false, 0);
    EXPECT_NEAR(v_pop.item<float>(), 8.0f / 3.0f, 1e-5);

    Array s_pop = nanstd(x);
    EXPECT_NEAR(s_pop.item<float>(), std::sqrt(8.0f / 3.0f), 1e-5);
}

TEST_F(ReductionTest, NanMedianNanQuantile) {
    std::vector<float> data = { 1.0f, std::nanf(""), 3.0f, std::nanf(""), 5.0f };
    Array x = to_array(data, Shape({ 5 }));
    EXPECT_NEAR(nanmedian(x).item<float>(), 3.0, 1e-5);
    EXPECT_NEAR(nanquantile(x, 0.25).item<float>(), 2.0, 1e-5);
}

// ========== 3D Tensors ==========

TEST_F(ReductionTest, Sum3D) {
    Array x = arange_3d(2, 3, 4);  // values 0..23

    Array s0 = sum(x, 0);
    EXPECT_EQ(s0.shape(), Shape({ 3, 4 }));
    const float* s0_data = s0.data<float>();
    for (int i = 0; i < 12; ++i) {
        EXPECT_NEAR(s0_data[i], static_cast<float>(i + (i + 12)), 1e-5);
    }

    Array s1 = sum(x, 1);
    EXPECT_EQ(s1.shape(), Shape({ 2, 4 }));

    Array s2 = sum(x, 2);
    EXPECT_EQ(s2.shape(), Shape({ 2, 3 }));
    const float* s2_data = s2.data<float>();
    for (int i = 0; i < 6; ++i) {
        float expected = static_cast<float>(i * 4 + (i * 4 + 1) + (i * 4 + 2) + (i * 4 + 3));
        EXPECT_NEAR(s2_data[i], expected, 1e-5);
    }
}

TEST_F(ReductionTest, CumSum3D) {
    Array x = arange_3d(2, 2, 3);  // slice0: [0,1,2; 3,4,5]; slice1: [6,7,8; 9,10,11]

    Array cs = cumsum(x, 2);
    const double* cs_data = cs.data<double>();

    EXPECT_NEAR(cs_data[0], 0.0, 1e-8);
    EXPECT_NEAR(cs_data[1], 1.0, 1e-8);
    EXPECT_NEAR(cs_data[2], 3.0, 1e-8);
    EXPECT_NEAR(cs_data[3], 3.0, 1e-8);
    EXPECT_NEAR(cs_data[4], 7.0, 1e-8);
    EXPECT_NEAR(cs_data[5], 12.0, 1e-8);
    EXPECT_NEAR(cs_data[6], 6.0, 1e-8);
    EXPECT_NEAR(cs_data[7], 13.0, 1e-8);
    EXPECT_NEAR(cs_data[8], 21.0, 1e-8);
    EXPECT_NEAR(cs_data[9], 9.0, 1e-8);
    EXPECT_NEAR(cs_data[10], 19.0, 1e-8);
    EXPECT_NEAR(cs_data[11], 30.0, 1e-8);
}

// ========== Keepdim Flag ==========

TEST_F(ReductionTest, KeepdimFlag) {
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

TEST_F(ReductionTest, DtypeConsistency) {
    std::vector<int> data = { 1, 2, 3, 4, 5 };
    Array x = to_array(data, Shape({ 5 }));

    Array s = sum(x);
    EXPECT_EQ(s.dtype(), DType::I32);
    EXPECT_EQ(s.item<int32_t>(), 15);

    Array m = mean(x);
    EXPECT_EQ(m.dtype(), DType::F64);
    EXPECT_NEAR(m.item<double>(), 3.0, 1e-5);
}