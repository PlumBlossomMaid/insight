// tests/cpu/test_signal.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "insight/insight.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif
using namespace ins;

class SignalTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ins::init();
        set_device(CPUPlace());
    }
};

// ========== unwrap tests ==========

TEST_F(SignalTest, UnwrapBasic) {
    std::vector<double> data = { 0.0, M_PI_2, M_PI, 3.0 * M_PI_2, 2.0 * M_PI, 3.0 * M_PI };
    Array a = to_array(data);
    Array u = unwrap(a);
    const double* u_data = u.data<double>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(u_data[i], data[i], 1e-6);
    }
}

TEST_F(SignalTest, UnwrapWithJumps) {
    std::vector<double> data = { 0.0, 0.1, 3.2, 3.3, 6.4, 6.5 };
    Array a = to_array(data);
    Array u = unwrap(a);
    // 默认阈值 π，3.2 不会触发跳变
    const double* u_data = u.data<double>();
    EXPECT_NEAR(u_data[0], 0.0, 1e-6);
    EXPECT_NEAR(u_data[1], 0.1, 1e-6);
    EXPECT_NEAR(u_data[2], 3.2, 1e-6);
    EXPECT_NEAR(u_data[3], 3.3, 1e-6);
    EXPECT_NEAR(u_data[4], 6.4, 1e-6);
    EXPECT_NEAR(u_data[5], 6.5, 1e-6);
}

TEST_F(SignalTest, UnwrapScalar) {
    Array a = to_array({ M_PI });
    Array u = unwrap(a);
    EXPECT_NEAR(u.item<double>(), M_PI, 1e-6);
}

TEST_F(SignalTest, Unwrap2DAxis1) {
    std::vector<double> data = {
        0.0, M_PI_2, M_PI, 3.0 * M_PI_2,
        0.1, M_PI_2, 3.2, 3.3
    };
    Array m = to_array(data, Shape({ 2, 4 }));
    Array u = unwrap(m, 1);
    EXPECT_EQ(u.shape(), Shape({ 2, 4 }));
    const double* u_data = u.data<double>();
    // 第一行：单调递增，不变
    EXPECT_NEAR(u_data[0], 0.0, 1e-6);
    EXPECT_NEAR(u_data[1], M_PI_2, 1e-6);
    EXPECT_NEAR(u_data[2], M_PI, 1e-6);
    EXPECT_NEAR(u_data[3], 3.0 * M_PI_2, 1e-6);
    // 第二行可能有调整
    EXPECT_NEAR(u_data[4], 0.1, 1e-6);
    EXPECT_NEAR(u_data[5], M_PI_2, 1e-6);
    EXPECT_NEAR(u_data[6], 3.2, 1e-6);
    EXPECT_NEAR(u_data[7], 3.3, 1e-6);
}

TEST_F(SignalTest, Unwrap2DAxis0) {
    std::vector<double> data = {
        0.0, M_PI_2, M_PI, 3.0 * M_PI_2,
        0.1, M_PI_2, 3.2, 3.3
    };
    Array m = to_array(data, Shape({ 2, 4 }));
    Array u = unwrap(m, 0);
    EXPECT_EQ(u.shape(), Shape({ 2, 4 }));
}

TEST_F(SignalTest, UnwrapCustomParams) {
    std::vector<double> data = { 0.0, 0.5, 1.0, 0.2, 0.7, 1.2 };
    Array a = to_array(data);
    Array u = unwrap(a, -1, 0.3, 1.0);
    const double* u_data = u.data<double>();
    // 阈值 0.3，周期 1.0，检测跳变并修正
    EXPECT_NEAR(u_data[0], 0.0, 1e-6);
    EXPECT_NEAR(u_data[1], -0.5, 1e-6);
    EXPECT_NEAR(u_data[2], -1.0, 1e-6);
    EXPECT_NEAR(u_data[3], -0.8, 1e-6);
    EXPECT_NEAR(u_data[4], -1.3, 1e-6);
    EXPECT_NEAR(u_data[5], -1.8, 1e-6);
}

TEST_F(SignalTest, UnwrapNegativeAxis) {
    std::vector<float> data = { 0.0f, 3.5f, 0.1f, 3.4f, 0.2f, 3.3f };
    Array a = to_array(data, Shape({ 2, 3 }));
    Array u = unwrap(a, -1);
    EXPECT_EQ(u.shape(), Shape({ 2, 3 }));
}

TEST_F(SignalTest, UnwrapContinuity) {
    // 创建连续相位，然后取模 2π，验证 unwrap 能恢复
    int n = 20;
    std::vector<double> original(n);
    std::vector<double> wrapped(n);
    for (int i = 0; i < n; ++i) {
        original[i] = static_cast<double>(i) * 0.6;
        wrapped[i] = std::fmod(original[i], 2.0 * M_PI);
    }
    Array orig = to_array(original);
    Array wrapped_arr = to_array(wrapped);
    Array recovered = unwrap(wrapped_arr);
    const double* orig_data = orig.data<double>();
    const double* rec_data = recovered.data<double>();
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(rec_data[i], orig_data[i], 1e-6);
    }
}

TEST_F(SignalTest, Sinc) {
    std::vector<float> data = { 0.0f, 1.0f, 2.0f, -1.0f, -2.0f, 0.5f };
    Array x = to_array(data);
    Array y = ins::sinc(x);

    const float* y_data = y.data<float>();
    // sinc(0) = 1
    EXPECT_NEAR(y_data[0], 1.0f, 1e-5);
    // sinc(1) = sin(π)/π = 0/π = 0
    EXPECT_NEAR(y_data[1], 0.0f, 1e-5);
    // sinc(2) = sin(2π)/2π = 0
    EXPECT_NEAR(y_data[2], 0.0f, 1e-5);
    // sinc(-1) = sin(-π)/(-π) = 0
    EXPECT_NEAR(y_data[3], 0.0f, 1e-5);
    // sinc(0.5) = sin(π/2)/(π/2) = 1/(π/2) = 2/π ≈ 0.6366
    EXPECT_NEAR(y_data[5], 0.6366198f, 1e-5);
}

TEST_F(SignalTest, Convolve) {
    std::vector<float> a_data = { 1.0f, 2.0f, 3.0f };
    std::vector<float> v_data = { 1.0f, 1.0f };
    Array a = to_array(a_data);
    Array v = to_array(v_data);

    Array full = ins::convolve(a, v, "full");
    const float* full_data = full.data<float>();
    EXPECT_EQ(full.numel(), 4);
    EXPECT_NEAR(full_data[0], 1.0f, 1e-5);
    EXPECT_NEAR(full_data[1], 3.0f, 1e-5);
    EXPECT_NEAR(full_data[2], 5.0f, 1e-5);
    EXPECT_NEAR(full_data[3], 3.0f, 1e-5);

    Array same = ins::convolve(a, v, "same");
    const float* same_data = same.data<float>();
    EXPECT_EQ(same.numel(), 3);
    // 对齐 NumPy 行为：返回 [1, 3, 5]
    EXPECT_NEAR(same_data[0], 1.0f, 1e-5);
    EXPECT_NEAR(same_data[1], 3.0f, 1e-5);
    EXPECT_NEAR(same_data[2], 5.0f, 1e-5);

    Array valid = ins::convolve(a, v, "valid");
    const float* valid_data = valid.data<float>();
    EXPECT_EQ(valid.numel(), 2);
    EXPECT_NEAR(valid_data[0], 3.0f, 1e-5);
    EXPECT_NEAR(valid_data[1], 5.0f, 1e-5);
}