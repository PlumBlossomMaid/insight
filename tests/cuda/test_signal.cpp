// tests/cuda/test_signal.cu
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

class SignalTestGPU : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ins::init();
        set_device(GPUPlace(0));
    }
};

// ========== unwrap tests ==========

TEST_F(SignalTestGPU, UnwrapBasic) {
    std::vector<double> data = { 0.0, M_PI_2, M_PI, 3.0 * M_PI_2, 2.0 * M_PI, 3.0 * M_PI };
    Array a_cpu = to_array(data);
    Array a = a_cpu.to(GPUPlace(0));
    Array u = unwrap(a);
    Array u_cpu = u.to(CPUPlace());
    const double* u_data = u_cpu.data<double>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(u_data[i], data[i], 1e-6);
    }
}

TEST_F(SignalTestGPU, UnwrapWithJumps) {
    std::vector<double> data = { 0.0, 0.1, 3.2, 3.3, 6.4, 6.5 };
    Array a_cpu = to_array(data);
    Array a = a_cpu.to(GPUPlace(0));
    Array u = unwrap(a);
    Array u_cpu = u.to(CPUPlace());
    const double* u_data = u_cpu.data<double>();
    EXPECT_NEAR(u_data[0], 0.0, 1e-6);
    EXPECT_NEAR(u_data[1], 0.1, 1e-6);
    EXPECT_NEAR(u_data[2], 3.2, 1e-6);
    EXPECT_NEAR(u_data[3], 3.3, 1e-6);
    EXPECT_NEAR(u_data[4], 6.4, 1e-6);
    EXPECT_NEAR(u_data[5], 6.5, 1e-6);
}

TEST_F(SignalTestGPU, UnwrapScalar) {
    Array a_cpu = to_array({ M_PI });
    Array a = a_cpu.to(GPUPlace(0));
    Array u = unwrap(a);
    EXPECT_NEAR(u.to(CPUPlace()).item<double>(), M_PI, 1e-6);
}

TEST_F(SignalTestGPU, Unwrap2DAxis1) {
    std::vector<double> data = {
        0.0, M_PI_2, M_PI, 3.0 * M_PI_2,
        0.1, M_PI_2, 3.2, 3.3
    };
    Array m_cpu = to_array(data, Shape({ 2, 4 }));
    Array m = m_cpu.to(GPUPlace(0));
    Array u = unwrap(m, 1);
    EXPECT_EQ(u.shape(), Shape({ 2, 4 }));
    Array u_cpu = u.to(CPUPlace());
    const double* u_data = u_cpu.data<double>();
    EXPECT_NEAR(u_data[0], 0.0, 1e-6);
    EXPECT_NEAR(u_data[1], M_PI_2, 1e-6);
    EXPECT_NEAR(u_data[2], M_PI, 1e-6);
    EXPECT_NEAR(u_data[3], 3.0 * M_PI_2, 1e-6);
    EXPECT_NEAR(u_data[4], 0.1, 1e-6);
    EXPECT_NEAR(u_data[5], M_PI_2, 1e-6);
    EXPECT_NEAR(u_data[6], 3.2, 1e-6);
    EXPECT_NEAR(u_data[7], 3.3, 1e-6);
}

TEST_F(SignalTestGPU, Unwrap2DAxis0) {
    std::vector<double> data = {
        0.0, M_PI_2, M_PI, 3.0 * M_PI_2,
        0.1, M_PI_2, 3.2, 3.3
    };
    Array m_cpu = to_array(data, Shape({ 2, 4 }));
    Array m = m_cpu.to(GPUPlace(0));
    Array u = unwrap(m, 0);
    EXPECT_EQ(u.shape(), Shape({ 2, 4 }));
}

TEST_F(SignalTestGPU, UnwrapCustomParams) {
    std::vector<double> data = { 0.0, 0.5, 1.0, 0.2, 0.7, 1.2 };
    Array a_cpu = to_array(data);
    Array a = a_cpu.to(GPUPlace(0));
    Array u = unwrap(a, -1, 0.3, 1.0);
    Array u_cpu = u.to(CPUPlace());
    const double* u_data = u_cpu.data<double>();
    EXPECT_NEAR(u_data[0], 0.0, 1e-6);
    EXPECT_NEAR(u_data[1], -0.5, 1e-6);
    EXPECT_NEAR(u_data[2], -1.0, 1e-6);
    EXPECT_NEAR(u_data[3], -0.8, 1e-6);
    EXPECT_NEAR(u_data[4], -1.3, 1e-6);
    EXPECT_NEAR(u_data[5], -1.8, 1e-6);
}

TEST_F(SignalTestGPU, UnwrapNegativeAxis) {
    std::vector<float> data = { 0.0f, 3.5f, 0.1f, 3.4f, 0.2f, 3.3f };
    Array a_cpu = to_array(data, Shape({ 2, 3 }));
    Array a = a_cpu.to(GPUPlace(0));
    Array u = unwrap(a, -1);
    EXPECT_EQ(u.shape(), Shape({ 2, 3 }));
}

TEST_F(SignalTestGPU, UnwrapContinuity) {
    int n = 20;
    std::vector<double> original(n);
    std::vector<double> wrapped(n);
    for (int i = 0; i < n; ++i) {
        original[i] = static_cast<double>(i) * 0.6;
        wrapped[i] = std::fmod(original[i], 2.0 * M_PI);
    }

    Array orig = to_array(original, Shape({ n }), DType::F64, CPUPlace());
    Array wrapped_cpu = to_array(wrapped, Shape({ n }), DType::F64, CPUPlace());
    Array wrapped_arr = wrapped_cpu.to(GPUPlace(0));
    Array recovered = unwrap(wrapped_arr);
    Array rec = recovered.to(CPUPlace());

    const double* orig_data = orig.data<double>();
    const double* rec_data = rec.data<double>();
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(rec_data[i], orig_data[i], 1e-6);
    }
}

TEST_F(SignalTestGPU, Sinc) {
    std::vector<float> data = { 0.0f, 1.0f, 2.0f, -1.0f, -2.0f, 0.5f };
    Array x_cpu = to_array(data);
    Array x = x_cpu.to(GPUPlace(0));
    Array y = ins::sinc(x);

    Array y_cpu = y.to(CPUPlace());
    const float* y_data = y_cpu.data<float>();
    EXPECT_NEAR(y_data[0], 1.0f, 1e-5);
    EXPECT_NEAR(y_data[1], 0.0f, 1e-5);
    EXPECT_NEAR(y_data[2], 0.0f, 1e-5);
    EXPECT_NEAR(y_data[3], 0.0f, 1e-5);
    EXPECT_NEAR(y_data[5], 0.6366198f, 1e-5);
}

TEST_F(SignalTestGPU, Convolve) {
    std::vector<float> a_data = { 1.0f, 2.0f, 3.0f };
    std::vector<float> v_data = { 1.0f, 1.0f };
    Array a_cpu = to_array(a_data);
    Array v_cpu = to_array(v_data);
    Array a = a_cpu.to(GPUPlace(0));
    Array v = v_cpu.to(GPUPlace(0));

    Array full = ins::convolve(a, v, "full");
    Array full_cpu = full.to(CPUPlace());
    const float* full_data = full_cpu.data<float>();
    EXPECT_EQ(full.numel(), 4);
    EXPECT_NEAR(full_data[0], 1.0f, 1e-5);
    EXPECT_NEAR(full_data[1], 3.0f, 1e-5);
    EXPECT_NEAR(full_data[2], 5.0f, 1e-5);
    EXPECT_NEAR(full_data[3], 3.0f, 1e-5);

    Array same = ins::convolve(a, v, "same");
    Array same_cpu = same.to(CPUPlace());
    const float* same_data = same_cpu.data<float>();
    EXPECT_EQ(same.numel(), 3);
    EXPECT_NEAR(same_data[0], 1.0f, 1e-5);
    EXPECT_NEAR(same_data[1], 3.0f, 1e-5);
    EXPECT_NEAR(same_data[2], 5.0f, 1e-5);

    Array valid = ins::convolve(a, v, "valid");
    Array valid_cpu = valid.to(CPUPlace());
    const float* valid_data = valid_cpu.data<float>();
    EXPECT_EQ(valid.numel(), 2);
    EXPECT_NEAR(valid_data[0], 3.0f, 1e-5);
    EXPECT_NEAR(valid_data[1], 5.0f, 1e-5);
}