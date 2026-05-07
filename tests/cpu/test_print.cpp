// tests/cpu/test_print.cpp
#include <gtest/gtest.h>
#include <sstream>
#include "insight/insight.h"
#include "insight/io/print.h"

using namespace ins;

class PrintTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ins::init();
    }
};

TEST_F(PrintTest, PrintScalar) {
    Array a(3.14159f);
    std::string str = to_string(a);

    EXPECT_TRUE(str.find("Array(shape=[], dtype=float32, place=cpu") != std::string::npos);
    EXPECT_TRUE(str.find("3.14") != std::string::npos);
}

TEST_F(PrintTest, Print1D) {
    Array a({ 5 }, DType::F32);
    float* data = a.data<float>();
    for (int i = 0; i < 5; ++i) data[i] = static_cast<float>(i);

    std::string str = to_string(a);

    EXPECT_TRUE(str.find("shape=[5]") != std::string::npos);
    EXPECT_TRUE(str.find("[0., 1., 2., 3., 4.]") != std::string::npos);
}

TEST_F(PrintTest, Print2D) {
    Array a({ 2, 3 }, DType::F32);
    float* data = a.data<float>();
    for (int i = 0; i < 6; ++i) data[i] = static_cast<float>(i);

    std::string str = to_string(a);

    EXPECT_TRUE(str.find("shape=[2, 3]") != std::string::npos);
    EXPECT_TRUE(str.find("[0., 1., 2.]") != std::string::npos);
    EXPECT_TRUE(str.find("[3., 4., 5.]") != std::string::npos);
}

TEST_F(PrintTest, Print3D) {
    Array a({ 2, 2, 2 }, DType::F32);
    float* data = a.data<float>();
    for (int i = 0; i < 8; ++i) data[i] = static_cast<float>(i);

    std::string str = to_string(a);

    EXPECT_TRUE(str.find("shape=[2, 2, 2]") != std::string::npos);
    // Check that output contains nested brackets
    EXPECT_TRUE(str.find("[[[0., 1.]") != std::string::npos);
}

TEST_F(PrintTest, PrintInt) {
    Array a({ 2, 3 }, DType::I32);
    int32_t* data = a.data<int32_t>();
    for (int i = 0; i < 6; ++i) data[i] = i;

    std::string str = to_string(a);

    EXPECT_TRUE(str.find("dtype=int32") != std::string::npos);
    EXPECT_TRUE(str.find("[0, 1, 2]") != std::string::npos);
    EXPECT_TRUE(str.find("[3, 4, 5]") != std::string::npos);
}

TEST_F(PrintTest, PrintBool) {
    Array a({ 2, 2 }, DType::BOOL);
    bool* data = a.data<bool>();
    data[0] = true;
    data[1] = false;
    data[2] = true;
    data[3] = false;

    std::string str = to_string(a);

    EXPECT_TRUE(str.find("dtype=bool") != std::string::npos);
    EXPECT_TRUE(str.find("[true, false]") != std::string::npos);
}

TEST_F(PrintTest, PrintComplex) {
    Array a({ 2 }, DType::C32);
    std::complex<float>* data = a.data<std::complex<float>>();
    data[0] = std::complex<float>(1.0f, 2.0f);
    data[1] = std::complex<float>(3.0f, -4.0f);

    std::string str = to_string(a);

    EXPECT_TRUE(str.find("dtype=complex64") != std::string::npos);
    EXPECT_TRUE(str.find("(1.00000000+2.00000000j)") != std::string::npos);
    EXPECT_TRUE(str.find("(3.00000000-4.00000000j)") != std::string::npos);
}

TEST_F(PrintTest, PrintWithSetPrecision) {
    Array a({ 1 }, DType::F32);
    a.data<float>()[0] = 3.14159265358979f;

    set_printoptions(4, -1, -1, -1, false);
    std::string str = to_string(a);
    EXPECT_TRUE(str.find("3.1416") != std::string::npos);

    set_printoptions(2, -1, -1, -1, false);
    str = to_string(a);
    EXPECT_TRUE(str.find("3.14") != std::string::npos);

    // Reset
    set_printoptions(8, -1, -1, -1, false);
}

TEST_F(PrintTest, PrintWithSummary) {
    // Create a large array to trigger summary mode
    Array a({ 20 }, DType::F32);
    float* data = a.data<float>();
    for (int i = 0; i < 20; ++i) data[i] = static_cast<float>(i);

    // Set low threshold to force summary
    set_printoptions(-1, 10, 3, -1, false);
    std::string str = to_string(a);

    EXPECT_TRUE(str.find("..., ") != std::string::npos);
    EXPECT_TRUE(str.find("0.") != std::string::npos);
    EXPECT_TRUE(str.find("19.") != std::string::npos);

    // Reset
    set_printoptions(-1, 1000, 3, -1, false);
}

TEST_F(PrintTest, PrintNonContiguous) {
    Array a({ 3, 4 }, DType::F32);
    float* data = a.data<float>();
    for (int i = 0; i < 12; ++i) data[i] = static_cast<float>(i);

    // Create non-contiguous view (rows 0 and 2)
    Array view = a.slice(0, 0, 3, 2);
    EXPECT_FALSE(view.is_contiguous());

    std::string str = to_string(view);

    EXPECT_TRUE(str.find("shape=[2, 4]") != std::string::npos);
    EXPECT_TRUE(str.find("[0., 1., 2., 3.]") != std::string::npos);
    EXPECT_TRUE(str.find("[8., 9., 10., 11.]") != std::string::npos);
}

TEST_F(PrintTest, PrintOperator) {
    Array a({ 2, 2 }, DType::F32);
    float* data = a.data<float>();
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    data[3] = 4.0f;

    std::ostringstream oss;
    oss << a;

    std::string str = oss.str();
    EXPECT_TRUE(str.find("Array(shape=[2, 2]") != std::string::npos);
    EXPECT_TRUE(str.find("1.") != std::string::npos);
    EXPECT_TRUE(str.find("2.") != std::string::npos);
    EXPECT_TRUE(str.find("3.") != std::string::npos);
    EXPECT_TRUE(str.find("4.") != std::string::npos);
}