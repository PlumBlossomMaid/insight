// tests/test_dtype.cpp
#include <gtest/gtest.h>
#include "insight/core/dtype.h"

using namespace ins;

// ========== DType 大小测试 ==========
TEST(DTypeTest, Size) {
    EXPECT_EQ(dtype_size(DType::F32), sizeof(float));
    EXPECT_EQ(dtype_size(DType::F64), sizeof(double));
    EXPECT_EQ(dtype_size(DType::I32), sizeof(int32_t));
    EXPECT_EQ(dtype_size(DType::I64), sizeof(int64_t));
    EXPECT_EQ(dtype_size(DType::U8), sizeof(uint8_t));
    EXPECT_EQ(dtype_size(DType::BOOL), sizeof(bool));
}

// ========== DType 名称测试 ==========
TEST(DTypeTest, Name) {
    EXPECT_STREQ(dtype_name(DType::F32), "float32");
    EXPECT_STREQ(dtype_name(DType::F64), "float64");
    EXPECT_STREQ(dtype_name(DType::I32), "int32");
    EXPECT_STREQ(dtype_name(DType::I64), "int64");
    EXPECT_STREQ(dtype_name(DType::U8), "uint8");
    EXPECT_STREQ(dtype_name(DType::BOOL), "bool");
    EXPECT_STREQ(dtype_name(static_cast<DType>(99)), "unknown");
}

// ========== DType 浮点判断测试 ==========
TEST(DTypeTest, IsFloatingPoint) {
    EXPECT_TRUE(is_floating_point(DType::F32));
    EXPECT_TRUE(is_floating_point(DType::F64));
    EXPECT_FALSE(is_floating_point(DType::I32));
    EXPECT_FALSE(is_floating_point(DType::I64));
    EXPECT_FALSE(is_floating_point(DType::U8));
    EXPECT_FALSE(is_floating_point(DType::BOOL));
}

// ========== DType 整数判断测试 ==========
TEST(DTypeTest, IsInteger) {
    EXPECT_FALSE(is_integer(DType::F32));
    EXPECT_FALSE(is_integer(DType::F64));
    EXPECT_TRUE(is_integer(DType::I32));
    EXPECT_TRUE(is_integer(DType::I64));
    EXPECT_TRUE(is_integer(DType::U8));
    EXPECT_FALSE(is_integer(DType::BOOL));
}

// ========== DType 有符号判断测试 ==========
TEST(DTypeTest, IsSigned) {
    EXPECT_TRUE(is_signed(DType::F32));
    EXPECT_TRUE(is_signed(DType::F64));
    EXPECT_TRUE(is_signed(DType::I32));
    EXPECT_TRUE(is_signed(DType::I64));
    EXPECT_FALSE(is_signed(DType::U8));
    EXPECT_FALSE(is_signed(DType::BOOL));
}