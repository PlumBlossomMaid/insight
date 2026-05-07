// tests/cpu/test_manipulation.cpp
#include <gtest/gtest.h>
#include <complex>
#include "insight/insight.h"
#include "insight/ops/manipulation.h"

using namespace ins;

class ManipulationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ins::init();
    }
};

// ========== Helper functions ==========

template<typename T>
void fill_sequential(Array& arr, T start = 0) {
    T* data = arr.data<T>();
    int64_t n = arr.numel();
    for (int64_t i = 0; i < n; ++i) {
        data[i] = static_cast<T>(start + i);
    }
}

template<typename T>
void expect_equal(const Array& arr, const std::vector<T>& expected) {
    ASSERT_EQ(arr.numel(), static_cast<int64_t>(expected.size()));
    const T* data = arr.data<T>();
    for (int64_t i = 0; i < arr.numel(); ++i) {
        EXPECT_EQ(data[i], expected[i]);
    }
}

template<typename T>
void expect_float_equal(const Array& arr, const std::vector<T>& expected, T tol = 1e-6) {
    ASSERT_EQ(arr.numel(), static_cast<int64_t>(expected.size()));
    const T* data = arr.data<T>();
    for (int64_t i = 0; i < arr.numel(); ++i) {
        EXPECT_NEAR(data[i], expected[i], tol);
    }
}

// ========== reshape ==========

TEST_F(ManipulationTest, Reshape2DTo3D) {
    Array a({ 2, 3 }, DType::F32);
    fill_sequential<float>(a, 0);

    Array b = reshape(a, { 3, 2 });
    EXPECT_EQ(b.shape(), Shape({ 3, 2 }));
    expect_float_equal<float>(b, { 0, 1, 2, 3, 4, 5 });
}

TEST_F(ManipulationTest, ReshapeTo1D) {
    Array a({ 2, 3 }, DType::F32);
    fill_sequential<float>(a, 0);

    Array b = reshape(a, { 6 });
    EXPECT_EQ(b.shape(), Shape({ 6 }));
    expect_float_equal<float>(b, { 0, 1, 2, 3, 4, 5 });
}

// ========== flatten / ravel ==========

TEST_F(ManipulationTest, Flatten) {
    Array a({ 2, 3 }, DType::F32);
    fill_sequential<float>(a, 0);

    Array b = flatten(a);
    EXPECT_EQ(b.shape(), Shape({ 6 }));
    expect_float_equal<float>(b, { 0, 1, 2, 3, 4, 5 });
}

TEST_F(ManipulationTest, Ravel) {
    Array a({ 2, 3 }, DType::F32);
    fill_sequential<float>(a, 0);

    Array b = ravel(a);
    EXPECT_EQ(b.shape(), Shape({ 6 }));
    expect_float_equal<float>(b, { 0, 1, 2, 3, 4, 5 });
}

// ========== squeeze / unsqueeze ==========

TEST_F(ManipulationTest, Squeeze) {
    Array a({ 1, 3, 1, 4 }, DType::F32);
    fill_sequential<float>(a, 0);

    Array b = squeeze(a);
    EXPECT_EQ(b.shape(), Shape({ 3, 4 }));

    const float* data = b.data<float>();
    EXPECT_FLOAT_EQ(data[0], 0);
    EXPECT_FLOAT_EQ(data[3], 3);
    EXPECT_FLOAT_EQ(data[4], 4);
    EXPECT_FLOAT_EQ(data[11], 11);
}

TEST_F(ManipulationTest, SqueezeAxis) {
    Array a({ 1, 3, 1, 4 }, DType::F32);
    fill_sequential<float>(a, 0);

    Array b = squeeze(a, 2);  // squeeze axis 2 (size 1)
    EXPECT_EQ(b.shape(), Shape({ 1, 3, 4 }));
}

TEST_F(ManipulationTest, Unsqueeze) {
    Array a({ 3, 4 }, DType::F32);
    fill_sequential<float>(a, 0);

    Array b = unsqueeze(a, 0);
    EXPECT_EQ(b.shape(), Shape({ 1, 3, 4 }));

    Array c = unsqueeze(a, -1);
    EXPECT_EQ(c.shape(), Shape({ 3, 4, 1 }));
}

// ========== transpose / permute / swapaxes / moveaxis ==========

TEST_F(ManipulationTest, Transpose) {
    Array a({ 2, 3 }, DType::F32);
    fill_sequential<float>(a, 0);

    Array b = transpose(a).contiguous();
    EXPECT_EQ(b.shape(), Shape({ 3, 2 }));
    // Expected: [[0, 3], [1, 4], [2, 5]] -> flattened: 0, 3, 1, 4, 2, 5
    expect_float_equal<float>(b, { 0, 3, 1, 4, 2, 5 });
}

TEST_F(ManipulationTest, Permute) {
    Array a({ 2, 3, 4 }, DType::F32);

    Array b = permute(a, { 2, 0, 1 });
    EXPECT_EQ(b.shape(), Shape({ 4, 2, 3 }));
}

TEST_F(ManipulationTest, Swapaxes) {
    Array a({ 2, 3, 4 }, DType::F32);

    Array b = swapaxes(a, 0, 2);
    EXPECT_EQ(b.shape(), Shape({ 4, 3, 2 }));
}

TEST_F(ManipulationTest, Moveaxis) {
    Array a({ 2, 3, 4 }, DType::F32);

    Array b = moveaxis(a, 0, 2);
    EXPECT_EQ(b.shape(), Shape({ 3, 4, 2 }));
}

// ========== flip / fliplr / flipud ==========

TEST_F(ManipulationTest, Flip) {
    Array a({ 2, 4 }, DType::F32);
    fill_sequential<float>(a, 0);

    Array b = flip(a, 0);
    expect_float_equal<float>(b, { 4, 5, 6, 7, 0, 1, 2, 3 });

    Array c = flip(a, 1);
    expect_float_equal<float>(c, { 3, 2, 1, 0, 7, 6, 5, 4 });
}

TEST_F(ManipulationTest, Fliplr) {
    Array a({ 2, 4 }, DType::F32);
    fill_sequential<float>(a, 0);

    Array b = fliplr(a);
    expect_float_equal<float>(b, { 3, 2, 1, 0, 7, 6, 5, 4 });
}

TEST_F(ManipulationTest, Flipud) {
    Array a({ 2, 4 }, DType::F32);
    fill_sequential<float>(a, 0);

    Array b = flipud(a);
    expect_float_equal<float>(b, { 4, 5, 6, 7, 0, 1, 2, 3 });
}

// ========== rot90 ==========

TEST_F(ManipulationTest, Rot90) {
    Array a({ 2, 3 }, DType::F32);
    fill_sequential<float>(a, 0);

    Array b = rot90(a);
    EXPECT_EQ(b.shape(), Shape({ 3, 2 }));
    // NumPy rot90 on 2x3: [[0,1,2],[3,4,5]] -> [[2,5],[1,4],[0,3]]
    expect_float_equal<float>(b, { 2, 5, 1, 4, 0, 3 });

    Array c = rot90(a, 2);
    EXPECT_EQ(c.shape(), Shape({ 2, 3 }));
    // 180-degree rotation
    expect_float_equal<float>(c, { 5, 4, 3, 2, 1, 0 });
}

// ========== concat ==========

TEST_F(ManipulationTest, ConcatAxis0) {
    Array a({ 2, 2 }, DType::F32);
    Array b({ 2, 2 }, DType::F32);
    fill_sequential<float>(a, 0);
    fill_sequential<float>(b, 4);

    Array c = concat({ a, b }, 0);
    EXPECT_EQ(c.shape(), Shape({ 4, 2 }));
    expect_float_equal<float>(c, { 0, 1, 2, 3, 4, 5, 6, 7 });
}

TEST_F(ManipulationTest, ConcatAxis1) {
    Array a({ 2, 2 }, DType::F32);
    Array b({ 2, 2 }, DType::F32);
    fill_sequential<float>(a, 0);
    fill_sequential<float>(b, 4);

    Array c = concat({ a, b }, 1);
    EXPECT_EQ(c.shape(), Shape({ 2, 4 }));
    expect_float_equal<float>(c, { 0, 1, 4, 5, 2, 3, 6, 7 });
}

TEST_F(ManipulationTest, ConcatSingle) {
    Array a({ 2, 3 }, DType::F32);
    fill_sequential<float>(a, 0);

    Array c = concat({ a }, 0);
    EXPECT_EQ(c.shape(), Shape({ 2, 3 }));
    expect_float_equal<float>(c, { 0, 1, 2, 3, 4, 5 });
}

// ========== stack ==========

TEST_F(ManipulationTest, StackAxis0) {
    Array a({ 2, 2 }, DType::F32);
    Array b({ 2, 2 }, DType::F32);
    fill_sequential<float>(a, 0);
    fill_sequential<float>(b, 4);

    Array c = stack({ a, b }, 0);
    EXPECT_EQ(c.shape(), Shape({ 2, 2, 2 }));
}

TEST_F(ManipulationTest, StackAxis1) {
    Array a({ 2, 2 }, DType::F32);
    Array b({ 2, 2 }, DType::F32);
    fill_sequential<float>(a, 0);
    fill_sequential<float>(b, 4);

    Array c = stack({ a, b }, 1);
    EXPECT_EQ(c.shape(), Shape({ 2, 2, 2 }));
}

// ========== split ==========

TEST_F(ManipulationTest, SplitEqualParts) {
    Array a({ 4, 6 }, DType::F32);
    fill_sequential<float>(a, 0);

    auto parts = split(a, 2, 0);
    ASSERT_EQ(parts.size(), 2);
    EXPECT_EQ(parts[0].shape(), Shape({ 2, 6 }));
    EXPECT_EQ(parts[1].shape(), Shape({ 2, 6 }));
}

TEST_F(ManipulationTest, SplitIndices) {
    Array a({ 4, 6 }, DType::F32);
    fill_sequential<float>(a, 0);

    auto parts = split(a, { 2, 4 }, 1);
    ASSERT_EQ(parts.size(), 3);
    EXPECT_EQ(parts[0].shape(), Shape({ 4, 2 }));
    EXPECT_EQ(parts[1].shape(), Shape({ 4, 2 }));
    EXPECT_EQ(parts[2].shape(), Shape({ 4, 2 }));
}

// ========== repeat ==========

TEST_F(ManipulationTest, RepeatAxis0) {
    Array a({ 2, 3 }, DType::F32);
    fill_sequential<float>(a, 0);

    Array b = repeat(a, 2, 0);
    EXPECT_EQ(b.shape(), Shape({ 4, 3 }));
    const float* data = b.data<float>();
    EXPECT_FLOAT_EQ(data[0], 0);
    EXPECT_FLOAT_EQ(data[3], 0);
    EXPECT_FLOAT_EQ(data[6], 3);
}

TEST_F(ManipulationTest, RepeatAxis1) {
    Array a({ 2, 3 }, DType::F32);
    fill_sequential<float>(a, 0);

    Array b = repeat(a, 2, 1);
    EXPECT_EQ(b.shape(), Shape({ 2, 6 }));
}

// ========== tile ==========

TEST_F(ManipulationTest, Tile2D) {
    Array a({ 2, 2 }, DType::F32);
    fill_sequential<float>(a, 0);

    Array b = tile(a, { 2, 2 });
    EXPECT_EQ(b.shape(), Shape({ 4, 4 }));
}

TEST_F(ManipulationTest, TileWithLeadingOnes) {
    Array a({ 2, 2 }, DType::F32);
    fill_sequential<float>(a, 0);

    Array b = tile(a, { 1, 3 });
    EXPECT_EQ(b.shape(), Shape({ 2, 6 }));
}

// ========== pad ==========

TEST_F(ManipulationTest, PadConstant) {
    Array a({ 2, 3 }, DType::F32);
    fill_sequential<float>(a, 0);

    Array b = pad(a, { 1, 1, 1, 1 }, 0);
    EXPECT_EQ(b.shape(), Shape({ 4, 5 }));

    // Expected 4x5 matrix:
    // [[0, 0, 0, 0, 0],
    //  [0, 0, 1, 2, 0],
    //  [0, 3, 4, 5, 0],
    //  [0, 0, 0, 0, 0]]
    std::vector<float> expected = {
        0, 0, 0, 0, 0,
        0, 0, 1, 2, 0,
        0, 3, 4, 5, 0,
        0, 0, 0, 0, 0
    };
    expect_float_equal<float>(b, expected);
}

// ========== roll ==========

TEST_F(ManipulationTest, RollAlongAxis) {
    Array a({ 2, 4 }, DType::F32);
    fill_sequential<float>(a, 0);  // [[0,1,2,3], [4,5,6,7]]

    // shift=1 向下滚动一行
    Array b = roll(a, 1, 0);
    // Expected: [[4,5,6,7], [0,1,2,3]]
    std::vector<float> expected = { 4, 5, 6, 7, 0, 1, 2, 3 };
    expect_float_equal<float>(b, expected);

    // shift=2 滚动两行，2行数组回到原位
    Array c = roll(a, 2, 0);
    expect_float_equal<float>(c, { 0,1,2,3,4,5,6,7 });
}

TEST_F(ManipulationTest, RollFlatten) {
    Array a({ 2, 4 }, DType::F32);
    fill_sequential<float>(a, 0);

    Array b = roll(a, 3);
    EXPECT_EQ(b.shape(), Shape({ 2, 4 }));
}

// ========== diag ==========

TEST_F(ManipulationTest, DiagExtract) {
    Array a({ 3, 3 }, DType::F32);
    fill_sequential<float>(a, 0);

    Array b = diag(a);
    EXPECT_EQ(b.shape(), Shape({ 3 }));
    expect_float_equal<float>(b, { 0, 4, 8 });

    Array c = diag(a, 1);
    EXPECT_EQ(c.shape(), Shape({ 2 }));
    expect_float_equal<float>(c, { 1, 5 });
}

TEST_F(ManipulationTest, DiagConstruct) {
    Array a({ 3 }, DType::F32);
    fill_sequential<float>(a, 0);

    Array b = diag(a);
    EXPECT_EQ(b.shape(), Shape({ 3, 3 }));
    const float* data = b.data<float>();
    EXPECT_FLOAT_EQ(data[0], 0);
    EXPECT_FLOAT_EQ(data[4], 1);
    EXPECT_FLOAT_EQ(data[8], 2);
}

// ========== tril / triu ==========

TEST_F(ManipulationTest, Tril) {
    Array a({ 3, 3 }, DType::F32);
    fill_sequential<float>(a, 0);

    Array b = tril(a);
    const float* data = b.data<float>();
    EXPECT_FLOAT_EQ(data[0], 0);
    EXPECT_FLOAT_EQ(data[1], 0);
    EXPECT_FLOAT_EQ(data[3], 3);
    EXPECT_FLOAT_EQ(data[4], 4);
    EXPECT_FLOAT_EQ(data[8], 8);
}

TEST_F(ManipulationTest, TrilWithK) {
    Array a({ 3, 3 }, DType::F32);
    fill_sequential<float>(a, 0);

    Array b = tril(a, 1);
    const float* data = b.data<float>();
    EXPECT_FLOAT_EQ(data[1], 1);
    EXPECT_FLOAT_EQ(data[5], 5);
}

TEST_F(ManipulationTest, Triu) {
    Array a({ 3, 3 }, DType::F32);
    fill_sequential<float>(a, 0);

    Array b = triu(a);
    const float* data = b.data<float>();
    EXPECT_FLOAT_EQ(data[0], 0);
    EXPECT_FLOAT_EQ(data[1], 1);
    EXPECT_FLOAT_EQ(data[3], 0);
    EXPECT_FLOAT_EQ(data[4], 4);
    EXPECT_FLOAT_EQ(data[8], 8);
}

// ========== vstack / hstack ==========

TEST_F(ManipulationTest, Vstack) {
    Array a({ 2, 2 }, DType::F32);
    Array b({ 2, 2 }, DType::F32);
    fill_sequential<float>(a, 0);
    fill_sequential<float>(b, 4);

    Array c = vstack({ a, b });
    EXPECT_EQ(c.shape(), Shape({ 4, 2 }));
    expect_float_equal<float>(c, { 0, 1, 2, 3, 4, 5, 6, 7 });
}

TEST_F(ManipulationTest, Hstack) {
    Array a({ 2, 2 }, DType::F32);
    Array b({ 2, 2 }, DType::F32);
    fill_sequential<float>(a, 0);
    fill_sequential<float>(b, 4);

    Array c = hstack({ a, b });
    EXPECT_EQ(c.shape(), Shape({ 2, 4 }));
    expect_float_equal<float>(c, { 0, 1, 4, 5, 2, 3, 6, 7 });
}

// ========== diff tests ==========

TEST_F(ManipulationTest, Diff1DBasic) {
    Array a = to_array({ 1.0, 2.0, 4.0, 7.0, 11.0 });
    Array d = diff(a);
    EXPECT_EQ(d.shape(), Shape({ 4 }));
    const double* data = d.data<double>();
    EXPECT_NEAR(data[0], 1.0, 1e-6);
    EXPECT_NEAR(data[1], 2.0, 1e-6);
    EXPECT_NEAR(data[2], 3.0, 1e-6);
    EXPECT_NEAR(data[3], 4.0, 1e-6);
}

TEST_F(ManipulationTest, Diff2ndOrder) {
    Array a = to_array({ 1.0, 2.0, 4.0, 7.0, 11.0 });
    Array d = diff(a, 2);
    EXPECT_EQ(d.shape(), Shape({ 3 }));
    const double* data = d.data<double>();
    EXPECT_NEAR(data[0], 1.0, 1e-6);
    EXPECT_NEAR(data[1], 1.0, 1e-6);
    EXPECT_NEAR(data[2], 1.0, 1e-6);
}

TEST_F(ManipulationTest, DiffInt) {
    Array a = to_array({ 1, 3, 5, 7, 9 });
    Array d = diff(a);
    const int32_t* data = d.data<int32_t>();
    EXPECT_EQ(data[0], 2);
    EXPECT_EQ(data[1], 2);
    EXPECT_EQ(data[2], 2);
    EXPECT_EQ(data[3], 2);
}

TEST_F(ManipulationTest, DiffWithNegativeAxis) {
    Array a = to_array({ 10.0, 20.0, 30.0, 40.0 });
    Array d = diff(a, 1, -1);
    EXPECT_EQ(d.shape(), Shape({ 3 }));
    const double* data = d.data<double>();
    EXPECT_NEAR(data[0], 10.0, 1e-6);
    EXPECT_NEAR(data[1], 10.0, 1e-6);
    EXPECT_NEAR(data[2], 10.0, 1e-6);
}

TEST_F(ManipulationTest, Diff2DAxis0) {
    std::vector<float> data = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    };
    Array m = to_array(data, Shape({ 3, 4 }));
    Array d = diff(m, 1, 0);
    EXPECT_EQ(d.shape(), Shape({ 2, 4 }));
    const float* d_data = d.data<float>();
    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(d_data[i], 4.0f);
    }
}

TEST_F(ManipulationTest, Diff2DAxis1) {
    std::vector<float> data = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    };
    Array m = to_array(data, Shape({ 3, 4 }));
    Array d = diff(m, 1, 1);
    EXPECT_EQ(d.shape(), Shape({ 3, 3 }));
    const float* d_data = d.data<float>();
    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(d_data[i], 1.0f);
    }
}