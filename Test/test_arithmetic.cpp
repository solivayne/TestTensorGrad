// test_arithmetic: elementwise add, elementwise sub, matmul, pow, sum, relu

#include <cmath>
#include <stdexcept>
#include <vector>
#include "Tensor.hpp"
#include "test.hpp"

using ArithmeticTest = EngineTest; // defining test suite

// TC_Arithmetic_001: Add_Elementwise
TEST_F(ArithmeticTest, TC_Arithmetic_001)
{
    for (int n = 1; n <= 17; n++)
    {
        SCOPED_TRACE("n = " + std::to_string(n));
        std::vector<float> av(n), bv(n);
        for (int i = 0; i < n; i++)
        {
            av[i] = static_cast<float>(i + 1);
            bv[i] = static_cast<float>(10 * (i + 1));
        }
        Tensor a = make(av, {n});
        Tensor b = make(bv, {n});
        Tensor c = a + b;
        ASSERT_EQ(c.size, static_cast<size_t>(n))
            << "[TC-Arithmetic-001] Possible issue: result shape in Tensor::operator+.";
        for (int i = 0; i < n; i++)
            EXPECT_EQ(c.data[i], av[i] + bv[i])
                << "[TC-Arithmetic-001] Possible issue: "
                << (i < n - n % 8 ? "AVX main loop" : "scalar tail")
                << " in Tensor::operator+.";
    }
}

// TC_Arithmetic_002: Sub_Elementwise
TEST_F(ArithmeticTest, TC_Arithmetic_002)
{
    for (int n = 1; n <= 17; n++)
    {
        SCOPED_TRACE("n = " + std::to_string(n));
        std::vector<float> av(n), bv(n);
        for (int i = 0; i < n; i++)
        {
            av[i] = static_cast<float>(i + 5);
            bv[i] = static_cast<float>(i + 2);
        }
        Tensor a = make(av, {n});
        Tensor b = make(bv, {n});
        Tensor c = a - b;
        for (int i = 0; i < n; i++)
            EXPECT_EQ(c.data[i], 3.0f)
                << "[TC-Arithmetic-002] Possible issue: "
                << (i < n - n % 8 ? "AVX main loop" : "scalar tail")
                << " in Tensor::operator-.";
    }

    Tensor pred = make({0.0f, 1.0f, 1.0f, 0.0f}, {4, 1});
    Tensor target = make({0.0f, 1.0f, 1.0f, 0.0f}, {4, 1});
    Tensor diff = pred - target;
    ASSERT_EQ(diff.size, 4u);
    for (int i = 0; i < 4; i++)
        EXPECT_EQ(diff.data[i], 0.0f)
            << "[TC-Arithmetic-002] Possible issue: scalar tail sign in Tensor::operator-.";
}

// TC_Arithmetic_003: AddSub_BroadCast
TEST_F(ArithmeticTest, TC_Arithmetic_003)
{
    // (2,3) + (1,3)
    Tensor x = make({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor bias = make({10, 20, 30}, {1, 3});
    Tensor r1 = x + bias;
    ASSERT_EQ(r1.shape, (std::vector<int>{2, 3}))
        << "[TC-Arithmetic-003] Possible issue: broadcast result shape in Tensor::operator+.";
    const float e1[] = {11, 22, 33, 14, 25, 36};
    for (int i = 0; i < 6; i++)
        EXPECT_EQ(r1.data[i], e1[i])
            << "[TC-Arithmetic-003] Possible issue: row-vector broadcast in Tensor::operator+.";

    // (3,2) - (3,1)
    Tensor y = make({1, 2, 3, 4, 5, 6}, {3, 2});
    Tensor col = make({1, 10, 100}, {3, 1});
    Tensor r2 = y - col;
    const float e2[] = {0, 1, -7, -6, -95, -94};
    for (int i = 0; i < 6; i++)
        EXPECT_EQ(r2.data[i], e2[i])
            << "[TC-Arithmetic-003] Possible issue: column-vector broadcast in Tensor::operator-.";

    // (3,3) - (1,3)
    Tensor z = make({1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 3});
    Tensor row = make({0, 1, 2}, {1, 3});
    Tensor r3 = z - row;
    const float e3[] = {1, 1, 1, 4, 4, 4, 7, 7, 7};
    for (int i = 0; i < 9; i++)
        EXPECT_EQ(r3.data[i], e3[i])
            << "[TC-Arithmetic-003] Possible issue: broadcast path with unaligned size in"
               " Tensor::operator-.";

    // (3,2) - (2)
    Tensor v = make({1, 1}, {2});
    Tensor r4 = y - v;
    for (int i = 0; i < 6; i++)
        EXPECT_EQ(r4.data[i], static_cast<float>(i))
            << "[TC-Arithmetic-003] Possible issue: rank-extension broadcast in Tensor::operator-.";
}

// TC-Arithmetic-004  AddSub_Transpose
TEST_F(ArithmeticTest, TC_Arithmetic_004)
{
    Tensor p = make({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor pt = p.transpose(); // [1,4,2,5,3,6],strides {1,3}
    Tensor y = make({10, 20, 30, 40, 50, 60}, {3, 2});

    Tensor s = pt + y;
    const float es[] = {11, 24, 32, 45, 53, 66};
    for (int i = 0; i < 6; i++)
        EXPECT_EQ(s.data[i], es[i])
            << "[TC-Arithmetic-004] Possible issue: "
               " Tensor::operator+ (transposed operand read in memory order).";

    Tensor d = pt - y;
    const float ed[] = {-9, -16, -28, -35, -47, -54};
    for (int i = 0; i < 6; i++)
        EXPECT_EQ(d.data[i], ed[i])
            << "[TC-Arithmetic-004] Possible issue: "
               " Tensor::operator- (transposed operand read in memory order).";
}

// TC-Arithmetic-005  Elementwise_Invalid
TEST_F(ArithmeticTest, TC_Arithmetic_005)
{
    Tensor a = make({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor b = make({1, 2, 3, 4, 5, 6}, {3, 2});
    EXPECT_THROW(a + b, std::runtime_error)
        << "[TC-Arithmetic-005] Possible issue: broadcast compatibility check in Tensor::operator+.";
    EXPECT_THROW(a - b, std::runtime_error)
        << "[TC-Arithmetic-005] Possible issue: broadcast compatibility check in Tensor::operator-.";

    Tensor g({2, 3}, false, Device::CUDA);
    EXPECT_THROW(a + g, std::runtime_error)
        << "[TC-Arithmetic-005] Possible issue: device guard in Tensor::operator+.";
    EXPECT_THROW(a - g, std::runtime_error)
        << "[TC-Arithmetic-005] Possible issue: device guard in Tensor::operator-.";
    EXPECT_THROW(a * g, std::runtime_error)
        << "[TC-Arithmetic-005] Possible issue: device guard in Tensor::operator*.";
}
