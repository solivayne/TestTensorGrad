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

// TC-Arithmetic-006  MatMul
TEST_F(ArithmeticTest, TC_Arithmetic_006)
{
    Tensor A = make({1, 2, 3, 4}, {2, 2});
    Tensor B = make({5, 6, 7, 8}, {2, 2});
    Tensor C = A * B;
    ASSERT_EQ(C.shape, (std::vector<int>{2, 2}))
        << "[TC-Arithmetic-006] Possible issue: result shape in Tensor::operator*.";
    const float e1[] = {19, 22, 43, 50};
    for (int i = 0; i < 4; i++)
        EXPECT_EQ(C.data[i], e1[i])
            << "[TC-Arithmetic-006] Possible issue: accumulation in Tensor::operator*.";

    Tensor M = make({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor I3 = make({1, 0, 0, 0, 1, 0, 0, 0, 1}, {3, 3});
    Tensor MI = M * I3;
    for (int i = 0; i < 6; i++)
        EXPECT_EQ(MI.data[i], static_cast<float>(i + 1))
            << "[TC-Arithmetic-006] Possible issue: indexing in Tensor::operator* (identity).";

    const int m = 3, k = 5, n = 7;
    std::vector<float> av(m * k), bv(k * n);
    for (int i = 0; i < m * k; i++)
        av[i] = static_cast<float>((i * 7) % 11 - 5);
    for (int i = 0; i < k * n; i++)
        bv[i] = static_cast<float>((i * 5) % 13 - 6);
    Tensor P = make(av, {m, k});
    Tensor Q = make(bv, {k, n});
    Tensor R = P * Q;
    std::vector<float> ref = refMatmul(av, bv, m, k, n);
    ASSERT_EQ(R.shape, (std::vector<int>{m, n}));
    for (int i = 0; i < m * n; i++)
        EXPECT_EQ(R.data[i], ref[i])
            << "[TC-Arithmetic-006] Possible issue: block boundary / AVX tail in Tensor::operator*.";
}

// TC-Arithmetic-007  MatMul_Transpose
TEST_F(ArithmeticTest, TC_Arithmetic_007)
{

    Tensor p = make({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor pt = p.transpose(); // (3,2) strides {1,3}
    Tensor I2 = make({1, 0, 0, 1}, {2, 2});
    Tensor L = pt * I2;
    const float el[] = {1, 4, 2, 5, 3, 6};
    for (int i = 0; i < 6; i++)
        EXPECT_EQ(L.data[i], el[i])
            << "[TC-Arithmetic-007] Possible issue: left-operand strides in Tensor::operator*.";

    Tensor A = make({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor W = make({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor Wt = W.transpose(); // (3,2) [1,4,2,5,3,6]
    Tensor R = A * Wt;
    std::vector<float> ref = refMatmul({1, 2, 3, 4, 5, 6}, {1, 4, 2, 5, 3, 6}, 2, 3, 2);
    for (int i = 0; i < 4; i++)
        EXPECT_EQ(R.data[i], ref[i])
            << "[TC-Arithmetic-007] Possible issue: right-operand strides in Tensor::operator*.";

    // batch mat mul
    std::vector<float> av(12), bv(12);
    for (int i = 0; i < 12; i++)
    {
        av[i] = static_cast<float>(i + 1);
        bv[i] = static_cast<float>(12 - i);
    }
    Tensor BA = make(av, {2, 2, 3});
    Tensor BB = make(bv, {2, 3, 2});
    Tensor BC = BA * BB;
    ASSERT_EQ(BC.shape, (std::vector<int>{2, 2, 2}))
        << "[TC-Arithmetic-007] Possible issue: batch result shape in Tensor::operator*.";
    for (int b = 0; b < 2; b++)
    {
        std::vector<float> sa(av.begin() + b * 6, av.begin() + b * 6 + 6);
        std::vector<float> sb(bv.begin() + b * 6, bv.begin() + b * 6 + 6);
        std::vector<float> rb = refMatmul(sa, sb, 2, 3, 2);
        for (int i = 0; i < 4; i++)
            EXPECT_EQ(BC.data[b * 4 + i], rb[i])
                << "[TC-Arithmetic-007] Possible issue: batch offset (batch " << b
                << ") in Tensor::operator*.";
    }
}

// TC-Arithmetic-008  MatMul_Invalid
// 代码设计拒绝向量与矩阵乘法 (Tensor.cpp: line 442)
TEST_F(ArithmeticTest, TC_Arithmetic_008)
{
    // (2,3)·(2,2)
    Tensor A = make({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor B = make({1, 2, 3, 4}, {2, 2});
    EXPECT_THROW(A * B, std::runtime_error)
        << "[TC-Arithmetic-008] Possible issue: inner-dimension check in Tensor::operator*.";

    // (2, )·(1, )
    Tensor v = make({1, 2}, {2});
    Tensor s = make({1}, {1});
    EXPECT_THROW(v * B, std::runtime_error)
        << "[TC-Arithmetic-008] Possible issue: rank check in Tensor::operator* (rank-1 lhs).";
    EXPECT_THROW(B * v, std::runtime_error)
        << "[TC-Arithmetic-008] Possible issue: rank check in Tensor::operator* (rank-1 rhs).";
    EXPECT_THROW(s * B, std::runtime_error)
        << "[TC-Arithmetic-008] Possible issue: rank check in Tensor::operator* (single-element lhs).";
}

// TC-Arithmetic-009  UnaryOps
TEST_F(ArithmeticTest, TC_Arithmetic_009)
{
    // pow for integer
    Tensor x = make({-3, -1, 0, 2, 5}, {5});
    Tensor sq = x.pow(2.0f);
    const float esq[] = {9, 1, 0, 4, 25};
    for (int i = 0; i < 5; i++)
        EXPECT_EQ(sq.data[i], esq[i])
            << "[TC-Arithmetic-009] Possible issue: integer exponent in Tensor::pow.";

    // pow for decimal
    Tensor y = make({1, 4, 9, 16}, {4});
    Tensor rt = y.pow(0.5f);
    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(rt.data[i], static_cast<float>(i + 1))
            << "[TC-Arithmetic-009] Possible issue: non-integer exponent in Tensor::pow.";

    // relu
    Tensor z = make({-5, -1, 0, 1, 5}, {5});
    Tensor r = z.relu();
    const float er[] = {0, 0, 0, 1, 5};
    for (int i = 0; i < 5; i++)
        EXPECT_EQ(r.data[i], er[i])
            << "[TC-Arithmetic-009] Possible issue: threshold in Tensor::relu.";

    // sum
    Tensor m = make({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor sm = m.sum();
    ASSERT_EQ(sm.size, 1u)
        << "[TC-Arithmetic-009] Possible issue: result shape in Tensor::sum.";
    EXPECT_EQ(sm.data[0], 21.0f)
        << "[TC-Arithmetic-009] Possible issue: reduction in Tensor::sum.";
    Tensor one = make({3.5f}, {1});
    EXPECT_EQ(one.sum().data[0], 3.5f)
        << "[TC-Arithmetic-009] Possible issue: single-element reduction in Tensor::sum.";

    // equal shape test
    EXPECT_EQ(m.relu().shape, (std::vector<int>{2, 3}))
        << "[TC-Arithmetic-009] Possible issue: shape propagation in Tensor::relu.";
    EXPECT_EQ(m.pow(2.0f).shape, (std::vector<int>{2, 3}))
        << "[TC-Arithmetic-009] Possible issue: shape propagation in Tensor::pow.";
    EXPECT_EQ(sm.shape, (std::vector<int>{1}))
        << "[TC-Arithmetic-009] Possible issue: reduced shape in Tensor::sum.";
}