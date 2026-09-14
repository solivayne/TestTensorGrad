// Autograd: per-operator backward rules, broadcast reduction, matmul

#include <vector>
#include "Tensor.hpp"
#include "test.hpp"

using AutogradTest = EngineTest;

// TC-Autograd-001  AddSub_Gradient_Backgrad
TEST_F(AutogradTest, TC_Autograd_001)
{
    // Add
    Tensor a = make({1, 2, 3, 4, 5, 6, 7, 8}, {8});
    Tensor b = make({1, 1, 1, 1, 1, 1, 1, 1}, {8});
    Tensor c = a + b;
    for (int i = 0; i < 8; i++)
        c.grad[i] = static_cast<float>(i + 1);
    c.backward();
    for (int i = 0; i < 8; i++)
    {
        EXPECT_EQ(a.grad[i], static_cast<float>(i + 1))
            << "[TC-Autograd-001] Possible issue: lhs gradient in Tensor::operator+ backward.";
        EXPECT_EQ(b.grad[i], static_cast<float>(i + 1))
            << "[TC-Autograd-001] Possible issue: rhs gradient in Tensor::operator+ backward.";
    }

    // Sub
    for (int n = 1; n <= 17; n++)
    {
        SCOPED_TRACE("n = " + std::to_string(n));
        std::vector<float> av(n, 5.0f), bv(n, 3.0f);
        Tensor p = make(av, {n});
        Tensor q = make(bv, {n});
        Tensor d = p - q;
        seed(d);
        d.backward();
        for (int i = 0; i < n; i++)
        {
            EXPECT_EQ(p.grad[i], 1.0f)
                << "[TC-Autograd-001] Possible issue: lhs sign in Tensor::operator- backward ("
                << (i < n - n % 8 ? "AVX main loop" : "scalar tail") << ").";
            EXPECT_EQ(q.grad[i], -1.0f)
                << "[TC-Autograd-001] Possible issue: rhs sign in Tensor::operator- backward ("
                << (i < n - n % 8 ? "AVX main loop" : "scalar tail") << ").";
        }
    }

    // y = 2x
    Tensor x = make({1, 2, 3, 4}, {4});
    Tensor y = x + x;
    seed(y);
    y.backward();
    for (int i = 0; i < 4; i++)
        EXPECT_EQ(x.grad[i], 2.0f)
            << "[TC-Autograd-001] Possible issue: gradient accumulation in"
               " Tensor::operator+ backward.";
}

// TC-Autograd-002  BroadCast_Gradient_Reduce
TEST_F(AutogradTest, TC_Autograd_002)
{
    // y = x(3,2) - row(1, 2) s = y.sum() reduce to row
    Tensor x = make({1, 2, 3, 4, 5, 6}, {3, 2});
    Tensor row = make({10, 20}, {1, 2});
    Tensor y = x - row;
    Tensor s = y.sum();
    seed(s);
    s.backward();
    for (int i = 0; i < 6; i++)
        EXPECT_EQ(x.grad[i], 1.0f)
            << "[TC-Autograd-002] Possible issue: non-broadcast side in Tensor::operator- backward.";
    for (int i = 0; i < 2; i++)
        EXPECT_EQ(row.grad[i], -3.0f)
            << "[TC-Autograd-002] Possible issue: broadcast reduction in Tensor::operator- backward.";

    // Y = X(3,2) · W(2,2) + bias(1,2), s = sum(Y²)。
    // X = [[1,0],[0,1],[1,1]], W = I, bias = [1,2] → Y = [[2,2],[1,3],[2,3]]
    // dY = 2Y = [[4,4],[2,6],[4,6]]; bias.grad = [10,16]; W.grad = Xᵀ·dY = [[8,10],[6,12]]。
    Tensor X = make({1, 0, 0, 1, 1, 1}, {3, 2});
    Tensor W = make({1, 0, 0, 1}, {2, 2});
    Tensor bias = make({1, 2}, {1, 2});
    Tensor XW = X * W;
    Tensor Y = XW + bias;
    Tensor sq = Y.pow(2.0f);
    Tensor loss = sq.sum();
    seed(loss);
    loss.backward();
    const float eb[] = {10, 16};
    for (int i = 0; i < 2; i++)
        EXPECT_FLOAT_EQ(bias.grad[i], eb[i])
            << "[TC-Autograd-002] Possible issue: bias batch reduction in Tensor::operator+ backward.";
    const float ew[] = {8, 10, 6, 12};
    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(W.grad[i], ew[i])
            << "[TC-Autograd-002] Possible issue: chain rule through Tensor::operator* backward.";
}

// TC-Autograd-003  Unary_Gradient_Backgrad
TEST_F(AutogradTest, TC_Autograd_003)
{
    // pow(2)
    Tensor x = make({1, 2, 3, 4}, {4});
    Tensor x2 = x.pow(2.0f);
    seed(x2);
    x2.backward();
    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(x.grad[i], 2.0f * (i + 1))
            << "[TC-Autograd-003] Possible issue: derivative formula in Tensor::pow backward (e=2).";

    // pow(3)' = 3x^2
    Tensor z = make({-2, -0.5f, 0.5f, 2, 3}, {5});
    Tensor z3 = z.pow(3.0f);
    Tensor zs = z3.sum();
    seed(zs);
    zs.backward();
    const float zv[] = {-2, -0.5f, 0.5f, 2, 3};
    for (int i = 0; i < 5; i++)
        EXPECT_FLOAT_EQ(z.grad[i], 3.0f * zv[i] * zv[i])
            << "[TC-Autograd-003] Possible issue: derivative formula in Tensor::pow backward"
               " (e=3, negative base).";

    // relu
    Tensor r = make({-5, -0.001f, 0, 0.001f, 5}, {5});
    Tensor rr = r.relu();
    seed(rr, 3.0f);
    rr.backward();
    const float er[] = {0, 0, 0, 3, 3};
    for (int i = 0; i < 5; i++)
        EXPECT_EQ(r.grad[i], er[i])
            << "[TC-Autograd-003] Possible issue: mask / pass-through in Tensor::relu backward"
               " (index "
            << i << ").";

    // sum
    Tensor m = make({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {2, 5});
    Tensor ms = m.sum();
    seed(ms);
    ms.backward();
    for (int i = 0; i < 10; i++)
        EXPECT_EQ(m.grad[i], 1.0f)
            << "[TC-Autograd-003] Possible issue: scatter in Tensor::sum backward.";
}

// TC-Autograd-004  MatMul_Gradients
TEST_F(AutogradTest, TC_Autograd_004)
{
    Tensor A = make({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor B = make({1, 2, 3, 4, 5, 6}, {3, 2});
    Tensor C = A * B;
    Tensor cs = C.sum();
    seed(cs);
    cs.backward();
    const float eA[] = {3, 7, 11, 3, 7, 11};
    const float eB[] = {5, 5, 7, 7, 9, 9};
    for (int i = 0; i < 6; i++)
    {
        EXPECT_FLOAT_EQ(A.grad[i], eA[i])
            << "[TC-Autograd-004] Possible issue: dA = dC·Bᵀ in Tensor::operator* backward.";
        EXPECT_FLOAT_EQ(B.grad[i], eB[i])
            << "[TC-Autograd-004] Possible issue: dB = Aᵀ·dC in Tensor::operator* backward.";
    }

    Tensor P = make({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor Pt = P.transpose();
    Tensor D = make({2, 0, 0, 3}, {2, 2});
    Tensor Z = Pt * D;
    Tensor zs = Z.sum();
    seed(zs);
    zs.backward();
    const float eP[] = {2, 2, 2, 3, 3, 3};
    for (int i = 0; i < 6; i++)
        EXPECT_FLOAT_EQ(P.grad[i], eP[i])
            << "[TC-Autograd-004] Possible issue: dA written back by linear index in"
               " Tensor::operator* backward (transposed lhs view).";

    Tensor Ab = make({1, 2, 3, 4, 5, 6}, {1, 2, 3});
    Tensor Bb = make({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 3, 2});
    Tensor Cb = Ab * Bb;
    Tensor cbs = Cb.sum();
    seed(cbs);
    cbs.backward();
    const float eAb[] = {18, 26, 34, 18, 26, 34};
    for (int i = 0; i < 6; i++)
        EXPECT_FLOAT_EQ(Ab.grad[i], eAb[i])
            << "[TC-Autograd-004] Possible issue: batch reduction of dA in"
               " Tensor::operator* backward (broadcast lhs).";

    Tensor M = make({1, 2, 3, 4}, {2, 2});
    Tensor I = make({1, 0, 0, 1}, {2, 2});
    Tensor M1 = M * I;
    Tensor M2 = M1 * I;
    Tensor m2s = M2.sum();
    seed(m2s);
    m2s.backward();
    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(M.grad[i], 1.0f)
            << "[TC-Autograd-004] Possible issue: chained matmul accumulation in"
               " Tensor::operator* backward.";
}

// TC-Autograd-005  Graph_BackGrad
TEST_F(AutogradTest, TC_Autograd_005)
{
    Tensor x = make({1, 2, 3, 4}, {4});
    Tensor a = x.pow(1.0f);
    Tensor b = x + x;
    Tensor y = a + b;
    seed(y);
    y.backward();
    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(x.grad[i], 3.0f)
            << "[TC-Autograd-005] Possible issue: topological visit-once / accumulation in"
               " Tensor::backward (diamond graph).";

    Tensor u = make({1, 2, 3, 4}, {4});
    Tensor v = u + u;
    seed(v);
    v.backward();
    v.backward();
    for (int i = 0; i < 4; i++)
        EXPECT_EQ(u.grad[i], 4.0f)
            << "[TC-Autograd-005] Possible issue: accumulate-across-calls semantics in Tensor::backward.";
    u.zeroGrad();
    for (int i = 0; i < 4; i++)
        EXPECT_EQ(u.grad[i], 0.0f)
            << "[TC-Autograd-005] Possible issue: Tensor::zeroGrad.";

    Tensor leaf = make({1, 2, 3}, {3});
    leaf.grad[0] = 1.0f;
    EXPECT_NO_THROW(leaf.backward())
        << "[TC-Autograd-005] Possible issue: empty _backward guard in Tensor::backward.";
    EXPECT_EQ(leaf.grad[0], 1.0f);
    EXPECT_EQ(leaf.grad[1], 0.0f);
    EXPECT_EQ(leaf.grad[2], 0.0f);

    Tensor A = make({1, 2, 3, 4}, {2, 2});
    Tensor O = make({1, 1, 1, 1}, {2, 2});
    Tensor prod = A * O;
    Tensor loss = prod.sum();
    ASSERT_EQ(loss.prev.size(), 1u)
        << "[TC-Autograd-005] Possible issue: prev bookkeeping in Tensor::sum.";
    EXPECT_EQ(loss.prev[0], &prod)
        << "[TC-Autograd-005] Possible issue: prev points to a different object than the named"
           " intermediate.";
    seed(loss);
    loss.backward();
    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(A.grad[i], 2.0f)
            << "[TC-Autograd-005] Possible issue: gradient through named intermediates.";

    Tensor p = make({1, 2, 3, 4}, {4});
    Tensor q = make({4, 3, 2, 1}, {4});
    Tensor sum = p + q;
    Tensor diff = p - q;
    Tensor total = sum + diff;
    seed(total);
    total.backward();
    for (int i = 0; i < 4; i++)
    {
        EXPECT_EQ(p.grad[i], 2.0f)
            << "[TC-Autograd-005] Possible issue: multi-operator composite gradient (p).";
        EXPECT_EQ(q.grad[i], 0.0f)
            << "[TC-Autograd-005] Possible issue: multi-operator composite gradient (q, +1−1).";
    }
}
