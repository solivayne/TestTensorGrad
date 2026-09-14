// NN: Linear forward / init / backward, MSELoss value & gradient, SGD,

#include <cmath>
#include <vector>
#include "Linear.hpp"
#include "Loss.hpp"
#include "Optimizer.hpp"
#include "Tensor.hpp"
#include "test.hpp"

using NNTest = EngineTest;

// TC-NN-001  Linear_Forward
TEST_F(NNTest, TC_NN_001)
{
    Linear l1(2, 3);
    const float w1[] = {1, 0, 0, 0, 1, 0};
    for (int i = 0; i < 6; i++)
        l1.weights.data[i] = w1[i];
    fill(l1.bias, 0.0f);
    Tensor x1 = make({7, 9}, {1, 2});
    Tensor y1 = l1(x1);
    ASSERT_EQ(y1.shape, (std::vector<int>{1, 3}))
        << "[TC-NN-001] Possible issue: output shape in Linear::operator().";
    const float e1[] = {7, 9, 0};
    for (int i = 0; i < 3; i++)
        EXPECT_FLOAT_EQ(y1.data[i], e1[i])
            << "[TC-NN-001] Possible issue: X·W in Linear::operator().";

    Linear l2(2, 2);
    fill(l2.weights, 0.0f);
    l2.bias.data[0] = 1.5f;
    l2.bias.data[1] = -2.5f;
    Tensor x2 = make({1, 2, 3, 4, 5, 6}, {3, 2});
    Tensor y2 = l2(x2);
    for (int r = 0; r < 3; r++)
    {
        EXPECT_FLOAT_EQ(y2.data[r * 2 + 0], 1.5f)
            << "[TC-NN-001] Possible issue: bias broadcast in Linear::operator() (row " << r << ").";
        EXPECT_FLOAT_EQ(y2.data[r * 2 + 1], -2.5f)
            << "[TC-NN-001] Possible issue: bias broadcast in Linear::operator() (row " << r << ").";
    }

    Linear l3(3, 4);
    std::vector<Tensor *> ps = l3.parameters();
    ASSERT_EQ(ps.size(), 2u)
        << "[TC-NN-001] Possible issue: parameter count in Linear::parameters.";
    EXPECT_EQ(ps[0], &l3.weights)
        << "[TC-NN-001] Possible issue: weights pointer in Linear::parameters.";
    EXPECT_EQ(ps[1], &l3.bias)
        << "[TC-NN-001] Possible issue: bias pointer in Linear::parameters.";
    for (size_t i = 0; i < l3.bias.size; i++)
        EXPECT_EQ(l3.bias.data[i], 0.0f)
            << "[TC-NN-001] Possible issue: bias initialisation in Linear::Linear.";
}

// ---------------------------------------------------------------------------
// TC-NN-002  Linear_Init
TEST_F(NNTest, TC_NN_002)
{
    Linear l(4, 12);
    const float limit = std::sqrt(6.0f / (4 + 12));
    bool allSame = true;
    for (size_t i = 0; i < l.weights.size; i++)
    {
        EXPECT_LE(std::fabs(l.weights.data[i]), limit)
            << "[TC-NN-002] Possible issue: Xavier bound in Linear::Linear (index " << i << ").";
        if (l.weights.data[i] != l.weights.data[0])
            allSame = false;
    }
    EXPECT_FALSE(allSame)
        << "[TC-NN-002] Possible issue: constant weight initialisation in Linear::Linear.";

    Linear a(4, 4);
    Linear b(4, 4);
    bool identical = true;
    for (size_t i = 0; i < a.weights.size; i++)
        if (a.weights.data[i] != b.weights.data[i])
        {
            identical = false;
            break;
        }
    EXPECT_FALSE(identical)
        << "[TC-NN-002] Possible issue: fixed RNG seed in Linear::Linear — same-shape layers"
           " receive identical initial weights.";
}

// TC-NN-003  Linear_Backward (原 009+010)
TEST_F(NNTest, TC_NN_003)
{
    Linear l(2, 2);
    fill(l.weights, 0.5f);
    fill(l.bias, 0.0f);
    Tensor X = make({1, 1, 1, 1}, {2, 2});
    Tensor y = l(X);
    Tensor loss = y.sum();
    seed(loss);
    loss.backward();
    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(l.weights.grad[i], 2.0f)
            << "[TC-NN-003] Possible issue: weight gradient through Linear::operator().";
    for (int i = 0; i < 2; i++)
        EXPECT_FLOAT_EQ(l.bias.grad[i], 2.0f)
            << "[TC-NN-003] Possible issue: bias gradient (batch reduction) through"
               " Linear::operator().";

    Linear m(2, 2);
    fill(m.weights, 0.5f);
    fill(m.bias, 0.0f);
    Tensor xa = make({1, 1, 1, 1}, {2, 2});
    Tensor xb = make({5, 5, 5, 5}, {2, 2});
    Tensor ya = m(xa);
    Tensor yb = m(xb);
    Tensor la = ya.sum();
    seed(la);
    la.backward();
    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(m.weights.grad[i], 2.0f)
            << "[TC-NN-003] Known limitation: Linear::matmulCache member is overwritten by a"
               " second forward before backward — first graph's matmul node is lost.";
}

// TC-NN-004  MSELoss
TEST_F(NNTest, TC_NN_004)
{
    MSELoss mse1;
    Tensor p1 = make({1, 2, 3, 4, 5, 6, 7, 8}, {8});
    Tensor t1 = make({1, 1, 1, 1, 1, 1, 1, 1}, {8});
    Tensor l1 = mse1(p1, t1);
    ASSERT_EQ(l1.size, 1u)
        << "[TC-NN-004] Possible issue: loss shape in MSELoss::operator().";
    EXPECT_FLOAT_EQ(l1.data[0], 17.5f)
        << "[TC-NN-004] Possible issue: mean of squared error in MSELoss::operator().";

    MSELoss mse2;
    Tensor p2 = make({1, 2, 3, 4}, {4});
    Tensor t2 = make({0, 0, 0, 0}, {4});
    Tensor l2 = mse2(p2, t2);
    seed(l2);
    l2.backward();
    const float eg[] = {0.5f, 1.0f, 1.5f, 2.0f};
    for (int i = 0; i < 4; i++)
    {
        EXPECT_FLOAT_EQ(p2.grad[i], eg[i])
            << "[TC-NN-004] Possible issue: 2/N scaling in MSELoss backward chain (pred).";
        EXPECT_FLOAT_EQ(t2.grad[i], -eg[i])
            << "[TC-NN-004] Possible issue: sign of target gradient in MSELoss backward chain.";
    }

    MSELoss mse3;
    Tensor p3 = make({1, 2, 3, 4, 5, 6, 7, 8}, {8});
    Tensor t3 = make({1, 2, 3, 4, 5, 6, 7, 8}, {8});
    Tensor l3 = mse3(p3, t3);
    EXPECT_EQ(l3.data[0], 0.0f)
        << "[TC-NN-004] Possible issue: constant term / normalisation in MSELoss::operator().";
}

// TC-NN-005  SGD_StepAndZeroGrad
TEST_F(NNTest, TC_NN_005)
{
    Tensor p = make({1, 2, 3, 4}, {4}, true);
    seed(p);
    SGD opt1({&p}, 0.1f);
    opt1.step();
    const float e1[] = {0.9f, 1.9f, 2.9f, 3.9f};
    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(p.data[i], e1[i])
            << "[TC-NN-005] Possible issue: update rule in SGD::step.";

    Tensor q = make({1, 1}, {2}, true);
    q.grad[0] = 2.0f;
    q.grad[1] = -2.0f;
    SGD opt2({&q}, 0.5f);
    opt2.step();
    EXPECT_FLOAT_EQ(q.data[0], 0.0f)
        << "[TC-NN-005] Possible issue: descent direction in SGD::step (positive grad).";
    EXPECT_FLOAT_EQ(q.data[1], 2.0f)
        << "[TC-NN-005] Possible issue: descent direction in SGD::step (negative grad).";

    Tensor r = make({1, 2}, {2}, true);
    r.grad[0] = 7.0f;
    SGD opt3({&r}, 0.0f);
    opt3.step();
    EXPECT_EQ(r.data[0], 1.0f)
        << "[TC-NN-005] Possible issue: lr-independent offset in SGD::step.";
    EXPECT_EQ(r.data[1], 2.0f);

    Linear l(2, 2);
    seed(l.weights, 5.0f);
    seed(l.bias, 5.0f);
    SGD opt4(l.parameters(), 0.1f);
    opt4.zeroGrad();
    for (size_t i = 0; i < l.weights.size; i++)
        EXPECT_EQ(l.weights.grad[i], 0.0f)
            << "[TC-NN-005] Possible issue: SGD::zeroGrad does not cover weights.";
    for (size_t i = 0; i < l.bias.size; i++)
        EXPECT_EQ(l.bias.grad[i], 0.0f)
            << "[TC-NN-005] Possible issue: SGD::zeroGrad does not cover bias.";
}

// TC-NN-006  EndToEnd_Training
TEST_F(NNTest, TC_NN_006)
{
    {
        Tensor X = make({0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1}, {8, 2}, true);
        Tensor Y = make({0, 1, 1, 0, 0, 1, 1, 0}, {8, 1}, true);
        Linear l1(2, 16);
        Linear l2(16, 1);
        MSELoss mse;
        std::vector<Tensor *> params = l1.parameters();
        std::vector<Tensor *> p2 = l2.parameters();
        params.insert(params.end(), p2.begin(), p2.end());
        SGD opt(params, 0.5f);

        float lastLoss = 0.0f;
        for (int epoch = 0; epoch < 2000; epoch++)
        {
            Tensor h = l1(X);
            Tensor a = h.relu();
            Tensor out = l2(a);
            Tensor loss = mse(out, Y);
            lastLoss = loss.data[0];
            opt.zeroGrad();
            seed(loss);
            loss.backward();
            opt.step();
            globalArena.reset();
        }
        EXPECT_LT(lastLoss, 1e-3f)
            << "[TC-NN-006] Possible issue: end-to-end XOR training does not converge.";

        Tensor h = l1(X);
        Tensor a = h.relu();
        Tensor out = l2(a);
        for (int i = 0; i < 8; i++)
            EXPECT_NEAR(out.data[i], Y.data[i], 5e-2f)
                << "[TC-NN-006] Possible issue: XOR prediction mismatch at sample " << i << ".";
        globalArena.reset();
    }

    {
        std::vector<float> xs(32), ys(32);
        for (int i = 0; i < 32; i++)
        {
            xs[i] = i / 32.0f;
            ys[i] = 3.0f * xs[i] + 1.0f;
        }
        Tensor X = make(xs, {32, 1}, true);
        Tensor Y = make(ys, {32, 1}, true);
        Linear l(1, 1);
        MSELoss mse;
        SGD opt(l.parameters(), 0.5f);

        float firstLoss = -1.0f, lastLoss = 0.0f;
        for (int epoch = 0; epoch < 300; epoch++)
        {
            Tensor out = l(X);
            Tensor loss = mse(out, Y);
            if (epoch == 0)
                firstLoss = loss.data[0];
            lastLoss = loss.data[0];
            opt.zeroGrad();
            seed(loss);
            loss.backward();
            opt.step();
            globalArena.reset();
        }
        EXPECT_LT(lastLoss, 0.05f * firstLoss)
            << "[TC-NN-006] Possible issue: optimisation loop fails to reduce loss (first="
            << firstLoss << ", last=" << lastLoss << ").";
    }

    {
        Tensor X = make({0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1}, {8, 2}, true);
        Tensor Y = make({0, 1, 1, 0, 0, 1, 1, 0}, {8, 1}, true);
        Linear l1(2, 8);
        Linear l2(8, 1);
        MSELoss mse;
        std::vector<Tensor *> params = l1.parameters();
        std::vector<Tensor *> p2 = l2.parameters();
        params.insert(params.end(), p2.begin(), p2.end());
        SGD opt(params, 0.1f);

        bool finite = true;
        for (int epoch = 0; epoch < 100 && finite; epoch++)
        {
            Tensor h = l1(X);
            Tensor out = l2(h);
            Tensor loss = mse(out, Y);
            opt.zeroGrad();
            seed(loss);
            loss.backward();
            for (Tensor *p : params)
                for (size_t i = 0; i < p->size; i++)
                    if (!std::isfinite(p->grad[i]))
                        finite = false;
            opt.step();
            globalArena.reset();
        }
        EXPECT_TRUE(finite)
            << "[TC-NN-006] Possible issue: NaN/Inf gradient during training.";
    }
}
