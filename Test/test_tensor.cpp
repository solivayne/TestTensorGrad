// Tensor: construction, shape/strides bookkeeping, indexing and views.

#include <stdexcept>
#include <vector>
#include "Tensor.hpp"
#include "test.hpp"

using TensorCoreTest = EngineTest;

// TC-Tensor-001  Tensor_Shape
TEST_F(TensorCoreTest, TC_Tensor_001)
{
    struct Case
    {
        const char *name;
        std::vector<int> shape;
        std::vector<int> strides;
    };
    const std::vector<Case> cases = {
        {"rank 1", {5}, {1}},
        {"rank 2", {2, 3}, {3, 1}},
        {"rank 3", {2, 3, 4}, {12, 4, 1}},
        {"rank 4", {2, 3, 4, 5}, {60, 20, 5, 1}},
        {"degenerate size-1 dims", {1, 3, 1}, {3, 1, 1}},
    };

    for (const Case &c : cases)
    {
        SCOPED_TRACE(c.name);
        ASSERT_EQ(c.strides, rowMajor(c.shape)) << "test table is internally inconsistent";

        Tensor t(c.shape);
        EXPECT_EQ(t.shape, c.shape)
            << "[TC-Tensor-001] Possible issue: shape not stored verbatim in Tensor::Tensor(shape).";
        EXPECT_EQ(t.size, product(c.shape))
            << "[TC-Tensor-001] Possible issue: size computation in Tensor::Tensor(shape).";

        EXPECT_EQ(t.strides.size(), c.strides.size())
            << "[TC-Tensor-001] Possible issue: strides rank in Tensor::Tensor(shape).";
        if (t.strides.size() != c.strides.size())
            continue;
        for (size_t i = 0; i < c.strides.size(); i++)
        {
            EXPECT_EQ(t.strides[i], c.strides[i])
                << "[TC-Tensor-001] Possible issue: stride derivation (dim " << i
                << ") in Tensor::Tensor(shape).";
        }
    }

    Tensor t({2, 3, 4});
    ASSERT_EQ(t.size, 24u);
    ASSERT_NE(t.data, nullptr);
    for (int f = 0; f < 24; f++)
        ASSERT_EQ(t.data[f], 0.0f) << "precondition: fresh tensor is zeroed";

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 4; k++)
                t.at({i, j, k}) = static_cast<float>(i * 12 + j * 4 + k);

    for (int f = 0; f < 24; f++)
        EXPECT_EQ(t.data[f], static_cast<float>(f))
            << "[TC-Tensor-001] Possible issue: stride addressing in Tensor::at /"
               " Tensor::Tensor(shape) (flat "
            << f << ").";

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 4; k++)
                EXPECT_EQ(t.at({i, j, k}), static_cast<float>(i * 12 + j * 4 + k))
                    << "[TC-Tensor-001] Possible issue: Tensor::at read/write asymmetry.";
}

// TC-Tensor-002  Construction_ZeroInit
TEST_F(TensorCoreTest, TC_Tensor_002)
{
    Tensor a({8});
    ASSERT_EQ(a.size, 8u);
    for (int i = 0; i < 8; i++)
    {
        a.data[i] = 7.0f;
        a.grad[i] = 9.0f;
    }
    for (int i = 0; i < 8; i++)
    {
        ASSERT_EQ(a.data[i], 7.0f) << "precondition: dirty data not written";
        ASSERT_EQ(a.grad[i], 9.0f) << "precondition: dirty grad not written";
    }
    float *const aData = a.data;
    float *const aGrad = a.grad;

    globalArena.reset();

    Tensor b({8});
    ASSERT_EQ(b.size, 8u);
    ASSERT_EQ(b.data, aData)
        << "[TC-Tensor-002] Possible issue: cpuOffset not reset in Arena::reset.";
    ASSERT_EQ(b.grad, aGrad)
        << "[TC-Tensor-002] Possible issue: cpuOffset not reset in Arena::reset.";
    for (int i = 0; i < 8; i++)
    {
        EXPECT_EQ(b.data[i], 0.0f)
            << "[TC-Tensor-002] Possible issue: data zero-initialisation in"
               " Tensor::Tensor(shape) (index "
            << i << ").";
        EXPECT_EQ(b.grad[i], 0.0f)
            << "[TC-Tensor-002] Possible issue: grad zero-initialisation in"
               " Tensor::Tensor(shape) (index "
            << i << ").";
    }

    Tensor g({8}, false, Device::CUDA);
    ASSERT_EQ(g.device, Device::CUDA);
    ASSERT_EQ(g.size, 8u);
    Tensor h = g.to(Device::CPU);
    ASSERT_EQ(h.device, Device::CPU)
        << "[TC-Tensor-002] Possible issue: device tag after Tensor::to.";
    ASSERT_EQ(h.size, 8u);
    ASSERT_NE(h.data, g.data) << "precondition: to() must produce a host copy, not alias VRAM";
    for (int i = 0; i < 8; i++)
    {
        EXPECT_EQ(h.data[i], 0.0f)
            << "[TC-Tensor-002] Possible issue: fillZerosVram path in"
               " Tensor::Tensor(shape) (data, index "
            << i << ").";
        EXPECT_EQ(h.grad[i], 0.0f)
            << "[TC-Tensor-002] Possible issue: fillZerosVram path in"
               " Tensor::Tensor(shape) (grad, index "
            << i << ").";
    }
}

// TC-Tensor-003  Tensor_Construct
TEST_F(TensorCoreTest, TC_Tensor_003)
{
    const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor t(values, {2, 3}, {3, 1});
    EXPECT_EQ(t.device, Device::CPU)
        << "[TC-Tensor-003] Possible issue: device tag in Tensor::Tensor(data,shape,strides).";
    EXPECT_EQ(t.shape, (std::vector<int>{2, 3}))
        << "[TC-Tensor-003] Possible issue: shape in Tensor::Tensor(data,shape,strides).";
    ASSERT_EQ(t.size, values.size())
        << "[TC-Tensor-003] Possible issue: size in Tensor::Tensor(data,shape,strides).";
    EXPECT_EQ(t.strides, (std::vector<int>{3, 1}))
        << "[TC-Tensor-003] Possible issue: strides preservation in"
           " Tensor::Tensor(data,shape,strides).";
    ASSERT_NE(t.data, values.data()) << "precondition: constructor must copy, not alias the input";
    for (size_t i = 0; i < values.size(); i++)
    {
        EXPECT_EQ(t.data[i], values[i])
            << "[TC-Tensor-003] Possible issue: data copy in"
               " Tensor::Tensor(data,shape,strides) (index "
            << i << ").";
        EXPECT_EQ(t.grad[i], 0.0f)
            << "[TC-Tensor-003] Possible issue: grad zero-initialisation in"
               " Tensor::Tensor(data,shape,strides) (index "
            << i << ").";
    }

    Tensor cm(values, {2, 3}, {1, 2});
    EXPECT_EQ(cm.strides, (std::vector<int>{1, 2}))
        << "[TC-Tensor-003] Possible issue: strides preservation in"
           " Tensor::Tensor(data,shape,strides).";
    EXPECT_EQ(cm.at({1, 1}), 4.0f)
        << "[TC-Tensor-003] Possible issue: stride addressing in Tensor::at.";
    EXPECT_NE(cm.at({1, 1}), 5.0f)
        << "[TC-Tensor-003] Possible issue: caller strides recomputed as row-major.";
}

// TC-Tensor-004  At_Invalid
TEST_F(TensorCoreTest, TC_Tensor_004)
{
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, {3, 1});
    ASSERT_NO_THROW(t.at({1, 2})) << "precondition: valid index must not throw";
    ASSERT_EQ(t.at({1, 2}), 6.0f) << "precondition: valid index must address correctly";

    EXPECT_THROW(t.at({1}), std::invalid_argument)
        << "[TC-Tensor-004] Possible issue: rank validation in Tensor::at (too few indices).";
    EXPECT_THROW(t.at({0, 0, 0}), std::invalid_argument)
        << "[TC-Tensor-004] Possible issue: rank validation in Tensor::at (too many indices).";

    EXPECT_THROW(t.at({2, 0}), std::out_of_range)
        << "[TC-Tensor-004] Possible issue: bounds check in Tensor::at (dim 0).";
    EXPECT_THROW(t.at({0, 3}), std::out_of_range)
        << "[TC-Tensor-004] Possible issue: bounds check in Tensor::at (dim 1).";

    EXPECT_THROW(t.at({-1, 0}), std::out_of_range)
        << "[TC-Tensor-004] Possible issue: bounds check in Tensor::at (negative index).";

    EXPECT_EQ(t.at({1, 2}), 6.0f)
        << "[TC-Tensor-004] Possible issue: rejected access mutated tensor state.";

    Tensor g({4}, false, Device::CUDA);
    ASSERT_EQ(g.device, Device::CUDA);
    EXPECT_THROW(g.at({0}), std::runtime_error)
        << "[TC-Tensor-004] Possible issue: device guard in Tensor::at.";
}

// TC-Tensor-005  Transpose
TEST_F(TensorCoreTest, TC_Tensor_005)
{
    Tensor p({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, {3, 1});
    float *const pData = p.data;
    float *const pGrad = p.grad;

    Tensor pt = p.transpose();
    EXPECT_EQ(pt.shape, (std::vector<int>{3, 2}))
        << "[TC-Tensor-005] Possible issue: shape swap in Tensor::transpose.";
    EXPECT_EQ(pt.strides, (std::vector<int>{1, 3}))
        << "[TC-Tensor-005] Possible issue: strides swap in Tensor::transpose.";
    EXPECT_EQ(pt.size, p.size)
        << "[TC-Tensor-005] Possible issue: size in Tensor::transpose.";
    EXPECT_EQ(pt.device, p.device)
        << "[TC-Tensor-005] Possible issue: device tag in Tensor::transpose.";
    EXPECT_EQ(pt.data, pData)
        << "[TC-Tensor-005] Possible issue: view storage sharing (data) in Tensor::transpose.";
    EXPECT_EQ(pt.grad, pGrad)
        << "[TC-Tensor-005] Possible issue: view storage sharing (grad) in Tensor::transpose.";
    EXPECT_EQ(p.shape, (std::vector<int>{2, 3}));
    EXPECT_EQ(p.strides, (std::vector<int>{3, 1}));

    ASSERT_EQ(pt.shape.size(), 2u);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2; j++)
            EXPECT_EQ(pt.at({i, j}), p.at({j, i}))
                << "[TC-Tensor-005] Possible issue: stride addressing in"
                   " Tensor::transpose / Tensor::at ("
                << i << "," << j << ").";
    pt.at({2, 0}) = 42.0f; // pt[2][0] ≡ p[0][2]
    EXPECT_EQ(p.at({0, 2}), 42.0f)
        << "[TC-Tensor-005] Possible issue: transpose view does not alias source storage.";
    p.at({0, 2}) = 3.0f;

    Tensor ptt = pt.transpose();
    EXPECT_EQ(ptt.shape, p.shape)
        << "[TC-Tensor-005] Possible issue: double transpose does not restore shape.";
    EXPECT_EQ(ptt.strides, p.strides)
        << "[TC-Tensor-005] Possible issue: double transpose does not restore strides.";
    EXPECT_EQ(ptt.data, pData);
    EXPECT_EQ(ptt.grad, pGrad);

    Tensor v({1.0f, 2.0f, 3.0f}, {3}, {1});
    Tensor vt = v.transpose();
    EXPECT_EQ(vt.shape, v.shape)
        << "[TC-Tensor-005] Possible issue: rank guard in Tensor::transpose.";
    EXPECT_EQ(vt.strides, v.strides);
    EXPECT_EQ(vt.size, v.size);
    EXPECT_EQ(vt.data, v.data);
    EXPECT_EQ(vt.grad, v.grad);
}

// TC-Tensor-006  Tensor_Broadcast
TEST_F(TensorCoreTest, TC_Tensor_006)
{
    Tensor row({1.0f, 2.0f, 3.0f}, {1, 3}, {3, 1});
    float *const rowData = row.data;
    float *const rowGrad = row.grad;
    Tensor b = row.broadcastTo({4, 3});
    EXPECT_EQ(b.shape, (std::vector<int>{4, 3}))
        << "[TC-Tensor-006] Possible issue: shape in Tensor::broadcastTo.";
    EXPECT_EQ(b.strides, (std::vector<int>{0, 1}))
        << "[TC-Tensor-006] Possible issue: zero-stride expansion in Tensor::broadcastTo.";
    EXPECT_EQ(b.size, 12u)
        << "[TC-Tensor-006] Possible issue: logical size in Tensor::broadcastTo.";
    EXPECT_EQ(b.data, rowData)
        << "[TC-Tensor-006] Possible issue: view storage sharing (data) in Tensor::broadcastTo.";
    EXPECT_EQ(b.grad, rowGrad)
        << "[TC-Tensor-006] Possible issue: view storage sharing (grad) in Tensor::broadcastTo.";
    ASSERT_EQ(b.shape.size(), 2u);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_EQ(b.at({i, j}), row.at({0, j}))
                << "[TC-Tensor-006] Possible issue: stride addressing in"
                   " Tensor::broadcastTo / Tensor::at ("
                << i << "," << j << ").";

    EXPECT_EQ(row.shape, (std::vector<int>{1, 3}));
    EXPECT_EQ(row.strides, (std::vector<int>{3, 1}));

    Tensor v({1.0f, 2.0f, 3.0f}, {3}, {1});
    Tensor r = v.broadcastTo({2, 4, 3});
    EXPECT_EQ(r.shape, (std::vector<int>{2, 4, 3}))
        << "[TC-Tensor-006] Possible issue: rank-extension shape in Tensor::broadcastTo.";
    EXPECT_EQ(r.strides, (std::vector<int>{0, 0, 1}))
        << "[TC-Tensor-006] Possible issue: prepended-dim strides in Tensor::broadcastTo.";
    EXPECT_EQ(r.size, 24u);
    EXPECT_EQ(r.data, v.data);

    Tensor t({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2}, {2, 1});
    Tensor same = t.broadcastTo({2, 2});
    EXPECT_EQ(same.shape, t.shape)
        << "[TC-Tensor-006] Possible issue: same-shape passthrough in Tensor::broadcastTo.";
    EXPECT_EQ(same.strides, t.strides)
        << "[TC-Tensor-006] Possible issue: same-shape passthrough rewrote strides.";
    EXPECT_EQ(same.size, t.size);
    EXPECT_EQ(same.data, t.data);
    EXPECT_EQ(same.grad, t.grad);

    Tensor w({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, {3, 1});
    EXPECT_THROW(w.broadcastTo({3, 3}), std::runtime_error)
        << "[TC-Tensor-006] Possible issue: compatibility check in Tensor::broadcastTo.";

    EXPECT_THROW(w.broadcastTo({3}), std::runtime_error)
        << "[TC-Tensor-006] Possible issue: rank check in Tensor::broadcastTo.";
    EXPECT_EQ(w.shape, (std::vector<int>{2, 3}))
        << "[TC-Tensor-006] Possible issue: rejected broadcast mutated source shape.";
    EXPECT_EQ(w.strides, (std::vector<int>{3, 1}))
        << "[TC-Tensor-006] Possible issue: rejected broadcast mutated source strides.";
}

// TC-Tensor-007  Tensor_ZeroGrad
TEST_F(TensorCoreTest, TC_Tensor_007)
{
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {5}, {1});
    for (int i = 0; i < 5; i++)
        t.grad[i] = 3.0f;
    t.zeroGrad();
    for (int i = 0; i < 5; i++)
    {
        EXPECT_EQ(t.grad[i], 0.0f)
            << "[TC-Tensor-007] Possible issue: grad zeroing in Tensor::zeroGrad (index " << i << ").";
        EXPECT_EQ(t.data[i], static_cast<float>(i + 1))
            << "[TC-Tensor-007] Possible issue: Tensor::zeroGrad touched data.";
    }

    Tensor p({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, {3, 1});
    for (int i = 0; i < 6; i++)
        p.grad[i] = 5.0f;
    Tensor pt = p.transpose();
    ASSERT_EQ(pt.grad, p.grad) << "precondition: view must share grad storage";
    pt.zeroGrad();
    for (int i = 0; i < 6; i++)
        EXPECT_EQ(p.grad[i], 0.0f)
            << "[TC-Tensor-007] Possible issue: grad zeroing on transposed view in"
               " Tensor::zeroGrad (index "
            << i << ").";

    Tensor a({10.0f, 20.0f, 30.0f}, {1, 3}, {3, 1});
    Tensor b({7.0f, 8.0f, 9.0f}, {3}, {1});

    ASSERT_EQ(a.grad, a.data + 3) << "precondition: a.grad must directly follow a.data";
    ASSERT_EQ(b.data, a.grad + 3) << "precondition: b.data must directly follow a.grad";
    for (int i = 0; i < 3; i++)
        a.grad[i] = 1.0f;

    Tensor ab = a.broadcastTo({4, 3});
    ASSERT_EQ(ab.size, 12u);
    ASSERT_EQ(ab.grad, a.grad) << "precondition: broadcast view must share grad storage";
    ab.zeroGrad();

    for (int i = 0; i < 3; i++)
        EXPECT_EQ(a.grad[i], 0.0f)
            << "[TC-Tensor-007] Possible issue: zeroGrad on broadcast view did not clear"
               " underlying storage.";

    for (int i = 0; i < 3; i++)
        EXPECT_EQ(b.data[i], static_cast<float>(7 + i))
            << "[TC-Tensor-007] Possible issue: zeroGrad traverses by size on"
               " broadcast views, overwriting neighbouring arena storage (b.data["
            << i << "]).";
}
