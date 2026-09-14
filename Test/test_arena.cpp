// test_arena: bump allocation, reset and capacity limits

#include <stdexcept>
#include <vector>
#include "Arena.hpp"
#include "Tensor.hpp"
#include "cuda_kernels.cuh"
#include "test.hpp"

using ArenaTest = EngineTest;

// TC-Arena-001  Allocation_Offset
TEST_F(ArenaTest, TC_Arena_001)
{
    Arena a(64);

    float *c1 = a.allocate(10, Device::CPU);
    float *c2 = a.allocate(10, Device::CPU);
    float *c3 = a.allocate(10, Device::CPU);
    ASSERT_NE(c1, nullptr) << "[TC-Arena-001] Possible issue: null return in Arena::allocate (CPU).";
    ASSERT_NE(c2, nullptr) << "[TC-Arena-001] Possible issue: null return in Arena::allocate (CPU).";
    ASSERT_NE(c3, nullptr) << "[TC-Arena-001] Possible issue: null return in Arena::allocate (CPU).";
    EXPECT_EQ(c2, c1 + 10)
        << "[TC-Arena-001] Possible issue: cpuOffset advance in Arena::allocate.";
    EXPECT_EQ(c3, c2 + 10)
        << "[TC-Arena-001] Possible issue: cpuOffset advance in Arena::allocate.";

    for (int i = 0; i < 10; i++)
    {
        c1[i] = 1.0f;
        c2[i] = 2.0f;
        c3[i] = 3.0f;
    }
    for (int i = 0; i < 10; i++)
    {
        EXPECT_EQ(c1[i], 1.0f) << "[TC-Arena-001] Possible issue: CPU block 1 overwritten by later allocation.";
        EXPECT_EQ(c2[i], 2.0f) << "[TC-Arena-001] Possible issue: CPU block 2 overwritten by later allocation.";
        EXPECT_EQ(c3[i], 3.0f) << "[TC-Arena-001] Possible issue: CPU block 3 overwritten by later allocation.";
    }

    float *g1 = a.allocate(10, Device::CUDA);
    float *g2 = a.allocate(10, Device::CUDA);
    float *g3 = a.allocate(10, Device::CUDA);
    ASSERT_NE(g1, nullptr) << "[TC-Arena-001] Possible issue: null return in Arena::allocate (GPU).";
    ASSERT_NE(g2, nullptr) << "[TC-Arena-001] Possible issue: null return in Arena::allocate (GPU).";
    ASSERT_NE(g3, nullptr) << "[TC-Arena-001] Possible issue: null return in Arena::allocate (GPU).";
    EXPECT_EQ(g2, g1 + 10)
        << "[TC-Arena-001] Possible issue: gpuOffset advance in Arena::allocate.";
    EXPECT_EQ(g3, g2 + 10)
        << "[TC-Arena-001] Possible issue: gpuOffset advance in Arena::allocate.";

    gpuFill(g1, 10, 1.0f);
    gpuFill(g2, 10, 2.0f);
    gpuFill(g3, 10, 3.0f);
    const std::vector<float> r1 = gpuRead(g1, 10);
    const std::vector<float> r2 = gpuRead(g2, 10);
    const std::vector<float> r3 = gpuRead(g3, 10);
    for (int i = 0; i < 10; i++)
    {
        EXPECT_EQ(r1[i], 1.0f) << "[TC-Arena-001] Possible issue: GPU block 1 overwritten by later allocation.";
        EXPECT_EQ(r2[i], 2.0f) << "[TC-Arena-001] Possible issue: GPU block 2 overwritten by later allocation.";
        EXPECT_EQ(r3[i], 3.0f) << "[TC-Arena-001] Possible issue: GPU block 3 overwritten by later allocation.";
    }
}

// TC-Arena-002  Capacity_Limits
TEST_F(ArenaTest, TC_Arena_002)
{
    Arena a(32);

    float *cFull = nullptr;
    ASSERT_NO_THROW(cFull = a.allocate(32, Device::CPU))
        << "[TC-Arena-002] Possible issue: capacity check boundary in Arena::allocate (CPU).";
    ASSERT_NE(cFull, nullptr);
    float *cEnd = a.allocate(0, Device::CPU); // 用满后的当前位置
    EXPECT_THROW(a.allocate(1, Device::CPU), std::runtime_error)
        << "[TC-Arena-002] Possible issue: capacity check in Arena::allocate (CPU).";
    EXPECT_EQ(a.allocate(0, Device::CPU), cEnd)
        << "[TC-Arena-002] Possible issue: rejected allocation advanced cpuOffset.";

    float *gFull = nullptr;
    ASSERT_NO_THROW(gFull = a.allocate(32, Device::CUDA))
        << "[TC-Arena-002] Possible issue: capacity check boundary in Arena::allocate (GPU).";
    ASSERT_NE(gFull, nullptr);
    float *gEnd = a.allocate(0, Device::CUDA);
    EXPECT_THROW(a.allocate(1, Device::CUDA), std::runtime_error)
        << "[TC-Arena-002] Possible issue: capacity check in Arena::allocate (GPU).";
    EXPECT_EQ(a.allocate(0, Device::CUDA), gEnd)
        << "[TC-Arena-002] Possible issue: rejected allocation advanced gpuOffset.";
}

// TC-Arena-003  Offset_Consistency
TEST_F(ArenaTest, TC_Arena_003)
{
    Arena a(16);

    float *z1 = a.allocate(0, Device::CPU);
    float *z2 = a.allocate(0, Device::CPU);
    EXPECT_EQ(z1, z2)
        << "[TC-Arena-003] Possible issue: cpuOffset advance in Arena::allocate.";

    float *full = nullptr;
    ASSERT_NO_THROW(full = a.allocate(16, Device::CPU))
        << "[TC-Arena-003] Possible issue: capacity accounting in Arena::allocate.";
    EXPECT_EQ(full, z1)
        << "[TC-Arena-003] Possible issue: zero-size allocation consumed capacity.";
}

// TC-Arena-004  Reset
TEST_F(ArenaTest, TC_Arena_004)
{
    Arena a(64);
    float *c1 = a.allocate(64, Device::CPU);  // Exhaustion of CPU
    float *g1 = a.allocate(64, Device::CUDA); // Exhaustion of GPU
    ASSERT_NE(c1, nullptr);
    ASSERT_NE(g1, nullptr);
    ASSERT_THROW(a.allocate(1, Device::CPU), std::runtime_error);
    ASSERT_THROW(a.allocate(1, Device::CUDA), std::runtime_error);

    a.reset();

    float *c2 = nullptr;
    float *g2 = nullptr;
    ASSERT_NO_THROW(c2 = a.allocate(64, Device::CPU))
        << "[TC-Arena-004] Possible issue: cpuOffset not reset in Arena::reset.";
    ASSERT_NO_THROW(g2 = a.allocate(64, Device::CUDA))
        << "[TC-Arena-004] Possible issue: gpuOffset not reset in Arena::reset.";
    EXPECT_EQ(c2, c1)
        << "[TC-Arena-004] Possible issue: cpuOffset not rewound to base in Arena::reset.";
    EXPECT_EQ(g2, g1)
        << "[TC-Arena-004] Possible issue: gpuOffset not rewound to base in Arena::reset.";

    Tensor keep({1.0f, 2.0f, 3.0f, 4.0f}, {4}, {1});
    ASSERT_EQ(keep.device, Device::CPU);
    for (int i = 0; i < 4; i++)
        ASSERT_EQ(keep.data[i], static_cast<float>(i + 1));
    globalArena.reset();
    Tensor fresh({4});
    ASSERT_NE(fresh.data, nullptr);
    EXPECT_EQ(keep.data, fresh.data)
        << "[TC-Arena-004] Possible issue: reset must rewind the arena for storage reuse.";
}
