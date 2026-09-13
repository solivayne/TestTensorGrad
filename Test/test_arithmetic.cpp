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