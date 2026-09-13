// Shared test fixture: resets globalArena before and after each test.
#pragma once

#include <gtest/gtest.h>
#include <vector>
#include "Arena.hpp"
#include "Tensor.hpp"
#include "cuda_kernels.cuh"

inline std::vector<int> rowMajor(const std::vector<int> &shape)
{
    std::vector<int> strides(shape.size());
    int current = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
    {
        strides[i] = current;
        current *= shape[i];
    }
    return strides;
}

inline size_t product(const std::vector<int> &shape)
{
    size_t result = 1;
    for (int dimension : shape)
        result *= static_cast<size_t>(dimension);
    return result;
}

inline Tensor make(const std::vector<float> &data,
                   const std::vector<int> &shape,
                   bool isParam = false)
{
    return Tensor(data, shape, rowMajor(shape), isParam);
}

inline void fill(Tensor &tensor, float value)
{
    for (size_t i = 0; i < tensor.size; ++i)
        tensor.data[i] = value;
}

inline void seed(Tensor &tensor, float value = 1.0f)
{
    for (size_t i = 0; i < tensor.size; ++i)
        tensor.grad[i] = value;
}

// naive mat multiply for evaluate the effectiveness of TensorGrad MatMul
inline std::vector<float> refMatmul(const std::vector<float> &a,
                                    const std::vector<float> &b,
                                    int rowsA, int inner, int colsB)
{
    std::vector<float> result(rowsA * colsB, 0.0f);
    for (int i = 0; i < rowsA; ++i)
        for (int j = 0; j < colsB; ++j)
            for (int k = 0; k < inner; ++k)
                result[i * colsB + j] += a[i * inner + k] * b[k * colsB + j];
    return result;
}

inline void gpuFill(float *destination, size_t count, float value)
{
    std::vector<float> buffer(count, value);
    copyMemory(destination, buffer.data(), count * sizeof(float), true);
}

inline std::vector<float> gpuRead(const float *source, size_t count)
{
    std::vector<float> buffer(count, -1.0f);
    copyMemory(buffer.data(), source, count * sizeof(float), false);
    return buffer;
}

class EngineTest : public ::testing::Test
{
protected:
    void SetUp() override { globalArena.reset(); }
    void TearDown() override { globalArena.reset(); }
};
