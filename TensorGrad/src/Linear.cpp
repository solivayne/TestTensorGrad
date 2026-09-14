#include "Linear.hpp"
#include <random>
#include <memory>
#include <cmath>

Linear::Linear(int inFeatures, int outFeatures) : 
    weights({inFeatures, outFeatures}, true), 
    bias({1, outFeatures}, true), 
    matmulCache({1})
{
    // A function-local generator restarts from seed 42 for every layer, so
    // layers with the same shape receive identical initial weights.
    // std::mt19937 gen(42);
    static thread_local std::mt19937 gen(42);
    float limit = std::sqrt(6.0 / (inFeatures + outFeatures));
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (size_t i = 0; i < weights.size; i++) {
        weights.data[i] = dist(gen);
    }
    for (size_t i = 0; i < bias.size; i++) {
        bias.data[i] = 0.0;
    }
}

Tensor Linear::operator()(const Tensor& input) {
    matmulCache = input * weights; 

    // Returning this expression directly stores a raw pointer to matmulCache
    // in the output graph. The next forward call overwrites that same object,
    // so an earlier output can no longer backpropagate through its own matmul.
    // return matmulCache + bias;

    // Give each output an independently owned matmul node. Capturing the node
    // in the backward closure keeps it alive for the graph's whole lifetime.
    auto node = std::make_shared<Tensor>(matmulCache);
    Tensor result = *node + bias;
    auto backward = result._backward;
    result._backward = [node, backward](const float* grad) {
        static_cast<void>(node);
        backward(grad);
    };
    return result;
}

std::vector<Tensor*> Linear::parameters() {
    return { &weights, &bias };
}
