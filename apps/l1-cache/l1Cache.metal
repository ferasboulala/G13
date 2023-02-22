#include <metal_stdlib>
using namespace metal;

static inline uint32_t rand(uint32_t prev) {
    return prev * 1664525 + 1013904223;
}

kernel void l1CacheIntrospection(
    device float* inputBuffer,
    device const uint64_t &bufferSize,
    device const uint32_t &nIterations)
{
    for (uint64_t i = 0; i < bufferSize; ++i) {
        ++inputBuffer[i];
    }

    for (uint32_t i = 0, index = 0; i < nIterations; ++i, index = rand(index)) {
        ++inputBuffer[index % bufferSize];
    }
}

