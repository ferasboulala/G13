#include <metal_stdlib>
using namespace metal;

inline uint32_t pointerChase(const device uint32_t *indices) {
    uint32_t ptr = indices[0];
    while (ptr) {
        ptr = indices[ptr];
    }

    return indices[ptr];
}

kernel void cacheIntrospection(
    device uint32_t *indices,
    device const uint32_t &nIterations)
{
    uint32_t dummy = 0;
    for (uint32_t iter = 0; iter < nIterations; ++iter) {
        dummy += pointerChase(indices);
    }

    indices[0] = dummy;
}