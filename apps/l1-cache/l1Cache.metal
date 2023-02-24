#include <metal_stdlib>
using namespace metal;

kernel void l1CacheIntrospection(
    device uint32_t *indices,
    device const uint32_t &nIterations)
{
    uint32_t iter = 0;
    uint32_t dummy = 123;
    do {
        uint32_t ptr = indices[0];
        while (ptr) {
            ptr = indices[ptr];
        }
        dummy += indices[ptr];
    } while (++iter < nIterations);

    indices[0] = dummy;
}