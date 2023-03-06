#include <metal_stdlib>
using namespace metal;

kernel void cacheSize(
    device uint64_t *indices,
    device const uint64_t &nIterations)
{
    uint64_t index = 0;
    do {
        const uint64_t nextIndex = indices[index];
        indices[index] = (uint64_t)(indices + nextIndex);
        index = nextIndex;
    } while (index);

    uint64_t iter = nIterations;
    device uint64_t *address = indices;
    while (--iter) {
        address = (device uint64_t*)(*address);
    }

    indices[0] = *address;
}