#include <metal_stdlib>

using namespace metal;

kernel void sharedMemory(device uint64_t       *indices,
                         device const uint64_t &nIterations) {
    constexpr uint64_t LOCAL_MEMORY_SIZE = 1 << 15;
    constexpr uint64_t LOCAL_MEMORY_N_ELEMENTS =
        LOCAL_MEMORY_SIZE / sizeof(uint64_t);

    threadgroup uint64_t localIndices[LOCAL_MEMORY_N_ELEMENTS];
    for (uint64_t i = 0; i < LOCAL_MEMORY_N_ELEMENTS; ++i) {
        localIndices[i] = indices[i];
    }

    uint64_t index = 0;
    do {
        const uint64_t nextIndex = localIndices[index];
        indices[index] = (uint64_t)(localIndices + nextIndex);
        index = nextIndex;
    } while (index);

    uint64_t iter = nIterations;
    threadgroup uint64_t *address = localIndices;
    while (--iter) {
        address = (threadgroup uint64_t*)(*address);
    }

    indices[0] = *address;
}