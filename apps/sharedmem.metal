#include <metal_stdlib>

using namespace metal;

kernel void sharedMemory(device uint32_t       *indices,
                        device const uint32_t &expectedNReads) {
    constexpr uint32_t LOCAL_MEMORY_SIZE = 1 << 15;
    constexpr uint32_t LOCAL_MEMORY_N_ELEMENTS = LOCAL_MEMORY_SIZE / sizeof(uint32_t);

    threadgroup uint32_t localIndices[LOCAL_MEMORY_N_ELEMENTS];
    for (uint32_t i = 0; i < LOCAL_MEMORY_N_ELEMENTS; ++i) {
        localIndices[i] = indices[i];
    }

    uint32_t nReads = 1;
    uint32_t ptr = localIndices[0];
    while (nReads < expectedNReads) {
        ++nReads;
        ptr = localIndices[ptr];
    }

    indices[ptr] = 0;
}