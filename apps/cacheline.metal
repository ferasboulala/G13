#include <metal_stdlib>
using namespace metal;

uint32_t pointerChase(const device uint32_t *indices) {
    uint32_t ptr = indices[0];
    while (ptr) {
        ptr = indices[ptr];
    }

    return indices[ptr];
}

kernel void cacheLine(device uint32_t *indices)
{
    const uint32_t result = pointerChase(indices);
    indices[result] = result;
}