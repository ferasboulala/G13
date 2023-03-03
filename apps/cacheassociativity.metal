#include <metal_stdlib>
using namespace metal;

kernel void cacheAssociativity(
    device uint32_t *indices,
    device const uint32_t &expectedNReads)
{
    uint32_t nReads = 1;
    uint32_t ptr = indices[0];
    while (nReads < expectedNReads) {
        ++nReads;
        ptr = indices[ptr];
    }

    indices[ptr] = 0;
}