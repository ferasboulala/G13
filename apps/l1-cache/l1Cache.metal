#include <metal_stdlib>
using namespace metal;

#define REPEAT2(x) x x 
#define REPEAT4(x) REPEAT2(x) REPEAT2(x)
#define REPEAT8(x) REPEAT4(x) REPEAT4(x)
#define REPEAT16(x) REPEAT8(x) REPEAT8(x)
#define REPEAT(x) REPEAT16(x) REPEAT16(x)

kernel void l1CacheIntrospection(
    device char *inputBuffer,
    device const uint32_t *indicesPermutation,
    device const uint64_t &bufferSize,
    device const uint64_t &nIterations)
{
    for (unsigned iter = 0; iter < nIterations; ++iter) {
        for (uint64_t i = 0; i < bufferSize; ++i) {
            inputBuffer[indicesPermutation[i++]]++;
        }
    }
}