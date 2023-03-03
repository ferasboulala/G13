#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"
#include "measure.hh"

template <typename T>
void prepareStridedIndices(T *buffer, uint64_t size, uint64_t stride) {
    assert(stride);
    assert(size % stride == 0);
    assert(std::numeric_limits<T>::max() >= size);

    const uint64_t nIterations = size / stride;
    for (uint64_t i = 0; i < nIterations - 1; ++i) {
        buffer[i * stride] = (i + 1) * stride;
    }
    buffer[(nIterations - 1) * stride] = 0;
}

template <typename T>
inline bool isAPowerOfTwo(T x) {
    return (x & (x - 1)) == 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("usage: %s L1CacheSizeInBytes\n", argv[0]);
        return -1;
    }

    const uint64_t L1CacheSizeInBytes = std::atoll(argv[1]);
    assert(L1CacheSizeInBytes);
    assert(isAPowerOfTwo(L1CacheSizeInBytes));

    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    assert(device);

    const std::string libraryPath = std::string(argv[0]) + "_library";
    printf("Compute shader library in %s\n", libraryPath.c_str());
    NS::Error    *error = nullptr;
    MTL::Library *computeShaderLibrary = device->newLibrary(
        NS::String::string(libraryPath.c_str(), NS::ASCIIStringEncoding), &error);
    assert(computeShaderLibrary);

    static const char *kernelName = "cacheAssociativity";
    const auto         kernelNameNS = NS::String::string(kernelName, NS::ASCIIStringEncoding);
    MTL::Function     *kernelHandle = computeShaderLibrary->newFunction(kernelNameNS);
    assert(kernelHandle);

    auto computePipelineState = device->newComputePipelineState(kernelHandle, &error);
    assert(computePipelineState);

    auto commandQueue = device->newCommandQueue();
    assert(commandQueue);

    static constexpr uint64_t maxAssociativity = 256;
    static constexpr uint32_t nExpectedReads = 1 << 24;
    static const uint64_t     bufferSize = 2 * L1CacheSizeInBytes / sizeof(uint32_t);
    assert(L1CacheSizeInBytes / maxAssociativity > sizeof(uint32_t));

    auto indicesBuffer = device->newBuffer(L1CacheSizeInBytes, MTL::ResourceStorageModeShared);

    for (uint64_t associativity = 1; associativity <= maxAssociativity; associativity <<= 1) {
        const uint64_t strideInBytes = L1CacheSizeInBytes / associativity;
        const uint64_t stride = strideInBytes / sizeof(uint32_t);

        prepareStridedIndices(
            reinterpret_cast<uint32_t *>(indicesBuffer->contents()), bufferSize, stride);

        MTL::CommandBuffer *commandBuffer = commandQueue->commandBuffer();
        assert(commandBuffer);

        MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
        assert(computeEncoder);

        computeEncoder->setComputePipelineState(computePipelineState);
        computeEncoder->setBuffer(indicesBuffer, 0, 0);
        computeEncoder->setBytes(&nExpectedReads, sizeof(nExpectedReads), 1);

        const MTL::Size threadgroupSize = MTL::Size::Make(1, 1, 1);
        const MTL::Size gridSize = MTL::Size::Make(1, 1, 1);
        computeEncoder->dispatchThreadgroups(gridSize, threadgroupSize);
        computeEncoder->endEncoding();

        const auto commit = [&]() {
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
        };

        const auto duration = g13::measureTime(commit);
        printf("%12llu %12lld ns\n", associativity, duration);
        fflush(stdout);
    }

    return 0;
}