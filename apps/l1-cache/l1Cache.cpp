#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"
#include "measure.hh"

template <typename T>
std::vector<T> generateRandomIndices(uint64_t size) {
    std::random_device               randomDevice;
    std::mt19937                     generator(randomDevice());
    std::uniform_int_distribution<T> distribution(0, size - 1);
    std::vector<T>                   indices(size);

    for (uint64_t i = 0; i < size; ++i) {
        indices[i] = distribution(generator);
    }

    return indices;
}

int main(int argc, char **argv) {
    printf("%s to determine M1 GPU L1 cache size\n", argv[0]);
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    assert(device);

    const std::string libraryPath = std::string(argv[0]) + "_library";
    printf("Compute shader library in %s\n", libraryPath.c_str());
    NS::Error    *error = nullptr;
    MTL::Library *computeShaderLibrary = device->newLibrary(
        NS::String::string(libraryPath.c_str(), NS::ASCIIStringEncoding), &error);
    assert(computeShaderLibrary);

    static const char *kernelName = "l1CacheIntrospection";
    const auto         kernelNameNS = NS::String::string(kernelName, NS::ASCIIStringEncoding);
    MTL::Function     *kernelHandle = computeShaderLibrary->newFunction(kernelNameNS);
    assert(kernelHandle);

    auto computePipelineState = device->newComputePipelineState(kernelHandle, &error);
    assert(computePipelineState);

    auto commandQueue = device->newCommandQueue();
    assert(commandQueue);

    static constexpr uint64_t maxSize = 1 << 22;
    static constexpr uint64_t nReads = maxSize * 8;
    for (uint64_t size = 1 << 10; size <= maxSize; size <<= 1) {
        const uint64_t nIterations = nReads / size;

        auto indicesPermutationBuffer =
            device->newBuffer(size * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        const auto indicesPermutation = generateRandomIndices<uint32_t>(size);
        std::memcpy(indicesPermutationBuffer->contents(),
                    indicesPermutation.data(),
                    indicesPermutationBuffer->allocatedSize());

        MTL::CommandBuffer *commandBuffer = commandQueue->commandBuffer();
        assert(commandBuffer);

        MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
        assert(computeEncoder);

        auto inputBuffer = device->newBuffer(size, MTL::ResourceStorageModeShared);

        computeEncoder->setComputePipelineState(computePipelineState);
        computeEncoder->setBuffer(inputBuffer, 0, 0);
        computeEncoder->setBuffer(indicesPermutationBuffer, 0, 1);
        computeEncoder->setBytes(&size, sizeof(size), 2);
        computeEncoder->setBytes(&nIterations, sizeof(nIterations), 3);

        const MTL::Size threadgroupSize = MTL::Size::Make(1, 1, 1);
        const MTL::Size gridSize = MTL::Size::Make(1, 1, 1);
        computeEncoder->dispatchThreadgroups(gridSize, threadgroupSize);
        computeEncoder->endEncoding();

        const auto commit = [&]() {
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
        };
        const auto duration = g13::measureTime(commit);
        printf("%12llu %12lld ns %4lld ns/iter\n", size, duration, duration / nReads);
        fflush(stdout);
    }

    return 0;
}