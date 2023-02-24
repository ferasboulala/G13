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
std::vector<T> satolloRandomIndices(uint64_t size) {
    assert(size >= 2);
    std::random_device               randomDevice;
    std::mt19937                     generator(randomDevice());
    std::uniform_int_distribution<T> distribution;
    std::vector<T>                   indices(size);

    std::iota(indices.begin(), indices.end(), 0);
    for (uint64_t i = size - 1; i > 1; --i) {
        std::swap(indices[i], indices[distribution(generator) % i]);
    }
    std::swap(indices[0], indices[1]);

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
    static constexpr uint32_t nExpectedReads = maxSize * 8;  // 128 MB
    for (uint64_t size = 1 << 8; size <= maxSize; size <<= 1) {
        const uint32_t nIterations = nExpectedReads / size;

        auto indicesBuffer =
            device->newBuffer(size * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        const auto indices = satolloRandomIndices<uint32_t>(size);
        std::memcpy(indicesBuffer->contents(), indices.data(), indicesBuffer->allocatedSize());

        MTL::CommandBuffer *commandBuffer = commandQueue->commandBuffer();
        assert(commandBuffer);

        MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
        assert(computeEncoder);

        computeEncoder->setComputePipelineState(computePipelineState);
        computeEncoder->setBuffer(indicesBuffer, 0, 0);
        computeEncoder->setBytes(&nIterations, sizeof(nIterations), 1);

        const MTL::Size threadgroupSize = MTL::Size::Make(1, 1, 1);
        const MTL::Size gridSize = MTL::Size::Make(1, 1, 1);
        computeEncoder->dispatchThreadgroups(gridSize, threadgroupSize);
        computeEncoder->endEncoding();

        const auto commit = [&]() {
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
        };
        const auto duration = g13::measureTime(commit);

        printf("%12u %12lld ns %4lld ns/read\n",
               uint32_t(indicesBuffer->allocatedSize()),
               duration,
               duration / nExpectedReads);
        fflush(stdout);
    }

    return 0;
}