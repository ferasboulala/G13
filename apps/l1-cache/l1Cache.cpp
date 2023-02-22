#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"
#include "measure.hh"

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

    static constexpr uint64_t fp32Size = sizeof(float);
    static const uint64_t     maxSize = 1 << 22;
    static_assert(fp32Size == 4);
    auto inputBuffer = device->newBuffer(maxSize * fp32Size, MTL::ResourceStorageModeShared);
    std::memset(inputBuffer->contents(), 0, inputBuffer->allocatedSize());

    const uint32_t nIterations = 1 << 20;
    for (uint64_t size = 1 << 10; size <= maxSize; size <<= 1) {
        MTL::CommandBuffer *commandBuffer = commandQueue->commandBuffer();
        assert(commandBuffer);
        MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
        assert(computeEncoder);

        computeEncoder->setComputePipelineState(computePipelineState);
        computeEncoder->setBuffer(inputBuffer, 0, 0);
        computeEncoder->setBytes(&size, sizeof(size), 1);
        computeEncoder->setBytes(&nIterations, sizeof(nIterations), 2);

        const MTL::Size threadgroupSize = MTL::Size::Make(1, 1, 1);
        const MTL::Size gridSize = MTL::Size::Make(1, 1, 1);
        computeEncoder->dispatchThreadgroups(gridSize, threadgroupSize);
        computeEncoder->endEncoding();

        const auto commit = [&]() {
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
        };
        const auto duration = g13::measureTime(commit);
        printf("size=%llu : %lld us\n", size, duration);
        fflush(stdout);
    }

    return 0;
}