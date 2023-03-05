#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"
#include "indices.hh"
#include "measure.hh"

int main(int argc, char **argv) {
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    assert(device);

    const std::string libraryPath = std::string(argv[0]) + "_library";
    NS::Error        *error = nullptr;
    MTL::Library     *computeShaderLibrary = device->newLibrary(
        NS::String::string(libraryPath.c_str(), NS::ASCIIStringEncoding), &error);
    assert(computeShaderLibrary);

    static const char *kernelName = "sharedMemory";
    const auto         kernelNameNS = NS::String::string(kernelName, NS::ASCIIStringEncoding);
    MTL::Function     *kernelHandle = computeShaderLibrary->newFunction(kernelNameNS);
    assert(kernelHandle);

    auto computePipelineState = device->newComputePipelineState(kernelHandle, &error);
    assert(computePipelineState);

    auto commandQueue = device->newCommandQueue();
    assert(commandQueue);

    MTL::CommandBuffer *commandBuffer = commandQueue->commandBuffer();
    assert(commandBuffer);

    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder);

    static constexpr uint32_t SHARED_MEMORY_SIZE = 1 << 15;
    static constexpr uint32_t SHARED_MEMORY_N_ELEMENTS = SHARED_MEMORY_SIZE / sizeof(uint32_t);
    static constexpr uint32_t EXPECTED_N_READS = 1 << 24;

    auto indicesBuffer = device->newBuffer(SHARED_MEMORY_SIZE, MTL::ResourceStorageModeShared);
    auto indices = g13::satolloRandomIndices<uint32_t>(SHARED_MEMORY_N_ELEMENTS);
    std::memcpy(indicesBuffer->contents(), indices.data(), indicesBuffer->allocatedSize());

    computeEncoder->setComputePipelineState(computePipelineState);
    computeEncoder->setBuffer(indicesBuffer, 0, 0);
    computeEncoder->setBytes(&EXPECTED_N_READS, sizeof(EXPECTED_N_READS), 1);

    const MTL::Size threadgroupSize = MTL::Size::Make(1, 1, 1);
    const MTL::Size gridSize = MTL::Size::Make(1, 1, 1);
    computeEncoder->dispatchThreadgroups(gridSize, threadgroupSize);
    computeEncoder->endEncoding();

    const auto commit = [&]() {
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
    };
    const auto duration = g13::measureTime(commit);

    printf("%lld ns -> %lld ns/item", duration, duration / EXPECTED_N_READS);

    return 0;
}