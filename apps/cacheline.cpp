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
#include "measure.hh"

template <typename T>
void prepareStridedIndices(T *buffer, uint64_t size, uint64_t stride) {
    assert(stride);
    assert(size % stride == 0);
    assert(std::numeric_limits<T>::max() >= size);

    const uint64_t sizePerStep = size / stride;
    for (uint64_t step = 0; step < stride; ++step) {
        for (uint64_t i = 0; i < sizePerStep - 1; ++i) {
            buffer[i * stride + step] = (i + 1) * stride + step;
        }
        buffer[(sizePerStep - 1) * stride + step] = step + 1;
    }
    buffer[(sizePerStep - 1) * stride + stride - 1] = 0;
}

int main(int argc, char **argv) {
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    assert(device);

    const std::string libraryPath = std::string(argv[0]) + "_library";
    NS::Error    *error = nullptr;
    MTL::Library *computeShaderLibrary = device->newLibrary(
        NS::String::string(libraryPath.c_str(), NS::ASCIIStringEncoding), &error);
    assert(computeShaderLibrary);

    static const char *kernelName = "cacheLine";
    const auto         kernelNameNS = NS::String::string(kernelName, NS::ASCIIStringEncoding);
    MTL::Function     *kernelHandle = computeShaderLibrary->newFunction(kernelNameNS);
    assert(kernelHandle);

    auto computePipelineState = device->newComputePipelineState(kernelHandle, &error);
    assert(computePipelineState);

    auto commandQueue = device->newCommandQueue();
    assert(commandQueue);

    static constexpr uint64_t MAX_STRIDE = 512;
    static constexpr uint64_t BUFFER_SIZE = 1 << 24;

    auto indicesBuffer =
        device->newBuffer(BUFFER_SIZE * sizeof(uint32_t), MTL::ResourceStorageModeShared);

    for (uint64_t stride = 1; stride <= MAX_STRIDE; stride <<= 1) {
        prepareStridedIndices(
            reinterpret_cast<uint32_t *>(indicesBuffer->contents()), BUFFER_SIZE, stride);

        MTL::CommandBuffer *commandBuffer = commandQueue->commandBuffer();
        assert(commandBuffer);

        MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
        assert(computeEncoder);

        computeEncoder->setComputePipelineState(computePipelineState);
        computeEncoder->setBuffer(indicesBuffer, 0, 0);

        const MTL::Size threadgroupSize = MTL::Size::Make(1, 1, 1);
        const MTL::Size gridSize = MTL::Size::Make(1, 1, 1);
        computeEncoder->dispatchThreadgroups(gridSize, threadgroupSize);
        computeEncoder->endEncoding();

        const auto commit = [&]() {
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
        };

        const auto duration = g13::measureTime(commit);
        printf("%12llu %12lld ns %12llu ns/element\n",
               stride * sizeof(uint32_t),
               duration,
               duration / BUFFER_SIZE);
        fflush(stdout);
    }

    return 0;
}