#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

    static const char *kernelName = "cacheSize";
    const auto         kernelNameNS = NS::String::string(kernelName, NS::ASCIIStringEncoding);
    MTL::Function     *kernelHandle = computeShaderLibrary->newFunction(kernelNameNS);
    assert(kernelHandle);

    auto computePipelineState = device->newComputePipelineState(kernelHandle, &error);
    assert(computePipelineState);

    auto commandQueue = device->newCommandQueue();
    assert(commandQueue);

    // The kernel will execute extremely fast on small inputs. To aleviate noisy measurements,
    // we execute the reads many times using a loop. Final duration is divided by the number of
    // iterations to accomodate for that.
    static constexpr uint64_t MAX_SIZE = 1 << 24;
    static constexpr uint32_t N_EXPECTED_READS = 1 << 24;

    for (uint64_t size = 1 << 8; size <= MAX_SIZE; size <<= 1) {
        const uint32_t nIterations = N_EXPECTED_READS / size;

        auto indicesBuffer =
            device->newBuffer(size * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        const auto indices = g13::satolloRandomIndices<uint32_t>(size);
        std::memcpy(indicesBuffer->contents(), indices.data(), indicesBuffer->allocatedSize());

        MTL::CommandBuffer *commandBuffer = commandQueue->commandBuffer();
        assert(commandBuffer);

        MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
        assert(computeEncoder);

        computeEncoder->setComputePipelineState(computePipelineState);
        computeEncoder->setBuffer(indicesBuffer, 0, 0);
        computeEncoder->setBytes(&nIterations, sizeof(nIterations), 1);

        const MTL::Size threadgroupSize = MTL::Size::Make(1, 1, 1);
        const MTL::Size gridSize = MTL::Size::Make(32, 1, 1);
        computeEncoder->dispatchThreadgroups(gridSize, threadgroupSize);
        computeEncoder->endEncoding();

        const auto commit = [&]() {
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
        };

        const auto     duration = g13::measureTime(commit);
        const uint64_t durationPerInnerLoopIteration = duration / N_EXPECTED_READS;
        printf("%12u %12lld ns %4lld ns/read\n",
               uint32_t(indicesBuffer->allocatedSize()),
               duration,
               durationPerInnerLoopIteration);
        fflush(stdout);
    }

    return 0;
}