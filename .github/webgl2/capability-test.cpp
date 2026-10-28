// Checks that the WebGL 2 adapter reports what it can do, and that the device
// rejects what it cannot do with validation errors instead of failing in GL.
#include <cstdint>
#include <string>

#include "test.h"

namespace {

using namespace test;

void TestLimits() {
    wgpu::Limits adapterLimits;
    adapter.GetLimits(&adapterLimits);
    wgpu::Limits deviceLimits;
    device.GetLimits(&deviceLimits);

    // WebGL 2 has no storage buffers, storage textures or compute, and caps strides at 255.
    struct Expected {
        const char* name;
        uint64_t adapterValue;
        uint64_t deviceValue;
        uint64_t max;
    };
    const Expected expected[] = {
        {"maxStorageBuffersPerShaderStage", adapterLimits.maxStorageBuffersPerShaderStage,
         deviceLimits.maxStorageBuffersPerShaderStage, 0},
        {"maxStorageTexturesPerShaderStage", adapterLimits.maxStorageTexturesPerShaderStage,
         deviceLimits.maxStorageTexturesPerShaderStage, 0},
        {"maxComputeInvocationsPerWorkgroup", adapterLimits.maxComputeInvocationsPerWorkgroup,
         deviceLimits.maxComputeInvocationsPerWorkgroup, 0},
        {"maxComputeWorkgroupsPerDimension", adapterLimits.maxComputeWorkgroupsPerDimension,
         deviceLimits.maxComputeWorkgroupsPerDimension, 0},
        {"maxVertexBufferArrayStride", adapterLimits.maxVertexBufferArrayStride,
         deviceLimits.maxVertexBufferArrayStride, 255},
    };
    for (const Expected& limit : expected) {
        std::string name = std::string("adapter ") + limit.name;
        Check(name.c_str(), limit.adapterValue <= limit.max,
              std::to_string(limit.adapterValue) + " > " + std::to_string(limit.max));
        name = std::string("device ") + limit.name;
        Check(name.c_str(), limit.deviceValue <= limit.adapterValue,
              std::to_string(limit.deviceValue) + " > adapter's " +
                  std::to_string(limit.adapterValue));
    }
}

void TestStorageBufferBinding() {
    std::string error = ValidationErrorOf([] {
        wgpu::BindGroupLayoutEntry entry{
            .binding = 0,
            .visibility = wgpu::ShaderStage::Fragment,
            .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage},
        };
        wgpu::BindGroupLayoutDescriptor descriptor{.entryCount = 1, .entries = &entry};
        device.CreateBindGroupLayout(&descriptor);
    });
    CheckRejected("storage buffer binding is rejected", error);
}

void TestComputePipeline() {
    std::string error = ValidationErrorOf([] {
        wgpu::ShaderSourceWGSL wgsl{{.code = "@compute @workgroup_size(1) fn main() {}"}};
        wgpu::ShaderModuleDescriptor shaderDescriptor{.nextInChain = &wgsl};
        wgpu::ComputePipelineDescriptor descriptor{
            .compute = {.module = device.CreateShaderModule(&shaderDescriptor),
                        .entryPoint = "main"},
        };
        device.CreateComputePipeline(&descriptor);
    });
    CheckRejected("compute pipeline is rejected", error);
}

void TestIndexBufferUsages() {
    std::string error = ValidationErrorOf([] {
        wgpu::BufferDescriptor descriptor{
            .usage = wgpu::BufferUsage::Index | wgpu::BufferUsage::Vertex, .size = 16};
        device.CreateBuffer(&descriptor);
    });
    CheckRejected("Index | Vertex buffer is rejected", error);

    error = ValidationErrorOf([] {
        wgpu::BufferDescriptor descriptor{
            .usage = wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst |
                     wgpu::BufferUsage::CopySrc,
            .size = 16};
        device.CreateBuffer(&descriptor);
    });
    Check("Index | CopyDst | CopySrc buffer is accepted", error.empty(), error);
}

}  // namespace

int main() {
    if (!Setup()) {
        return 1;
    }
    TestLimits();
    TestStorageBufferBinding();
    TestComputePipeline();
    TestIndexBufferUsages();
    FinishWhen([] { return true; });
    return 0;
}
