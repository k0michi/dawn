// Checks the index 0xFFFF on WebGL 2, which always takes it as primitive
// restart: list topologies must draw it as vertex 65535, as WebGPU does, and
// strip topologies must restart at it. Vertices 0, 1 and 65535 make a triangle
// covering the target; every other vertex is outside it.
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "test.h"

namespace {

using namespace test;

constexpr uint32_t kSize = 4;
constexpr uint32_t kVertexCount = 65536;
int pending = 0;

const char* kShader = R"(
@vertex fn vertexMain(@location(0) position: vec2f) -> @builtin(position) vec4f {
    return vec4f(position, 0.0, 1.0);
}
@fragment fn fragmentMain() -> @location(0) vec4f {
    return vec4f(1.0, 0.0, 0.0, 1.0);
}
)";

wgpu::Buffer vertexBuffer;

wgpu::Buffer MakeBuffer(wgpu::BufferUsage usage, const std::vector<uint16_t>& data) {
    wgpu::BufferDescriptor descriptor{.usage = usage | wgpu::BufferUsage::CopyDst,
                                      .size = (data.size() * 2 + 3) & ~size_t(3)};
    wgpu::Buffer buffer = device.CreateBuffer(&descriptor);
    std::vector<uint16_t> padded = data;
    padded.resize(descriptor.size / 2);
    device.GetQueue().WriteBuffer(buffer, 0, padded.data(), descriptor.size);
    return buffer;
}

wgpu::RenderPipeline MakePipeline(wgpu::PrimitiveTopology topology) {
    wgpu::ShaderSourceWGSL wgsl{{.code = kShader}};
    wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl};
    wgpu::ShaderModule module = device.CreateShaderModule(&moduleDescriptor);
    wgpu::VertexAttribute attribute{.format = wgpu::VertexFormat::Float32x2};
    wgpu::VertexBufferLayout layout{
        .arrayStride = 8, .attributeCount = 1, .attributes = &attribute};
    wgpu::ColorTargetState target{.format = wgpu::TextureFormat::RGBA8Unorm};
    wgpu::FragmentState fragment{.module = module, .targetCount = 1, .targets = &target};
    const bool strip = topology == wgpu::PrimitiveTopology::TriangleStrip;
    wgpu::RenderPipelineDescriptor descriptor{
        .vertex = {.module = module, .bufferCount = 1, .buffers = &layout},
        .primitive = {.topology = topology,
                      .stripIndexFormat =
                          strip ? wgpu::IndexFormat::Uint16 : wgpu::IndexFormat::Undefined},
        .fragment = &fragment,
    };
    return device.CreateRenderPipeline(&descriptor);
}

// Draws |indexCount| indices from |firstIndex| of |indexBuffer| at |offset| and checks whether
// the target is red everywhere or black everywhere.
void Draw(const char* name,
          wgpu::PrimitiveTopology topology,
          wgpu::Buffer indexBuffer,
          uint64_t offset,
          uint32_t indexCount,
          uint32_t firstIndex,
          bool expectDrawn) {
    wgpu::TextureDescriptor targetDescriptor{
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc,
        .size = {kSize, kSize},
        .format = wgpu::TextureFormat::RGBA8Unorm};
    wgpu::Texture target = device.CreateTexture(&targetDescriptor);
    wgpu::RenderPassColorAttachment attachment{.view = target.CreateView(),
                                               .loadOp = wgpu::LoadOp::Clear,
                                               .storeOp = wgpu::StoreOp::Store,
                                               .clearValue = {0, 0, 0, 1}};
    wgpu::RenderPassDescriptor passDescriptor{.colorAttachmentCount = 1,
                                              .colorAttachments = &attachment};
    wgpu::BufferDescriptor bufferDescriptor{
        .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, .size = 256 * kSize};
    wgpu::Buffer readback = device.CreateBuffer(&bufferDescriptor);

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDescriptor);
    pass.SetPipeline(MakePipeline(topology));
    pass.SetVertexBuffer(0, vertexBuffer);
    pass.SetIndexBuffer(indexBuffer, wgpu::IndexFormat::Uint16, offset);
    pass.DrawIndexed(indexCount, 1, firstIndex);
    pass.End();
    wgpu::TexelCopyTextureInfo from{.texture = target};
    wgpu::TexelCopyBufferInfo to{.layout = {.bytesPerRow = 256}, .buffer = readback};
    wgpu::Extent3D size{kSize, kSize};
    encoder.CopyTextureToBuffer(&from, &to, &size);
    wgpu::CommandBuffer commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);

    pending++;
    readback.MapAsync(
        wgpu::MapMode::Read, 0, 256 * kSize, wgpu::CallbackMode::AllowProcessEvents,
        [name, readback, expectDrawn](wgpu::MapAsyncStatus status, wgpu::StringView) {
            pending--;
            if (status != wgpu::MapAsyncStatus::Success) {
                Check(name, false, "mapAsync failed");
                return;
            }
            const auto* data =
                static_cast<const uint8_t*>(readback.GetConstMappedRange(0, 256 * kSize));
            std::string why;
            for (uint32_t y = 0; y < kSize && why.empty(); y++) {
                for (uint32_t x = 0; x < kSize && why.empty(); x++) {
                    if ((data[y * 256 + x * 4] == 255) != expectDrawn) {
                        why = std::string("texel ") + std::to_string(x) + "," + std::to_string(y) +
                              (expectDrawn ? " is not drawn" : " is drawn");
                    }
                }
            }
            Check(name, why.empty(), why);
            readback.Unmap();
        });
}

}  // namespace

int main() {
    if (!Setup()) {
        return 1;
    }
    std::vector<float> positions(kVertexCount * 2, 2.0f);
    const float corners[3][2] = {{-1, -1}, {3, -1}, {-1, 3}};
    for (uint32_t i : {0u, 1u}) {
        positions[i * 2] = corners[i][0];
        positions[i * 2 + 1] = corners[i][1];
    }
    positions[0xffff * 2] = corners[2][0];
    positions[0xffff * 2 + 1] = corners[2][1];
    wgpu::BufferDescriptor vertexDescriptor{
        .usage = wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst,
        .size = positions.size() * 4};
    vertexBuffer = device.CreateBuffer(&vertexDescriptor);
    device.GetQueue().WriteBuffer(vertexBuffer, 0, positions.data(), vertexDescriptor.size);

    const auto kList = wgpu::PrimitiveTopology::TriangleList;
    const std::vector<uint16_t> triangle = {0, 1, 0xffff};

    Draw("a triangle list draws index 0xFFFF as vertex 65535", kList,
         MakeBuffer(wgpu::BufferUsage::Index, triangle), 0, 3, 0, true);

    // Behind an offset and a first index: 7 is outside the target.
    Draw("with an index buffer offset and a first index", kList,
         MakeBuffer(wgpu::BufferUsage::Index, {7, 7, 7, 7, 0, 1, 0xffff}), 2, 3, 3, true);

    {
        wgpu::BufferDescriptor descriptor{
            .usage = wgpu::BufferUsage::Index, .size = 8, .mappedAtCreation = true};
        wgpu::Buffer buffer = device.CreateBuffer(&descriptor);
        // Mapped ranges are multiples of 4 bytes.
        auto* mapped = static_cast<uint16_t*>(buffer.GetMappedRange(0, 8));
        std::memcpy(mapped, triangle.data(), 6);
        buffer.Unmap();
        Draw("indices written with mappedAtCreation", kList, buffer, 0, 3, 0, true);
    }

    for (bool fromIndexBuffer : {false, true}) {
        wgpu::Buffer source =
            MakeBuffer(fromIndexBuffer ? wgpu::BufferUsage::Index | wgpu::BufferUsage::CopySrc
                                       : wgpu::BufferUsage::CopySrc,
                       triangle);
        wgpu::Buffer destination = MakeBuffer(wgpu::BufferUsage::Index, {0, 0, 0});
        wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
        encoder.CopyBufferToBuffer(source, 0, destination, 0, 8);
        wgpu::CommandBuffer commands = encoder.Finish();
        device.GetQueue().Submit(1, &commands);
        Draw(fromIndexBuffer ? "indices copied from another index buffer"
                             : "indices copied from a buffer that isn't an index buffer",
             kList, destination, 0, 3, 0, true);
    }

    // A strip restarts at 0xFFFF: 0 1 then 2 alone make no triangle.
    Draw("a triangle strip restarts at 0xFFFF", wgpu::PrimitiveTopology::TriangleStrip,
         MakeBuffer(wgpu::BufferUsage::Index, {0, 1, 0xffff, 2}), 0, 4, 0, false);

    FinishWhen([] { return pending == 0; });
    return 0;
}
