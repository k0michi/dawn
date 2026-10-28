// Checks indirect draws on WebGL 2, which has neither indirect draws nor the
// compute shaders Dawn validates them with: the backend reads the arguments and
// draws directly. Vertices 3 to 5 cover the target and 0 to 2 are outside it,
// so a draw only shows when its first vertex or base vertex reaches the shader.
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "test.h"

namespace {

using namespace test;

constexpr uint32_t kSize = 4;
int pending = 0;

const char* kShader = R"(
@vertex fn vertexMain(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let positions = array(vec2f(2.0, 2.0), vec2f(2.0, 2.0), vec2f(2.0, 2.0),
                          vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    return vec4f(positions[min(i, 5u)], 0.0, 1.0);
}
@fragment fn fragmentMain() -> @location(0) vec4f {
    return vec4f(1.0, 0.0, 0.0, 1.0);
}
)";

wgpu::RenderPipeline pipeline;

wgpu::Buffer MakeBuffer(wgpu::BufferUsage usage, std::vector<uint32_t> data) {
    wgpu::BufferDescriptor descriptor{.usage = usage | wgpu::BufferUsage::CopyDst,
                                      .size = data.size() * 4};
    wgpu::Buffer buffer = device.CreateBuffer(&descriptor);
    device.GetQueue().WriteBuffer(buffer, 0, data.data(), descriptor.size);
    return buffer;
}

// Records |draw| in a render pass and checks whether the target is red everywhere or black.
void Draw(const char* name,
          std::function<void(wgpu::CommandEncoder, wgpu::RenderPassEncoder)> draw,
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

    std::string error = ValidationErrorOf([&] {
        wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
        wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDescriptor);
        pass.SetPipeline(pipeline);
        draw(encoder, pass);
        pass.End();
        wgpu::TexelCopyTextureInfo from{.texture = target};
        wgpu::TexelCopyBufferInfo to{.layout = {.bytesPerRow = 256}, .buffer = readback};
        wgpu::Extent3D size{kSize, kSize};
        encoder.CopyTextureToBuffer(&from, &to, &size);
        wgpu::CommandBuffer commands = encoder.Finish();
        device.GetQueue().Submit(1, &commands);
    });
    if (!error.empty()) {
        Check(name, false, error.substr(0, error.find('\n')));
        return;
    }

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
            bool drawn = true;
            bool blank = true;
            for (uint32_t y = 0; y < kSize; y++) {
                for (uint32_t x = 0; x < kSize; x++) {
                    drawn = drawn && data[y * 256 + x * 4] == 255;
                    blank = blank && data[y * 256 + x * 4] == 0;
                }
            }
            Check(name, expectDrawn ? drawn : blank, expectDrawn ? "not drawn" : "drawn");
            readback.Unmap();
        });
}

}  // namespace

int main() {
    if (!Setup()) {
        return 1;
    }
    wgpu::ShaderSourceWGSL wgsl{{.code = kShader}};
    wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl};
    wgpu::ShaderModule module = device.CreateShaderModule(&moduleDescriptor);
    wgpu::ColorTargetState colorTarget{.format = wgpu::TextureFormat::RGBA8Unorm};
    wgpu::FragmentState fragment{.module = module, .targetCount = 1, .targets = &colorTarget};
    wgpu::RenderPipelineDescriptor pipelineDescriptor{.vertex = {.module = module},
                                                      .fragment = &fragment};
    pipeline = device.CreateRenderPipeline(&pipelineDescriptor);

    const auto kIndirect = wgpu::BufferUsage::Indirect;
    const auto kIndex = wgpu::BufferUsage::Index;

    // drawIndirect arguments: vertexCount, instanceCount, firstVertex, firstInstance.
    Draw(
        "drawIndirect from the first vertex 3",
        [&](wgpu::CommandEncoder, wgpu::RenderPassEncoder pass) {
            pass.DrawIndirect(MakeBuffer(kIndirect, {3, 1, 3, 0}), 0);
        },
        true);
    Draw(
        "drawIndirect from the first vertex 0 draws outside",
        [&](wgpu::CommandEncoder, wgpu::RenderPassEncoder pass) {
            pass.DrawIndirect(MakeBuffer(kIndirect, {3, 1, 0, 0}), 0);
        },
        false);
    Draw(
        "drawIndirect at an offset",
        [&](wgpu::CommandEncoder, wgpu::RenderPassEncoder pass) {
            pass.DrawIndirect(MakeBuffer(kIndirect, {0, 0, 0, 0, 3, 1, 3, 0}), 16);
        },
        true);
    Draw(
        "drawIndirect with a first instance is skipped",
        [&](wgpu::CommandEncoder, wgpu::RenderPassEncoder pass) {
            pass.DrawIndirect(MakeBuffer(kIndirect, {3, 1, 3, 1}), 0);
        },
        false);
    {
        // The arguments come from a copy on the GPU, submitted before the draw.
        wgpu::Buffer source = MakeBuffer(wgpu::BufferUsage::CopySrc, {3, 1, 3, 0});
        wgpu::Buffer indirect = MakeBuffer(kIndirect, {0, 0, 0, 0});
        wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
        encoder.CopyBufferToBuffer(source, 0, indirect, 0, 16);
        wgpu::CommandBuffer commands = encoder.Finish();
        device.GetQueue().Submit(1, &commands);
        Draw(
            "drawIndirect with arguments copied on the GPU",
            [&](wgpu::CommandEncoder, wgpu::RenderPassEncoder pass) {
                pass.DrawIndirect(indirect, 0);
            },
            true);
    }

    // drawIndexedIndirect arguments: indexCount, instanceCount, firstIndex, baseVertex,
    // firstInstance.
    Draw(
        "drawIndexedIndirect with the base vertex 3",
        [&](wgpu::CommandEncoder, wgpu::RenderPassEncoder pass) {
            pass.SetIndexBuffer(MakeBuffer(kIndex, {0, 1, 2}), wgpu::IndexFormat::Uint32);
            pass.DrawIndexedIndirect(MakeBuffer(kIndirect, {3, 1, 0, 3, 0}), 0);
        },
        true);
    Draw(
        "drawIndexedIndirect from the first index 3",
        [&](wgpu::CommandEncoder, wgpu::RenderPassEncoder pass) {
            pass.SetIndexBuffer(MakeBuffer(kIndex, {0, 0, 0, 3, 4, 5}), wgpu::IndexFormat::Uint32);
            pass.DrawIndexedIndirect(MakeBuffer(kIndirect, {3, 1, 3, 0, 0}), 0);
        },
        true);
    Draw(
        "drawIndexedIndirect past the end of the index buffer is skipped",
        [&](wgpu::CommandEncoder, wgpu::RenderPassEncoder pass) {
            pass.SetIndexBuffer(MakeBuffer(kIndex, {3, 4, 5}), wgpu::IndexFormat::Uint32);
            pass.DrawIndexedIndirect(MakeBuffer(kIndirect, {3, 1, 1, 0, 0}), 0);
        },
        false);
    Draw(
        "drawIndexedIndirect with a first instance is skipped",
        [&](wgpu::CommandEncoder, wgpu::RenderPassEncoder pass) {
            pass.SetIndexBuffer(MakeBuffer(kIndex, {3, 4, 5}), wgpu::IndexFormat::Uint32);
            pass.DrawIndexedIndirect(MakeBuffer(kIndirect, {3, 1, 0, 0, 1}), 0);
        },
        false);

    Draw(
        "drawIndirect in a render bundle",
        [&](wgpu::CommandEncoder, wgpu::RenderPassEncoder pass) {
            wgpu::TextureFormat format = wgpu::TextureFormat::RGBA8Unorm;
            wgpu::RenderBundleEncoderDescriptor descriptor{.colorFormatCount = 1,
                                                           .colorFormats = &format};
            wgpu::RenderBundleEncoder bundle = device.CreateRenderBundleEncoder(&descriptor);
            bundle.SetPipeline(pipeline);
            bundle.DrawIndirect(MakeBuffer(kIndirect, {3, 1, 3, 0}), 0);
            wgpu::RenderBundle finished = bundle.Finish();
            pass.ExecuteBundles(1, &finished);
        },
        true);

    FinishWhen([] { return pending == 0; });
    return 0;
}
