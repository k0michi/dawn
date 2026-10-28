// Checks multisampled rendering on WebGL 2, which has multisampled
// renderbuffers but no multisampled textures: drawing with 4 samples and
// resolving, keeping a multisampled attachment's contents between passes, and
// rejecting multisampled textures for shaders. Copy usages are allowed, as in
// compatibility mode, which rejects the copies themselves.
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "test.h"

namespace {

using namespace test;

constexpr uint32_t kSize = 8;
constexpr uint32_t kBytesPerRow = 256;
int pending = 0;

const char* kShader = R"(
@vertex fn vertexMain(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    // The lower-left half of the target, split along the diagonal.
    let positions = array(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0));
    return vec4f(positions[i], 0.0, 1.0);
}
@fragment fn fragmentMain() -> @location(0) vec4f {
    return vec4f(1.0, 0.0, 0.0, 1.0);
}
)";

wgpu::Texture MakeTexture(wgpu::TextureFormat format, uint32_t sampleCount, wgpu::TextureUsage usage) {
    wgpu::TextureDescriptor descriptor{
        .usage = usage, .size = {kSize, kSize}, .format = format, .sampleCount = sampleCount};
    return device.CreateTexture(&descriptor);
}

wgpu::RenderPipeline MakePipeline(bool withDepth) {
    wgpu::ShaderSourceWGSL wgsl{{.code = kShader}};
    wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl};
    wgpu::ShaderModule module = device.CreateShaderModule(&moduleDescriptor);
    wgpu::ColorTargetState target{.format = wgpu::TextureFormat::RGBA8Unorm};
    wgpu::FragmentState fragment{.module = module, .targetCount = 1, .targets = &target};
    wgpu::DepthStencilState depth{.format = wgpu::TextureFormat::Depth24Plus,
                                  .depthWriteEnabled = wgpu::OptionalBool::True,
                                  .depthCompare = wgpu::CompareFunction::Less};
    wgpu::RenderPipelineDescriptor descriptor{
        .vertex = {.module = module},
        .depthStencil = withDepth ? &depth : nullptr,
        .multisample = {.count = 4},
        .fragment = &fragment,
    };
    return device.CreateRenderPipeline(&descriptor);
}

// Copies |texture| to a buffer and calls |check| with its rows.
void ReadBack(const char* name,
              wgpu::Texture texture,
              std::function<std::string(const uint8_t*)> check) {
    wgpu::BufferDescriptor bufferDescriptor{
        .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
        .size = kBytesPerRow * kSize};
    wgpu::Buffer buffer = device.CreateBuffer(&bufferDescriptor);
    wgpu::TexelCopyTextureInfo source{.texture = texture};
    wgpu::TexelCopyBufferInfo destination{.layout = {.bytesPerRow = kBytesPerRow}, .buffer = buffer};
    wgpu::Extent3D size{kSize, kSize};
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.CopyTextureToBuffer(&source, &destination, &size);
    wgpu::CommandBuffer commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);
    pending++;
    buffer.MapAsync(wgpu::MapMode::Read, 0, kBytesPerRow * kSize,
                    wgpu::CallbackMode::AllowProcessEvents,
                    [name, buffer, check](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                        pending--;
                        if (status != wgpu::MapAsyncStatus::Success) {
                            Check(name, false, "mapAsync failed: " + ToString(message));
                            return;
                        }
                        std::string why = check(static_cast<const uint8_t*>(
                            buffer.GetConstMappedRange(0, kBytesPerRow * kSize)));
                        Check(name, why.empty(), why);
                        buffer.Unmap();
                    });
}

// Draws a half-covering triangle with 4 samples and resolves it: pixels on the diagonal are
// partly covered, so some must resolve to a red between none and full.
void TestDrawAndResolve(bool withDepth) {
    const char* name = withDepth ? "draw with 4 samples and a depth attachment, then resolve"
                                 : "draw with 4 samples, then resolve";
    // Copy usages don't change how the texture is stored.
    wgpu::Texture multisampled = MakeTexture(
        wgpu::TextureFormat::RGBA8Unorm, 4,
        wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc);
    wgpu::Texture resolved =
        MakeTexture(wgpu::TextureFormat::RGBA8Unorm, 1,
                    wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc);
    wgpu::RenderPassColorAttachment color{
        .view = multisampled.CreateView(),
        .resolveTarget = resolved.CreateView(),
        .loadOp = wgpu::LoadOp::Clear,
        .storeOp = wgpu::StoreOp::Discard,
        .clearValue = {0, 0, 0, 1},
    };
    wgpu::Texture depthTexture;
    wgpu::RenderPassDepthStencilAttachment depth;
    wgpu::RenderPassDescriptor passDescriptor{.colorAttachmentCount = 1,
                                              .colorAttachments = &color};
    if (withDepth) {
        depthTexture =
            MakeTexture(wgpu::TextureFormat::Depth24Plus, 4, wgpu::TextureUsage::RenderAttachment);
        depth = {.view = depthTexture.CreateView(),
                 .depthLoadOp = wgpu::LoadOp::Clear,
                 .depthStoreOp = wgpu::StoreOp::Discard,
                 .depthClearValue = 1.0};
        passDescriptor.depthStencilAttachment = &depth;
    }
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDescriptor);
    pass.SetPipeline(MakePipeline(withDepth));
    pass.Draw(3);
    pass.End();
    wgpu::CommandBuffer commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);

    ReadBack(name, resolved, [](const uint8_t* data) -> std::string {
        int partial = 0;
        int full = 0;
        for (uint32_t y = 0; y < kSize; y++) {
            for (uint32_t x = 0; x < kSize; x++) {
                uint8_t red = data[y * kBytesPerRow + x * 4];
                partial += red > 0 && red < 255;
                full += red == 255;
            }
        }
        if (full == 0) {
            return "nothing was drawn";
        }
        return partial > 0 ? "" : "no partly covered pixels: not multisampled";
    });
}

// Clears a multisampled attachment in one pass and loads and resolves it in another: the
// attachment must keep its contents in between.
void TestLoadBetweenPasses() {
    const char* name = "a multisampled attachment keeps its contents between passes";
    wgpu::Texture multisampled = MakeTexture(wgpu::TextureFormat::RGBA8Unorm, 4,
                                             wgpu::TextureUsage::RenderAttachment);
    wgpu::Texture resolved =
        MakeTexture(wgpu::TextureFormat::RGBA8Unorm, 1,
                    wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc);
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::RenderPassColorAttachment clear{
        .view = multisampled.CreateView(),
        .loadOp = wgpu::LoadOp::Clear,
        .storeOp = wgpu::StoreOp::Store,
        .clearValue = {0, 1, 0, 1},
    };
    wgpu::RenderPassDescriptor clearPass{.colorAttachmentCount = 1, .colorAttachments = &clear};
    encoder.BeginRenderPass(&clearPass).End();
    wgpu::RenderPassColorAttachment load{
        .view = multisampled.CreateView(),
        .resolveTarget = resolved.CreateView(),
        .loadOp = wgpu::LoadOp::Load,
        .storeOp = wgpu::StoreOp::Store,
    };
    wgpu::RenderPassDescriptor loadPass{.colorAttachmentCount = 1, .colorAttachments = &load};
    encoder.BeginRenderPass(&loadPass).End();
    wgpu::CommandBuffer commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);

    ReadBack(name, resolved, [](const uint8_t* data) -> std::string {
        for (uint32_t y = 0; y < kSize; y++) {
            for (uint32_t x = 0; x < kSize; x++) {
                const uint8_t* texel = data + y * kBytesPerRow + x * 4;
                if (texel[0] != 0 || texel[1] != 255 || texel[2] != 0 || texel[3] != 255) {
                    return "pixel " + std::to_string(x) + "," + std::to_string(y) + " is not green";
                }
            }
        }
        return "";
    });
}

}  // namespace

int main() {
    if (!Setup()) {
        return 1;
    }
    TestDrawAndResolve(false);
    TestDrawAndResolve(true);
    TestLoadBetweenPasses();
    // WebGL 2 shaders can't read multisampled textures.
    CheckRejected("a multisampled texture for shaders is rejected", ValidationErrorOf([] {
                      MakeTexture(wgpu::TextureFormat::RGBA8Unorm, 4,
                                  wgpu::TextureUsage::RenderAttachment |
                                      wgpu::TextureUsage::TextureBinding);
                  }));
    // As in compatibility mode elsewhere, copy usages are allowed and copies rejected.
    const auto kCopyUsages = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc |
                             wgpu::TextureUsage::CopyDst;
    wgpu::Texture source;
    wgpu::Texture destination;
    std::string error = ValidationErrorOf([&] {
        source = MakeTexture(wgpu::TextureFormat::RGBA8Unorm, 4, kCopyUsages);
        destination = MakeTexture(wgpu::TextureFormat::RGBA8Unorm, 4, kCopyUsages);
    });
    Check("a multisampled texture with copy usages is accepted", error.empty(), error);
    CheckRejected("copying a multisampled texture is rejected", ValidationErrorOf([&] {
                      wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
                      wgpu::TexelCopyTextureInfo from{.texture = source};
                      wgpu::TexelCopyTextureInfo to{.texture = destination};
                      wgpu::Extent3D size{kSize, kSize};
                      encoder.CopyTextureToTexture(&from, &to, &size);
                      encoder.Finish();
                  }));
    FinishWhen([] { return pending == 0; });
    return 0;
}
