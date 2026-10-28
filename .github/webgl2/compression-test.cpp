// Checks the compressed texture formats WebGL 2 exposes as extensions: each
// test uploads blocks of one known color, reads every texel in a shader and
// checks the color.
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "test.h"

namespace {

using namespace test;

constexpr uint32_t kSize = 8;
constexpr uint32_t kBytesPerRow = 256;
int pending = 0;

const char* kShader = R"(
@group(0) @binding(0) var image: texture_2d<f32>;
@vertex fn vertexMain(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let positions = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    return vec4f(positions[i], 0.0, 1.0);
}
@fragment fn fragmentMain(@builtin(position) position: vec4f) -> @location(0) vec4f {
    return textureLoad(image, vec2i(position.xy), 0);
}
)";

struct Case {
    const char* name;
    wgpu::FeatureName feature;
    wgpu::TextureFormat format;
    // One block of a single color, repeated over the texture.
    std::vector<uint8_t> block;
    uint32_t blockBytes;
    uint8_t expected[4];
};

// Packs fields least significant bit first, as BC7 blocks are laid out.
struct BitWriter {
    std::vector<uint8_t> bytes = std::vector<uint8_t>(16, 0);
    uint32_t position = 0;
    void Write(uint32_t value, uint32_t bits) {
        for (uint32_t i = 0; i < bits; i++, position++) {
            if ((value >> i) & 1) {
                bytes[position / 8] |= uint8_t(1 << (position % 8));
            }
        }
    }
};

// A BC7 mode 6 block with equal endpoints: 7-bit channels and a shared low bit of 0.
std::vector<uint8_t> BC7Block(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    BitWriter writer;
    writer.Write(1 << 6, 7);  // Mode 6.
    for (uint8_t channel : {r, g, b, a}) {
        writer.Write(channel >> 1, 7);
        writer.Write(channel >> 1, 7);
    }
    // P bits and indices are 0.
    return writer.bytes;
}

const std::vector<Case> kCases = {
    // Equal 565 endpoints: red.
    {"bc1-rgba-unorm", wgpu::FeatureName::TextureCompressionBC,
     wgpu::TextureFormat::BC1RGBAUnorm, {0x00, 0xf8, 0x00, 0xf8, 0, 0, 0, 0}, 8,
     {255, 0, 0, 255}},
    // Opaque alpha and green.
    {"bc3-rgba-unorm", wgpu::FeatureName::TextureCompressionBC,
     wgpu::TextureFormat::BC3RGBAUnorm,
     {0xff, 0xff, 0, 0, 0, 0, 0, 0, 0xe0, 0x07, 0xe0, 0x07, 0, 0, 0, 0}, 16, {0, 255, 0, 255}},
    {"bc4-r-unorm", wgpu::FeatureName::TextureCompressionBC, wgpu::TextureFormat::BC4RUnorm,
     {0x80, 0x80, 0, 0, 0, 0, 0, 0}, 8, {128, 0, 0, 255}},
    {"bc5-rg-unorm", wgpu::FeatureName::TextureCompressionBC, wgpu::TextureFormat::BC5RGUnorm,
     {0x40, 0x40, 0, 0, 0, 0, 0, 0, 0xc0, 0xc0, 0, 0, 0, 0, 0, 0}, 16, {64, 192, 0, 255}},
    {"bc7-rgba-unorm", wgpu::FeatureName::TextureCompressionBC,
     wgpu::TextureFormat::BC7RGBAUnorm, BC7Block(128, 64, 192, 254), 16, {128, 64, 192, 254}},
    // Individual mode with 4-bit colors F, 8, 0 and the +2 modifier.
    {"etc2-rgb8-unorm", wgpu::FeatureName::TextureCompressionETC2,
     wgpu::TextureFormat::ETC2RGB8Unorm, {0xff, 0x88, 0x00, 0, 0, 0, 0, 0}, 8,
     {255, 138, 2, 255}},
    // A void-extent block: one color for the whole block, as 16-bit channels.
    {"astc-4x4-unorm", wgpu::FeatureName::TextureCompressionASTC,
     wgpu::TextureFormat::ASTC4x4Unorm,
     {0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x80, 0x00, 0x00, 0xff,
      0xff},
     16,
     {255, 128, 0, 255}},
};

void TestFormat(const Case& c, wgpu::RenderPipeline pipeline) {
    wgpu::TextureDescriptor textureDescriptor{
        .usage = wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::TextureBinding,
        .size = {kSize, kSize},
        .format = c.format,
    };
    wgpu::Texture texture;
    std::string error =
        ValidationErrorOf([&] { texture = device.CreateTexture(&textureDescriptor); });
    if (!error.empty()) {
        Check(c.name, false, "creating the texture: " + error.substr(0, error.find('\n')));
        return;
    }
    // 2x2 blocks of 4x4 texels.
    std::vector<uint8_t> data;
    for (uint32_t i = 0; i < 4; i++) {
        data.insert(data.end(), c.block.begin(), c.block.end());
    }
    wgpu::TexelCopyTextureInfo destination{.texture = texture};
    wgpu::TexelCopyBufferLayout layout{.bytesPerRow = 2 * c.blockBytes, .rowsPerImage = kSize};
    wgpu::Extent3D size{kSize, kSize};
    device.GetQueue().WriteTexture(&destination, data.data(), data.size(), &layout, &size);

    wgpu::TextureDescriptor targetDescriptor{
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc,
        .size = {kSize, kSize},
        .format = wgpu::TextureFormat::RGBA8Unorm,
    };
    wgpu::Texture target = device.CreateTexture(&targetDescriptor);
    wgpu::BindGroupEntry entry{.binding = 0, .textureView = texture.CreateView()};
    wgpu::BindGroupDescriptor bindGroupDescriptor{
        .layout = pipeline.GetBindGroupLayout(0), .entryCount = 1, .entries = &entry};
    wgpu::BindGroup bindGroup = device.CreateBindGroup(&bindGroupDescriptor);
    wgpu::RenderPassColorAttachment attachment{
        .view = target.CreateView(),
        .loadOp = wgpu::LoadOp::Clear,
        .storeOp = wgpu::StoreOp::Store,
    };
    wgpu::RenderPassDescriptor passDescriptor{.colorAttachmentCount = 1,
                                              .colorAttachments = &attachment};
    wgpu::BufferDescriptor bufferDescriptor{
        .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
        .size = kBytesPerRow * kSize};
    wgpu::Buffer buffer = device.CreateBuffer(&bufferDescriptor);
    wgpu::TexelCopyTextureInfo from{.texture = target};
    wgpu::TexelCopyBufferInfo to{.layout = {.bytesPerRow = kBytesPerRow}, .buffer = buffer};

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDescriptor);
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bindGroup);
    pass.Draw(3);
    pass.End();
    encoder.CopyTextureToBuffer(&from, &to, &size);
    wgpu::CommandBuffer commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);

    pending++;
    buffer.MapAsync(
        wgpu::MapMode::Read, 0, kBytesPerRow * kSize, wgpu::CallbackMode::AllowProcessEvents,
        [c, buffer](wgpu::MapAsyncStatus status, wgpu::StringView message) {
            pending--;
            if (status != wgpu::MapAsyncStatus::Success) {
                Check(c.name, false, "mapAsync failed: " + ToString(message));
                return;
            }
            const auto* data =
                static_cast<const uint8_t*>(buffer.GetConstMappedRange(0, kBytesPerRow * kSize));
            std::string why;
            for (uint32_t y = 0; y < kSize && why.empty(); y++) {
                for (uint32_t x = 0; x < kSize && why.empty(); x++) {
                    const uint8_t* texel = data + y * kBytesPerRow + x * 4;
                    for (uint32_t i = 0; i < 4; i++) {
                        // Decoders may round differently by one step.
                        if (std::abs(int(texel[i]) - int(c.expected[i])) > 1) {
                            char text[64];
                            std::snprintf(text, sizeof(text),
                                          "texel %u,%u is %u,%u,%u,%u, expected %u,%u,%u,%u", x,
                                          y, texel[0], texel[1], texel[2], texel[3],
                                          c.expected[0], c.expected[1], c.expected[2],
                                          c.expected[3]);
                            why = text;
                            break;
                        }
                    }
                }
            }
            Check(c.name, why.empty(), why);
            buffer.Unmap();
        });
}

}  // namespace

int main() {
    instance = wgpu::CreateInstance();
    wgpu::RequestAdapterOptions adapterOptions{
        .featureLevel = wgpu::FeatureLevel::Compatibility,
        .backendType = wgpu::BackendType::OpenGLES,
    };
    instance.RequestAdapter(&adapterOptions, wgpu::CallbackMode::AllowProcessEvents,
                            [](wgpu::RequestAdapterStatus, wgpu::Adapter result,
                               wgpu::StringView) { adapter = std::move(result); });
    instance.ProcessEvents();
    if (!adapter) {
        std::printf("FAIL requestAdapter\n1 test(s) failed\n");
        return 1;
    }
    // The CI's SwiftShader has all three. A browser without one skips its formats.
    std::vector<wgpu::FeatureName> features;
    for (wgpu::FeatureName feature :
         {wgpu::FeatureName::TextureCompressionBC, wgpu::FeatureName::TextureCompressionETC2,
          wgpu::FeatureName::TextureCompressionASTC}) {
        if (adapter.HasFeature(feature)) {
            features.push_back(feature);
        }
    }
    Check("the adapter has BC, ETC2 and ASTC", features.size() == 3,
          std::to_string(features.size()) + " of them");
    wgpu::DeviceDescriptor deviceDescriptor;
    deviceDescriptor.requiredFeatureCount = features.size();
    deviceDescriptor.requiredFeatures = features.data();
    deviceDescriptor.SetUncapturedErrorCallback(
        [](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView message) {
            uncapturedErrors += ToString(message) + "\n";
        });
    deviceDescriptor.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowProcessEvents,
        [](const wgpu::Device&, wgpu::DeviceLostReason, wgpu::StringView message) {
            deviceLost = true;
            std::printf("device lost: %s\n", ToString(message).c_str());
        });
    adapter.RequestDevice(&deviceDescriptor, wgpu::CallbackMode::AllowProcessEvents,
                          [](wgpu::RequestDeviceStatus, wgpu::Device result, wgpu::StringView) {
                              device = std::move(result);
                          });
    instance.ProcessEvents();
    if (!device) {
        std::printf("FAIL requestDevice\n%d test(s) failed\n", failures + 1);
        return 1;
    }

    wgpu::ShaderSourceWGSL wgsl{{.code = kShader}};
    wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl};
    wgpu::ShaderModule module = device.CreateShaderModule(&moduleDescriptor);
    wgpu::ColorTargetState target{.format = wgpu::TextureFormat::RGBA8Unorm};
    wgpu::FragmentState fragment{.module = module, .targetCount = 1, .targets = &target};
    wgpu::RenderPipelineDescriptor pipelineDescriptor{.vertex = {.module = module},
                                                      .fragment = &fragment};
    wgpu::RenderPipeline pipeline = device.CreateRenderPipeline(&pipelineDescriptor);

    for (const Case& c : kCases) {
        if (device.HasFeature(c.feature)) {
            TestFormat(c, pipeline);
        }
    }
    FinishWhen([] { return pending == 0; });
    return 0;
}
