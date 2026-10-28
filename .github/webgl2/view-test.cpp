// Checks texture views on WebGL 2: rendering into views of mip levels, layers
// and 3D slices, and sampling views of them, of cube textures and of the depth
// and stencil of depth-stencil textures. Run it also with
// HIDE_WEBGL_EXTENSIONS=WEBGL_stencil_texturing, where stencil sampling is
// emulated.
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "test.h"

namespace {

using namespace test;

int pending = 0;
// Readbacks to map once their command buffer is submitted.
std::vector<std::function<void()>> readbacks;

wgpu::Texture MakeTexture(wgpu::TextureDescriptor descriptor) {
    return device.CreateTexture(&descriptor);
}

void Clear(wgpu::CommandEncoder encoder,
           wgpu::TextureView view,
           wgpu::Color color,
           uint32_t depthSlice = WGPU_DEPTH_SLICE_UNDEFINED) {
    wgpu::RenderPassColorAttachment attachment{.view = view,
                                               .depthSlice = depthSlice,
                                               .loadOp = wgpu::LoadOp::Clear,
                                               .storeOp = wgpu::StoreOp::Store,
                                               .clearValue = color};
    wgpu::RenderPassDescriptor pass{.colorAttachmentCount = 1, .colorAttachments = &attachment};
    encoder.BeginRenderPass(&pass).End();
}

void ClearDepthStencil(wgpu::CommandEncoder encoder,
                       wgpu::Texture texture,
                       float depth,
                       uint32_t stencil) {
    wgpu::RenderPassDepthStencilAttachment attachment{.view = texture.CreateView(),
                                                      .depthLoadOp = wgpu::LoadOp::Clear,
                                                      .depthStoreOp = wgpu::StoreOp::Store,
                                                      .depthClearValue = depth,
                                                      .stencilLoadOp = wgpu::LoadOp::Clear,
                                                      .stencilStoreOp = wgpu::StoreOp::Store,
                                                      .stencilClearValue = stencil};
    wgpu::RenderPassDescriptor pass{.depthStencilAttachment = &attachment};
    encoder.BeginRenderPass(&pass).End();
}

// Draws |fragmentBody|, which returns a vec4f, over a |size|x|size| RGBA8 target with |view|
// bound at binding 0 of |declarations| (and a sampler at binding 1 if |sampler| is set), and
// calls |check| with the red channel of each texel.
void Sample(const char* name,
            wgpu::CommandEncoder encoder,
            const std::string& declarations,
            const std::string& fragmentBody,
            wgpu::TextureView view,
            wgpu::Sampler sampler,
            uint32_t size,
            std::function<std::string(uint32_t x, uint32_t y, int red)> check) {
    std::string code = declarations + R"(
@vertex fn vertexMain(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let positions = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    return vec4f(positions[i], 0.0, 1.0);
}
@fragment fn fragmentMain(@builtin(position) position: vec4f) -> @location(0) vec4f {
)" + fragmentBody + "\n}\n";
    wgpu::RenderPipeline pipeline;
    wgpu::BindGroup bindGroup;
    std::string error = ValidationErrorOf([&] {
        wgpu::ShaderSourceWGSL wgsl{{.code = code.c_str()}};
        wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl};
        wgpu::ShaderModule module = device.CreateShaderModule(&moduleDescriptor);
        wgpu::ColorTargetState target{.format = wgpu::TextureFormat::RGBA8Unorm};
        wgpu::FragmentState fragment{.module = module, .targetCount = 1, .targets = &target};
        wgpu::RenderPipelineDescriptor pipelineDescriptor{.vertex = {.module = module},
                                                          .fragment = &fragment};
        pipeline = device.CreateRenderPipeline(&pipelineDescriptor);
        std::vector<wgpu::BindGroupEntry> entries = {{.binding = 0, .textureView = view}};
        if (sampler) {
            entries.push_back({.binding = 1, .sampler = sampler});
        }
        wgpu::BindGroupDescriptor bindGroupDescriptor{.layout = pipeline.GetBindGroupLayout(0),
                                                      .entryCount = entries.size(),
                                                      .entries = entries.data()};
        bindGroup = device.CreateBindGroup(&bindGroupDescriptor);
    });
    if (!error.empty()) {
        Check(name, false, error.substr(0, error.find('\n')));
        return;
    }
    wgpu::Texture target =
        MakeTexture({.usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc,
                     .size = {size, size},
                     .format = wgpu::TextureFormat::RGBA8Unorm});
    wgpu::BufferDescriptor bufferDescriptor{
        .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, .size = 256 * size};
    wgpu::Buffer buffer = device.CreateBuffer(&bufferDescriptor);
    wgpu::RenderPassColorAttachment attachment{.view = target.CreateView(),
                                               .loadOp = wgpu::LoadOp::Clear,
                                               .storeOp = wgpu::StoreOp::Store};
    wgpu::RenderPassDescriptor passDescriptor{.colorAttachmentCount = 1,
                                              .colorAttachments = &attachment};
    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDescriptor);
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bindGroup);
    pass.Draw(3);
    pass.End();
    wgpu::TexelCopyTextureInfo from{.texture = target};
    wgpu::TexelCopyBufferInfo to{.layout = {.bytesPerRow = 256}, .buffer = buffer};
    wgpu::Extent3D extent{size, size};
    encoder.CopyTextureToBuffer(&from, &to, &extent);

    readbacks.push_back([name, buffer, size, check] {
        pending++;
        buffer.MapAsync(wgpu::MapMode::Read, 0, 256 * size, wgpu::CallbackMode::AllowProcessEvents,
                        [name, buffer, size, check](wgpu::MapAsyncStatus status, wgpu::StringView) {
                            pending--;
                            if (status != wgpu::MapAsyncStatus::Success) {
                                Check(name, false, "mapAsync failed");
                                return;
                            }
                            const auto* data = static_cast<const uint8_t*>(
                                buffer.GetConstMappedRange(0, 256 * size));
                            std::string why;
                            for (uint32_t y = 0; y < size && why.empty(); y++) {
                                for (uint32_t x = 0; x < size && why.empty(); x++) {
                                    why = check(x, y, data[y * 256 + x * 4]);
                                }
                            }
                            Check(name, why.empty(), why);
                            buffer.Unmap();
                        });
    });
}

void Submit(wgpu::CommandEncoder encoder) {
    wgpu::CommandBuffer commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);
    for (auto& readback : readbacks) {
        readback();
    }
    readbacks.clear();
}

// Checks that every texel's red channel is |expected|, give or take one step.
std::function<std::string(uint32_t, uint32_t, int)> All(int expected) {
    return [expected](uint32_t x, uint32_t y, int red) -> std::string {
        if (std::abs(red - expected) <= 1) {
            return "";
        }
        return "texel " + std::to_string(x) + "," + std::to_string(y) + " is " +
               std::to_string(red) + ", expected " + std::to_string(expected);
    };
}

const auto kUsage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding;
const char* kTexture2D = "@group(0) @binding(0) var t: texture_2d<f32>;";

void TestMipLevelView() {
    wgpu::Texture texture = MakeTexture({.usage = kUsage,
                                         .size = {4, 4},
                                         .format = wgpu::TextureFormat::RGBA8Unorm,
                                         .mipLevelCount = 3});
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    for (uint32_t level = 0; level < 3; level++) {
        wgpu::TextureViewDescriptor view{.baseMipLevel = level, .mipLevelCount = 1};
        Clear(encoder, texture.CreateView(&view), {level * 0.25, 0, 0, 1});
    }
    wgpu::TextureViewDescriptor view{.baseMipLevel = 1, .mipLevelCount = 1};
    Sample("a view of mip level 1", encoder, kTexture2D, "return textureLoad(t, vec2i(0), 0);",
           texture.CreateView(&view), nullptr, 1, All(64));
    Submit(encoder);
}

void TestArrayLayers() {
    wgpu::Texture texture = MakeTexture(
        {.usage = kUsage, .size = {4, 4, 3}, .format = wgpu::TextureFormat::RGBA8Unorm});
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    for (uint32_t layer = 0; layer < 3; layer++) {
        wgpu::TextureViewDescriptor view{.dimension = wgpu::TextureViewDimension::e2D,
                                         .baseArrayLayer = layer,
                                         .arrayLayerCount = 1};
        Clear(encoder, texture.CreateView(&view), {layer * 0.25, 0, 0, 1});
    }
    Sample("layer 2 of a 2D array view", encoder,
           "@group(0) @binding(0) var t: texture_2d_array<f32>;",
           "return textureLoad(t, vec2i(0), 2, 0);", texture.CreateView(), nullptr, 1, All(128));
    Submit(encoder);
}

void TestCube() {
    wgpu::TextureBindingViewDimension cube;
    cube.textureBindingViewDimension = wgpu::TextureViewDimension::Cube;
    wgpu::Texture texture = MakeTexture({.nextInChain = &cube,
                                         .usage = kUsage,
                                         .size = {4, 4, 6},
                                         .format = wgpu::TextureFormat::RGBA8Unorm});
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    for (uint32_t face = 0; face < 6; face++) {
        wgpu::TextureViewDescriptor view{.dimension = wgpu::TextureViewDimension::e2D,
                                         .baseArrayLayer = face,
                                         .arrayLayerCount = 1};
        Clear(encoder, texture.CreateView(&view), {face * 0.125, 0, 0, 1});
    }
    wgpu::TextureViewDescriptor view{.dimension = wgpu::TextureViewDimension::Cube};
    // +Y is face 2.
    Sample("the +Y face of a cube view", encoder,
           "@group(0) @binding(0) var t: texture_cube<f32>;"
           "@group(0) @binding(1) var s: sampler;",
           "return textureSampleLevel(t, s, vec3f(0.0, 1.0, 0.0), 0.0);", texture.CreateView(&view),
           device.CreateSampler(), 1, All(64));
    Submit(encoder);
}

void Test3DSlice() {
    wgpu::Texture texture = MakeTexture({.usage = kUsage,
                                         .dimension = wgpu::TextureDimension::e3D,
                                         .size = {4, 4, 3},
                                         .format = wgpu::TextureFormat::RGBA8Unorm});
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    for (uint32_t z = 0; z < 3; z++) {
        Clear(encoder, texture.CreateView(), {z * 0.25, 0, 0, 1}, z);
    }
    Sample("slice 1 of a 3D view", encoder, "@group(0) @binding(0) var t: texture_3d<f32>;",
           "return textureLoad(t, vec3i(0, 0, 1), 0);", texture.CreateView(), nullptr, 1, All(64));
    Submit(encoder);
}

// Compares against the depth of a depth-stencil texture, as shadow maps do.
void TestDepthOfDepthStencil() {
    wgpu::Texture texture = MakeTexture(
        {.usage = kUsage, .size = {4, 4}, .format = wgpu::TextureFormat::Depth24PlusStencil8});
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    ClearDepthStencil(encoder, texture, 0.5, 0);
    wgpu::TextureViewDescriptor view{.aspect = wgpu::TextureAspect::DepthOnly};
    wgpu::SamplerDescriptor samplerDescriptor{.compare = wgpu::CompareFunction::Less};
    // 0.25 < 0.5 and not 0.75 < 0.5.
    Sample("depth comparison on depth24plus-stencil8", encoder,
           "@group(0) @binding(0) var t: texture_depth_2d;"
           "@group(0) @binding(1) var s: sampler_comparison;",
           "return vec4f(textureSampleCompareLevel(t, s, vec2f(0.5), 0.25),"
           "             textureSampleCompareLevel(t, s, vec2f(0.5), 0.75), 0.0, 1.0);",
           texture.CreateView(&view), device.CreateSampler(&samplerDescriptor), 1, All(255));
    Submit(encoder);
}

// Writes a different stencil value to each texel and reads them in a shader.
void TestStencil(const char* name, wgpu::TextureFormat format) {
    constexpr uint32_t kSize = 4;
    wgpu::Texture texture = MakeTexture(
        {.usage = kUsage | wgpu::TextureUsage::CopyDst, .size = {kSize, kSize}, .format = format});
    std::vector<uint8_t> stencil(kSize * kSize);
    for (uint32_t i = 0; i < stencil.size(); i++) {
        stencil[i] = uint8_t(i * 17 + 3);
    }
    stencil[5] = 0x80;
    stencil[6] = 0xff;
    wgpu::TexelCopyTextureInfo destination{.texture = texture,
                                           .aspect = wgpu::TextureAspect::StencilOnly};
    wgpu::TexelCopyBufferLayout layout{.bytesPerRow = kSize, .rowsPerImage = kSize};
    wgpu::Extent3D size{kSize, kSize};
    device.GetQueue().WriteTexture(&destination, stencil.data(), stencil.size(), &layout, &size);

    wgpu::TextureViewDescriptor view{.aspect = wgpu::TextureAspect::StencilOnly};
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    Sample(name, encoder, "@group(0) @binding(0) var t: texture_2d<u32>;",
           "return vec4f(f32(textureLoad(t, vec2i(position.xy), 0).r) / 255.0, 0.0, 0.0, 1.0);",
           texture.CreateView(&view), nullptr, kSize,
           [stencil](uint32_t x, uint32_t y, int red) -> std::string {
               int expected = stencil[y * kSize + x];
               if (red == expected) {
                   return "";
               }
               return "texel " + std::to_string(x) + "," + std::to_string(y) + " is " +
                      std::to_string(red) + ", expected " + std::to_string(expected);
           });
    Submit(encoder);
}

// Samples the stencil, changes it and samples it again in the same command buffer: a copy of the
// stencil made for sampling must not go stale.
void TestStencilChangesBetweenPasses() {
    wgpu::Texture texture = MakeTexture(
        {.usage = kUsage, .size = {4, 4}, .format = wgpu::TextureFormat::Depth24PlusStencil8});
    wgpu::TextureViewDescriptor view{.aspect = wgpu::TextureAspect::StencilOnly};
    const char* declarations = "@group(0) @binding(0) var t: texture_2d<u32>;";
    const char* body = "return vec4f(f32(textureLoad(t, vec2i(0), 0).r) / 255.0, 0.0, 0.0, 1.0);";
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    ClearDepthStencil(encoder, texture, 1.0, 7);
    Sample("stencil before it changes", encoder, declarations, body, texture.CreateView(&view),
           nullptr, 1, All(7));
    ClearDepthStencil(encoder, texture, 1.0, 200);
    Sample("stencil after it changes", encoder, declarations, body, texture.CreateView(&view),
           nullptr, 1, All(200));
    Submit(encoder);
}

}  // namespace

int main() {
    if (!Setup()) {
        return 1;
    }
    TestMipLevelView();
    TestArrayLayers();
    TestCube();
    Test3DSlice();
    TestDepthOfDepthStencil();
    TestStencil("stencil of stencil8", wgpu::TextureFormat::Stencil8);
    TestStencil("stencil of depth24plus-stencil8", wgpu::TextureFormat::Depth24PlusStencil8);
    TestStencilChangesBetweenPasses();
    FinishWhen([] { return pending == 0; });
    return 0;
}
