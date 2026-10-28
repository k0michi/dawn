// Checks texture-to-buffer copies of the formats that glReadPixels can't read
// on WebGL 2, which Dawn copies with a compute shader elsewhere: each test
// writes known bytes to a texture, copies part of it to a buffer and checks
// the copied bytes and that the bytes around them kept their values.
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "test.h"

namespace {

using namespace test;

int pending = 0;

struct Case {
    const char* name;
    wgpu::TextureFormat format;
    uint32_t bytesPerTexel;
    // Bytes of one texel, repeated with a changing first byte over the texture. Floats must not
    // be NaN and must survive a round trip through the GPU.
    std::vector<uint8_t> texel;
};

std::vector<uint8_t> Floats(std::vector<float> values) {
    std::vector<uint8_t> bytes(values.size() * 4);
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

std::vector<uint8_t> Halves(std::vector<uint16_t> values) {
    std::vector<uint8_t> bytes(values.size() * 2);
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

const std::vector<Case> kCases = {
    {"r8unorm", wgpu::TextureFormat::R8Unorm, 1, {0x80}},
    {"rg8unorm", wgpu::TextureFormat::RG8Unorm, 2, {0x01, 0xfe}},
    {"bgra8unorm", wgpu::TextureFormat::BGRA8Unorm, 4, {0x10, 0x20, 0x30, 0x40}},
    {"r8uint", wgpu::TextureFormat::R8Uint, 1, {0xc3}},
    {"rg8sint", wgpu::TextureFormat::RG8Sint, 2, {0x7f, 0x80}},
    {"rgba8uint", wgpu::TextureFormat::RGBA8Uint, 4, {0x01, 0x02, 0xfd, 0xfe}},
    {"r16uint", wgpu::TextureFormat::R16Uint, 2, {0x34, 0xf2}},
    {"rg16sint", wgpu::TextureFormat::RG16Sint, 4, {0x00, 0x80, 0xff, 0x7f}},
    {"rgba16uint", wgpu::TextureFormat::RGBA16Uint, 8, {1, 2, 3, 4, 5, 6, 7, 8}},
    {"r32uint", wgpu::TextureFormat::R32Uint, 4, {0x78, 0x56, 0x34, 0xf2}},
    {"rg32sint", wgpu::TextureFormat::RG32Sint, 8, {1, 0, 0, 0x80, 0xff, 0xff, 0xff, 0x7f}},
    {"rgb10a2uint", wgpu::TextureFormat::RGB10A2Uint, 4, {0xff, 0x13, 0x57, 0xc6}},
    {"r32float", wgpu::TextureFormat::R32Float, 4, Floats({-1.5f})},
    {"rg32float", wgpu::TextureFormat::RG32Float, 8, Floats({3.25f, -1e-3f})},
    {"r16float", wgpu::TextureFormat::R16Float, 2, Halves({0x3c00})},
    {"rg16float", wgpu::TextureFormat::RG16Float, 4, Halves({0xbe00, 0x0001})},
    {"rgba16float", wgpu::TextureFormat::RGBA16Float, 8, Halves({0x7bff, 0x8400, 0x3555, 0x0000})},
    // -128 isn't used: it reads as -1.0 like -127.
    {"rgba8snorm", wgpu::TextureFormat::RGBA8Snorm, 4, {0x01, 0x7f, 0x81, 0x00}},
    {"r8snorm", wgpu::TextureFormat::R8Snorm, 1, {0x01}},
    // r = 1.328125 (exponent 15, mantissa 0x15), g = a denormal, b = 2.625.
    {"rg11b10ufloat", wgpu::TextureFormat::RG11B10Ufloat, 4, {0xd5, 0x1b, 0x80, 0x82}},
    // r = 300, g = 5, b = 511 with exponent 20: the encoding with the smallest exponent.
    {"rgb9e5ufloat", wgpu::TextureFormat::RGB9E5Ufloat, 4, {0x2c, 0x0b, 0xfc, 0xa7}},
    {"depth16unorm", wgpu::TextureFormat::Depth16Unorm, 2, {0x34, 0x12}},
};

constexpr uint32_t kWidth = 5;
constexpr uint32_t kHeight = 3;
constexpr uint32_t kLayers = 2;
constexpr uint32_t kBytesPerRow = 256;
constexpr uint8_t kFill = 0xaa;

std::string Bytes(const uint8_t* data, size_t size) {
    std::string text;
    for (size_t i = 0; i < size; i++) {
        char byte[4];
        std::snprintf(byte, sizeof(byte), "%02x", data[i]);
        text += byte;
    }
    return text;
}

// The texel at (x, y, layer): the case's texel with its first byte changed, so that every texel
// differs.
std::vector<uint8_t> Texel(const Case& c, uint32_t x, uint32_t y, uint32_t layer) {
    std::vector<uint8_t> texel = c.texel;
    // Float texels keep their bytes, since changing them could make NaNs or lose precision.
    bool isFloat = c.format == wgpu::TextureFormat::RG11B10Ufloat ||
                   c.format == wgpu::TextureFormat::RGB9E5Ufloat ||
                   c.format == wgpu::TextureFormat::R32Float ||
                   c.format == wgpu::TextureFormat::RG32Float ||
                   c.format == wgpu::TextureFormat::R16Float ||
                   c.format == wgpu::TextureFormat::RG16Float ||
                   c.format == wgpu::TextureFormat::RGBA16Float;
    if (!isFloat) {
        texel[0] = uint8_t(texel[0] + x + 8 * y + 32 * layer);
    }
    return texel;
}

// Writes every texel of a 2-layer texture, copies the region from (1, 1) of layer 1 to a buffer
// at |offset|, and checks the copied rows and the bytes around them.
void TestCopy(const Case& c, uint64_t offset) {
    std::string name = std::string(c.name) + " at offset " + std::to_string(offset);
    wgpu::TextureDescriptor textureDescriptor{
        .usage = wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::CopySrc,
        .size = {kWidth, kHeight, kLayers},
        .format = c.format,
    };
    wgpu::Texture texture;
    std::string error = ValidationErrorOf([&] { texture = device.CreateTexture(&textureDescriptor); });
    if (!error.empty()) {
        Check(name.c_str(), false, "creating the texture: " + error.substr(0, error.find('\n')));
        return;
    }

    std::vector<uint8_t> data;
    for (uint32_t layer = 0; layer < kLayers; layer++) {
        for (uint32_t y = 0; y < kHeight; y++) {
            for (uint32_t x = 0; x < kWidth; x++) {
                std::vector<uint8_t> texel = Texel(c, x, y, layer);
                data.insert(data.end(), texel.begin(), texel.end());
            }
        }
    }
    wgpu::TexelCopyTextureInfo whole{.texture = texture};
    wgpu::TexelCopyBufferLayout dataLayout{.bytesPerRow = kWidth * c.bytesPerTexel,
                                           .rowsPerImage = kHeight};
    wgpu::Extent3D wholeSize{kWidth, kHeight, kLayers};
    device.GetQueue().WriteTexture(&whole, data.data(), data.size(), &dataLayout, &wholeSize);

    // Two rows of four texels from (1, 1) of layer 1. Depth copies must cover whole layers.
    const bool isDepth = c.format == wgpu::TextureFormat::Depth16Unorm;
    const uint32_t start = isDepth ? 0 : 1;
    const uint32_t kCopyWidth = kWidth - start;
    const uint32_t kCopyHeight = kHeight - start;
    uint64_t size = offset + kBytesPerRow * kCopyHeight;
    std::vector<uint8_t> fill(size, kFill);
    wgpu::BufferDescriptor bufferDescriptor{
        .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, .size = (size + 3) & ~3ull};
    wgpu::Buffer buffer = device.CreateBuffer(&bufferDescriptor);
    fill.resize(bufferDescriptor.size, kFill);
    device.GetQueue().WriteBuffer(buffer, 0, fill.data(), fill.size());

    wgpu::TexelCopyTextureInfo source{.texture = texture, .origin = {start, start, 1}};
    wgpu::TexelCopyBufferInfo destination{
        .layout = {.offset = offset, .bytesPerRow = kBytesPerRow, .rowsPerImage = kCopyHeight},
        .buffer = buffer};
    wgpu::Extent3D copySize{kCopyWidth, kCopyHeight, 1};
    error = ValidationErrorOf([&] {
        wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
        encoder.CopyTextureToBuffer(&source, &destination, &copySize);
        wgpu::CommandBuffer commands = encoder.Finish();
        device.GetQueue().Submit(1, &commands);
    });
    if (!error.empty()) {
        Check(name.c_str(), false, "copying: " + error.substr(0, error.find('\n')));
        return;
    }

    std::vector<uint8_t> expected = fill;
    for (uint32_t y = 0; y < kCopyHeight; y++) {
        for (uint32_t x = 0; x < kCopyWidth; x++) {
            std::vector<uint8_t> texel = Texel(c, start + x, start + y, 1);
            std::memcpy(expected.data() + offset + y * kBytesPerRow + x * c.bytesPerTexel,
                        texel.data(), texel.size());
        }
    }

    pending++;
    buffer.MapAsync(
        wgpu::MapMode::Read, 0, expected.size(), wgpu::CallbackMode::AllowProcessEvents,
        [name, buffer, expected](wgpu::MapAsyncStatus status, wgpu::StringView message) {
            pending--;
            if (status != wgpu::MapAsyncStatus::Success) {
                Check(name.c_str(), false, "mapAsync failed: " + ToString(message));
                return;
            }
            const auto* data =
                static_cast<const uint8_t*>(buffer.GetConstMappedRange(0, expected.size()));
            size_t first = 0;
            while (first < expected.size() && data[first] == expected[first]) {
                first++;
            }
            if (first == expected.size()) {
                Check(name.c_str(), true);
            } else {
                size_t start = first & ~size_t(7);
                size_t count = std::min<size_t>(16, expected.size() - start);
                Check(name.c_str(), false,
                      "byte " + std::to_string(first) + ": got " + Bytes(data + start, count) +
                          " expected " + Bytes(expected.data() + start, count));
            }
            buffer.Unmap();
        });
}

// Copies BGRA8 texels from a buffer to a texture and back, which uploads from a buffer on the GPU
// rather than from the CPU as writeTexture does.
void TestBGRA8BufferToTexture() {
    const char* name = "bgra8unorm from a buffer";
    constexpr uint32_t kSize = 4 * 3 * 4;
    std::vector<uint8_t> texels(kSize);
    for (uint32_t i = 0; i < kSize; i++) {
        texels[i] = uint8_t(i * 7 + 1);
    }
    wgpu::BufferDescriptor sourceDescriptor{.usage = wgpu::BufferUsage::CopySrc,
                                            .size = kBytesPerRow * 3,
                                            .mappedAtCreation = true};
    wgpu::Buffer source = device.CreateBuffer(&sourceDescriptor);
    auto* mapped = static_cast<uint8_t*>(source.GetMappedRange(0, kBytesPerRow * 3));
    for (uint32_t y = 0; y < 3; y++) {
        std::memcpy(mapped + y * kBytesPerRow, texels.data() + y * 16, 16);
    }
    source.Unmap();
    wgpu::TextureDescriptor textureDescriptor{
        .usage = wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::CopySrc,
        .size = {4, 3},
        .format = wgpu::TextureFormat::BGRA8Unorm,
    };
    wgpu::Texture texture = device.CreateTexture(&textureDescriptor);
    wgpu::BufferDescriptor readbackDescriptor{
        .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, .size = kBytesPerRow * 3};
    wgpu::Buffer readback = device.CreateBuffer(&readbackDescriptor);

    wgpu::TexelCopyBufferInfo from{.layout = {.bytesPerRow = kBytesPerRow, .rowsPerImage = 3},
                                   .buffer = source};
    wgpu::TexelCopyTextureInfo textureCopy{.texture = texture};
    wgpu::TexelCopyBufferInfo to{.layout = {.bytesPerRow = kBytesPerRow, .rowsPerImage = 3},
                                 .buffer = readback};
    wgpu::Extent3D size{4, 3};
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.CopyBufferToTexture(&from, &textureCopy, &size);
    encoder.CopyTextureToBuffer(&textureCopy, &to, &size);
    wgpu::CommandBuffer commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);

    pending++;
    readback.MapAsync(
        wgpu::MapMode::Read, 0, kBytesPerRow * 3, wgpu::CallbackMode::AllowProcessEvents,
        [name, readback, texels](wgpu::MapAsyncStatus status, wgpu::StringView message) {
            pending--;
            if (status != wgpu::MapAsyncStatus::Success) {
                Check(name, false, "mapAsync failed: " + ToString(message));
                return;
            }
            const auto* data =
                static_cast<const uint8_t*>(readback.GetConstMappedRange(0, kBytesPerRow * 3));
            bool same = true;
            for (uint32_t y = 0; y < 3; y++) {
                same = same && std::memcmp(data + y * kBytesPerRow, texels.data() + y * 16, 16) == 0;
            }
            Check(name, same, "got " + Bytes(data, 16) + " expected " + Bytes(texels.data(), 16));
            readback.Unmap();
        });
}

// Clears each layer of a depth32float texture, which can't be written to, and copies layer 1 to a
// buffer.
void TestDepth32Float() {
    const char* name = "depth32float";
    wgpu::TextureDescriptor textureDescriptor{
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc,
        .size = {kWidth, kHeight, kLayers},
        .format = wgpu::TextureFormat::Depth32Float,
    };
    wgpu::Texture texture = device.CreateTexture(&textureDescriptor);
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    const float depths[kLayers] = {0.75f, 0.375f};
    for (uint32_t layer = 0; layer < kLayers; layer++) {
        wgpu::TextureViewDescriptor viewDescriptor{.dimension = wgpu::TextureViewDimension::e2D,
                                                   .baseArrayLayer = layer,
                                                   .arrayLayerCount = 1};
        wgpu::RenderPassDepthStencilAttachment depth{
            .view = texture.CreateView(&viewDescriptor),
            .depthLoadOp = wgpu::LoadOp::Clear,
            .depthStoreOp = wgpu::StoreOp::Store,
            .depthClearValue = depths[layer],
        };
        wgpu::RenderPassDescriptor passDescriptor{.depthStencilAttachment = &depth};
        encoder.BeginRenderPass(&passDescriptor).End();
    }
    wgpu::BufferDescriptor bufferDescriptor{
        .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, .size = kBytesPerRow * kHeight};
    wgpu::Buffer buffer = device.CreateBuffer(&bufferDescriptor);
    wgpu::TexelCopyTextureInfo source{.texture = texture, .origin = {0, 0, 1}};
    wgpu::TexelCopyBufferInfo destination{
        .layout = {.bytesPerRow = kBytesPerRow, .rowsPerImage = kHeight}, .buffer = buffer};
    wgpu::Extent3D size{kWidth, kHeight, 1};
    encoder.CopyTextureToBuffer(&source, &destination, &size);
    wgpu::CommandBuffer commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);

    pending++;
    buffer.MapAsync(
        wgpu::MapMode::Read, 0, kBytesPerRow * kHeight, wgpu::CallbackMode::AllowProcessEvents,
        [name, buffer](wgpu::MapAsyncStatus status, wgpu::StringView message) {
            pending--;
            if (status != wgpu::MapAsyncStatus::Success) {
                Check(name, false, "mapAsync failed: " + ToString(message));
                return;
            }
            const auto* data =
                static_cast<const uint8_t*>(buffer.GetConstMappedRange(0, kBytesPerRow * kHeight));
            const float expected = 0.375f;
            bool same = true;
            for (uint32_t y = 0; y < kHeight; y++) {
                for (uint32_t x = 0; x < kWidth; x++) {
                    same = same && std::memcmp(data + y * kBytesPerRow + x * 4, &expected, 4) == 0;
                }
            }
            Check(name, same, "got " + Bytes(data, 16));
            buffer.Unmap();
        });
}

}  // namespace

int main() {
    if (!Setup()) {
        return 1;
    }
    for (const Case& c : kCases) {
        TestCopy(c, 0);
        // An offset that is only a multiple of the texel size, where texels straddle words.
        // Depth copies need offsets that are multiples of 4.
        if (c.bytesPerTexel < 4 && c.format != wgpu::TextureFormat::Depth16Unorm) {
            TestCopy(c, c.bytesPerTexel);
        }
    }
    TestBGRA8BufferToTexture();
    TestDepth32Float();
    FinishWhen([] { return pending == 0; });
    return 0;
}
