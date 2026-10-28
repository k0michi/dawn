// Checks buffer mapping and copies on WebGL 2, which has no glMapBufferRange
// and keeps index buffers apart from other buffers, by reading the results
// back with mapAsync.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "test.h"

namespace {

using namespace test;

int pending = 0;

wgpu::Buffer MakeBuffer(wgpu::BufferUsage usage, uint64_t size, bool mappedAtCreation = false) {
    wgpu::BufferDescriptor descriptor{
        .usage = usage, .size = size, .mappedAtCreation = mappedAtCreation};
    return device.CreateBuffer(&descriptor);
}

void Submit(wgpu::CommandEncoder encoder) {
    wgpu::CommandBuffer commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);
}

std::string Bytes(const uint8_t* data, size_t size) {
    std::string text;
    for (size_t i = 0; i < size; i++) {
        char byte[4];
        std::snprintf(byte, sizeof(byte), "%02x ", data[i]);
        text += byte;
    }
    return text;
}

// Maps |buffer| for reading and checks that it holds |expected|.
void CheckContents(const char* name, wgpu::Buffer buffer, std::vector<uint8_t> expected) {
    pending++;
    buffer.MapAsync(
        wgpu::MapMode::Read, 0, expected.size(), wgpu::CallbackMode::AllowProcessEvents,
        [name, buffer, expected](wgpu::MapAsyncStatus status, wgpu::StringView message) {
            pending--;
            if (status != wgpu::MapAsyncStatus::Success) {
                Check(name, false, "mapAsync failed: " + ToString(message));
                return;
            }
            const auto* data =
                static_cast<const uint8_t*>(buffer.GetConstMappedRange(0, expected.size()));
            bool same = data != nullptr && std::memcmp(data, expected.data(), expected.size()) == 0;
            Check(name, same,
                  data == nullptr ? "no mapped range"
                                  : "got " + Bytes(data, expected.size()) + "expected " +
                                        Bytes(expected.data(), expected.size()));
            buffer.Unmap();
        });
}

const std::vector<uint8_t> kPattern = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

// Writes with mappedAtCreation, copies through an index buffer, and reads back.
void TestMappedAtCreationThroughIndexBuffer() {
    wgpu::Buffer source = MakeBuffer(wgpu::BufferUsage::CopySrc, kPattern.size(), true);
    std::memcpy(source.GetMappedRange(0, kPattern.size()), kPattern.data(), kPattern.size());
    source.Unmap();
    wgpu::Buffer index = MakeBuffer(
        wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc,
        kPattern.size());
    wgpu::Buffer readback =
        MakeBuffer(wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst, kPattern.size());

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(source, 0, index, 0, kPattern.size());
    encoder.CopyBufferToBuffer(index, 0, readback, 0, kPattern.size());
    Submit(encoder);
    CheckContents("mappedAtCreation data copied through an index buffer", readback, kPattern);
}

// Writes an index buffer with writeBuffer and copies it to a mappable buffer.
void TestWriteBufferToIndexBuffer() {
    wgpu::Buffer index = MakeBuffer(
        wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc,
        kPattern.size());
    device.GetQueue().WriteBuffer(index, 0, kPattern.data(), kPattern.size());
    wgpu::Buffer readback =
        MakeBuffer(wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst, kPattern.size());

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(index, 0, readback, 0, kPattern.size());
    Submit(encoder);
    CheckContents("writeBuffer data copied out of an index buffer", readback, kPattern);
}

// Maps the second half of a buffer for writing: the first half must keep its contents.
void TestMapWritePartially() {
    wgpu::Buffer buffer =
        MakeBuffer(wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc, 16, true);
    std::memset(buffer.GetMappedRange(0, 16), 0x11, 16);
    buffer.Unmap();

    pending++;
    buffer.MapAsync(
        wgpu::MapMode::Write, 8, 8, wgpu::CallbackMode::AllowProcessEvents,
        [buffer](wgpu::MapAsyncStatus status, wgpu::StringView message) {
            pending--;
            if (status != wgpu::MapAsyncStatus::Success) {
                Check("mapAsync(Write) of part of a buffer", false,
                      "mapAsync failed: " + ToString(message));
                return;
            }
            std::memset(buffer.GetMappedRange(8, 8), 0x22, 8);
            buffer.Unmap();

            wgpu::Buffer readback =
                MakeBuffer(wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst, 16);
            wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
            encoder.CopyBufferToBuffer(buffer, 0, readback, 0, 16);
            Submit(encoder);
            std::vector<uint8_t> expected(16, 0x11);
            std::fill(expected.begin() + 8, expected.end(), 0x22);
            CheckContents("mapAsync(Write) of part of a buffer", readback, expected);
        });
}

// Unmaps before a mapAsync completes.
void TestUnmapBeforeMapCompletes() {
    wgpu::Buffer buffer = MakeBuffer(wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst, 16);
    pending++;
    buffer.MapAsync(wgpu::MapMode::Read, 0, 16, wgpu::CallbackMode::AllowProcessEvents,
                    [](wgpu::MapAsyncStatus status, wgpu::StringView) {
                        pending--;
                        Check("unmap before mapAsync completes aborts it",
                              status != wgpu::MapAsyncStatus::Success, "mapAsync succeeded");
                    });
    buffer.Unmap();
}

// Clears an RGBA8 texture and reads it back with copyTextureToBuffer.
void TestTextureReadback() {
    wgpu::TextureDescriptor textureDescriptor{
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc,
        .size = {4, 4},
        .format = wgpu::TextureFormat::RGBA8Unorm,
    };
    wgpu::Texture texture = device.CreateTexture(&textureDescriptor);
    wgpu::RenderPassColorAttachment attachment{
        .view = texture.CreateView(),
        .loadOp = wgpu::LoadOp::Clear,
        .storeOp = wgpu::StoreOp::Store,
        .clearValue = {0.25, 0.5, 0.75, 1.0},
    };
    wgpu::RenderPassDescriptor passDescriptor{.colorAttachmentCount = 1,
                                              .colorAttachments = &attachment};
    constexpr uint32_t kBytesPerRow = 256;
    wgpu::Buffer readback =
        MakeBuffer(wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst, kBytesPerRow * 4);

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.BeginRenderPass(&passDescriptor).End();
    wgpu::TexelCopyTextureInfo source{.texture = texture};
    wgpu::TexelCopyBufferInfo destination{
        .layout = {.bytesPerRow = kBytesPerRow, .rowsPerImage = 4}, .buffer = readback};
    wgpu::Extent3D size{4, 4};
    encoder.CopyTextureToBuffer(&source, &destination, &size);
    Submit(encoder);

    pending++;
    readback.MapAsync(
        wgpu::MapMode::Read, 0, kBytesPerRow * 4, wgpu::CallbackMode::AllowProcessEvents,
        [readback](wgpu::MapAsyncStatus status, wgpu::StringView message) {
            pending--;
            const char* name = "copyTextureToBuffer of a cleared RGBA8 texture";
            if (status != wgpu::MapAsyncStatus::Success) {
                Check(name, false, "mapAsync failed: " + ToString(message));
                return;
            }
            const auto* data = static_cast<const uint8_t*>(
                readback.GetConstMappedRange(0, kBytesPerRow * 4));
            // 0.25, 0.5 and 0.75 round to 64, 128 and 191; allow one step either way.
            const uint8_t expected[] = {64, 128, 191, 255};
            bool close = true;
            for (uint32_t row = 0; row < 4; row++) {
                for (uint32_t x = 0; x < 4; x++) {
                    for (uint32_t c = 0; c < 4; c++) {
                        int value = data[row * kBytesPerRow + x * 4 + c];
                        close = close && value >= expected[c] - 1 && value <= expected[c] + 1;
                    }
                }
            }
            Check(name, close, "got " + Bytes(data, 4) + "for the first texel");
            readback.Unmap();
        });
}

}  // namespace

int main() {
    if (!Setup()) {
        return 1;
    }
    TestMappedAtCreationThroughIndexBuffer();
    TestWriteBufferToIndexBuffer();
    TestMapWritePartially();
    TestUnmapBeforeMapCompletes();
    TestTextureReadback();
    FinishWhen([] { return pending == 0; });
    return 0;
}
